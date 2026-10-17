"""Users, passwords and SSH keys of a web instance (step P14.2, REQ-WEB-9, ADR-0019).

- Passwords are stored as Argon2id hashes (``argon2-cffi``), never in clear.
- An SSH key proves its owner by signing a challenge of the server in the SSHSIG format of OpenSSH
  (``ssh-keygen -Y sign -n lumos``), verified here with ``cryptography``. Ed25519, ECDSA (NIST P-256, P-384,
  P-521) and RSA keys (signatures ``rsa-sha2-256`` and ``rsa-sha2-512``) are accepted.
- Users are listed in a TOML file managed with ``lumos users``::

      [users.alice]
      role = "operator"
      password = "$argon2id$v=19$m=65536,t=3,p=4$..."
      ssh_keys = ["ssh-ed25519 AAAAC3Nza... alice@laptop"]
"""

from __future__ import annotations

import base64
import hashlib
import struct
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from lumos.web.config import ConfigError

__all__ = [
    "ROLES",
    "ConfigError",
    "User",
    "hash_password",
    "load_users",
    "role_allows",
    "save_users",
    "verify_password",
    "verify_ssh_signature",
]

#: Roles, from the least to the most privileged.
ROLES: tuple[str, ...] = ("viewer", "simulator", "operator")

#: Namespace of the SSH signatures of LUMOS (``ssh-keygen -Y sign -n lumos``).
NAMESPACE = "lumos"

_DUMMY_HASH: list[str] = []


def role_allows(role: str | None, needed: str) -> bool:
    """Whether ``role`` has the rights of ``needed`` (each role includes the rights of the previous ones)."""
    return role in ROLES and ROLES.index(role) >= ROLES.index(needed)


# --- Passwords ----------------------------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return the Argon2id hash of ``password`` (``argon2-cffi`` parameters, RFC 9106 low-memory profile)."""
    from argon2 import PasswordHasher

    return PasswordHasher().hash(password)


def verify_password(hashed: str, password: str) -> bool:
    """Whether ``password`` matches the Argon2id hash ``hashed``."""
    from argon2 import PasswordHasher
    from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

    try:
        return PasswordHasher().verify(hashed, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def dummy_verify(password: str) -> None:
    """Spend the time of a verification, so that unknown users answer as slowly as known ones."""
    if not _DUMMY_HASH:
        _DUMMY_HASH.append(hash_password("not a password of any user"))
    verify_password(_DUMMY_HASH[0], password)


# --- Users --------------------------------------------------------------------------------------------------


@dataclass
class User:
    """A registered user.

    Attributes
    ----------
    name
        User name.
    role
        ``"viewer"``, ``"simulator"`` or ``"operator"``.
    password
        Argon2id hash of the password, if the user logs in with one.
    ssh_keys
        Public keys (lines of ``id_*.pub``) the user logs in with.
    """

    name: str
    role: str
    password: str | None = None
    ssh_keys: list[str] = field(default_factory=list)


def load_users(path: str | Path) -> dict[str, User]:
    """Read a users file.

    Raises
    ------
    ConfigError
        Missing or invalid file, unknown role or setting, password not given as an Argon2id hash.
    """
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"no users file at {path} (create it with lumos users add)") from None
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read the users file {path}: {error}") from error
    users = {}
    for name, entry in data.get("users", {}).items():
        unknown = set(entry) - {"role", "password", "ssh_keys"}
        if unknown:
            raise ConfigError(f"user {name}: unknown settings {', '.join(sorted(unknown))}")
        role = entry.get("role")
        if role not in ROLES:
            raise ConfigError(f"user {name}: role must be one of {', '.join(ROLES)}, not {role!r}")
        password = entry.get("password")
        if password is not None and not str(password).startswith("$argon2id$"):
            raise ConfigError(f"user {name}: the password must be an Argon2id hash (use lumos users add)")
        keys = entry.get("ssh_keys", [])
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ConfigError(f"user {name}: ssh_keys must be a list of public key lines")
        for key in keys:
            _key_blob(key)  # checks the format
        users[name] = User(name=name, role=role, password=password, ssh_keys=list(keys))
    return users


def _toml_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def save_users(path: str | Path, users: dict[str, User]) -> None:
    """Write a users file."""
    lines = ["# Users of a LUMOS web instance (lumos users add/remove/list).", ""]
    for user in sorted(users.values(), key=lambda u: u.name):
        lines.append(f"[users.{_toml_string(user.name)}]")
        lines.append(f"role = {_toml_string(user.role)}")
        if user.password:
            lines.append(f"password = {_toml_string(user.password)}")
        if user.ssh_keys:
            keys = ", ".join(_toml_string(k) for k in user.ssh_keys)
            lines.append(f"ssh_keys = [{keys}]")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# --- SSH signatures (OpenSSH PROTOCOL.sshsig) ---------------------------------------------------------------


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def raw(self, size: int) -> bytes:
        if self.offset + size > len(self.data):
            raise ValueError("truncated SSH data")
        chunk = self.data[self.offset : self.offset + size]
        self.offset += size
        return chunk

    def uint32(self) -> int:
        return int(struct.unpack(">I", self.raw(4))[0])

    def string(self) -> bytes:
        return self.raw(self.uint32())

    def mpint(self) -> int:
        return int.from_bytes(self.string(), "big", signed=True)

    def done(self) -> bool:
        return self.offset == len(self.data)


def _string(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def _key_blob(line: str) -> bytes:
    parts = line.split()
    if len(parts) < 2:
        raise ConfigError(f"not an SSH public key line: {line[:40]!r}")
    try:
        blob = base64.b64decode(parts[1], validate=True)
        kind = _Reader(blob).string().decode()
    except (ValueError, UnicodeDecodeError) as error:
        raise ConfigError(f"not an SSH public key line: {line[:40]!r}") from error
    if kind != parts[0]:
        raise ConfigError(f"SSH public key of type {kind} labelled {parts[0]}")
    return blob


def _armored(signature: str) -> bytes:
    lines = [line.strip() for line in signature.strip().splitlines()]
    if not lines or lines[0] != "-----BEGIN SSH SIGNATURE-----" or lines[-1] != "-----END SSH SIGNATURE-----":
        raise ValueError("not an SSH signature (ssh-keygen -Y sign)")
    return base64.b64decode("".join(lines[1:-1]), validate=True)


def verify_ssh_signature(
    public_key: str, message: bytes, signature: str, *, namespace: str = NAMESPACE
) -> bool:
    """Whether ``signature`` (SSHSIG, from ``ssh-keygen -Y sign``) signs ``message`` with ``public_key``.

    Parameters
    ----------
    public_key
        Line of an OpenSSH public key file (``ssh-ed25519 AAAA... comment``).
    message
        The signed bytes (the challenge of the server).
    signature
        Armored signature (``-----BEGIN SSH SIGNATURE-----`` ...).
    namespace
        Namespace the signature must have been made for (``-n`` of ``ssh-keygen``).
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

    try:
        expected_blob = _key_blob(public_key)
        reader = _Reader(_armored(signature))
        if reader.raw(6) != b"SSHSIG" or reader.uint32() != 1:
            return False
        key_blob = reader.string()
        signed_namespace = reader.string()
        reserved = reader.string()
        hash_name = reader.string()
        signature_blob = reader.string()
        if not reader.done() or key_blob != expected_blob or signed_namespace != namespace.encode():
            return False
        digests = {b"sha256": hashlib.sha256, b"sha512": hashlib.sha512}
        if hash_name not in digests:
            return False
        signed = (
            b"SSHSIG"
            + _string(signed_namespace)
            + _string(reserved)
            + _string(hash_name)
            + _string(digests[hash_name](message).digest())
        )
        inner = _Reader(signature_blob)
        algorithm = inner.string()
        raw = inner.string()
        key = serialization.load_ssh_public_key(public_key.encode())
        if isinstance(key, ed25519.Ed25519PublicKey) and algorithm == b"ssh-ed25519":
            key.verify(raw, signed)
        elif isinstance(key, ec.EllipticCurvePublicKey) and algorithm.startswith(b"ecdsa-sha2-"):
            curve_hash = {256: hashes.SHA256(), 384: hashes.SHA384(), 521: hashes.SHA512()}[
                key.curve.key_size
            ]
            numbers = _Reader(raw)
            r, s = numbers.mpint(), numbers.mpint()
            key.verify(encode_dss_signature(r, s), signed, ec.ECDSA(curve_hash))
        elif isinstance(key, rsa.RSAPublicKey) and algorithm in (b"rsa-sha2-256", b"rsa-sha2-512"):
            digest = hashes.SHA256() if algorithm == b"rsa-sha2-256" else hashes.SHA512()
            key.verify(raw, signed, padding.PKCS1v15(), digest)
        else:
            return False  # SHA-1 RSA signatures, security keys and other algorithms are not accepted
    except (InvalidSignature, ValueError, KeyError, ConfigError, TypeError):
        return False
    return True
