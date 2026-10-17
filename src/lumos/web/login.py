"""``lumos login``: log in to a web instance with an SSH key (step P14.2, REQ-WEB-9).

The server sends a challenge; OpenSSH signs it (``ssh-keygen -Y sign -n lumos``, which uses the SSH agent
for keys protected by a passphrase); the server checks the signature against the public keys of the user and
answers with a session token, usable as ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["LoginError", "login", "sign_challenge"]


class LoginError(RuntimeError):
    """The login failed (server refusal, missing ssh-keygen, unreadable key)."""


def sign_challenge(challenge: str, key: str | Path, *, namespace: str = "lumos") -> str:
    """Sign ``challenge`` with the private key ``key`` (or its ``.pub`` file with an SSH agent).

    Returns
    -------
    str
        The armored SSHSIG signature written by ``ssh-keygen -Y sign``.
    """
    program = shutil.which("ssh-keygen")
    if program is None:
        raise LoginError("ssh-keygen (OpenSSH) is not installed")
    result = subprocess.run(
        [program, "-Y", "sign", "-n", namespace, "-f", str(Path(key).expanduser())],
        input=challenge.encode(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise LoginError(f"ssh-keygen could not sign: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode()


def login(url: str, user: str, key: str | Path, *, client: Any = None) -> str:
    """Log in to the instance at ``url`` as ``user`` with an SSH key and return the session token.

    Parameters
    ----------
    url
        Base URL of the instance, e.g. ``https://lumos.example.org``.
    user
        User name registered on the instance.
    key
        Private key file (``~/.ssh/id_ed25519``), or its public key when the private key is in the SSH agent.
    client
        HTTP client with ``post`` (``httpx.Client``); by default a new one with TLS verification.
    """
    import httpx

    own = client is None
    http = client if client is not None else httpx.Client(base_url=url, timeout=30)
    try:
        response = http.post(f"{url.rstrip('/')}/api/auth/challenge", json={"username": user})
        if response.status_code != 200:
            raise LoginError(f"the server refused to give a challenge: {response.text}")
        challenge = response.json()
        signature = sign_challenge(challenge["challenge"], key, namespace=challenge["namespace"])
        response = http.post(
            f"{url.rstrip('/')}/api/auth/ssh",
            json={
                "username": user,
                "challenge": challenge["challenge"],
                "signature": signature,
                "bearer": True,
            },
        )
        if response.status_code != 200:
            raise LoginError(f"login refused: {response.json().get('error', response.text)}")
        return str(response.json()["token"])
    finally:
        if own:
            http.close()
