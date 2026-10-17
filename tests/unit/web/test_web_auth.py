"""Tests for the authentication and authorisation of the web instance (step P14.2, REQ-WEB-9, NF-7, ADR-0019).

Requirements: passwords are hashed with Argon2id; an SSH key proves its owner by signing a challenge of the
server (``ssh-keygen -Y sign`` or ``lumos login``); a challenge is single-use and expires; sessions expire;
roles (viewer, simulator, operator) limit what a session may do, whatever the client claims; login attempts
and requests are rate-limited; security events go to an audit journal; TLS is required as soon as the
operation is enabled outside the local machine. Tests of the plan: access refused without authentication,
elevation of role, replay of an SSH challenge.

Independent references: password hashes are made with ``argon2-cffi`` itself, SSH keys and signatures with
OpenSSH's ``ssh-keygen`` (the reference implementation of the SSHSIG format, PROTOCOL.sshsig).
"""

import json
import shutil
import subprocess
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("argon2")

from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from lumos.cli import main
from lumos.hardware.emulators import EmulatorServer, create_emulator
from lumos.service import ServiceConfig
from lumos.web import WebConfig, create_app
from lumos.web.auth import (
    ROLES,
    ConfigError,
    hash_password,
    load_users,
    verify_password,
    verify_ssh_signature,
)

SSH_KEYGEN = shutil.which("ssh-keygen")
needs_ssh = pytest.mark.skipif(SSH_KEYGEN is None, reason="OpenSSH ssh-keygen is not installed")

HOST = "https://lumos.example"
PASSWORDS = {"vera": "viewer password", "simon": "simulator password", "olga": "operator password"}


# --- Keys and users ----------------------------------------------------------------------------------------


def make_key(directory, name, kind="ed25519"):
    path = directory / name
    options = ["-b", "3072"] if kind == "rsa" else []
    subprocess.run(
        [SSH_KEYGEN, "-q", "-t", kind, *options, "-N", "", "-C", name, "-f", str(path)],
        check=True,
        capture_output=True,
    )
    return path


def ssh_sign(key, message, namespace="lumos"):
    data = key.parent / f"{key.name}.message"
    data.write_bytes(message.encode())
    subprocess.run(
        [SSH_KEYGEN, "-Y", "sign", "-n", namespace, "-f", str(key), str(data)],
        check=True,
        capture_output=True,
    )
    return (key.parent / f"{key.name}.message.sig").read_text()


def public(key):
    return (key.parent / f"{key.name}.pub").read_text().strip()


@pytest.fixture
def keys(tmp_path):
    if SSH_KEYGEN is None:
        return {}
    return {"olga": make_key(tmp_path, "olga"), "intruder": make_key(tmp_path, "intruder")}


@pytest.fixture
def users_file(tmp_path, keys):
    hasher = PasswordHasher()
    roles = {"vera": "viewer", "simon": "simulator", "olga": "operator"}
    lines = []
    for name, role in roles.items():
        lines += [f"[users.{name}]", f'role = "{role}"', f'password = "{hasher.hash(PASSWORDS[name])}"']
        if name in keys:
            lines.append(f'ssh_keys = ["{public(keys[name])}"]')
        lines.append("")
    path = tmp_path / "users.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def public_config(users_file, tmp_path, **options):
    service = options.pop("service", ServiceConfig())
    settings = {
        "profile": "public",
        "allowed_hosts": ["lumos.example"],
        "users_file": str(users_file),
        "audit_log": str(tmp_path / "audit.log"),
        "trust_proxy": True,  # TLS terminated by a reverse proxy (HTTPS in these tests)
        "service": service,
    }
    settings.update(options)
    return WebConfig(**settings)


def client_for(config, base_url=HOST):
    return TestClient(create_app(config), base_url=base_url)


def two_clients(config):
    """Two clients (a browser and a script, or a user and an attacker) of the same instance."""
    app = create_app(config)
    return TestClient(app, base_url=HOST), TestClient(app, base_url=HOST)


def login(client, name, password=None):
    return client.post("/api/auth/login", json={"username": name, "password": password or PASSWORDS[name]})


# --- Password hashes and users file ------------------------------------------------------------------------


def test_passwords_are_hashed_with_argon2id():
    hashed = hash_password("correct horse")
    assert hashed.startswith("$argon2id$")
    assert verify_password(hashed, "correct horse")
    assert not verify_password(hashed, "wrong horse")
    assert PasswordHasher().verify(hashed, "correct horse")  # readable by the reference implementation


def test_users_file(users_file):
    users = load_users(users_file)
    assert {name: user.role for name, user in users.items()} == {
        "vera": "viewer",
        "simon": "simulator",
        "olga": "operator",
    }
    assert ROLES == ("viewer", "simulator", "operator")


@pytest.mark.parametrize(
    "content",
    [
        '[users.eve]\nrole = "admin"\n',
        '[users.eve]\nrole = "viewer"\npassword = "plain text"\n',
        '[users.eve]\nrole = "viewer"\ncolour = 1\n',
    ],
)
def test_invalid_users_files_are_refused(tmp_path, content):
    path = tmp_path / "users.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_users(path)


def test_lumos_users_adds_and_lists_users(tmp_path, monkeypatch, capsys):
    import io

    path = tmp_path / "users.toml"
    monkeypatch.setattr("sys.stdin", io.StringIO("a long passphrase\n"))
    assert (
        main(["users", "add", "carol", "--role", "simulator", "--file", str(path), "--password-stdin"]) == 0
    )
    users = load_users(path)
    assert users["carol"].role == "simulator"
    assert verify_password(users["carol"].password, "a long passphrase")
    assert "a long passphrase" not in path.read_text(encoding="utf-8")
    assert main(["users", "list", "--file", str(path)]) == 0
    assert "carol" in capsys.readouterr().out
    assert main(["users", "remove", "carol", "--file", str(path)]) == 0
    assert "carol" not in load_users(path)


# --- Password login and sessions ---------------------------------------------------------------------------


def test_login_with_a_password_gives_the_role_of_the_user_and_a_new_session(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path)) as client:
        assert client.get("/api/auth/me").json() == {"user": None, "role": "simulator"}
        before = client.cookies["lumos_session"]
        response = login(client, "olga")
        assert response.status_code == 200, response.text
        assert response.json()["user"] == "olga"
        assert response.json()["role"] == "operator"
        assert client.cookies["lumos_session"] != before  # no session fixation
        assert client.get("/api/auth/me").json() == {"user": "olga", "role": "operator"}
        assert client.post("/api/auth/logout").status_code == 204
        assert client.get("/api/auth/me").json()["user"] is None


@pytest.mark.parametrize(
    ("name", "password"), [("olga", "wrong"), ("nobody", "operator password"), ("olga", "simulator password")]
)
def test_wrong_credentials_are_refused_without_saying_which_part_is_wrong(
    users_file, tmp_path, name, password
):
    with client_for(public_config(users_file, tmp_path)) as client:
        response = login(client, name, password)
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"
        assert response.json()["error"] == "wrong user name or password"
        assert client.get("/api/auth/me").json()["user"] is None


def test_sessions_expire_after_their_idle_time(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path, session_idle_minutes=0.005)) as client:  # 0.3 s
        login(client, "simon")
        assert client.get("/api/auth/me").json()["user"] == "simon"
        time.sleep(0.6)
        assert client.get("/api/auth/me").json()["user"] is None


def test_authenticated_sessions_expire_after_their_lifetime_even_when_used(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path, session_max_hours=0.6 / 3600)) as client:
        login(client, "simon")
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            me = client.get("/api/auth/me").json()
            if me["user"] is None:
                break
            time.sleep(0.1)
        assert me["user"] is None


def test_a_bearer_token_authenticates_api_clients(users_file, tmp_path):
    browser, script = two_clients(public_config(users_file, tmp_path))
    with browser, script:
        token = browser.post(
            "/api/auth/login", json={"username": "simon", "password": PASSWORDS["simon"], "bearer": True}
        ).json()["token"]
        me = script.get("/api/auth/me", headers={"authorization": f"Bearer {token}"}).json()
        assert me == {"user": "simon", "role": "simulator"}


def test_a_forged_session_cookie_is_an_anonymous_session(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path)) as client:
        client.cookies.set("lumos_session", "forged-token-value")
        assert client.get("/api/auth/me").json() == {"user": None, "role": "simulator"}


# --- Roles --------------------------------------------------------------------------------------------------


def test_access_is_refused_without_authentication_when_the_instance_requires_it(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path, anonymous_role=None)) as client:
        info = client.get("/api/info").json()
        assert info["auth"]["required"] is True
        assert set(info["auth"]["methods"]) == {"password", "ssh"}
        for method, path in [
            ("get", "/api/presets"),
            ("get", "/api/palette"),
            ("post", "/api/contexts"),
            ("get", "/api/contexts"),
        ]:
            response = client.request(method, path)
            assert response.status_code == 401, path
            assert response.json()["code"] == "unauthorized"
            assert "log in" in response.json()["hint"]
        login(client, "vera")
        assert client.get("/api/presets").status_code == 200


def test_a_viewer_reads_but_does_not_edit_or_run(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path, anonymous_role=None)) as client:
        login(client, "vera")
        cid = client.post("/api/presets/first_simulation/load").json()["id"]
        assert client.get(f"/api/contexts/{cid}").status_code == 200
        assert client.get(f"/api/contexts/{cid}/export.lumos").status_code == 200
        element = client.get(f"/api/contexts/{cid}").json()["elements"][0]["id"]
        refused = [
            client.post("/api/contexts", json={}),
            client.delete(f"/api/contexts/{cid}/elements/{element}"),
            client.post(f"/api/contexts/{cid}/runs", json={}),
            client.get(f"/api/contexts/{cid}/ports/{element}.out"),
        ]
        for response in refused:
            assert response.status_code == 403, response.request.url
            assert response.json()["code"] == "forbidden"
            assert "simulator" in response.json()["hint"]


def test_the_client_cannot_raise_its_own_role(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path, anonymous_role="viewer")) as client:
        login(client, "vera")
        client.cookies.set("lumos_role", "operator")
        response = client.post(
            "/api/contexts",
            json={"name": "x", "role": "operator"},
            headers={"x-lumos-role": "operator", "x-lumos-user": "olga"},
        )
        assert response.status_code == 403
        assert client.post("/api/contexts", params={"role": "operator"}, json={}).status_code == 403
        assert client.get("/api/auth/me").json() == {"user": "vera", "role": "viewer"}


@pytest.fixture
def stage():
    server = EmulatorServer(create_emulator("LL-LS100", latency=0.0)).start()
    yield server
    server.stop()
    time.sleep(0.05)


def operation_bench(client, url):
    cid = client.post("/api/contexts", json={"name": "bench", "mode": "operation"}).json()["id"]
    params = {"velocity": "50 mm/s", "acceleration": "500 mm/s2"}
    assert client.post(
        f"/api/contexts/{cid}/elements",
        json={"type_id": "lumos.lab.LinearStage", "name": "stage", "params": params},
    ).is_success
    return cid


def test_only_operators_drive_equipment_and_every_action_is_confirmed(users_file, tmp_path, stage):
    config = public_config(users_file, tmp_path, service=ServiceConfig(operation_enabled=True))
    with client_for(config) as client:
        cid = operation_bench(client, stage.url)  # anonymous: simulator
        assert client.post("/api/devices/discover", json={"tcp": [stage.url]}).status_code == 403
        login(client, "simon")
        cid = operation_bench(client, stage.url)
        assert client.post("/api/devices/discover", json={"tcp": [stage.url]}).status_code == 403
        login(client, "olga")
        cid = operation_bench(client, stage.url)
        assert client.post("/api/devices/discover", json={"tcp": [stage.url]}).status_code == 200
        assert client.post(f"/api/contexts/{cid}/devices/associate", json={}).status_code == 200
        confirmation = client.get(f"/api/contexts/{cid}/elements/stage/confirmation").json()
        assert confirmation["device"]  # the interface names the device in its confirmation dialog
        action = {"element": "stage", "action": "home"}
        unconfirmed = client.post(f"/api/contexts/{cid}/actions", json=action)
        assert unconfirmed.status_code == 403
        done = client.post(f"/api/contexts/{cid}/actions", json={**action, "confirmed": True})
        assert done.status_code == 200, done.text
        assert done.json()["mode"] == "operation"
        devices = client.get(f"/api/contexts/{cid}/devices").json()
        assert devices[0]["element"] == "stage"
        journal = client.get(f"/api/contexts/{cid}/journal", params={"kind": "action"}).json()
        assert journal[-1]["origin"] == "web"


def test_a_simulator_cannot_drive_equipment_even_with_a_confirmation(users_file, tmp_path, stage):
    service = ServiceConfig(operation_enabled=True)
    with client_for(public_config(users_file, tmp_path, service=service)) as client:
        login(client, "simon")
        cid = operation_bench(client, stage.url)
        response = client.post(
            f"/api/contexts/{cid}/actions", json={"element": "stage", "action": "home", "confirmed": True}
        )
        assert response.status_code == 403
        assert "operator" in response.json()["hint"]


def test_the_emergency_stop_is_allowed_to_everyone(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path, anonymous_role="viewer")) as client:
        cid = client.post("/api/presets/first_simulation/load").json()["id"]
        assert client.post(f"/api/contexts/{cid}/stop").status_code == 204


def test_the_local_profile_lets_its_user_operate_with_a_confirmation(stage):
    config = WebConfig(service=ServiceConfig(operation_enabled=True))
    with TestClient(create_app(config), base_url="http://localhost") as client:
        assert client.get("/api/auth/me").json() == {"user": None, "role": "operator"}
        cid = operation_bench(client, stage.url)
        assert client.post("/api/devices/discover", json={"tcp": [stage.url]}).status_code == 200
        assert client.post(f"/api/contexts/{cid}/devices/associate", json={}).status_code == 200
        action = {"element": "stage", "action": "home"}
        assert client.post(f"/api/contexts/{cid}/actions", json=action).status_code == 403
        assert (
            client.post(f"/api/contexts/{cid}/actions", json={**action, "confirmed": True}).status_code == 200
        )


def test_operation_routes_refuse_when_the_operation_is_disabled(users_file, tmp_path, stage):
    with client_for(public_config(users_file, tmp_path)) as client:
        login(client, "olga")
        response = client.post("/api/devices/discover", json={"tcp": [stage.url]})
        assert response.status_code == 403
        assert "disabled" in response.json()["error"]


# --- TLS ----------------------------------------------------------------------------------------------------


def test_logging_in_needs_tls_on_a_public_instance(users_file, tmp_path):
    config = public_config(users_file, tmp_path, service=ServiceConfig(operation_enabled=True))
    with client_for(config, base_url="http://lumos.example") as client:
        response = login(client, "olga")
        assert response.status_code == 403
        assert "HTTPS" in response.json()["error"]
        forwarded = client.post(
            "/api/auth/login",
            json={"username": "olga", "password": PASSWORDS["olga"]},
            headers={"x-forwarded-proto": "https"},
        )
        assert forwarded.status_code == 200  # TLS terminated by the trusted reverse proxy


def test_cookies_are_secure_over_https(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path)) as client:
        response = login(client, "simon")
        assert "secure" in response.headers["set-cookie"].lower()


def test_operation_off_the_local_machine_cannot_be_configured_without_tls_and_users(users_file, tmp_path):
    operation = ServiceConfig(operation_enabled=True)
    with pytest.raises(ConfigError, match="TLS"):
        create_app(WebConfig(profile="public", users_file=str(users_file), service=operation))
    with pytest.raises(ConfigError, match="operator"):
        create_app(WebConfig(profile="public", trust_proxy=True, service=operation))
    with pytest.raises(ConfigError, match="anonymous"):
        create_app(public_config(users_file, tmp_path, anonymous_role="operator", service=operation))


# --- SSH keys -----------------------------------------------------------------------------------------------


@needs_ssh
@pytest.mark.parametrize("kind", ["ed25519", "ecdsa", "rsa"])
def test_ssh_signatures_made_by_openssh_are_verified(tmp_path, kind):
    key = make_key(tmp_path, f"key-{kind}", kind)
    signature = ssh_sign(key, "challenge-text")
    assert verify_ssh_signature(public(key), b"challenge-text", signature, namespace="lumos")
    assert not verify_ssh_signature(public(key), b"other-text", signature, namespace="lumos")
    assert not verify_ssh_signature(public(key), b"challenge-text", signature, namespace="git")
    other = make_key(tmp_path, f"other-{kind}", kind)
    assert not verify_ssh_signature(public(other), b"challenge-text", signature, namespace="lumos")


@needs_ssh
def test_a_signature_for_another_namespace_is_refused(tmp_path):
    key = make_key(tmp_path, "key")
    signature = ssh_sign(key, "challenge-text", namespace="file")
    assert not verify_ssh_signature(public(key), b"challenge-text", signature, namespace="lumos")


@needs_ssh
def test_a_tampered_signature_is_refused(tmp_path):
    key = make_key(tmp_path, "key")
    lines = ssh_sign(key, "challenge-text").splitlines()
    body = list(lines[3])
    body[10] = "A" if body[10] != "A" else "B"
    lines[3] = "".join(body)
    assert not verify_ssh_signature(public(key), b"challenge-text", "\n".join(lines), namespace="lumos")


def ssh_login(client, name, key, challenge=None):
    if challenge is None:
        response = client.post("/api/auth/challenge", json={"username": name})
        assert response.status_code == 200
        challenge = response.json()
        assert challenge["namespace"] == "lumos"
    signature = ssh_sign(key, challenge["challenge"], namespace=challenge["namespace"])
    return challenge, client.post(
        "/api/auth/ssh", json={"username": name, "challenge": challenge["challenge"], "signature": signature}
    )


@needs_ssh
def test_login_with_an_ssh_key(users_file, tmp_path, keys):
    with client_for(public_config(users_file, tmp_path)) as client:
        _, response = ssh_login(client, "olga", keys["olga"])
        assert response.status_code == 200, response.text
        assert client.get("/api/auth/me").json() == {"user": "olga", "role": "operator"}


@needs_ssh
def test_a_challenge_cannot_be_replayed(users_file, tmp_path, keys):
    client, attacker = two_clients(public_config(users_file, tmp_path))
    with client, attacker:
        challenge, first = ssh_login(client, "olga", keys["olga"])
        assert first.status_code == 200
        signature = ssh_sign(keys["olga"], challenge["challenge"])
        replay = attacker.post(
            "/api/auth/ssh",
            json={"username": "olga", "challenge": challenge["challenge"], "signature": signature},
        )
        assert replay.status_code == 401
        assert attacker.get("/api/auth/me").json()["user"] is None


@needs_ssh
def test_a_failed_attempt_consumes_the_challenge(users_file, tmp_path, keys):
    with client_for(public_config(users_file, tmp_path)) as client:
        challenge = client.post("/api/auth/challenge", json={"username": "olga"}).json()
        _, wrong_key = ssh_login(client, "olga", keys["intruder"], challenge)
        assert wrong_key.status_code == 401
        _, right_key = ssh_login(client, "olga", keys["olga"], challenge)
        assert right_key.status_code == 401


@needs_ssh
def test_a_challenge_is_bound_to_its_user_and_expires(users_file, tmp_path, keys):
    with client_for(public_config(users_file, tmp_path, challenge_seconds=0.3)) as client:
        challenge = client.post("/api/auth/challenge", json={"username": "simon"}).json()
        _, other_user = ssh_login(client, "olga", keys["olga"], challenge)
        assert other_user.status_code == 401
        challenge = client.post("/api/auth/challenge", json={"username": "olga"}).json()
        time.sleep(0.5)
        _, late = ssh_login(client, "olga", keys["olga"], challenge)
        assert late.status_code == 401


@needs_ssh
def test_challenges_are_given_for_unknown_users_too(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path)) as client:
        response = client.post("/api/auth/challenge", json={"username": "nobody"})
        assert response.status_code == 200  # the answer does not reveal which users exist


@needs_ssh
def test_lumos_login_signs_the_challenge_with_ssh_keygen(users_file, tmp_path, keys):
    from lumos.web.login import login as ssh_keygen_login

    client, script = two_clients(public_config(users_file, tmp_path))
    with client, script:
        token = ssh_keygen_login(HOST, "olga", keys["olga"], client=client)
        me = script.get("/api/auth/me", headers={"authorization": f"Bearer {token}"}).json()
        assert me == {"user": "olga", "role": "operator"}


# --- Rate limiting and audit --------------------------------------------------------------------------------


def test_login_attempts_are_rate_limited(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path, login_attempts_per_minute=3)) as client:
        for _ in range(3):
            assert login(client, "olga", "wrong").status_code == 401
        blocked = login(client, "olga")
        assert blocked.status_code == 429
        assert blocked.json()["code"] == "rate_limited"
        assert int(blocked.headers["retry-after"]) > 0


def test_requests_are_rate_limited(users_file, tmp_path):
    with client_for(public_config(users_file, tmp_path, requests_per_minute=5)) as client:
        statuses = [client.get("/api/info").status_code for _ in range(8)]
        assert statuses[:5] == [200] * 5
        assert statuses[-1] == 429


def test_security_events_are_audited_without_secrets(users_file, tmp_path):
    config = public_config(users_file, tmp_path, anonymous_role="viewer")
    with client_for(config) as client:
        login(client, "simon", "a wrong password")
        login(client, "vera")
        token = client.cookies["lumos_session"]
        client.post("/api/contexts", json={})
        client.post("/api/auth/logout")
    text = (tmp_path / "audit.log").read_text(encoding="utf-8")
    events = [json.loads(line) for line in text.splitlines()]
    summary = [(e["event"], e["outcome"], e.get("user")) for e in events]
    assert ("login", "failure", "simon") in summary
    assert ("login", "success", "vera") in summary
    assert ("forbidden", "failure", "vera") in summary
    assert ("logout", "success", "vera") in summary
    assert all({"time", "ip", "event", "outcome"} <= set(e) for e in events)
    assert "a wrong password" not in text
    assert PASSWORDS["vera"] not in text
    assert token not in text
