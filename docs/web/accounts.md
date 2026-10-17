# Accounts and security

On your own machine (`lumos web`), there is nothing to set up: the interface only listens on `localhost` and you
have every right. Accounts matter on a shared or public instance.

## Roles

| Role | Can |
|------|-----|
| viewer | browse the elements and the documentation, load presets, inspect contexts, download `.lumos` files and scripts |
| simulator | everything above, plus edit contexts, run them and inspect any port |
| operator | everything above, plus discover devices and act on real equipment (each action confirmed) |

Anyone can press the emergency stop. On a public instance, visitors who did not log in are simulators by
default (`anonymous_role` in the configuration: `viewer`, `simulator`, or none to require a login).

## Managing users

Users are stored in a TOML file, with a hash of their password (Argon2id) and their public SSH keys. Manage it
with `lumos users`:

```console
$ lumos users add alice --role operator --password --file users.toml
password of alice:
again:
$ lumos users add bob --role simulator --ssh-key ~/bob_id_ed25519.pub --file users.toml
$ lumos users list --file users.toml
alice                operator   password
bob                  simulator  ssh
$ lumos users remove bob --file users.toml
```

Start the instance with `lumos web --users users.toml ...` (or `users_file` in its configuration). The file is
read again when it changes: no restart is needed to add a user.

## Logging in

- **With a password**: the *Log in* button of the interface.
- **With an SSH key**: the interface shows a challenge and the command that signs it; paste the signature it
  prints:

  ```console
  $ echo -n "lumos-login:…" | ssh-keygen -Y sign -n lumos -f ~/.ssh/id_ed25519
  ```

- **From a script**: `lumos login` does the same and prints a token to send as `Authorization: Bearer <token>`:

  ```console
  $ lumos login https://lumos.example.org --user bob --key ~/.ssh/id_ed25519
  ```

  A key protected by a passphrase is used through your SSH agent (give the `.pub` file as `--key`).

A challenge is valid for two minutes and only once. Sessions end after a period of inactivity and, once logged
in, after a maximum duration (12 h by default).

## Safeguards

- Passwords are never stored or logged; wrong credentials give the same answer whatever was wrong.
- Login attempts and requests are rate-limited per address; logging in needs HTTPS outside your own machine.
- Security events (logins, refusals, actions on equipment, emergency stops) are written to an audit journal
  (`audit_log`), without passwords or tokens.
- The operation of equipment on a public instance needs an explicit configuration, TLS and registered operators
  (see the hosting guide and ADR-0019).
