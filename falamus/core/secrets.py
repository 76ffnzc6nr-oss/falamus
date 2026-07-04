"""Provider-generic API-key storage (at-rest obfuscation encryption, AES-256-GCM).

GOAL (narrow, honestly stated): keep the plaintext api-key from being SEEN or pattern-scanned — `cat`,
a shoulder-surfer, logs/screen-share, or an infostealer grepping for `sk-…`. It does NOT try to stop an
attacker with full local access (root, a whole-disk/SD-card image) who deliberately reconstructs it — the
token and the ciphertext live on the same machine, so that is out of scope by design.

Crypto: AES-256-GCM (authenticated encryption) via `cryptography`. A random 256-bit key (the "token") is
generated once; each api-key is encrypted with a fresh random 96-bit nonce; the stored blob is
base64(nonce ‖ ciphertext+tag). GCM's tag makes tampering detectable (a corrupt blob decrypts to None).

Layout (no obvious "secrets" folder):
  - CIPHERTEXT: inline in `config.ini` under a `[keys]` section (`<provider> = <blob>`). It's ciphertext,
    so it's safe there — not a greppable plaintext key. config.ini keeps its normal perms (pointless to
    restrict; the ciphertext is useless without the token).
  - TOKEN: the 256-bit key in `~/.falamus/token` (0600) — reuses the existing hidden `~/.falamus/` dir,
    kept in a DIFFERENT directory from the ciphertext. THIS is the secret boundary.

The decrypted key is only ever returned into memory (callers put it in an HTTP header) — never written to
`os.environ`, so shell subprocesses (run_command / the persistent shell) don't inherit it.

`cryptography` is an OPTIONAL dependency (the `[cloud]` extra); everything fails cleanly with an
actionable message when it isn't installed — the pure-local base never imports it.
"""

from __future__ import annotations

import base64
import configparser
import os
from pathlib import Path

from falamus.settings import CONFIG_PATH

_TOKEN_FILE = Path("~/.falamus/token").expanduser()   # 256-bit key, hidden dir (reuses ~/.falamus/)
_CONFIG_FILE = CONFIG_PATH                             # ciphertext lives inline here, in a [keys] section
_KEYS_SECTION = "keys"
_NONCE = 12                                            # AES-GCM nonce length (96 bits, the standard)


class SecretsUnavailable(RuntimeError):
    """Raised when the crypto backend (the `[cloud]` extra) isn't installed."""


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:  # the [cloud] extra isn't installed
        raise SecretsUnavailable(
            "cloud API-key storage needs the 'cryptography' package — install the cloud extra: "
            "pip install 'falamus[cloud]'"
        ) from e
    return AESGCM


def _chmod600(path) -> None:
    """Restrict to the owner. POSIX: 0600. Windows: best-effort (chmod is limited); not fatal."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _load_or_make_token() -> bytes:
    aesgcm = _aesgcm()
    if _TOKEN_FILE.exists():
        return _TOKEN_FILE.read_bytes()
    _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(_TOKEN_FILE.parent, 0o700)
    except OSError:
        pass
    token = aesgcm.generate_key(bit_length=256)   # 32 random bytes
    _TOKEN_FILE.write_bytes(token)
    _chmod600(_TOKEN_FILE)
    return token


def _read_config() -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    if _CONFIG_FILE.exists():
        try:
            cp.read(_CONFIG_FILE, encoding="utf-8")
        except configparser.Error:
            pass
    return cp


def _write_config(cp: configparser.ConfigParser) -> None:
    # preserves the sections read in (Config.save() likewise preserves [keys] + restores its header
    # comments next time it writes). config.ini keeps its normal perms — the ciphertext is useless
    # without the token, so restricting config.ini adds nothing; the SECRET boundary is the 0600 token.
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _CONFIG_FILE.open("w", encoding="utf-8") as f:
        cp.write(f)


def save_api_key(provider: str, key: str) -> None:
    """Encrypt `key` for `provider` (AES-256-GCM) and store the ciphertext in config.ini `[keys]`. ONE
    entry per provider — re-saving OVERWRITES it. Raises SecretsUnavailable without the extra."""
    aesgcm = _aesgcm()
    token = _load_or_make_token()
    nonce = os.urandom(_NONCE)
    ct = aesgcm(token).encrypt(nonce, key.strip().encode("utf-8"), None)
    blob = base64.urlsafe_b64encode(nonce + ct).decode("ascii")
    cp = _read_config()
    if not cp.has_section(_KEYS_SECTION):
        cp.add_section(_KEYS_SECTION)
    cp.set(_KEYS_SECTION, provider, blob)
    _write_config(cp)


def load_api_key(provider: str) -> str | None:
    """Decrypt and return the stored key for `provider`, or None. In-memory only (never env). Raises
    SecretsUnavailable without the extra; returns None on a missing/corrupt/tampered entry or missing token."""
    blob = _read_config().get(_KEYS_SECTION, provider, fallback=None)
    if not blob or not _TOKEN_FILE.exists():
        return None
    try:
        aesgcm = _aesgcm()
        token = _TOKEN_FILE.read_bytes()
        raw = base64.urlsafe_b64decode(blob.encode("ascii"))
        return aesgcm(token).decrypt(raw[:_NONCE], raw[_NONCE:], None).decode("utf-8")
    except Exception:  # noqa: BLE001 — corrupt/tampered/rotated → "not set", caller re-prompts
        return None


def has_api_key(provider: str) -> bool:
    """Whether a ciphertext entry + token exist (does not decrypt / need the extra)."""
    return _read_config().has_option(_KEYS_SECTION, provider) and _TOKEN_FILE.exists()


def stored_providers() -> list[str]:
    """Canonical ids that currently have a stored key."""
    cp = _read_config()
    return sorted(cp.options(_KEYS_SECTION)) if cp.has_section(_KEYS_SECTION) else []


def delete_api_key(provider: str) -> bool:
    """Remove a stored key. Returns True if one was there."""
    cp = _read_config()
    if cp.has_option(_KEYS_SECTION, provider):
        cp.remove_option(_KEYS_SECTION, provider)
        _write_config(cp)
        return True
    return False


# NOTE: there is deliberately NO function to display/return a key for the user (write-only from the user's
# view — "只進不出"). load_api_key() exists only for the client to put the key in an auth header at send
# time; nothing surfaces it to the UI. has_api_key() reports only WHETHER one is set, never its value.
