from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlencode, urlparse


@dataclass(frozen=True)
class OAuthTokens:
    access_token: str = field(repr=False)
    token_type: str = "Bearer"
    expires_in: int | None = None
    refresh_token: str | None = field(default=None, repr=False)


class OAuthProvider(Protocol):
    def authorization_url(
        self,
        *,
        state: str,
        redirect_uri: str,
        code_challenge: str,
    ) -> str: ...

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> OAuthTokens: ...


@dataclass(frozen=True)
class PendingAuthorization:
    redirect_uri: str
    code_verifier: str = field(repr=False)


class OAuthFlow:
    """In-memory interface example. Production state storage is caller-owned."""

    def __init__(self, provider: OAuthProvider) -> None:
        self.provider = provider
        self._pending: dict[str, PendingAuthorization] = {}

    @staticmethod
    def _validate_redirect_uri(redirect_uri: str) -> None:
        parsed = urlparse(redirect_uri)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("redirect_uri must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname != "localhost":
            raise ValueError("non-local OAuth callbacks must use HTTPS")

    def begin(self, redirect_uri: str) -> str:
        self._validate_redirect_uri(redirect_uri)
        state = secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        self._pending[state] = PendingAuthorization(redirect_uri, verifier)
        return self.provider.authorization_url(
            state=state,
            redirect_uri=redirect_uri,
            code_challenge=challenge,
        )

    def finish(self, *, state: str, code: str) -> OAuthTokens:
        pending = self._pending.pop(state, None)
        if pending is None:
            raise ValueError("unknown or already consumed OAuth state")
        if not code:
            raise ValueError("authorization code is required")
        return self.provider.exchange_code(
            code=code,
            redirect_uri=pending.redirect_uri,
            code_verifier=pending.code_verifier,
        )


class ExampleInvalidProvider:
    """Non-network example showing parameter shape; intentionally unusable."""

    authorization_endpoint = "https://identity.example.invalid/authorize"

    def authorization_url(
        self,
        *,
        state: str,
        redirect_uri: str,
        code_challenge: str,
    ) -> str:
        return self.authorization_endpoint + "?" + urlencode(
            {
                "response_type": "code",
                "client_id": "YOUR_CLIENT_ID_HERE",
                "redirect_uri": redirect_uri,
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
        )

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> OAuthTokens:
        raise RuntimeError("example.invalid providers never exchange credentials")
