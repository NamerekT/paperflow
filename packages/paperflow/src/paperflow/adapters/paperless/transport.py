"""HTTP, authentication, API versioning and pagination.

The single place that talks to the network. Everything above it works in terms
of `request` and `paginate` and never builds a URL or a header itself.
"""

import logging
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from paperflow.errors import PaperlessError, PaperlessUnavailable
from paperflow.settings import PaperlessSettings

logger = logging.getLogger(__name__)


class PaperlessTransport:
    def __init__(self, settings: PaperlessSettings, *, session: requests.Session | None = None):
        self.settings = settings
        self.base_url = settings.url.rstrip("/")
        # A Session rather than bare module-level calls: there used to be
        # fifteen of those, each re-doing the TCP and TLS handshake against a
        # host reached over a tunnel. Walking the document list paid that cost
        # once per page.
        self._session = session or requests.Session()
        self._session.headers["Authorization"] = f"Token {settings.token}"

        # Determined at runtime, because instances support different ranges and
        # asking for a version that is not supported answers 406 Not
        # Acceptable. From API v7 a select field's value is the option's id, a
        # string; before v7 it is the option's index, a number, and the options
        # themselves are bare strings rather than {id, label} objects.
        self.api_version = 1

    # ── low level ─────────────────────────────────────────────────────────

    def url_for(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def headers(self, *, versioned: bool = False) -> dict[str, str]:
        if not versioned:
            return {}
        return {"Accept": f"application/json; version={self.api_version}"}

    def fix_next_url(self, next_url: str | None) -> str | None:
        """Rebuild a pagination link onto the host we actually use.

        Paperless builds absolute "next" links from what Django believes about
        its own scheme and host. Behind a TLS-terminating proxy that it has not
        been told to trust, Django sees a plain HTTP request from the proxy and
        emits `http://...:80` - even though the request that produced the
        response arrived over HTTPS. The client then dutifully tries port 80,
        where nothing is listening, and reports a connection refused that looks
        exactly like the tunnel having dropped.

        So only the path and query of the server's link are trusted; the scheme
        and host come from our own base URL.
        """
        if not next_url:
            return None
        base = urlsplit(self.base_url)
        parts = urlsplit(next_url)
        return urlunsplit((base.scheme, base.netloc, parts.path, parts.query, parts.fragment))

    def request(
        self,
        method: str,
        path: str,
        *,
        versioned: bool = False,
        timeout: float | None = None,
        absolute_url: str | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        url = absolute_url or self.url_for(path)
        headers = {**self.headers(versioned=versioned), **kwargs.pop("headers", {})}
        try:
            response = self._session.request(
                method,
                url,
                headers=headers,
                timeout=timeout or self.settings.timeout_default,
                **kwargs,
            )
        except requests.exceptions.ConnectionError as exc:
            raise PaperlessUnavailable(f"cannot reach Paperless at {url}: {exc}") from exc
        except requests.exceptions.Timeout as exc:
            raise PaperlessUnavailable(f"Paperless did not answer {method} {url} in time") from exc

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise PaperlessError(
                f"{method} {url} -> HTTP {response.status_code}: {response.text[:300]}"
            ) from exc
        return response

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("PATCH", path, **kwargs)

    # ── pagination ────────────────────────────────────────────────────────

    def paginate(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        versioned: bool = True,
        timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield every result across every page.

        The four hand-written `while url:` loops this replaces each applied the
        link rewriting above - or, in one case, did not paginate at all and
        silently returned the first page.

        `page_size` is requested large so the common case is a single round
        trip; following `next` remains the correctness guarantee rather than an
        optimisation, because a page size is a request, not a promise.
        """
        query = {"page_size": self.settings.page_size, **(params or {})}
        url: str | None = self.url_for(path)
        first = True

        while url:
            response = self.request(
                "GET",
                path,
                absolute_url=url,
                versioned=versioned,
                timeout=timeout or self.settings.timeout_long,
                params=query if first else None,
            )
            first = False
            data = response.json()
            if isinstance(data, list):
                # Some endpoints answer with a bare list on older versions.
                yield from data
                return
            yield from data.get("results", [])
            url = self.fix_next_url(data.get("next"))

    # ── startup ───────────────────────────────────────────────────────────

    def detect_api_version(self) -> int:
        """Ask the server which API version it supports.

        Any authenticated endpoint reports it in the X-Api-Version response
        header. An instance old enough not to send one is treated as v1.
        """
        response = self.get("/api/", timeout=self.settings.timeout_default)
        version = int(response.headers.get("X-Api-Version", "1"))
        server_version = response.headers.get("X-Version", "?")
        logger.info(f"Paperless server {server_version}, API version {version}")
        self.api_version = version
        return version

    def document_url(self, doc_id: int) -> str:
        return f"{self.base_url}/documents/{doc_id}/details"

    def close(self) -> None:
        self._session.close()
