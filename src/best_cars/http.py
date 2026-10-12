"""Bounded, cached HTTP access with offline operation and explicit failures."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class CachedClient:
    """HTTP transport with checksummed local responses and bounded retries.

    Attributes:
        cache_dir: Folder for downloaded responses, outside the repository.
        offline: Forbid all network activity and require a verified cached response.
        refresh: Fetch again even if a verified cache response exists.
        attempts: Maximum attempts per request.
        max_bytes: Maximum accepted response size.
        session: Injectable requests session for tests.
        last_acquired_at: Acquisition time of the last response, including cached reads.
    """

    cache_dir: Path
    offline: bool = False
    refresh: bool = False
    attempts: int = 3
    max_bytes: int = 64 * 1024 * 1024
    session: requests.Session = field(default_factory=requests.Session, repr=False)
    last_acquired_at: str | None = field(default=None, init=False)

    def request_bytes(self, *, url: str, query: dict | None = None) -> bytes:
        """Return a verified cache entry or a bounded successful HTTP response.

        Args:
            url: Public HTTPS data endpoint.
            query: Optional public PxWeb POST body; never credentials.

        Returns:
            Downloaded or cache-verified bytes.

        Raises:
            ValueError: For invalid configuration or oversized/empty responses.
            OSError: For corrupt/unavailable offline caches or network failure.
        """
        if not url.startswith("https://") or self.attempts < 1 or self.max_bytes <= 0:
            raise ValueError(
                "Require HTTPS, positive attempts and a positive byte limit."
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        identity = json.dumps(obj={"url": url, "query": query}, sort_keys=True).encode()
        key = hashlib.sha256(identity).hexdigest()
        path = self.cache_dir / f"{key}.bin"
        digest_path = self.cache_dir / f"{key}.sha256"
        acquired_path = self.cache_dir / f"{key}.acquired_at"
        if (
            path.exists()
            and digest_path.exists()
            and (self.offline or not self.refresh)
        ):
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest_path.read_text().strip():
                raise OSError(f"Cached response checksum failed: {path.name}")
            if not data or len(data) > self.max_bytes:
                raise OSError("Cached response exceeds its size limit or is empty.")
            LOGGER.info("Using cached data from %s", url.split("?")[0])
            self.last_acquired_at = (
                acquired_path.read_text().strip()
                if acquired_path.exists()
                else datetime.fromtimestamp(
                    timestamp=path.stat().st_mtime, tz=UTC
                ).isoformat(timespec="seconds")
            )
            return data
        if self.offline:
            raise OSError(f"No verified cached response for {url.split('?')[0]}")
        last_error = None
        for attempt in range(self.attempts):
            try:
                with self.session.request(
                    method="POST" if query is not None else "GET",
                    url=url,
                    json=query,
                    timeout=(10, 60),
                    stream=True,
                    headers={
                        "User-Agent": "best-cars/2.0 (public reliability data research)"
                    },
                ) as response:
                    response.raise_for_status()
                    chunks = []
                    size = 0
                    for chunk in response.iter_content(chunk_size=65536):
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ValueError(
                                "Data response exceeds the configured byte limit."
                            )
                        chunks.append(chunk)
                    data = b"".join(chunks)
                if not data:
                    raise ValueError("Source returned an empty data response.")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(target=path)
                digest_path.write_text(data=hashlib.sha256(data).hexdigest() + "\n")
                self.last_acquired_at = datetime.now(tz=UTC).isoformat(
                    timespec="seconds"
                )
                acquired_path.write_text(data=self.last_acquired_at + "\n")
                LOGGER.info("Cached %s bytes from %s", len(data), url.split("?")[0])
                return data
            except requests.RequestException as error:
                last_error = error
                LOGGER.warning(
                    "Request attempt %s/%s failed: %s",
                    attempt + 1,
                    self.attempts,
                    type(error).__name__,
                )
                if attempt + 1 < self.attempts:
                    time.sleep(min(2**attempt, 4))
        raise OSError(
            f"Data request failed after {self.attempts} attempts: {url.split('?')[0]}"
        ) from last_error

    def request_json(self, *, url: str, query: dict | None = None) -> dict:
        """Decode a cached or downloaded JSON mapping.

        Args:
            url: Public source endpoint.
            query: Optional PxWeb selection body.

        Returns:
            A JSON object.

        Raises:
            ValueError: If the source returns malformed JSON or a non-object.
        """
        data = json.loads(s=self.request_bytes(url=url, query=query))
        if not isinstance(data, dict):
            raise ValueError("Data endpoint did not return a JSON object.")
        return data
