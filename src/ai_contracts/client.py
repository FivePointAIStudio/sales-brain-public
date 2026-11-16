import json
import os
from pathlib import Path
from urllib.parse import urlparse, unquote
import httpx


class ServiceError(RuntimeError):
    def __init__(self, status_code, code, message=None):
        self.status_code = status_code
        self.code = str(code)
        super().__init__(message or self.code)


class ServiceClient:
    """Per-caller allowlisted endpoints. No redirects or implicit mutation retries."""
    def __init__(self, service, *, registry=None):
        self.service = service
        self._registry = registry

    def config(self):
        try:
            registry = self._registry or json.loads(Path(os.environ["SA_SERVICE_REGISTRY_FILE"]).read_text())
            item = registry.get("services", registry)[self.service]
            url = urlparse(item["base_url"])
            local = url.hostname in {"127.0.0.1", "localhost", "::1"}
            if url.scheme != "https" and not (local and url.scheme == "http"):
                raise ValueError("insecure_endpoint")
            if url.username or url.password or url.query or url.fragment:
                raise ValueError("invalid_endpoint")
            if len(item["token"]) < 24:
                raise ValueError("invalid_credential")
            return item
        except (KeyError, ValueError, OSError, TypeError) as exc:
            raise ServiceError(503, "service_not_configured") from exc

    def request(self, method, path, *, params=None, json=None, timeout=15.0, headers=None):
        decoded = unquote(path)
        if not path.startswith("/v1/") or ".." in decoded.split("/") or "?" in path or "#" in path or "\\" in path:
            raise ServiceError(400, "invalid_service_path")
        item = self.config()
        auth_headers = {"Authorization": "Bearer " + item["token"], "Accept": "application/json", "X-Contract-Version": "1.0"}
        if headers:
            for key in ["X-Request-Id", "X-Operator-Subject"]:
                if key in headers:
                    auth_headers[key] = headers[key][:160]
        try:
            with httpx.Client(follow_redirects=False, trust_env=False, timeout=httpx.Timeout(timeout, connect=min(3.0, timeout))) as client:
                with client.stream(method, item["base_url"].rstrip("/") + path, params=params, json=json, headers=auth_headers) as response:
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        data.extend(chunk)
                        if len(data) > 4_000_000:
                            raise ServiceError(502, "service_response_too_large")
                    try:
                        result = __import__("json").loads(data)
                    except ValueError as exc:
                        raise ServiceError(502, "invalid_service_response") from exc
                    if response.status_code >= 300:
                        err = result.get("error", {}) if isinstance(result, dict) else {}
                        raise ServiceError(response.status_code, err.get("code", "upstream_error"))
                    return result
        except httpx.TimeoutException as exc:
            raise ServiceError(504, "service_timeout") from exc
        except httpx.HTTPError as exc:
            raise ServiceError(503, "service_unavailable") from exc

    def get(self, path, *, params=None, timeout=15.0, **kwargs):
        return self.request("GET", path, params=params, timeout=timeout, **kwargs)

    def post(self, path, *, json=None, timeout=15.0, **kwargs):
        return self.request("POST", path, json=json, timeout=timeout, **kwargs)

    def put(self, path, *, json=None, timeout=15.0, **kwargs):
        return self.request("PUT", path, json=json, timeout=timeout, **kwargs)

    def delete(self, path, *, json=None, timeout=15.0, **kwargs):
        return self.request("DELETE", path, json=json, timeout=timeout, **kwargs)
