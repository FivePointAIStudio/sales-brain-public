"""Direct DeepSeek HTTPS inference; business memory and tool execution remain in this app."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
import socket
import ssl
import stat
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from jsonschema import Draft202012Validator, ValidationError
from .errors import BrainError
from .messages import ModelPrompt
from ..core.diagnostics import schema_failure

ENDPOINT = "https://api.deepseek.com/chat/completions"
MODELS = {"deepseek-flash", "deepseek-v4-pro"}
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_INFLIGHT = threading.BoundedSemaphore(2)
BASE = Path(__file__).resolve().parents[1]


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def strict_json(value):
    def pairs(items):
        output = {}
        for key, item in items:
            if key in output:
                raise ValueError("duplicate_json_key")
            output[key] = item
        return output
    def constant(value):
        raise ValueError("non_finite_json_number")
    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def _private_json(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077 or info.st_size > 16_384:
            raise ValueError("private_config_permissions")
        result = strict_json(stream.read(16_385))
    if not isinstance(result, dict):
        raise ValueError("private_config_shape")
    return result


def _write_private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("private_config_symlink")
    path.parent.chmod(0o700)
    temporary = path.with_name(path.name + ".tmp-" + os.urandom(8).hex())
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class DeepSeekProvider:
    provider_id = "deepseek"
    transport = "https_api"

    def __init__(self, private_dir=None, *, opener=None, clock=time.monotonic):
        self.private_dir = Path(private_dir or os.environ.get("BRAIN_PRIVATE_DIR", str(BASE / ".private-data")))
        self.model = os.environ.get("DEEPSEEK_BRAIN_MODEL", "deepseek-flash")
        self.key = ""
        self.config_error = False
        self.clock = clock
        self.last_usage = {}
        self.last_model = ""
        self.last_elapsed_ms = 0
        self.opener = opener if opener is not None else build_opener(
            ProxyHandler({}), NoRedirectHandler(), HTTPSHandler(context=ssl.create_default_context()))
        try:
            if self.private_dir.exists() and (self.private_dir.is_symlink() or
                    stat.S_IMODE(self.private_dir.stat().st_mode) & 0o077):
                raise ValueError("private_directory_permissions")
            env_key = os.environ.get("DEEPSEEK_BRAIN_API_KEY")
            if env_key is not None:
                self.key = env_key
            else:
                path = self.private_dir / "deepseek-api.json"
                if path.exists() or path.is_symlink():
                    data = _private_json(path)
                    if set(data) - {"api_key", "model"}:
                        raise ValueError("unexpected_config_fields")
                    self.key = data.get("api_key", "")
                    self.model = os.environ.get("DEEPSEEK_BRAIN_MODEL", data.get("model", "deepseek-flash"))
            if not isinstance(self.key, str) or (self.key and (
                    len(self.key) < 8 or len(self.key) > 4096 or any(c.isspace() for c in self.key))):
                raise ValueError("invalid_key_format")
            if self.model not in MODELS:
                raise ValueError("unsupported_model")
        except (OSError, ValueError, TypeError):
            self.key = ""
            if not isinstance(self.model, str) or self.model not in MODELS:
                self.model = "deepseek-flash"
            self.config_error = True

    def fingerprint(self):
        from ..core.contracts import MODEL_STEP_SCHEMA as MODEL_SCHEMA
        source = Path(__file__).read_bytes()
        material = (self.model + "\0" + self.key + json.dumps(MODEL_SCHEMA, sort_keys=True)).encode()
        return hashlib.sha256(source + material).hexdigest()

    def health(self):
        configured = bool(self.key) and not self.config_error
        verified = False
        returned_model = ""
        if configured:
            try:
                gate = _private_json(self.private_dir / "deepseek-api-probe.json")
                verified = gate.get("passed") is True and gate.get("fingerprint") == self.fingerprint()
                if verified and isinstance(gate.get("returned_model"), str):
                    returned_model = gate["returned_model"][:160]
            except (OSError, ValueError, TypeError):
                pass
        reason = ("DeepSeek API配置格式或私有权限不正确" if self.config_error else
                  "尚未配置本项目DeepSeek API Key" if not configured else
                  "已配置，待完成匿名API连通和动作格式验证" if not verified else
                  "DeepSeek HTTPS API已验证；记忆与业务工具由本系统管理")
        return {"ready": verified, "configured": configured, "verified": verified,
                "provider": "deepseek", "model": self.model, "transport": "https_api",
                "last_verified_returned_model": returned_model,
                "label": f"DeepSeek API · {self.model}", "reason": reason}

    def _read_response(self, request, timeout):
        deadline = self.clock() + timeout
        try:
            with self.opener.open(request, timeout=timeout) as response:
                status = response.getcode()
                if status != 200 or response.geturl() != ENDPOINT:
                    raise self._http_error(status)
                chunks, total = [], 0
                while True:
                    if self.clock() >= deadline:
                        raise BrainError("model_timeout", "DeepSeek请求超时，本轮未应用；可查询状态后重试", 504)
                    read = getattr(response, "read1", response.read)
                    chunk = read(min(65_536, MAX_RESPONSE_BYTES + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        raise BrainError("model_invalid_result", "DeepSeek返回内容超过本轮大小限制", 422)
                if self.clock() >= deadline:
                    raise BrainError("model_timeout", "DeepSeek请求超时，本轮未应用", 504)
                return b"".join(chunks)
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise self._http_error(code) from None
        except (TimeoutError, socket.timeout):
            raise BrainError("model_timeout", "DeepSeek请求超时，本轮未应用", 504) from None
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise BrainError("model_timeout", "DeepSeek请求超时，本轮未应用", 504) from None
            raise BrainError("deepseek_unavailable", "DeepSeek连接暂不可用，本轮未应用", 503) from None

    @staticmethod
    def _http_error(status):
        if status in (401, 403):
            return BrainError("deepseek_auth_failed", "DeepSeek API鉴权失败，请核对本项目Key", 503)
        if status == 402:
            return BrainError("deepseek_balance_insufficient", "DeepSeek账户余额不足", 503)
        if status == 429:
            return BrainError("deepseek_rate_limited", "DeepSeek接口限流，请稍后重试原轮次", 503)
        if 300 <= status < 400:
            return BrainError("deepseek_redirect_blocked", "DeepSeek接口返回重定向，未转发请求", 503)
        return BrainError("deepseek_unavailable", "DeepSeek接口暂不可用，本轮未应用", 503)

    def _bounded_request(self, request, timeout):
        if not _INFLIGHT.acquire(blocking=False):
            raise BrainError("deepseek_busy", "已有模型请求在处理，请稍后查询或重试", 503)
        completed = threading.Event()
        outcome = {}
        def worker():
            try:
                outcome["body"] = self._read_response(request, timeout)
            except BrainError as exc:
                outcome["error"] = exc
            except Exception:
                outcome["error"] = BrainError("deepseek_unavailable", "DeepSeek连接或响应异常，本轮未应用", 503)
            finally:
                _INFLIGHT.release()
                completed.set()
        threading.Thread(target=worker, name="deepseek-api-request", daemon=True).start()
        if not completed.wait(timeout):
            raise BrainError("model_timeout", "DeepSeek请求达到本轮期限；未应用结果，请查询状态后重试", 504)
        if "error" in outcome:
            raise outcome["error"]
        return outcome["body"]

    def complete(self, prompt, schema, timeout=90, *, _probe=False):
        self.last_usage = {}
        self.last_model = ""
        self.last_elapsed_ms = 0
        self.last_validation_error = None
        if self.config_error or not self.key:
            raise BrainError("provider_not_configured", self.health()["reason"], 503)
        if not _probe and not self.health()["ready"]:
            raise BrainError("model_not_verified", self.health()["reason"], 503)
        if not isinstance(prompt, str) or len(prompt.encode()) > 120_000:
            raise BrainError("context_budget_exceeded", "本轮上下文超过处理上限", 422)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise BrainError("invalid_timeout", "模型处理期限不正确", 400)
        timeout = min(float(timeout), 90)
        Draft202012Validator.check_schema(schema)
        system = prompt.system_message if isinstance(prompt, ModelPrompt) else ""
        user = prompt.user_message if isinstance(prompt, ModelPrompt) else prompt
        content = {"model": self.model, "messages": [
            {"role": "system", "content": system + "\nReturn exactly one JSON object, no Markdown or tool execution. "
             "The JSON must satisfy this schema: " + json.dumps(schema, ensure_ascii=False)},
            {"role": "user", "content": user}], "stream": False, "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"}, "max_tokens": 8192, "temperature": 0.2}
        request = Request(ENDPOINT, data=json.dumps(content, ensure_ascii=False).encode(), method="POST",
                          headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        started = time.monotonic()
        try:
            body = self._bounded_request(request, timeout)
        finally:
            self.last_elapsed_ms = round((time.monotonic() - started) * 1000)
        try:
            envelope = strict_json(body.decode("utf-8"))
            if not isinstance(envelope, dict) or not isinstance(envelope.get("choices"), list) or len(envelope["choices"]) != 1:
                raise ValueError("invalid_envelope")
            returned = envelope.get("model")
            if isinstance(returned, str) and (returned in MODELS or re.fullmatch(
                    r"deepseek-v\d+(?:\.\d+)?-(?:flash|pro)(?:-\d{4,8})?", returned, re.I)):
                self.last_model = returned
            usage = envelope.get("usage", {})
            if isinstance(usage, dict):
                self.last_usage = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                                                        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
                                   if type(usage.get(k)) is int and usage[k] >= 0}
            choice = envelope["choices"][0]
            message = choice.get("message")
            if choice.get("finish_reason") != "stop" or not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("incomplete_message")
            if message.get("tool_calls") or message.get("function_call") or message.get("refusal"):
                raise ValueError("unexpected_native_action")
            text = message.get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty_message")
            result = strict_json(text)
            if not isinstance(result, dict):
                raise ValueError("object_required")
            Draft202012Validator(schema).validate(result)
        except ValidationError as error:
            stage = 'intent_resolution' if 'intents' in schema.get('properties', {}) else 'model_step'
            self.last_validation_error = schema_failure(schema, error, stage=stage)
            rejected = BrainError("model_invalid_result", "DeepSeek未返回完整且符合合同的JSON，本轮未应用", 422)
            rejected.diagnostics = self.last_validation_error
            raise rejected from None
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeError) as error:
            stage = 'intent_resolution' if 'intents' in schema.get('properties', {}) else 'model_step'
            # Exception messages and JSON decoder documents can contain customer
            # text. Persist only a fixed category and an empty field path.
            self.last_validation_error = {'stage': stage, 'code': 'json_envelope_or_decode_invalid',
                                          'error_type': type(error).__name__, 'field_path': []}
            rejected = BrainError("model_invalid_result", "DeepSeek未返回完整且符合合同的JSON，本轮未应用", 422)
            rejected.diagnostics = self.last_validation_error
            raise rejected from None
        return result


def get_provider(private_dir=None):
    return DeepSeekProvider(private_dir)


def configure_api_key(key, private_dir=None, model="deepseek-flash"):
    if not isinstance(key, str) or len(key) < 8 or len(key) > 4096 or any(c.isspace() for c in key) or model not in MODELS:
        raise BrainError("invalid_api_configuration", "API Key或模型配置不正确", 400)
    root = Path(private_dir or os.environ.get("BRAIN_PRIVATE_DIR", str(BASE / ".private-data")))
    _write_private_json(root / "deepseek-api.json", {"api_key": key, "model": model})


def run_api_probe(private_dir=None):
    from ..core.contracts import MODEL_STEP_SCHEMA as MODEL_SCHEMA
    provider = DeepSeekProvider(private_dir)
    value = provider.complete(
        "Anonymous API contract test. Return step final; no tool_requests. Exactly one NO_ACTION with "
        "reason no_response_needed, proposal_key api-probe, empty depends_on. summary is API contract verified.",
        MODEL_SCHEMA, timeout=60, _probe=True)
    passed = (value["step"] == "final" and not value["tool_requests"] and len(value["actions"]) == 1 and
              value["actions"][0]["type"] == "NO_ACTION")
    if not passed:
        raise BrainError("api_probe_failed", "DeepSeek API动作合同验证未通过", 503)
    proof = {"passed": True, "provider": "deepseek", "model": provider.model, "transport": "https_api",
             "fingerprint": provider.fingerprint(), "checked_at": time.time(), "usage": provider.last_usage,
             "returned_model": provider.last_model, "elapsed_ms": provider.last_elapsed_ms,
             "scope": "anonymous direct-API connectivity and full action JSON contract"}
    _write_private_json(provider.private_dir / "deepseek-api-probe.json", proof)
    return {k: v for k, v in proof.items() if k != "fingerprint"}
