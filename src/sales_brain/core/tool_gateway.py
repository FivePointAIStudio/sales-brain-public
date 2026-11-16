"""Read-only tool gateway with injected transports; no real HTTP/MCP setup.

Adapters are async and must honor cancellation and their own network/DB timeout.
Python cannot forcibly stop arbitrary blocking work. A real DB adapter must set
statement_timeout; HTTP/MCP implementations must pass the deadline downstream.
Model arguments never contain credentials, a URL, a SQL statement or scope.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import inspect
import math
import re
import time
from typing import Any, Awaitable, Callable, Mapping, Protocol

from .contracts import ContractError, REF, REFS, TIMESTAMP, canonical, check_schema, obj, validate_json


class ScopeError(ValueError):
    pass


@dataclass(frozen=True)
class RequestScope:
    """Created by authentication/caller registration, never by model/request body."""
    tenant_ref: str
    caller_ref: str
    environment: str
    allowed_customer_refs: frozenset[str]
    allowed_regions: frozenset[str]
    permissions: frozenset[str]

    def __post_init__(self):
        if not self.tenant_ref or not self.caller_ref or self.environment not in {"sandbox", "shadow", "live"}:
            raise ScopeError("invalid trusted scope")
        for name in ("allowed_customer_refs", "allowed_regions", "permissions"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not all(isinstance(v, str) and v for v in value):
                raise ScopeError("scope grants must be explicit string collections")
            object.__setattr__(self, name, frozenset(value))
        if not self.allowed_customer_refs or not self.allowed_regions:
            raise ScopeError("customer and region grants cannot be empty")

    def require(self, context: "ToolContext", permission: str | None = None) -> None:
        if context.tenant_ref != self.tenant_ref or context.environment != self.environment:
            raise ScopeError("tenant/environment mismatch")
        if context.customer_ref not in self.allowed_customer_refs or context.region not in self.allowed_regions:
            raise ScopeError("customer/region outside grant")
        if permission and permission not in self.permissions:
            raise ScopeError("tool permission missing")


@dataclass(frozen=True)
class ToolContext:
    tenant_ref: str
    customer_ref: str
    region: str
    environment: str
    conversation_ref: str
    context_version: str
    # Optional adapter-side handles; excluded from repr/model-facing descriptions.
    private: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def public_scope(self) -> dict:
        return {k: getattr(self, k) for k in (
            "tenant_ref", "customer_ref", "region", "environment", "conversation_ref", "context_version")}


RESULT_STATUSES = ["ok", "not_found", "partial", "stale", "conflict", "unavailable",
                   "forbidden", "invalid_request", "timeout", "error"]
TOOL_RESULT_SCHEMA = obj({
    "status": {"enum": RESULT_STATUSES}, "data": {"type": "object"}, "source_refs": REFS,
    "code": {"anyOf": [REF, {"type": "null"}]}, "retryable": {"type": "boolean"},
    "as_of": {"anyOf": [TIMESTAMP, {"type": "null"}]}, "truncated": {"type": "boolean"},
})


@dataclass(frozen=True)
class ToolResult:
    status: str
    data: dict = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    code: str | None = None
    retryable: bool = False
    as_of: str | None = None
    truncated: bool = False

    def to_dict(self) -> dict:
        return {"status": self.status, "data": deepcopy(self.data), "source_refs": list(self.source_refs),
                "code": self.code, "retryable": self.retryable, "as_of": self.as_of, "truncated": self.truncated}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ToolResult":
        raw = deepcopy(dict(value))
        validate_json(raw, TOOL_RESULT_SCHEMA)
        raw["source_refs"] = tuple(raw["source_refs"])
        return cls(**raw)


class ReadAdapter(Protocol):
    async def invoke(self, arguments: dict, context: ToolContext, timeout: float) -> ToolResult: ...


@dataclass(frozen=True)
class LocalReadAdapter:
    """An async Python function/DB service. Blocking functions are not accepted."""
    handler: Callable[[dict, ToolContext, float], Awaitable[ToolResult]]

    async def invoke(self, arguments: dict, context: ToolContext, timeout: float) -> ToolResult:
        if not (inspect.iscoroutinefunction(self.handler) or inspect.iscoroutinefunction(getattr(self.handler, "__call__", None))):
            raise TypeError("local read handlers must be cancellation-cooperative async callables")
        return await self.handler(arguments, context, timeout)


class HttpReadTransport(Protocol):
    """Application port, not the signature of a particular HTTP library.

    The implementation maps service_ref/operation to allowlisted endpoints and
    private credentials. It owns TLS, retries for safe reads and response decoding.
    """
    async def read(self, *, service_ref: str, operation: str, arguments: dict,
                   trusted_scope: dict, timeout: float) -> Mapping[str, Any]: ...


class McpReadTransport(Protocol):
    """Application port over a configured MCP client, not a fake SDK implementation.

    Discover/register an approved read-tool subset at deployment. Map
    trusted_scope to server-side session identity/metadata, never model headers.
    Decode the server's structuredContent into TOOL_RESULT_SCHEMA explicitly.
    """
    async def call_tool(self, *, server_ref: str, tool_name: str, arguments: dict,
                        trusted_scope: dict, timeout: float) -> Mapping[str, Any]: ...


ScopeMapper = Callable[[ToolContext], dict]


def default_scope_mapping(context: ToolContext) -> dict:
    return context.public_scope()


@dataclass(frozen=True)
class HttpReadAdapter:
    transport: HttpReadTransport
    service_ref: str
    operation: str
    scope_mapper: ScopeMapper = default_scope_mapping

    async def invoke(self, arguments: dict, context: ToolContext, timeout: float) -> ToolResult:
        raw = await self.transport.read(service_ref=self.service_ref, operation=self.operation,
                                        arguments=deepcopy(arguments),
                                        trusted_scope=deepcopy(self.scope_mapper(context)), timeout=timeout)
        return ToolResult.from_mapping(raw)


@dataclass(frozen=True)
class McpReadAdapter:
    transport: McpReadTransport
    server_ref: str
    tool_name: str
    scope_mapper: ScopeMapper = default_scope_mapping

    async def invoke(self, arguments: dict, context: ToolContext, timeout: float) -> ToolResult:
        raw = await self.transport.call_tool(server_ref=self.server_ref, tool_name=self.tool_name,
                                             arguments=deepcopy(arguments),
                                             trusted_scope=deepcopy(self.scope_mapper(context)), timeout=timeout)
        return ToolResult.from_mapping(raw)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    data_schema: dict
    adapter: ReadAdapter
    permission: str
    timeout_seconds: float = 5.0
    revision: str = "1"
    read_only: bool = True

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_.]{0,95}", self.name) or not self.permission or not self.revision:
            raise ValueError("invalid tool manifest")
        if self.read_only is not True:
            raise ValueError("the brain gateway accepts read-only tools only")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 60:
            raise ValueError("tool timeout must be finite and bounded")
        check_schema(self.parameters)
        check_schema(self.data_schema)
        object.__setattr__(self, "parameters", deepcopy(self.parameters))
        object.__setattr__(self, "data_schema", deepcopy(self.data_schema))

    def description_for_model(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": deepcopy(self.parameters)}


RESERVED_ARGUMENTS = frozenset({"tenant_ref", "customer_ref", "region", "environment", "scope",
                                "headers", "authorization", "token", "url", "sql", "trusted_scope"})


class ToolGateway:
    """Runtime dispatch, independent tool timeouts, partial-batch preservation.

    Instantiate per event loop; the semaphore bounds active adapter reads. This
    layer does not guess whether one model call depends on another. call_many is
    only for already-validated, independent read requests from the same turn.
    """
    def __init__(self, tools: list[ToolSpec], *, max_concurrency: int = 4,
                 max_result_bytes: int = 48000, clock: Callable[[], float] = time.monotonic):
        if not 1 <= max_concurrency <= 32 or not 1024 <= max_result_bytes <= 1_000_000:
            raise ValueError("invalid gateway limits")
        self.tools = {t.name: t for t in tools}
        if len(self.tools) != len(tools):
            raise ValueError("tool names must be unique")
        self._slots = asyncio.Semaphore(max_concurrency)
        self.max_result_bytes, self.clock = max_result_bytes, clock

    def descriptions(self, scope: RequestScope, context: ToolContext) -> list[dict]:
        scope.require(context)
        return [t.description_for_model() for t in self.tools.values() if t.permission in scope.permissions]

    async def call(self, name: str, arguments: dict, *, scope: RequestScope,
                   context: ToolContext, deadline: float) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult("invalid_request", code="unknown_tool")
        try:
            scope.require(context, tool.permission)
        except ScopeError:
            return ToolResult("forbidden", code="scope_denied")
        if not isinstance(arguments, dict) or set(arguments) & RESERVED_ARGUMENTS:
            return ToolResult("invalid_request", code="model_scope_override")
        try:
            validate_json(arguments, tool.parameters)
        except ContractError:
            return ToolResult("invalid_request", code="invalid_arguments")
        remaining = deadline - self.clock()
        if remaining <= 0:
            return ToolResult("timeout", code="run_deadline")
        acquired = False
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=remaining)
            acquired = True
            remaining = deadline - self.clock()
            if remaining <= 0:
                return ToolResult("timeout", code="run_deadline")
            timeout = min(tool.timeout_seconds, remaining)
            result = await asyncio.wait_for(
                tool.adapter.invoke(deepcopy(arguments), context, timeout), timeout=timeout)
            if not isinstance(result, ToolResult):
                return ToolResult("error", code="adapter_contract_violation")
            raw = result.to_dict()
            validate_json(raw, TOOL_RESULT_SCHEMA)
            if result.status in {"ok", "not_found", "partial", "stale", "conflict"}:
                # Stale/conflicting facts remain typed diagnostic data, not a
                # confirmed current fact or a missing record. Domain validation
                # must stop them being used as verified public contact details.
                validate_json(result.data, tool.data_schema)
            if len(canonical(raw).encode("utf-8")) > self.max_result_bytes:
                # A domain adapter may deliberately return status=partial with
                # a cursor. The gateway never clips JSON and pretends completeness.
                return ToolResult("unavailable", code="result_too_large")
            return ToolResult.from_mapping(raw)
        except asyncio.TimeoutError:
            return ToolResult("timeout", code="tool_timeout", retryable=True)
        except asyncio.CancelledError:
            raise  # Caller cancellation must not become a model-visible fake success.
        except ContractError:
            return ToolResult("error", code="adapter_contract_violation")
        except Exception:
            # Never include exception text, connection strings, SQL or credentials.
            return ToolResult("unavailable", code="adapter_unavailable", retryable=True)
        finally:
            if acquired:
                self._slots.release()

    async def call_many(self, calls: list[dict], *, scope: RequestScope,
                        context: ToolContext, deadline: float) -> list[dict]:
        if not 1 <= len(calls) <= 4 or len({c["call_id"] for c in calls}) != len(calls):
            raise ContractError("invalid_batch", "one turn permits 1-4 unique call IDs")

        async def one(call: dict) -> dict:
            result = await self.call(call["name"], call["arguments"], scope=scope, context=context, deadline=deadline)
            return {"call_id": call["call_id"], "tool": call["name"], "result": result.to_dict()}

        # Each read converts its own expected failure into an explicit result;
        # successful siblings survive a timeout/unavailable response.
        return list(await asyncio.gather(*(one(call) for call in calls)))


# Compatibility name for design discussions; ToolSpec is the actual manifest.
ReadTool = ToolSpec
