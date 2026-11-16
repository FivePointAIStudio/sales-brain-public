"""Dedicated DeepSeek HTTPS adapter and independent v2.1 anonymous format gate.

Uses only the existing project's DEEPSEEK_BRAIN_API_KEY/private configuration.
Health checks never call the network. probe_v21 is an explicit operator action.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import asyncio
import json
import logging
from .transport.deepseek_provider import DeepSeekProvider, _private_json, _write_private_json
from .transport.errors import BrainError as ProviderError
from .transport.messages import ModelPrompt

from .core.contracts import ContractError, assemble_plan, canonical, model_step_schema, validate_model_step
from .core.diagnostics import safe_failure


class V21DeepSeekProvider(DeepSeekProvider):
    gate_filename = "nextbrain-v21-probe.json"

    def fingerprint(self):
        root = Path(__file__).parent / "core"
        files = ("contracts.py", "agent_loop.py", "tool_gateway.py", "policies.py", "sales_policy.py", "evidence.py", "diagnostics.py", "plan_compile.py")
        material = super().fingerprint().encode() + Path(__file__).read_bytes()
        material += b"runtime.py" + (root.parent / "runtime.py").read_bytes()
        for name in files:
            path = root / name
            if path.exists():
                material += name.encode() + path.read_bytes()
        return sha256(material).hexdigest()

    def health(self):
        configured = bool(self.key) and not self.config_error
        gate = {}
        if configured:
            try:
                gate = _private_json(self.private_dir / self.gate_filename)
            except (OSError, ValueError, TypeError):
                pass
        verified = configured and gate.get("passed") is True and gate.get("fingerprint") == self.fingerprint()
        return {"ready": bool(verified), "configured": configured, "verified": bool(verified),
                "provider": "deepseek", "model": self.model, "transport": "https_api",
                "schema_version": "2.1", "reason": "v2.1匿名格式已验证" if verified else "v2.1模型配置或匿名格式验证尚未完成"}

    def complete_json(self, *, system: str, user: dict, schema: dict, timeout: float, _probe=False) -> dict:
        try:
            return super().complete(ModelPrompt(system, canonical(user)), schema, timeout=timeout, _probe=_probe)
        except ProviderError as exc:
            if exc.code == "model_invalid_result":
                error = ContractError("invalid_model_output", "provider response violates the v2.1 JSON contract")
                error.diagnostics = safe_failure(exc)
                logging.getLogger('sales-brain.contracts').warning('model_contract_rejected %s',
                    json.dumps(error.diagnostics, ensure_ascii=False, separators=(',', ':')))
                raise error from None
            raise

    def probe_v21(self):
        """Two anonymous model calls: interpreter format, then final-plan format."""
        from .core.agent_loop import INTENT_RESULT_SCHEMA, SYSTEM, TrustedContext, bind_resolved_intents
        context = TrustedContext(tenant_ref="anonymous-probe", customer_ref="probe-customer", region="BKK",
            environment="sandbox", conversation_ref="probe-conversation", task_ref="probe-task", plan_revision=1,
            context_version="probe-context", config_version="probe-config", trigger_event_refs=("probe-event",),
            messages=({"event_ref": "probe-event", "role": "customer", "text": "你好"},),
            agent_config={"language": "zh", "public_identity": "Example Assistant"})
        interpretation = self.complete_json(system="Return one explicit general_question intent for this anonymous greeting, with its exact source quote. No tools or business effects.",
            user={"messages": list(context.messages), "trigger_event_refs": ["probe-event"]},
            schema=INTENT_RESULT_SCHEMA, timeout=30, _probe=True)
        if len(interpretation.get("intents", [])) != 1 or interpretation["intents"][0].get("kind") != "general_question":
            raise ValueError("anonymous_v21_intent_probe_failed")
        context = bind_resolved_intents(context, interpretation["intents"])
        if len(context.confirmed_intents) != 1:
            raise ValueError("anonymous_v21_intent_probe_failed")
        step = self.complete_json(system=SYSTEM,
            user={"context": context.to_public_dict(), "read_tools": [], "observations": [],
                  "instruction": "仅验证匿名问候格式。final里恰好一个REPLY_TEXT，不含事实、产品、写动作或执行成功承诺；覆盖提供的确认意图。"},
            schema=model_step_schema([]), timeout=30, _probe=True)
        validate_model_step(step, [])
        actions = step["plan"]["actions"]
        if len(actions) != 1 or actions[0]["type"] != "REPLY_TEXT" or actions[0]["payload"]["fact_refs"]:
            raise ValueError("anonymous_v21_plan_probe_failed")
        assemble_plan(step["plan"], context.to_public_dict(), run_id="probe-run", decision_id="probe-decision",
                      now=datetime.now(timezone.utc), ttl_seconds=60)
        _write_private_json(self.private_dir / self.gate_filename,
            {"passed": True, "fingerprint": self.fingerprint(), "returned_model": self.last_model})
        return self.health()


class V21JsonModel:
    """Async port over the existing bounded synchronous DeepSeek transport."""
    def __init__(self, transport: V21DeepSeekProvider):
        self.transport = transport

    async def complete(self, *, system: str, user: dict, schema: dict, timeout: float) -> dict:
        # Provider owns a bounded HTTPS deadline. This thread never touches ORM
        # state; cancellation cannot turn a late response into a released plan.
        return await asyncio.to_thread(self.transport.complete_json,
            system=system, user=user, schema=schema, timeout=timeout)

    def health(self):
        return self.transport.health()
