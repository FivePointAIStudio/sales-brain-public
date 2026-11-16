"""Concrete evidence/capability/contact/presentation checks before release.

Only trusted context and registered read-tool observations populate the evidence
catalog. The model draft is never ingested as evidence. This checks reference and
business invariants; arbitrary natural-language claims still require model eval.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import re
import unicodedata
from typing import Mapping

from jsonschema import Draft202012Validator, ValidationError

from .evidence import Plan as LegacyPlan
from .evidence import Evidence, RunError, _conflicts_with_explicit, _explicit_customer_codes
from .products import normalize_product_reference
from .contracts import ContractError, canonical, validate_plan
from .diagnostics import safe_failure
from .presentation import PENDING, SHOWN
from .sales_policy import (CustomerScope, PolicyContext, VerifiedContactIntent, check_contact_coverage,
                           contact_reply_is_independent, required_contact_actions)


_PRODUCT_TOOLS = {"resolve_product_reference", "search_products", "read_presented_product"}
_CONTACT_TOOLS = {"get_contact_routes", "get_public_contacts"}
_CONTACT_LITERAL = re.compile(r"https?://[^\s<>\]\[，。；]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)")
_DONE = r"(?:已经|已)(?:成功)?(?:帮[您你]|为[您你]|替[您你]|给[您你])?(?:(?:把|将)[^。！？\n]{0,24})?"
_COMPLETION = re.compile(_DONE + r"(?:登记|保存|记录|记下|通知|联系|提交|转交|提醒|告知|转告|安排)"
                         r"|(?:登记|保存|记录|记)好了|(?:have|has)\s+(?:already\s+)?(?:notified|contacted|saved|recorded|submitted)", re.I)
_NOTICE_COMPLETION = re.compile(_DONE + r"(?:通知|提醒|告知|转告|联系|转交)|(?:have|has)\s+(?:already\s+)?(?:notified|contacted)", re.I)
_DISPLAY_COMPLETION = re.compile(_DONE + r"(?:展示|显示|发给|发送|发出|送达)"
                                 r"|(?:already|have|has)\s+(?:already\s+)?(?:been\s+)?(?:shown|displayed|sent|delivered)", re.I)
_INTERNAL_IDENTIFIER = re.compile(r"\berp:\d+\b|\b(?:product_ref|product_key|display_key|source_event_refs|intent_ref|from_action_id)\s*[:=]", re.I)
_SALES_PROMISE = re.compile(r"(?:我会|我将|马上|稍后)(?:帮[您你]|为[您你])?(?:通知|提醒|转告|安排|联系).{0,10}(?:销售|业务|客户经理)"
                            r"|(?:销售|业务员|客户经理)(?:会|将|马上|很快|尽快).{0,10}(?:联系|回复|回电)")


def _visible_text(action):
    payload = action['payload']
    if action['type'] == 'REPLY_TEXT':
        yield payload['body'], None
        for segment in payload.get('conditional_segments', []):
            yield segment['text'], segment['when']
    elif action['type'] == 'ASK_CLARIFICATION':
        yield payload['question'], None
    elif action['type'] == 'RECOMMEND_PRODUCTS':
        yield payload.get('display_reason', ''), None


def _fold(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value))).lower()


def _copied_card_fields(text, product):
    """Narrow, verifiable duplication detection; no guessed product facts."""
    facts, normalized = product.get('facts', {}), _fold(text)
    copied = set()
    composition = _fold(facts.get('component') or '').replace('涤纶', '涤')
    if len(composition) >= 4 and composition in normalized.replace('涤纶', '涤'):
        copied.add('component')
    for category, keys, units in (
        ('width', ('width_cm', 'width'), r'(?:cm|厘米|公分)'),
        ('weight', ('weight_gsm', 'weight'), r'(?:g/?m(?:2|²)|gsm|克(?:/平方米)?)'),
    ):
        for key in keys:
            value = facts.get(key)
            if value is None or value == '':
                continue
            literal = _fold(value)
            if len(literal) >= 3 and re.search(r'[a-z\u3400-\u9fff]', literal) and literal in normalized:
                copied.add(category)
                break
            number = re.fullmatch(r'(\d+(?:\.\d+)?)(?:cm|厘米|公分|g/?m2|gsm|克)?', literal)
            if number:
                amount = format(float(number.group(1)), 'g')
                suffix = r'(?:\.0+)?' if '.' not in amount else ''
                if re.search(r'(?<!\d)' + re.escape(amount) + suffix + units, normalized):
                    copied.add(category)
                    break
    return copied


def validate_customer_text(plan, context, products, owner=None):
    """Customer-visible prose is subject to the same truth boundary as cards."""
    by_id = {action['action_id']: action for action in plan['actions']}
    current_refs = set(context.get('trigger_event_refs', []))
    requests = '\n'.join(m.get('text', '') for m in context.get('messages', [])
                         if m.get('role') == 'customer' and m.get('event_ref') in current_refs)
    explain_attributes = (any(i.get('kind') == 'product_compare' for i in context.get('confirmed_intents', []))
                          or bool(re.search(r'比较|对比|解释|区别|参数|规格|详细|成分.*(?:什么|多少)|(?:门幅|幅宽|克重).*(?:多少|什么)|有多宽|多重|compare|explain|specification', requests, re.I)))
    displayed = {item['product_ref'] for action in plan['actions'] if action['type'] == 'RECOMMEND_PRODUCTS'
                 for item in action['payload']['items']}
    displayed.update(str(row.get('product_ref') or row.get('product_id')) for row in context.get('presentations', [])
                     if row.get('status') in SHOWN | PENDING)
    display_products = [product for ref, product in products.items() if ref in displayed]
    for action in plan['actions']:
        for text, when in _visible_text(action):
            if _INTERNAL_IDENTIFIER.search(text):
                raise ContractError('internal_identifier_in_customer_text', 'Customer text must use public product codes, not ERP IDs or contract fields')
            display_claim, business_claim = bool(_DISPLAY_COMPLETION.search(text)), bool(_COMPLETION.search(text))
            if display_claim or business_claim:
                if when is None:
                    code = 'premature_display_claim' if display_claim else 'premature_completion_claim'
                    raise ContractError(code, 'Completion text requires an actual successful execution receipt; use future/neutral wording in the base reply')
                target = by_id.get(when['action_id'], {})
                if set(when.get('execution_status', [])) != {'succeeded'}:
                    raise ContractError('completion_condition_not_success', 'Completion text cannot match failed, pending or unknown execution')
                if display_claim:
                    delivered = {'sent','delivered','read'} if re.search(r'发送|发出|发给|送达|sent|delivered', text, re.I) else {'rendered','sent','delivered','read'}
                    if (target.get('type') not in {'RECOMMEND_PRODUCTS','REPLY_TEXT','ASK_CLARIFICATION'}
                        or not when.get('delivery_status') or not set(when['delivery_status']) <= delivered):
                        raise ContractError('display_receipt_required', 'Accepted or queued is not evidence that a card was rendered or delivered')
                elif _NOTICE_COMPLETION.search(text):
                    if (target.get('type') != 'NOTIFY_SALES' or not when.get('delivery_status')
                        or not set(when['delivery_status']) <= {'sent','delivered','read'}):
                        raise ContractError('notification_delivery_receipt_required', 'A stored request or console notice is not a completed sales notification')
                elif target.get('type') not in {'PROPOSE_PROJECT_PATCH','HANDOFF_SALES','PROPOSE_CUSTOMER_FACT'}:
                    raise ContractError('business_receipt_required', 'Recording a need requires a business-write receipt, not a reply receipt')
            if owner is None and _SALES_PROMISE.search(text) and not re.search(r'如果|等待|待分配|分配后|确认后|尚未|暂未|不能|无法', text):
                raise ContractError('unassigned_sales_promise', 'An unassigned followup cannot promise a salesperson notification or callback')
            if not explain_attributes and any(len(_copied_card_fields(text, product)) >= 2 for product in display_products):
                raise ContractError('product_card_facts_repeated', 'Do not copy multiple card attributes into prose; connect the product to the customer need and ask the next useful question')


def _records(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _records(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _records(child)


def _current(row, now):
    if row.get("verified") is False or row.get("is_demo") is True or row.get("valid") is False:
        return False
    if row.get("status") in {"stale", "conflict", "deleted", "revoked", "draft", "expired"}:
        return False
    expiry = row.get("valid_until") or row.get("expires_at")
    if expiry:
        try:
            parsed = datetime.fromisoformat(str(expiry).replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed <= now:
                return False
        except (ValueError, TypeError):
            return False
    return True


def _scope_matches(row, context):
    scope = row.get("scope") if isinstance(row.get("scope"), dict) else row
    for key in ("tenant_ref", "customer_ref", "region", "environment"):
        if key == "region" and scope.get("kind") == "personal" and scope.get("region") is None:
            continue  # The ContextProvider already authorized the subject.
        if scope.get(key, context[key]) != context[key]:
            return False
    return True


def _trusted_tree(value, context, now):
    if isinstance(value, dict):
        if not _current(value, now) or not _scope_matches(value, context):
            return None
        return {key: _trusted_tree(child, context, now) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean for child in value if (clean := _trusted_tree(child, context, now)) is not None]
    return value


def _contact_policy_context(context, latest, owner, capabilities):
    """One policy derivation for planning guidance and final enforcement."""
    control = context['control']
    all_paused = control.get('opted_out') or (control.get('manual_takeover') and not control.get('internal_assist_during_takeover'))
    result = latest.get('read_sales_capabilities', {})
    data = result.get('data', {}) if result.get('status') == 'ok' else {}
    enabled = (not all_paused and data.get('case_write_available') is True
               and any(row.get('request_type') == 'sales_followup' for row in capabilities.values())
               and control.get('business_writes_enabled') is not False)
    notice = data.get('notification_queue_available') is True and control.get('sales_notify_enabled') is not False
    return PolicyContext(
        customer_send_enabled=not control.get('opted_out',False) and control.get('customer_send_enabled') is not False,
        business_writes_enabled=enabled, sales_notify_enabled=notice,
        manual_takeover=bool(control.get('manual_takeover')),
        internal_assist_during_takeover=bool(control.get('internal_assist_during_takeover')),
        sales_owner_available=owner is not None, no_outbound_contact=bool(control.get('no_outbound_contact')))


def _scoped_contact_intent(context, intent):
    return VerifiedContactIntent(CustomerScope(context['tenant_ref'],context['environment'],context['customer_ref'],context['region']),
                                 intent['intent_ref'],tuple(intent['source_event_refs']))


class CandidateValidator:
    def __init__(self, *, snapshot, legacy_scope, utcnow=lambda: datetime.now(timezone.utc)):
        self.snapshot, self.scope, self.utcnow = snapshot, legacy_scope, utcnow
        self.catalog = {}

    def export(self):
        return deepcopy(self.catalog)

    def _collect(self, context, observations):
        now = self.utcnow()
        evidence = Evidence()
        # Explicitly select context evidence; no commands/recipients can be
        # laundered through a customer profile or free-form pending interpretation.
        evidence.ingest(_trusted_tree({"messages": context["messages"], "facts": context.get("facts", []), "cases": context.get("cases", [])}, context, now))
        products, media, facts, quotes, capabilities, requests = {}, {}, {}, {}, {}, {}
        contact_facts, latest, owner = {}, {}, None
        for row in context.get("facts", []):
            if isinstance(row, dict) and row.get("fact_ref") and _current(row, now) and _scope_matches(row, context):
                facts[row["fact_ref"]] = deepcopy(row)
        for observation in observations:
            name, result = observation.get("tool"), observation.get("result")
            if not name or not isinstance(result, dict):
                continue
            data = result.get("data", {})
            if not isinstance(data, dict):
                continue
            latest[name] = result
            if name == "read_sales_capabilities":
                capabilities.clear()
                evidence.commands.clear()
            if name == "read_sales_owner":
                owner = None
            if result.get("status") not in {"ok", "partial"}:
                continue
            # Identity/attribute rows in a partial catalog result can be valid
            # candidates, but partial contact/quote/capability data is not enough
            # to authorize a business command or publish a contact value.
            if result["status"] != "ok" and name not in _PRODUCT_TOOLS:
                continue
            valid_data = deepcopy(data)
            if name in _PRODUCT_TOOLS | {"resolve_quote", "search_company_knowledge", "read_workflow_context"} | _CONTACT_TOOLS:
                evidence.ingest(_trusted_tree(valid_data, context, now))
            if name == "read_sales_capabilities":
                for row in _records(data.get("capabilities", [])):
                    if row.get("command_ref") and _current(row, now) and _scope_matches(row, context):
                        capabilities[row["command_ref"]] = deepcopy(row)
                        evidence.commands.add(row["command_ref"])
            if name == "read_sales_owner" and data.get("recipient_ref") and data.get("channel_route_ref"):
                if _scope_matches(data, context) and _current(data, now) and data.get("source_ref"):
                    owner = deepcopy(data)
                    evidence.recipients.add(owner["recipient_ref"])
                    evidence.routes.add(owner["channel_route_ref"])
                    evidence.ingest({k: owner[k] for k in ("fact_ref", "source_ref") if k in owner})
            for row in _records(data):
                if not _current(row, now) or not _scope_matches(row, context):
                    continue
                if name in _PRODUCT_TOOLS and row.get("product_ref") and row.get("product_key") and row.get("display_key") and row.get("source_refs"):
                    products[row["product_ref"]] = deepcopy(row)
                if name == "get_product_media" and row.get("media_ref") and row.get("product_ref") and row.get("source_refs"):
                    media[row["media_ref"]] = deepcopy(row)
                if name in _CONTACT_TOOLS | {"search_company_knowledge"} and row.get("fact_ref") and row.get("source_ref"):
                    facts[row["fact_ref"]] = deepcopy(row)
                    if name in _CONTACT_TOOLS and row.get("public") is not False and row.get("value"):
                        contact_facts[row["fact_ref"]] = deepcopy(row)
                if name == "resolve_quote" and row.get("quote_ref") and row.get("valid") is True:
                    quotes[row["quote_ref"]] = deepcopy(row)
                if name == "read_workflow_context" and row.get("request_ref"):
                    requests[row["request_ref"]] = deepcopy(row)
        # A later expired/conflicting contact result revokes any earlier result
        # from the same run instead of publishing a stale number from the ledger.
        for name in _CONTACT_TOOLS:
            if name in latest and latest[name]["status"] not in {"ok"}:
                contact_facts.clear()
        evidence.recipients = {owner["recipient_ref"]} if owner else set()
        evidence.routes = {owner["channel_route_ref"]} if owner else set()
        confirmed = {i["intent_ref"]: i for i in context.get("confirmed_intents", [])}
        source_events = {m["event_ref"] for m in context["messages"] if m.get("role") == "customer"
                         and not m.get("removed") and not m.get("body_truncated")}
        evidence.products, evidence.quotes = set(products), set(quotes)
        evidence.quote_rows = deepcopy(quotes)
        evidence.facts, evidence.events = set(facts), set(source_events)
        return {'now': now, 'evidence': evidence, 'products': products, 'media': media, 'facts': facts, 'quotes': quotes, 'capabilities': capabilities, 'requests': requests, 'contact_facts': contact_facts, 'latest': latest, 'owner': owner, 'confirmed': confirmed, 'source_events': source_events}

    def reference_catalog(self, context, observations):
        """Typed reference table built by the same authority filter as validation."""
        values = self._collect(context, observations)
        evidence = values['evidence']
        contact_requirements = []
        policy = _contact_policy_context(context, values['latest'], values['owner'], values['capabilities'])
        for intent in values['confirmed'].values():
            if intent['kind'] != 'contact_request':
                continue
            scoped = _scoped_contact_intent(context, intent)
            requirements = required_contact_actions(scoped, policy, verify_intent=lambda value: value == scoped)
            unread = []
            if not any(name in values['latest'] for name in _CONTACT_TOOLS):
                unread.append('get_contact_routes')
            if 'read_sales_capabilities' not in values['latest']:
                unread.append('read_sales_capabilities')
            if policy.business_writes_enabled and policy.sales_notify_enabled and 'read_sales_owner' not in values['latest']:
                unread.append('read_sales_owner')
            contact_requirements.append({
                'intent_ref': intent['intent_ref'], 'read_before_final': unread,
                'public_contact_state': ('unread' if not any(name in values['latest'] for name in _CONTACT_TOOLS)
                                         else 'available' if values['contact_facts'] else 'unavailable'),
                'additional_coverage_reasons': (['public_contacts_unavailable']
                    if any(name in values['latest'] for name in _CONTACT_TOOLS) and not values['contact_facts'] else []),
                'requirements': [{'name': req.name,'action_type': req.action_type,'required': req.required,
                                  'request_type': req.request_type,'blocked_reason': req.blocked_reason} for req in requirements],
                'coverage_rule': 'Every required action must be in THIS intent_coverage.action_ids. A case action may have another confirmed intent_ref; include that same action_id in both coverage rows instead of creating duplicate cases.',
            })
        return {
            'intents': [{key: row[key] for key in ('intent_ref','kind','source_event_refs')}
                        for row in values['confirmed'].values()],
            'product_refs': sorted(values['products']),
            'products': [{key: row.get(key) for key in ('product_ref','product_key','display_key','dcode','product_code')}
                         for row in values['products'].values()],
            'presented_products': [{
                'presentation_ref': row.get('presentation_ref') or row.get('presentation_id'),
                'product_ref': row.get('product_ref') or row.get('product_id'),
                'display_key': row.get('display_key'), 'status': row.get('status'),
            } for row in context.get('presentations', []) if row.get('status') in SHOWN
                and _scope_matches(row, context) and (row.get('presentation_ref') or row.get('presentation_id'))],
            'fact_refs': sorted(evidence.facts | evidence.sources),
            'quote_refs': sorted(values['quotes']),
            'ranking_evidence_refs': sorted(evidence.rankings),
            'source_event_refs': sorted(values['source_events']),
            'case_refs': sorted(evidence.cases),
            'request_refs': sorted(values['requests']),
            'command_refs': sorted(values['capabilities']),
            'recipient_refs': sorted(evidence.recipients),
            'channel_route_refs': sorted(evidence.routes),
            'contact_action_requirements': contact_requirements,
            'note': 'Presented products are history identities, not refreshed product evidence. Use read_presented_product(presentation_ref) before writing an interest if no current product_refs exist. This table does not authorize arbitrary uses: intent, scope, source freshness, capability, command parameters and contact value checks still apply. Do not copy internal references into customer-visible text.',
        }

    def __call__(self, plan, context, observations):
        validate_plan(plan)
        for key in ("tenant_ref", "customer_ref", "region", "environment", "conversation_ref", "context_version", "config_version"):
            if plan[key] != context[key]:
                raise ContractError("plan_scope_mismatch", "plan metadata differs from trusted runtime context")
        collected = self._collect(context, observations)
        now = collected['now']
        evidence = collected['evidence']
        products = collected['products']
        media = collected['media']
        facts = collected['facts']
        quotes = collected['quotes']
        capabilities = collected['capabilities']
        requests = collected['requests']
        contact_facts = collected['contact_facts']
        latest = collected['latest']
        owner = collected['owner']
        confirmed = collected['confirmed']
        source_events = collected['source_events']
        # Let the proven v2 verifier check facts/quotes/explicit product prefixes,
        # while substituting only source-verified v2.1 pending interpretations.
        adapted_snapshot = replace(self.snapshot, messages=tuple(context["messages"]),
            pending_intents=tuple(confirmed.values()), selected_event_refs=tuple(context["trigger_event_refs"]))
        try:
            evidence.verify(LegacyPlan(deepcopy(plan)), adapted_snapshot, now=now)
        except RunError as exc:
            error = ContractError(exc.code, "unverified evidence or omitted customer intent")
            error.diagnostics = safe_failure(exc)
            raise error from None
        for action_index, action in enumerate(plan["actions"]):
            kind, payload = action["type"], action["payload"]
            action_intent = confirmed.get(action["intent_ref"])
            if kind not in {"NO_ACTION", "ASK_CLARIFICATION"} and action_intent is None:
                error = ContractError("confirmed_intent_required", "planner-only intent labels cannot authorize actions")
                error.diagnostics = {'code': error.code, 'field_path': ['actions', action_index, 'intent_ref'],
                    'required_binding': 'confirmed_intent_ref', 'verified_candidate_count': len(confirmed)}
                raise error
            if kind == "PROPOSE_CUSTOMER_FACT":
                if action_intent["kind"] != "customer_fact":
                    raise ContractError("customer_fact_intent_required", "a temporary sales need is not a long-term customer fact")
                content = payload.get("content")
                if content is not None and not any(content["source_quote"] in m.get("text", "")
                        for m in context["messages"] if m["event_ref"] in payload["source_event_refs"] and m["event_ref"] in source_events):
                    raise ContractError("customer_fact_quote_mismatch", "memory proposal must preserve an exact authorized customer quote")
                if isinstance(payload.get("fact_ref"), str):
                    fact = facts.get(payload["fact_ref"])
                    if fact is None or fact.get("resource_version", fact.get("revision")) != payload.get("expected_version"):
                        raise ContractError("customer_fact_version_unverified", "existing memory mutation needs a current fact/version")
            if kind == "HANDOFF_SALES":
                manifest = capabilities.get(payload["command_ref"])
                if manifest is None:
                    raise ContractError("sales_capability_unverified", "command was not returned by the sales capability tool")
                operations = manifest.get("operations") or [manifest.get("operation")]
                targets = manifest.get("target_results") or [manifest.get("target_result")]
                if (manifest.get("request_type") != payload["request_type"] or payload["operation"] not in operations
                        or payload["target_result"] not in targets):
                    raise ContractError("sales_capability_mismatch", "command intent, operation or target is outside its manifest")
                schema = manifest.get("parameters_schema")
                if not isinstance(schema, dict):
                    raise ContractError("sales_capability_schema_missing", "command manifest has no argument schema")
                try:
                    Draft202012Validator.check_schema(schema)
                    Draft202012Validator(schema).validate(payload["parameters"])
                except Exception:
                    # Schema and validation exceptions are both non-authority.
                    raise ContractError("sales_capability_arguments", "arguments do not match the registered command") from None
                intent = confirmed.get(action["intent_ref"])
                if payload["request_type"] == "callback" and (not intent or intent["kind"] != "callback_request"):
                    raise ContractError("callback_intent_required", "requesting contact information does not authorize callback")
            if kind in {"PROPOSE_PROJECT_PATCH", "HANDOFF_SALES"}:
                target, version = payload.get("case_ref"), payload.get("expected_version")
                if isinstance(target, str):
                    rows = [c for c in context.get("cases", []) if c.get("case_ref") == target]
                    if not rows or not any(c.get("resource_version", c.get("revision")) == version for c in rows):
                        raise ContractError("case_version_unverified", "case version does not match the current workflow read")
                    if not any(c.get("authority") == "sales-workflows" for c in rows):
                        raise ContractError("case_owner_mismatch", "legacy preview projects cannot be written through the v2.1 case adapter")
            if kind == "PROPOSE_PROJECT_PATCH":
                actual_caps = latest.get("read_sales_capabilities", {})
                if (actual_caps.get("status") != "ok" or actual_caps.get("data", {}).get("case_write_available") is not True
                        or context["control"].get("business_writes_enabled") is False):
                    raise ContractError("case_capability_unverified", "case writes require an actual enabled workflow capability read")
            if kind == "NOTIFY_SALES":
                if owner is None or (payload["recipient_ref"], payload["channel_route_ref"]) != (owner["recipient_ref"], owner["channel_route_ref"]):
                    raise ContractError("sales_owner_unverified", "notification recipient/route must be resolved by the owner service")
                if isinstance(payload["request_ref"], str) and payload["request_ref"] not in requests:
                    raise ContractError("sales_request_unverified", "notification needs a real request or a receipt binding")
            if kind == "REPLY_TEXT":
                allowed_contact_values = [str(r["value"]) for ref, r in contact_facts.items() if ref in payload["fact_refs"]]
                for literal in _CONTACT_LITERAL.findall(payload["body"]):
                    token = re.sub(r"\s+", "", literal).rstrip(".,;，。；")
                    if not any(token in re.sub(r"\s+", "", value) for value in allowed_contact_values):
                        # An address/phone/web contact cannot be invented or
                        # cited using an unrelated generic fact.
                        raise ContractError("public_contact_literal_unverified", "contact literal is not backed by a current public contact fact")
            if kind == "RECOMMEND_PRODUCTS":
                self._check_presentations(payload, context, confirmed, products, media)
        validate_customer_text(plan, context, products, owner)
        self._check_contact_obligations(plan, context, confirmed, latest, contact_facts, owner, capabilities)
        self.catalog = {"products": products, "media": media, "facts": facts, "quotes": quotes,
                        "capabilities": capabilities, "sales_owner": owner,
                        "source_event_refs": sorted(source_events),
                        "confirmed_intents": deepcopy(confirmed),
                        "source_messages": {m["event_ref"]: {
                            "text_sha256": sha256(m["text"].encode("utf-8")).hexdigest(),
                            "source_revision": deepcopy(m.get("source_revision")),
                            "role": m["role"], "customer_ref": context["customer_ref"],
                            "removed": bool(m.get("removed")), "body_truncated": bool(m.get("body_truncated")),
                        } for m in context["messages"] if m["event_ref"] in source_events}}

    def _check_contact_obligations(self, plan, context, confirmed, latest, contact_facts, owner, capabilities):
        control = context["control"]
        for intent in confirmed.values():
            if intent["kind"] != "contact_request":
                continue
            all_paused = control.get("opted_out") or (control.get("manual_takeover") and not control.get("internal_assist_during_takeover"))
            if not all_paused:
                if not any(name in latest for name in _CONTACT_TOOLS):
                    raise ContractError("contact_information_not_read", "read public company contact facts before answering")
                if "read_sales_capabilities" not in latest:
                    raise ContractError("contact_capabilities_not_read", "read actual internal workflow capabilities")
            policy = _contact_policy_context(context, latest, owner, capabilities)
            if policy.business_writes_enabled and policy.sales_notify_enabled and "read_sales_owner" not in latest:
                raise ContractError("sales_owner_not_read", "resolve the actual notification owner")
            scoped = _scoped_contact_intent(context, intent)
            issues = check_contact_coverage(plan, scoped, policy, verify_intent=lambda value:
                value.intent_ref in confirmed and confirmed[value.intent_ref]["kind"] == "contact_request"
                and set(value.source_event_refs) == set(confirmed[value.intent_ref]["source_event_refs"]))
            if issues:
                issue = issues[0]
                error = ContractError(issue.code, issue.requirement)
                row_index = next(index for index,row in enumerate(plan['intent_coverage']) if row['intent_ref']==intent['intent_ref'])
                requirement = next((req for req in required_contact_actions(scoped,policy,verify_intent=lambda value:value==scoped)
                                    if req.name==issue.requirement),None)
                error.diagnostics = {'code': error.code, 'field_path':['intent_coverage',row_index,'action_ids'],
                    'requirement':issue.requirement,
                    'required_action_type':requirement.action_type if requirement else None,
                    'required_request_type':requirement.request_type if requirement else None,
                    'blocked_reason':requirement.blocked_reason if requirement else None,
                    'matching_action_indices': [index for index,action in enumerate(plan['actions'])
                        if requirement and action['type']==requirement.action_type and (not requirement.request_type
                            or action['payload'].get('request_type')==requirement.request_type)]}
                if issue.code in {'missing_blocked_reason','contact_unavailability_not_recorded'}:
                    error.diagnostics['field_path'][-1]='reason'
                if issue.code=='action_sources_missing':
                    action_index=next((index for index,action in enumerate(plan['actions']) if action['action_id']==issue.message),None)
                    if action_index is not None:
                        error.diagnostics['field_path']=['actions',action_index,'payload','source_event_refs']
                raise error
            row = next(r for r in plan["intent_coverage"] if r["intent_ref"] == intent["intent_ref"])
            if policy.customer_send_enabled and not policy.manual_takeover:
                all_by_id = {a["action_id"]: a for a in plan["actions"]}
                replies = [a for a in plan["actions"] if a["action_id"] in row["action_ids"]
                           and contact_reply_is_independent(a, all_by_id)]
                if contact_facts:
                    if not any(any(ref in a["payload"]["fact_refs"] and str(fact["value"]) in a["payload"]["body"]
                                   for ref, fact in contact_facts.items()) for a in replies):
                        raise ContractError("contact_details_not_grounded", "reply must actually provide a cited current public contact value")
                elif "public_contacts_unavailable" not in row["reason"]:
                    raise ContractError("contact_unavailability_not_recorded", "retain unresolved contact facts and answer without inventing them")

    @staticmethod
    def _check_presentations(payload, context, confirmed, products, media):
        rows = [p for p in context.get("presentations", []) if isinstance(p, dict) and _scope_matches(p, context)]
        for item in payload["items"]:
            product = products.get(item["product_ref"])
            if product is None or product["product_key"] != item["product_key"] or product["display_key"] != item["display_key"]:
                raise ContractError("product_identity_unverified", "presentation identity must match a real catalog result")
            existing = [p for p in rows if p.get("display_key") == item["display_key"] and p.get("status") in SHOWN | PENDING]
            targeted = [p for p in existing if p.get("presentation_id", p.get("presentation_ref")) == item["existing_presentation_ref"]]
            product_codes = {normalize_product_reference(product.get(key)) for key in ("dcode", "public_code", "product_code")} - {None}
            def matches_product(intent):
                reference = normalize_product_reference(intent.get("product_reference"))
                if reference is None:
                    return True  # Generic requests still need a source-bound kind.
                return reference in product_codes or (reference.isdigit() and any(
                    re.fullmatch(r"[A-Z]+[-_]?" + re.escape(reference), code) for code in product_codes))
            intents = [i for i in confirmed.values() if set(item["source_event_refs"]).issubset(i["source_event_refs"])
                       and matches_product(i)]
            kinds = {i["kind"] for i in intents}
            mode = item["mode"]
            if mode == "new_card":
                if existing:
                    raise ContractError("duplicate_product_presentation", "same display identity is already sent or pending")
                if not kinds.intersection({"product_search", "product_compare"}):
                    raise ContractError("display_intent_required", "interest/contact alone does not request another product card")
            if mode in {"reference_existing", "update_existing", "explicit_redisplay"} and not targeted:
                raise ContractError("presentation_reference_unverified", "existing presentation needs a trusted actual delivery/pending record")
            if mode == "explicit_redisplay" and "product_reshow" not in kinds:
                raise ContractError("reshow_intent_required", "explicit source-bound reshow intent is required")
            if mode == "media_only" and "product_media_request" not in kinds:
                raise ContractError("media_intent_required", "source-bound image request is required")
            if mode == "update_existing":
                if not product.get("fact_version") or any(p.get("fact_version") == product["fact_version"] for p in targeted):
                    raise ContractError("product_update_not_verified", "update needs a verified material fact version change")
            for ref in item["media_refs"]:
                record = media.get(ref)
                if record is None or record.get("product_ref") != item["product_ref"] or record.get("product_key") != item["product_key"]:
                    raise ContractError("product_media_unverified", "media must belong to the same verified product")
