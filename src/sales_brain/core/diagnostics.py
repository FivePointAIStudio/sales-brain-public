"""Bounded contract diagnostics without model text, values or dynamic keys."""
from copy import deepcopy
import re


def _schema_names(schema):
    names = set()
    def visit(value):
        if isinstance(value, dict):
            names.update(value.get('properties', {}))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schema)
    return names


def schema_failure(schema, error, *, stage):
    names = _schema_names(schema)
    rows, pending = [], [error]
    while pending and len(rows) < 16:
        item = pending.pop(0)
        row = {
            'validator': str(item.validator)[:80],
            'field_path': [part if isinstance(part, int) or part in names else '<dynamic-key>'
                           for part in item.absolute_path][:32],
            # This path belongs to our server-owned schema, not a data value.
            'schema_path': list(item.absolute_schema_path)[:48],
        }
        if item.validator == 'required' and isinstance(item.instance, dict):
            row['missing_fields'] = [key for key in item.validator_value
                                     if key in names and key not in item.instance][:32]
        if item.validator == 'additionalProperties' and isinstance(item.instance, dict):
            row['extra_field_count'] = len(set(item.instance) - set(item.schema.get('properties', {})))
        rows.append(row)
        pending.extend(item.context)
    return {'stage': stage, 'code': 'json_schema_violation', 'errors': rows}


def safe_failure(error, *, stage=None):
    """Only internally constructed metadata is carried; never stringify errors."""
    value = getattr(error, 'diagnostics', None)
    result = deepcopy(value) if isinstance(value, dict) else {}
    code = getattr(error, 'code', 'invalid_model_output')
    result.setdefault('code', code if isinstance(code, str) and re.fullmatch(r'[a-zA-Z0-9_.:-]{1,160}', code) else 'contract_error')
    if stage:
        result['stage'] = stage
    return result


def field_failure(code, path, schema, *, stage, **metadata):
    """Attach positions/types only; never serialize a rejected reference value."""
    names = _schema_names(schema)
    return {'code': code, 'stage': stage,
            'field_path': [part if isinstance(part, int) or part in names else '<dynamic-key>' for part in path][:32],
            **metadata}


_REPAIR_HINTS = {
    'unbound_action': 'Use an exact allowed_references.intents[].intent_ref for the action. Include this action_id in the intent_coverage row with the same intent_ref; one action may additionally cover other intents.',
    'confirmed_intent_required': 'The action.intent_ref must copy a complete current confirmed intent_ref, not its kind, event_ref, a new label, or an example action ID. Read allowed_references.intents.',
    'unverified_reference': 'The indicated reference field contains a value not returned by a trusted read. Use only the matching allowed_references category. product_ref is not fact_ref; event_ref is not fact_ref. A neutral reply can use fact_refs: []. Read the relevant tool if evidence is missing; never guess a replacement.',
    'invalid_condition': 'Each conditional_segments[].when.action_id must name an existing other action, also present in this reply depends_on. Do not cite an intent, event, resource or receipt ID in action_id.',
    'contact_reply_blocked_by_internal_action': 'Keep a separate independent REPLY_TEXT for actual contact details or their honest absence. Put saved-need confirmation in another reply waiting for the business receipt.',
    'missing_intent_coverage': 'Copy every current allowed_references.intents[] row with its exact intent_ref, kind and source_event_refs; link the actual actions that fulfill it.',
    'intent_binding_mismatch': 'Do not reclassify a confirmed intent or replace its source_event_refs. Copy the exact current allowed_references.intents[] binding.',
    'missing_required_action': 'Read the indicated requirement and allowed_references.contact_action_requirements. Include an existing matching action_id in this contact coverage row, or propose the missing authorized action. A shared case can be linked in both purchase and contact rows; keep it in its own intent row too. Do not duplicate a case merely to change coverage.',
    'missing_blocked_reason': 'Preserve the exact blocked_reason from contact_action_requirements in the contact intent coverage reason. Do not fabricate a recipient or send an unauthorized action to avoid this reason.',
    'action_sources_missing': 'The indicated business action must retain the contact intent source_event_refs, including when shared with a purchase intent. Use only the verified source bindings supplied by the runtime.',
    'contact_unavailability_not_recorded': 'The public contact read returned no current verified value. State that absence honestly and preserve public_contacts_unavailable in the contact coverage reason, alongside any owner/permission blocked reasons. Authorized internal case and followup actions still apply.',
}


def candidate_failure(error):
    result = safe_failure(error, stage='candidate_validation')
    if error.code in _REPAIR_HINTS:
        result['repair_hint'] = _REPAIR_HINTS[error.code]
    return result
