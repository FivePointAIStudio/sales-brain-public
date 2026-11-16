"""Conservative compilation of redundant receipt dependencies.

This module never creates a business action, chooses an intent/evidence binding,
or removes a constraint. The assembled plan still passes every contract,
provenance, capability, scope, cycle and policy check before release.
"""
from copy import deepcopy


def normalize_conditional_dependencies(draft):
    result = deepcopy(draft)
    actions = result['actions']
    by_id = {action['action_id']: index for index, action in enumerate(actions)}
    if len(by_id) != len(actions):
        return result, []  # The regular duplicate-action rejection remains authoritative.
    changes = []
    for index, action in enumerate(actions):
        if action['type'] != 'REPLY_TEXT':
            continue
        existing = {dep['action_id'] for dep in action['depends_on']}
        conditions = {}
        for segment_index, segment in enumerate(action['payload']['conditional_segments']):
            target = segment['when']['action_id']
            if target in existing or target not in by_id or target == action['action_id']:
                continue
            conditions.setdefault(target, []).append((segment_index, segment['when']['execution_status']))
        for target, segments in conditions.items():
            condition = 'success' if all(set(statuses) == {'succeeded'} for _, statuses in segments) else 'terminal'
            action['depends_on'].append({'action_id': target, 'condition': condition})
            changes.append({'action_index': index, 'target_action_index': by_id[target],
                            'segment_indices': [item[0] for item in segments], 'condition': condition})
    return result, changes


def binding_examples(context):
    """Show exact current reference spelling without choosing business content."""
    intents = context.get('confirmed_intents', [])
    if not intents:
        return {}
    intent = intents[0]
    result = {
        'instruction': 'These are format fragments, not new facts or commands. Replace local action IDs as needed; copy real intent/source refs exactly.',
        'covered_action_link': {
            'coverage': {'intent_ref': intent['intent_ref'], 'kind': intent['kind'], 'confirmation': 'candidate',
                         'source_event_refs': list(intent['source_event_refs']), 'status': 'planned',
                         'action_ids': ['save_need', 'acknowledge_need'], 'reason': ''},
            'each_action_intent_ref': intent['intent_ref'],
        },
        'receipt_acknowledgement_fragment': {
            'action_id': 'acknowledge_need', 'type': 'REPLY_TEXT', 'intent_ref': intent['intent_ref'],
            'depends_on': [{'action_id': 'save_need', 'condition': 'success'}], 'guards': [],
            'payload': {'body': '', 'fact_refs': [], 'reply_group': 'need_acknowledgement',
                        'conditional_segments': [{'text': '已为您登记这次需求。',
                            'when': {'action_id': 'save_need', 'execution_status': ['succeeded']}}]},
        },
        'contact_answer_rule': 'For contact_request use a separate independent REPLY_TEXT containing a grounded public contact or honest absence. The receipt acknowledgement above does not replace that answer.',
        'binding_type_rule': 'product_refs/product_interests.product_ref use allowed_references.product_refs. fact_refs use allowed_references.fact_refs, never product_ref, event_ref or command_ref. source_event_refs use allowed_references.source_event_refs. Every action.intent_ref is one exact allowed_references.intents[].intent_ref.',
    }
    result['case_and_receipt_subgraph'] = {
        'instruction': 'This is a schema-valid two-action subgraph showing a new case and its acknowledgement, not a complete answer. Populate real need/product evidence and check live case capability first. Existing cases need their current case_ref/version. Contact intent also needs an independent answer and the supported followup.',
        'actions': [{
            'action_id': 'save_need', 'type': 'PROPOSE_PROJECT_PATCH', 'intent_ref': intent['intent_ref'],
            'depends_on': [], 'guards': [],
            'payload': {'operation': 'create', 'case_ref': None, 'expected_version': None,
                        'product_interests': [], 'requirements': {}, 'open_intents': [intent['intent_ref']],
                        'clear_fields': [], 'source_event_refs': list(intent['source_event_refs'])},
        }, deepcopy(result['receipt_acknowledgement_fragment'])],
    }
    return result
