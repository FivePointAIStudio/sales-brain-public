"""Real transport decoding and bounded repairs; all HTTPS bodies are fixtures."""
import asyncio
from copy import deepcopy
import json

import pytest

from sales_brain.core.agent_loop import BrainError, INTENT_RESULT_SCHEMA, JsonIntentResolver, TrustedContext
from sales_brain.core.contracts import ContractError
from sales_brain.provider import V21DeepSeekProvider, V21JsonModel


def context():
    return TrustedContext('tenant:test', 'customer:test', 'BKK', 'live', 'conv:test', 'task:test',
                          1, 'ctx:test', 'cfg:test', ('event:current',), (
                              {'event_ref': 'event:old', 'role': 'customer', 'text': '请看看10001'},
                              {'event_ref': 'assistant:old', 'role': 'assistant', 'text': '需要比较其他产品吗？'},
                              {'event_ref': 'event:current', 'role': 'customer', 'text': 'SYN-10001我想采购，请记下采购意向。'},
                          ), intent_resolution_event_refs=('event:current',))


def valid():
    return {'intents': [{'kind': 'product_interest', 'certainty': 'explicit',
                         'source_event_refs': ['event:current'],
                         'source_quotes': [{'event_ref': 'event:current', 'quote': 'SYN-10001我想采购'}],
                         'product_reference': 'SYN-10001'}]}


def transport(responses):
    provider = V21DeepSeekProvider.__new__(V21DeepSeekProvider)
    provider.key = 'offline-test-key-not-a-real-credential'
    provider.model = 'deepseek-flash'
    provider.config_error = False
    provider.health = lambda: {'ready': True}
    output = iter(responses)
    calls = []

    def respond(request, timeout):
        calls.append({'request': json.loads(request.data), 'timeout': timeout})
        value = next(output)
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        return json.dumps({'model': 'deepseek-flash', 'choices': [
            {'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': value}}
        ]}, ensure_ascii=False).encode()

    provider._bounded_request = respond
    return provider, calls


def test_raw_invalid_json_schema_gets_one_repair_without_logging_customer_text(caplog):
    bad = valid()
    bad['intents'][0]['customer-private-dynamic-key'] = 'SYNTHETIC_PRIVATE_TEXT'
    provider, calls = transport([bad, valid()])
    resolver = JsonIntentResolver(V21JsonModel(provider))
    result = asyncio.run(resolver.resolve(context(), 10))
    assert result == valid()['intents'] and resolver.attempts == 2
    assert len(calls) == 2
    second = json.loads(calls[1]['request']['messages'][1]['content'])
    assert second['previous_rejection']['code'] == 'json_schema_violation'
    assert any(error['validator'] == 'additionalProperties' for error in second['previous_rejection']['errors'])
    assert 'PRIVATE_CUSTOMER_TEXT' not in caplog.text
    assert 'customer-private-dynamic-key' not in caplog.text
    assert 'PRIVATE_CUSTOMER_TEXT' not in json.dumps(resolver.diagnostics)
    assert 'customer-private-dynamic-key' not in json.dumps(resolver.diagnostics)


def test_envelope_decode_diagnostic_never_persists_rejected_content(caplog):
    provider, _ = transport(['PRIVATE_CUSTOMER_TEXT this is not JSON'])
    with pytest.raises(ContractError) as error:
        provider.complete_json(system='fixture', user={}, schema=INTENT_RESULT_SCHEMA, timeout=1)
    assert error.value.diagnostics['code'] == 'json_envelope_or_decode_invalid'
    assert error.value.diagnostics['field_path'] == []
    assert 'PRIVATE_CUSTOMER_TEXT' not in caplog.text


def test_only_current_customer_events_are_classification_sources_and_history_has_no_ids():
    class Model:
        calls = []
        async def complete(self, *, system, user, schema, timeout):
            self.calls.append(deepcopy(user))
            if len(self.calls) == 1:
                return {'intents': [{'kind': 'product_search', 'certainty': 'explicit',
                                    'source_event_refs': ['event:old'],
                                    'source_quotes': [{'event_ref': 'event:old', 'quote': '请看看10001'}]}]}
            return valid()

    model = Model()
    resolver = JsonIntentResolver(model)
    assert asyncio.run(resolver.resolve(context(), 5)) == valid()['intents']
    assert [row['event_ref'] for row in model.calls[0]['messages']] == ['event:current']
    assert all(row['role'] == 'customer' for row in model.calls[0]['messages'])
    assert all(set(row) == {'role', 'text'} for row in model.calls[0]['conversation_context_only'])
    assert model.calls[1]['previous_rejection']['field_path'] == ['intents', 0, 'source_event_refs']
    assert model.calls[1]['intent_resolution_event_refs'] == ['event:current']


def test_repair_shares_original_deadline_instead_of_starting_another_full_timeout():
    clock = [0.0]
    class Model:
        budgets = []
        async def complete(self, *, system, user, schema, timeout):
            self.budgets.append(timeout)
            if len(self.budgets) == 1:
                clock[0] += 7.0
                raise ContractError('invalid_model_output', 'fixture failure')
            return valid()

    model = Model()
    resolver = JsonIntentResolver(model, clock=lambda: clock[0])
    assert asyncio.run(resolver.resolve(context(), 10)) == valid()['intents']
    assert model.budgets == [10.0, 3.0]


def test_second_invalid_interpretation_is_rejected_and_never_confers_source_authority():
    class Model:
        calls = 0
        async def complete(self, **kwargs):
            self.calls += 1
            wrong = valid()
            wrong['intents'][0]['source_quotes'][0]['quote'] = 'not present in any customer message'
            return wrong

    model = Model()
    with pytest.raises(BrainError, match='intent_quote_mismatch') as error:
        asyncio.run(JsonIntentResolver(model).resolve(context(), 5))
    assert model.calls == 2
    assert error.value.diagnostics['attempts'] == 2
    assert 'not present in any' not in json.dumps(error.value.diagnostics)


def test_expired_original_budget_cannot_start_repair():
    clock = [0.0]
    class Model:
        calls = 0
        async def complete(self, **kwargs):
            self.calls += 1
            clock[0] = 11.0
            raise ContractError('invalid_model_output', 'fixture failure')

    model = Model()
    with pytest.raises(BrainError, match='stage_timeout'):
        asyncio.run(JsonIntentResolver(model, clock=lambda: clock[0]).resolve(context(), 10))
    assert model.calls == 1
