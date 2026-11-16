"""A company source can verify multiple independent contacts or documents."""
import asyncio
import time

import pytest

from sales_brain.core.agent_loop import TrustedContext
from sales_brain.core.tool_gateway import RequestScope
from sales_brain.runtime import DomainTools


@pytest.mark.parametrize('name,args,records', [
    ('get_contact_routes', {}, [
        {'contact_id': 'phone', 'kind': 'phone', 'value': '+12025550123',
         'public': True, 'region': 'BKK', 'revision': 2,
         'valid_until': '2099-01-01T00:00:00Z'},
        {'contact_id': 'web', 'kind': 'web', 'value': 'https://example.invalid/contact',
         'public': True, 'region': 'BKK', 'revision': 3,
         'valid_until': '2099-01-01T00:00:00Z'},
    ]),
    ('search_company_knowledge', {'query': '公司业务范围'}, [
        {'document_id': 'company-profile:one', 'revision': 2, 'text': '范围一'},
        {'document_id': 'doc:two', 'revision': 4, 'text': '范围二'},
    ]),
])
def test_gateway_keeps_distinct_facts_sharing_one_verified_source(name, args, records):
    context = TrustedContext('tenant:test', 'customer:1', 'BKK', 'live',
                             'conversation:1', 'task:1', 1, 'ctx:1', 'cfg:1', (), ())
    scope = RequestScope('tenant:test', 'test', 'live', frozenset({'customer:1'}),
                         frozenset({'BKK'}), frozenset({'read'}))
    rows = [{**record, 'source_ref': 'company-source:one'} for record in records]

    class Client:
        def get(self, path, **kwargs):
            assert path == '/v1/contacts'
            return {'items': rows}

        def post(self, path, **kwargs):
            assert path == '/v1/search'
            return {'items': rows}

    def clients(service):
        assert service == 'company-knowledge'
        return Client()

    gateway = DomainTools(context, None, clients).build()
    result = asyncio.run(gateway.call(name, args, scope=scope,
                                     context=context.tool_context(), deadline=time.monotonic() + 5))
    assert result.status == 'ok', result.code
    assert result.source_refs == ('company-source:one',)
    assert len(result.data['facts']) == 2
    assert len({fact['fact_ref'] for fact in result.data['facts']}) == 2
