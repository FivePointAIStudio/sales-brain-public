"""A shown card may be re-read as evidence; the read never selects an ambiguous item or sends a new card."""
import asyncio
from copy import deepcopy
from dataclasses import replace

import pytest

from sales_brain.core.agent_loop import TrustedContext
from sales_brain.core.contracts import ContractError, validate_json
from sales_brain.core.tool_gateway import RequestScope
from sales_brain.runtime import DomainTools


def presentation(**updates):
    return {
        'tenant_id':'tenant-test','customer_ref':'customer:1','region':'BKK',
        'presentation_ref':'presentation:1','product_id':'demo:1','product_ref':'demo:1',
        'product_key':'demo:1','display_key':'demo:1','fact_version':'v1','revision':2,
        'status':'rendered',**updates,
    }


def product(**updates):
    return {
        'product_id':'demo:1','stable_id':'demo:1','display_key':'demo:1','code':'SYN-10001',
        'version':'v2','facts':{'component':'verified composition'},
        'verified_fields':['component'],'source':{'record_id':'1','row_sha256':'source-hash'},
        'active':True,'public':True,**updates,
    }


def context(rows=None, text='我要这个，帮我记下需求。'):
    return TrustedContext(
        'tenant-test','customer:1','BKK','live','conversation:1','task:1',1,'ctx:1','cfg:1',
        ('event:1',),({'event_ref':'event:1','customer_ref':'customer:1','role':'customer','text':text},),
        presentations=tuple(rows if rows is not None else [presentation()]),
        control={'presentation_ledger_available':True},
    )


class Services:
    def __init__(self, rows=None, products=None):
        self.rows=rows if rows is not None else [presentation()]
        self.products=products if products is not None else [product()]
        self.calls=[]

    def __call__(self, service):
        owner=self
        class Client:
            def get(self,path,**kwargs):
                owner.calls.append((service,'get',path,deepcopy(kwargs)))
                assert service=='customer-orchestrator'
                assert path=='/v1/customers/customer%3A1/presentations'
                return {'items':deepcopy(owner.rows)}
            def post(self,path,**kwargs):
                owner.calls.append((service,'post',path,deepcopy(kwargs)))
                # This is the explicit read-only hydrate API, never a command or channel send.
                assert service=='product-catalog' and path=='/v1/products/hydrate'
                assert kwargs['json']=={'keys':['demo:1'],'region':'BKK'}
                return {'items':deepcopy(owner.products),'generation':'catalog-generation:2'}
        return Client()


def read(initial=None, services=None, reference='presentation:1'):
    initial=initial or context()
    services=services or Services()
    tools=DomainTools(initial,None,services)
    result=asyncio.run(tools.presented_product({'presentation_ref':reference},initial.tool_context(),5))
    return result,tools,services


def test_single_shown_product_can_be_refreshed_after_pronoun_only_request():
    result,tools,services=read()
    assert result.status=='ok'
    assert result.data['products'][0]['product_ref']=='demo:1'
    assert result.data['products'][0]['product_code']=='SYN-10001'
    assert result.data['products'][0]['fact_version']=='v2'
    assert result.data['facts_changed_since_display'] is True
    assert result.data['displayed_fact_version']=='v1'
    assert result.data['selection_basis']=='single_shown_product'
    assert set(tools.products)=={'demo:1'}
    assert [call[:3] for call in services.calls]==[
        ('customer-orchestrator','get','/v1/customers/customer%3A1/presentations'),
        ('product-catalog','post','/v1/products/hydrate'),
    ]
    assert not any(service in {'channel-and-voice','sales-workflows'} for service,*_ in services.calls)


@pytest.mark.parametrize('updates',[
    {'tenant_id':'other-tenant'}, {'tenant_ref':'other-tenant'},
    {'customer_ref':'customer:other'}, {'region':'HCM'}, {'environment':'shadow'},
    {'scope':{'tenant_ref':'other-tenant'}}, {'scope':{'tenant_id':'other-tenant'}},
    {'scope':{'customer_ref':'customer:other'}}, {'scope':{'region':'HCM'}},
    {'status':'accepted'}, {'status':'queued'}, {'status':'unknown'}, {'status':'confirmed_not_sent'},
    {'revision':True}, {'revision':0}, {'fact_version':''}, {'product_id':'erp:2'},
])
def test_unscoped_pending_or_unversioned_presentation_cannot_authorize_a_read(updates):
    result,tools,services=read(context([presentation(**updates)]))
    assert result.status=='invalid_request'
    assert not tools.products and not services.calls


def test_missing_reference_and_unavailable_ledger_are_not_guessed():
    result,tools,services=read(reference='presentation:unknown')
    assert result.code=='shown_presentation_unverified'
    assert not tools.products and not services.calls
    initial=replace(context(),control={'presentation_ledger_available':False})
    result,tools,services=read(initial)
    assert result.code=='presentation_ledger_unavailable'
    assert not tools.products and not services.calls


@pytest.mark.parametrize('rows',[
    [], [presentation(revision=3)], [presentation(fact_version='changed')],
    [presentation(customer_ref='customer:other')], [presentation(status='confirmed_not_sent')],
    [presentation(),presentation()],
])
def test_current_ledger_is_rechecked_before_catalog_read(rows):
    result,tools,services=read(services=Services(rows=rows))
    assert result.code=='presentation_changed'
    assert not tools.products
    assert len(services.calls)==1


@pytest.mark.parametrize('updates',[
    {'product_id':'erp:2'}, {'stable_id':'erp:2'}, {'display_key':'different-product'},
    {'active':False}, {'public':False}, {'region':'HCM'}, {'tenant_id':'other-tenant'},
])
def test_hydration_must_preserve_current_stable_identity_and_scope(updates):
    result,tools,_=read(services=Services(products=[product(**updates)]))
    assert result.code=='presented_product_identity_changed'
    assert not tools.products


def test_removed_product_is_honestly_unavailable_and_not_registered_as_evidence():
    result,tools,_=read(services=Services(products=[]))
    assert result.status=='not_found' and result.code=='presented_product_no_longer_available'
    assert not tools.products


def test_multiple_shown_products_do_not_allow_model_to_guess_a_pronoun_selection():
    rows=[presentation(),presentation(presentation_ref='presentation:2',product_id='erp:2',
        product_ref='erp:2',product_key='erp:2',display_key='erp:2')]
    result,tools,_=read(context(rows),Services(rows=rows))
    assert result.code=='presented_product_selection_ambiguous'
    assert result.data=={'needs_clarification':True,'shown_product_count':2}
    assert not tools.products and 'products' not in result.data
    explicit,tools,_=read(context(rows,text='我要SYN-10001。'),Services(rows=rows))
    assert explicit.status=='ok' and explicit.data['selection_basis']=='explicit_current_product_code'
    wrong,tools,_=read(context(rows,text='我要ALT-10001。'),Services(rows=rows))
    assert wrong.code=='presented_product_selection_ambiguous' and not tools.products


def test_tool_schema_accepts_only_the_presentation_reference_not_a_caller_product_id():
    initial=context()
    tools=DomainTools(initial,None,Services()).build()
    scope=RequestScope('tenant-test','test','live',frozenset({'customer:1'}),frozenset({'BKK'}),frozenset({'read'}))
    registered=next(item for item in tools.descriptions(scope,initial.tool_context()) if item['name']=='read_presented_product')
    validate_json({'presentation_ref':'presentation:1'},registered['parameters'])
    with pytest.raises(ContractError):
        validate_json({'presentation_ref':'presentation:1','product_id':'erp:other'},registered['parameters'])
