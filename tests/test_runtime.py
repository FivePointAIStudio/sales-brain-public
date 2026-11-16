"""Offline boundary tests; test doubles live only here, never in runtime configuration."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import pytest
from fastapi import HTTPException
from ai_contracts import Actor, ServiceError
from sales_brain.core.agent_loop import INTENT_RESULT_SCHEMA, BrainError, TrustedContext
from sales_brain.core.contracts import REQUEST_PARAMETERS, assemble_plan, ContractError
from sales_brain.core.tool_gateway import RequestScope
from sales_brain.runtime import plan, DomainTools, HttpContextProvider
from sales_brain.agents import AgentConfig

ACTOR=Actor('test','tenant-test',frozenset({'execute','read'}),frozenset({'BKK'}))
CONTACT='https://sales.example.invalid/contact'


def agent(actor,customer):
    return {**AgentConfig(name='测试接待',public_identity='Example Assistant',region='BKK',language='zh').model_dump(),
            'id':'bkk','config_version':'config:1'}


class Services:
    def __init__(self):
        self.requests=[]
        self.context={'customer':{'tenant_id':'tenant-test','customer_ref':'customer:1','region':'BKK','active':True,'agent_id':'bkk'},
            'context_version':'ctx:1','messages':[{'tenant_id':'tenant-test','customer_ref':'customer:1','event_ref':'event:1','role':'customer','text':'怎么联系你们？','source_revision':1}],
            'facts':[],'restrictions':[],'cases':[],'presentations':[],'receipts':[],
            'sources':{'presentations':{'status':'ok'}},'control':{},'partial':False}
        self.owner_available=True
    def __call__(self,service):
        owner=self
        class Client:
            def get(self,path,**kw):
                owner.requests.append((service,'get',path,kw))
                if service=='customer-context':
                    return deepcopy(owner.context)
                if service=='company-knowledge':
                    return {'items':[{'contact_id':'phone','kind':'web','value':CONTACT,'label':'销售入口','region':'BKK','public':True,'source_ref':'verified-contact:1','revision':2,'valid_until':'2099-01-01T00:00:00Z'}]}
                if path.endswith('/capabilities'):
                    return {'status':'ok','case_write_available':True,'notification_queue_available':True,
                        'capabilities':[{'command_ref':'sales-workflows:sales_followup:upsert:v1','request_type':'sales_followup',
                         'operations':['create','update','upsert'],'target_results':['followup_registered'],'parameters_schema':REQUEST_PARAMETERS['sales_followup']}]}
                if path.endswith('/owner'):
                    return {'recipient_ref':'owner:1','channel_route_ref':'route:1','source_ref':'assignment:1'} if owner.owner_available else {'status':'unavailable'}
                if service=='product-catalog' and path=='/v1/products':
                    return {'items':[{'product_id':'demo:1','stable_id':'demo:1','display_key':'demo:1','code':'ALT-10001','version':'v1','facts':{'width_cm':150},'verified_fields':['width_cm'],'source':{'record_id':'1','row_sha256':'h'}}],
                            'generation':'gen:1','complete':True}
                raise AssertionError((service,path))
            def post(self,*args,**kw):
                raise AssertionError('Planning made an unexpected business/transport POST')
        return Client()


def action(aid,kind,payload,intent,deps=()):
    return {'action_id':aid,'type':kind,'intent_ref':intent,'depends_on':[{'action_id':d,'condition':'success'} for d in deps], 'guards':[],'payload':payload}


def contact_draft(context,omit_followup=False,owner=True):
    intent=context['confirmed_intents'][0]
    ir,refs=intent['intent_ref'],intent['source_event_refs']
    bind=lambda aid,field:{'from_action_id':aid,'result_path':field}
    fact='company-contact:phone:2'
    actions=[action('reply','REPLY_TEXT',{'body':'联系入口：'+CONTACT,'fact_refs':[fact],'reply_group':'reply','conditional_segments':[]},ir),
        action('case','PROPOSE_PROJECT_PATCH',{'operation':'create','case_ref':None,'expected_version':None,'product_interests':[],'requirements':{'need_text':'客户询问公司联系方式'},'open_intents':[ir],'clear_fields':[],'source_event_refs':refs},ir)]
    if not omit_followup:
        actions.append(action('followup','HANDOFF_SALES',{'request_type':'sales_followup','operation':'upsert','command_ref':'sales-workflows:sales_followup:upsert:v1','case_ref':bind('case','resource_ref'),'expected_version':bind('case','resource_version'),'request_ref':None,'target_result':'followup_registered','parameters':{'summary':'客户询问联系方式','reason':'customer_requested_contact','product_refs':[]},'source_event_refs':refs},ir,['case']))
        if owner:
            actions.append(action('notify','NOTIFY_SALES',{'request_ref':bind('followup','resource_ref'),'recipient_ref':'owner:1','channel_route_ref':'route:1','body':'客户询问联系方式，请内部跟进。','source_event_refs':refs,'fact_refs':[fact]},ir,['followup']))
    return {'status':'ready','intent_coverage':[{'intent_ref':ir,'kind':intent['kind'],'confirmation':'candidate','source_event_refs':refs,'status':'planned','action_ids':[a['action_id'] for a in actions],'reason':'' if owner else 'sales_owner_unavailable'}], 'actions':actions}


class Model:
    def __init__(self,services,omit=False):
        self.round=0;self.services=services;self.omit=omit;self.calls=[];self.change_context=False
    async def complete(self,*,system,user,schema,timeout):
        self.calls.append(deepcopy(user))
        if schema==INTENT_RESULT_SCHEMA:
            return {'intents':[{'kind':'contact_request','certainty':'explicit','source_event_refs':['event:1'],
                'source_quotes':[{'event_ref':'event:1','quote':'怎么联系你们'}]}]}
        self.round+=1
        if self.round==1:
            return {'kind':'read_tools','calls':[{'call_id':name,'name':name,'arguments':{}} for name in ('get_contact_routes','read_sales_capabilities','read_sales_owner')]}
        if self.change_context:
            self.services.context['context_version']='ctx:changed'
        return {'kind':'final','plan':contact_draft(user['context'],self.omit,self.services.owner_available)}


def test_real_loop_reads_tools_and_reasons_again_before_four_action_plan():
    services=Services();model=Model(services)
    result=asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',model,services,agent))
    assert [a['type'] for a in result['plan']['actions']]==['REPLY_TEXT','PROPOSE_PROJECT_PATCH','HANDOFF_SALES','NOTIFY_SALES']
    assert result['tool_calls']==3 and result['model_rounds']==2
    assert result['observations'] and result['evidence_catalog']['facts']
    assert all(method=='get' for _,method,_,_ in services.requests)
    assert result['execution_receipts']==[]


def test_missing_contact_followup_is_rejected_not_silently_reduced_to_reply():
    services=Services();model=Model(services,omit=True)
    with pytest.raises(BrainError,match='round_budget_exhausted'):
        asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',model,services,agent))
    assert any(any(o.get('code')=='missing_required_action' for o in call.get('observations',[])) for call in model.calls)


def test_final_context_recheck_rejects_new_input():
    services=Services();model=Model(services);model.change_context=True
    with pytest.raises(BrainError,match='context_changed'):
        asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',model,services,agent))


def test_no_sales_owner_does_not_invent_a_notification_route():
    services=Services();services.owner_available=False
    result=asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',Model(services),services,agent))
    assert [a['type'] for a in result['plan']['actions']]==['REPLY_TEXT','PROPOSE_PROJECT_PATCH','HANDOFF_SALES']
    assert 'sales_owner_unavailable' in result['plan']['intent_coverage'][0]['reason']


def test_product_reference_preserves_prefix_and_accepts_english_word_spaces():
    services=Services();services.context['messages'][0]['text']='I want ALT-10001 please'
    provider=HttpContextProvider(ACTOR,'task:1',services,agent)
    scope=RequestScope('tenant-test','test','live',frozenset({'customer:1'}),frozenset({'BKK'}),frozenset({'read'}))
    context=asyncio.run(provider.load('customer:1',scope,'event:1',5))
    tools=DomainTools(context,None,services)
    result=asyncio.run(tools.resolve({'reference':'ALT-10001','source_event_refs':['event:1']},context.tool_context(),5))
    assert result.status=='ok' and result.data['products'][0]['product_code']=='ALT-10001'
    wrong=asyncio.run(tools.resolve({'reference':'SYN-10001','source_event_refs':['event:1']},context.tool_context(),5))
    assert wrong.status=='invalid_request'


def test_scope_does_not_accept_another_customers_source():
    services=Services();services.context['messages'][0]['customer_ref']='customer:other'
    with pytest.raises(BrainError,match='cross_customer_source'):
        asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',Model(services),services,agent))


def test_agent_identity_editable_and_unknown_fields_rejected():
    cfg=AgentConfig(name='后台名称',public_identity='Lisa',region='BKK',language='zh')
    assert cfg.public_identity=='Lisa'
    with pytest.raises(ValueError):
        AgentConfig(name='x',public_identity='x',region='BKK',language='zh',arbitrary_tool='send')


def test_extracted_wire_contract_matches_shared_versioned_sdk():
    import json
    from importlib.resources import files
    from sales_brain.core.contracts import PLAN_SCHEMA
    shared=json.loads(files('ai_contracts').joinpath('schemas/plan-v2.1/plan.schema.json').read_text())
    assert PLAN_SCHEMA==shared


def test_authoritative_catalog_code_preserves_internal_spaces_and_identity():
    from sales_brain.runtime import catalog_code
    assert catalog_code('  8W QUAN MIAN  ')=='8W QUAN MIAN'
    assert catalog_code('FRM B22-81')=='FRM B22-81'
    context=TrustedContext('tenant-test','customer:1','BKK','live','conv','task',1,'ctx','cfg',(),())
    tools=DomainTools(context,None,Services())
    value=tools.product({'product_id':'erp:23','stable_id':'erp:23','code':'Example weave','version':'v1','source':{'record_id':'23'},'facts':{}},context.tool_context())
    assert value['dcode']=='EXAMPLE WEAVE' and value['display_key']=='erp:23'


class ReceiptAcknowledgementModel(Model):
    """Exercise the complete planner contract, not just a graph predicate."""
    def __init__(self, services, variant='independent_answer'):
        super().__init__(services)
        self.variant=variant

    async def complete(self, **kwargs):
        step=await super().complete(**kwargs)
        if step.get('kind')!='final':
            return step
        draft=step['plan']
        reply=draft['actions'][0]
        if self.variant=='blocked_answer':
            reply['depends_on']=[{'action_id':'case','condition':'success'}]
            return step
        acknowledgement=action('ack','REPLY_TEXT',{
            'body':'','fact_refs':[],'reply_group':'ack',
            'conditional_segments':[{'text':'已为您记下这次联系需求。',
                'when':{'action_id':'case','execution_status':['succeeded']}}],
        },reply['intent_ref'],['case'])
        if self.variant=='placeholder_answer':
            reply['payload']['body']='您好。'
            reply['payload']['fact_refs']=[]
            acknowledgement['payload']['body']='联系入口：'+CONTACT
            acknowledgement['payload']['fact_refs']=['company-contact:phone:2']
        draft['actions'].append(acknowledgement)
        draft['intent_coverage'][0]['action_ids'].append('ack')
        return step


def test_contact_answer_allows_a_separate_receipt_bound_saved_need_acknowledgement():
    services=Services()
    result=asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',
        ReceiptAcknowledgementModel(services),services,agent))
    actions={a['action_id']:a for a in result['plan']['actions']}
    assert actions['reply']['depends_on']==[]
    assert CONTACT in actions['reply']['payload']['body']
    assert actions['ack']['depends_on']==[{'action_id':'case','condition':'success'}]
    assert actions['ack']['payload']['conditional_segments'][0]['when']['execution_status']==['succeeded']
    assert result['model_rounds']==2 and result['execution_receipts']==[]


@pytest.mark.parametrize('variant,code',[
    ('blocked_answer','contact_reply_blocked_by_internal_action'),
    ('placeholder_answer','contact_details_not_grounded'),
])
def test_contact_answer_cannot_hide_real_details_behind_internal_write(variant,code):
    services=Services();model=ReceiptAcknowledgementModel(services,variant)
    with pytest.raises(BrainError,match='round_budget_exhausted'):
        asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',model,services,agent))
    assert any(any(o.get('code')==code for o in call.get('observations',[])) for call in model.calls)


def test_real_loop_compiles_only_a_redundant_receipt_dependency_then_validates_the_whole_plan():
    class MissingDependency(ReceiptAcknowledgementModel):
        async def complete(self, **kwargs):
            value=await super().complete(**kwargs)
            if value.get('kind')=='final':
                value['plan']['actions'][-1]['depends_on']=[]
            return value
    services=Services();model=MissingDependency(services);checkpoints=[]
    async def checkpoint(row):checkpoints.append(row)
    result=asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',model,services,agent,checkpoint))
    assert result['plan']['actions'][-1]['depends_on']==[{'action_id':'case','condition':'success'}]
    assert any(row['stage']=='conditional_dependencies_compiled' for row in checkpoints)
    assert result['model_rounds']==2


@pytest.mark.parametrize('variant,code,path',[
    ('fact_ref','unverified_reference',['actions',0,'payload','fact_refs',0]),
    ('intent_ref','confirmed_intent_required',['actions',0,'intent_ref']),
    ('coverage_link','unbound_action',['actions',0,'intent_ref']),
])
def test_repair_receives_typed_real_references_and_exact_safe_field_without_log_values(variant,code,path):
    class BrokenThenFixed(Model):
        async def complete(self, **kwargs):
            value=await super().complete(**kwargs)
            if value.get('kind')=='final' and self.round==2:
                candidate=value['plan'];reply=candidate['actions'][0]
                if variant=='fact_ref':
                    reply['payload']['fact_refs']=['customer-private-value-do-not-log']
                elif variant=='intent_ref':
                    reply['intent_ref']='customer-private-value-do-not-log'
                    row=deepcopy(candidate['intent_coverage'][0]);row['intent_ref']=reply['intent_ref'];row['action_ids']=['reply']
                    candidate['intent_coverage'].append(row)
                else:
                    candidate['intent_coverage'][0]['action_ids'].remove('reply')
            return value
    services=Services();model=BrokenThenFixed(services);checkpoints=[]
    async def checkpoint(row):checkpoints.append(row)
    result=asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',model,services,agent,checkpoint))
    assert result['model_rounds']==3
    repair=model.calls[-1]
    assert 'rejected_candidate' in repair
    assert repair['allowed_references']['intents'][0]['intent_ref'].startswith('intent:')
    assert 'company-contact:phone:2' in repair['allowed_references']['fact_refs']
    assert 'sales-workflows:sales_followup:upsert:v1' in repair['allowed_references']['command_refs']
    assert 'event:1' not in repair['allowed_references']['fact_refs']
    diagnostic=next(o['diagnostics'] for o in repair['observations'] if o.get('code')==code)
    assert diagnostic['field_path']==path and diagnostic['repair_hint']
    import json
    assert 'customer-private-value-do-not-log' not in json.dumps(checkpoints)


@pytest.mark.parametrize('owner_available',[False,True])
def test_purchase_and_contact_share_one_case_with_live_requirements_and_executor_compatibility(owner_available):
    class MissingContacts(Services):
        def __call__(self,service):
            delegate=super().__call__(service)
            if service!='company-knowledge':return delegate
            class Knowledge:
                def get(self,path,**kwargs):return {'items':[]}
            return Knowledge()

    class CombinedModel(Model):
        async def complete(self,*,system,user,schema,timeout):
            self.calls.append(deepcopy(user))
            if schema==INTENT_RESULT_SCHEMA:
                return {'intents':[
                    {'kind':'product_interest','certainty':'explicit','source_event_refs':['event:1'],
                     'product_reference':'ALT-10001','source_quotes':[{'event_ref':'event:1','quote':'我要 ALT-10001'}]},
                    {'kind':'contact_request','certainty':'explicit','source_event_refs':['event:1'],
                     'source_quotes':[{'event_ref':'event:1','quote':'怎么联系你们'}]},
                ]}
            self.round+=1
            if self.round==1:
                return {'kind':'read_tools','calls':[
                    {'call_id':name,'name':name,'arguments':{}} for name in ('get_contact_routes','read_sales_capabilities','read_sales_owner')]
                    +[{'call_id':'product','name':'resolve_product_reference','arguments':{'reference':'ALT-10001','source_event_refs':['event:1']}}]}
            context=user['context'];by_kind={item['kind']:item for item in context['confirmed_intents']}
            contact=by_kind['contact_request'];purchase=by_kind['product_interest']
            contact_context=deepcopy(context);contact_context['confirmed_intents']=[contact]
            value=contact_draft(contact_context,owner=owner_available)
            value['actions'][0]['payload'].update(body='目前暂无已核实的公开联系方式；您预计需要多少米？',fact_refs=[])
            case=value['actions'][1]
            case['intent_ref']=purchase['intent_ref']
            case['payload']['open_intents']=[purchase['intent_ref'],contact['intent_ref']]
            case['payload']['product_interests']=[{'product_ref':'demo:1','source_event_refs':['event:1'],'status':'interested'}]
            case['payload']['requirements']={'need_text':'客户希望采购ALT-10001并询问联系方式'}
            for item in value['actions']:
                if item['type']=='NOTIFY_SALES':item['payload']['fact_refs']=[]
            value['intent_coverage'][0]['reason']+=';public_contacts_unavailable'
            value['intent_coverage'].append({'intent_ref':purchase['intent_ref'],'kind':'product_interest','confirmation':'candidate',
                'source_event_refs':['event:1'],'status':'planned','action_ids':['case'],'reason':''})
            if self.round==2:
                value['intent_coverage'][0]['action_ids'].remove('case')
            return {'kind':'final','plan':value}

    services=MissingContacts();services.owner_available=owner_available
    services.context['messages'][0]['text']='我要 ALT-10001，怎么联系你们？'
    model=CombinedModel(services)
    result=asyncio.run(plan(ACTOR,'customer:1','event:1','run:1',model,services,agent))
    assert result['model_rounds']==3
    value=result['plan'];cases=[action for action in value['actions'] if action['type']=='PROPOSE_PROJECT_PATCH']
    assert len(cases)==1 and cases[0]['payload']['product_interests'][0]['product_ref']=='demo:1'
    assert all('case' in row['action_ids'] for row in value['intent_coverage'])
    assert next(action for action in value['actions'] if action['type']=='REPLY_TEXT')['depends_on']==[]
    assert any(action['type']=='HANDOFF_SALES' for action in value['actions'])
    assert any(action['type']=='NOTIFY_SALES' for action in value['actions']) is owner_available
    repair=model.calls[-1]
    obligations=repair['allowed_references']['contact_action_requirements'][0]
    assert obligations['read_before_final']==[]
    assert obligations['public_contact_state']=='unavailable'
    assert obligations['additional_coverage_reasons']==['public_contacts_unavailable']
    reqs={req['name']:req for req in obligations['requirements']}
    assert reqs['save_interest']['required'] and reqs['sales_followup']['required']
    assert reqs['notify_sales']['required'] is owner_available
    if not owner_available:assert reqs['notify_sales']['blocked_reason']=='sales_owner_unavailable'
    rejected=next(row['diagnostics'] for row in repair['observations'] if row.get('code')=='missing_required_action')
    assert rejected['field_path']==['intent_coverage',0,'action_ids']
    assert rejected['requirement']=='save_interest' and rejected['required_action_type']=='PROPOSE_PROJECT_PATCH'
    assert rejected['matching_action_indices']==[1]
