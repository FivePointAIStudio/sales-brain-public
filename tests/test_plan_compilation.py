from copy import deepcopy
from datetime import datetime, timezone

import pytest

from sales_brain.core.agent_loop import TrustedContext
from sales_brain.core.contracts import ContractError, assemble_plan
from sales_brain.core.plan_compile import binding_examples, normalize_conditional_dependencies


def current_context():
    return TrustedContext('test-tenant','test-customer','BKK','sandbox','conversation','task',1,'context','config',
        ('event:1',), ({'event_ref':'event:1','role':'customer','text':'我想采购这款产品。'},),
        confirmed_intents=({'intent_ref':'intent:1','kind':'product_interest','source_event_refs':['event:1']},)).to_public_dict()


def draft():
    sample=binding_examples(current_context())
    return {'status':'ready','intent_coverage':[sample['covered_action_link']['coverage']],
            'actions':deepcopy(sample['case_and_receipt_subgraph']['actions'])}


def assemble(value):
    return assemble_plan(value,current_context(),run_id='run:test',decision_id='decision:test',
                         now=datetime.now(timezone.utc))


def test_multi_action_example_is_a_valid_structural_plan_and_preserves_exact_intent_binding():
    value=assemble(draft())
    assert [action['intent_ref'] for action in value['actions']]==['intent:1','intent:1']
    assert value['intent_coverage'][0]['confirmation']=='confirmed'


def test_compiler_only_adds_missing_existing_receipt_dependency_and_is_idempotent():
    original=draft()
    original['actions'][1]['depends_on']=[]
    saved=deepcopy(original)
    compiled,changes=normalize_conditional_dependencies(original)
    assert original==saved
    assert compiled['actions'][1]['depends_on']==[{'action_id':'save_need','condition':'success'}]
    assert compiled['actions'][0]==original['actions'][0]
    assert compiled['actions'][1]['payload']==original['actions'][1]['payload']
    assert changes==[{'action_index':1,'target_action_index':0,'segment_indices':[0],'condition':'success'}]
    assert normalize_conditional_dependencies(compiled)==(compiled,[])
    assemble(compiled)


@pytest.mark.parametrize('target',['nonexistent_business_action','acknowledge_need'])
def test_compiler_never_invents_an_unknown_target_or_self_dependency(target):
    original=draft()
    original['actions'][1]['depends_on']=[]
    original['actions'][1]['payload']['conditional_segments'][0]['when']['action_id']=target
    compiled,changes=normalize_conditional_dependencies(original)
    assert not changes and compiled==original
    with pytest.raises(ContractError,match='invalid_condition') as error:
        assemble(compiled)
    assert error.value.diagnostics['field_path']==['actions',1,'payload','conditional_segments',0,'when','action_id']


def test_added_wait_cannot_bypass_cycle_detection_or_weaken_an_existing_wait():
    original=draft()
    original['actions'][1]['depends_on']=[]
    original['actions'][0]['depends_on']=[{'action_id':'acknowledge_need','condition':'success'}]
    compiled,_=normalize_conditional_dependencies(original)
    assert compiled['actions'][0]['depends_on']==original['actions'][0]['depends_on']
    with pytest.raises(ContractError,match='cyclic_dependency'):
        assemble(compiled)


def test_compiler_does_not_rewrite_failed_status_condition_as_success_or_authorize_its_claim():
    original=draft()
    original['actions'][1]['depends_on']=[]
    original['actions'][1]['payload']['conditional_segments'][0]['when']['execution_status']=['failed']
    compiled,_=normalize_conditional_dependencies(original)
    assert compiled['actions'][1]['depends_on']==[{'action_id':'save_need','condition':'terminal'}]
    from sales_brain.core.policies import validate_customer_text
    with pytest.raises(ContractError,match='completion_condition_not_success'):
        validate_customer_text(assemble(compiled),current_context(),{})
