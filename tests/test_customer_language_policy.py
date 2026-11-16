from copy import deepcopy

import pytest

from sales_brain.core.contracts import ContractError
from sales_brain.core.policies import validate_customer_text


PRODUCTS = {'demo:1': {'facts': {'component': '50%棉50%涤纶', 'width_cm': 146, 'weight_gsm': 130}}}


def context(text='我看10001这个产品不错，请展示。'):
    return {'trigger_event_refs': ['event:1'], 'messages': [{'event_ref': 'event:1', 'role': 'customer', 'text': text}],
            'confirmed_intents': [], 'presentations': []}


def plan(body='这款供您查看。您预计需要多少米？'):
    return {'actions': [
        {'action_id': 'card', 'type': 'RECOMMEND_PRODUCTS', 'payload': {
            'items': [{'product_ref': 'demo:1'}], 'display_reason': '', 'reply_group': 'cards'}},
        {'action_id': 'reply', 'type': 'REPLY_TEXT', 'payload': {
            'body': body, 'conditional_segments': [], 'reply_group': 'reply'}},
    ]}


@pytest.mark.parametrize('text', ['卡片已为您展示。', '我已帮您发送产品卡片。', 'The product has already been displayed.'])
def test_current_card_cannot_be_claimed_shown_before_actual_delivery(text):
    with pytest.raises(ContractError, match='premature_display_claim'):
        validate_customer_text(plan(text), context(), PRODUCTS)


def test_exact_live_reply_spec_repetition_is_rejected_across_different_reply_groups():
    body = 'SYN-10001是衬衫用色织面料，50% 棉 50% 涤纶，130g/m²，幅宽146cm。需要再看相似产品吗？'
    with pytest.raises(ContractError, match='product_card_facts_repeated'):
        validate_customer_text(plan(body), context(), PRODUCTS)
    validate_customer_text(plan(body), context('请解释这款的门幅和克重参数。'), PRODUCTS)


def test_public_prose_does_not_expose_erp_identity_but_wire_identity_is_retained():
    value = plan()
    validate_customer_text(value, context(), PRODUCTS)
    assert value['actions'][0]['payload']['items'][0]['product_ref'] == 'demo:1'
    value['actions'][0]['payload']['display_reason'] = '客户指定货号，唯一匹配SYN-10001（product_ref=demo:1）。'
    with pytest.raises(ContractError, match='internal_identifier_in_customer_text'):
        validate_customer_text(value, context(), PRODUCTS)


def test_saved_need_claim_requires_a_successful_business_receipt_not_a_reply_receipt():
    value = plan('我已为您记下采购意向。')
    with pytest.raises(ContractError, match='premature_completion_claim'):
        validate_customer_text(value, context(), PRODUCTS)
    value['actions'][1]['payload']['body'] = '您预计需要多少米？'
    value['actions'].append({'action_id': 'case', 'type': 'PROPOSE_PROJECT_PATCH', 'payload': {}})
    segment = {'text': '已为您记下SYN-10001的采购意向。', 'when': {'action_id': 'case', 'execution_status': ['succeeded']}}
    value['actions'][1]['payload']['conditional_segments'] = [segment]
    validate_customer_text(value, context(), PRODUCTS)
    segment['when']['action_id'] = 'card'
    with pytest.raises(ContractError, match='business_receipt_required'):
        validate_customer_text(value, context(), PRODUCTS)


def test_display_success_segment_requires_rendered_receipt_and_never_acceptance():
    value = plan()
    segment = {'text': '卡片已为您展示。', 'when': {'action_id': 'card', 'execution_status': ['succeeded'], 'delivery_status': ['accepted']}}
    value['actions'][1]['payload']['conditional_segments'] = [segment]
    with pytest.raises(ContractError, match='display_receipt_required'):
        validate_customer_text(value, context(), PRODUCTS)
    segment['when']['delivery_status'] = ['rendered']
    validate_customer_text(value, context(), PRODUCTS)
    segment['text'] = '卡片已发送给您。'
    with pytest.raises(ContractError, match='display_receipt_required'):
        validate_customer_text(value, context(), PRODUCTS)


def test_missing_sales_owner_cannot_be_replaced_by_a_future_promise():
    with pytest.raises(ContractError, match='unassigned_sales_promise'):
        validate_customer_text(plan('销售会尽快联系您。'), context(), PRODUCTS)
    validate_customer_text(plan('跟进需求待分配销售，目前无法承诺具体联系时间。'), context(), PRODUCTS)


def test_purchase_followup_does_not_repeat_fields_from_an_existing_card():
    value = plan('SYN-10001幅宽146cm，克重130gsm，您准备采购多少？')
    value['actions'] = value['actions'][1:]
    ctx = context('这款我想采购。')
    ctx['presentations'] = [{'product_ref': 'demo:1', 'status': 'rendered'}]
    with pytest.raises(ContractError, match='product_card_facts_repeated'):
        validate_customer_text(value, ctx, PRODUCTS)
