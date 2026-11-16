from copy import deepcopy
from typing import Literal
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from psycopg.types.json import Jsonb
from ai_contracts import digest
from .db import connect


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class Communication(StrictModel):
    max_questions: int = Field(default=1, ge=0, le=3)
    explain_next_step: bool = True


class Style(StrictModel):
    verbosity: Literal['concise','balanced','detailed'] = 'balanced'
    emoji: Literal['none','limited'] = 'none'
    avoid_card_repetition: bool = True


class AgentConfig(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    public_identity: str = Field(min_length=1, max_length=100)
    region: Literal['BKK','HN','HCM','IDK']
    language: Literal['zh','en','th','vi','id']
    brand: str = Field(default='', max_length=120)
    role: str = Field(default='面料销售助手', min_length=1, max_length=160)
    tone: str = Field(default='自然、专业，结合客户已经表达的需求推进下一步。', max_length=2000)
    greeting: str = Field(default='', max_length=600)
    greeting_rule: Literal['first_customer_turn','never','return_after_gap'] = 'first_customer_turn'
    communication: Communication = Field(default_factory=Communication)
    response_style: Style = Field(default_factory=Style)
    active: bool = True


FIELD_EFFECTS = {
    'name': '后台管理名称，不改变公开身份或目录权限',
    'public_identity': '模型公开自称；与后台名称分别维护',
    'region': '客户与Agent地区必须一致；工具仍按服务端授权过滤',
    'language': '没有明确客户语言偏好时使用的默认回复语言',
    'brand': '展示品牌背景，不授予跨品牌商品访问权限',
    'role': '接待角色与工作范围的提示配置，不扩大实际工具能力',
    'tone': '回复语气指导，不能覆盖事实、来源与执行政策',
    'greeting': '满足问候条件时供模型使用的问候文案',
    'greeting_rule': '首次、从不或间隔返回的问候策略',
    'communication': '最多追问数量与是否说明下一步',
    'response_style': '长度、表情及避免重复卡片的表达偏好；账本防重始终执行',
    'active': '发布为false后，新计划被拒绝',
}


def seed_defaults(tenant):
    labels = {'BKK': ('曼谷接待','th'), 'HN': ('河内接待','vi'), 'HCM': ('胡志明接待','vi'), 'IDK': ('印尼接待','id')}
    with connect() as conn:
        for region, (name, language) in labels.items():
            config = AgentConfig(name=name, public_identity='Example Assistant', region=region, language=language).model_dump()
            conn.execute('''INSERT INTO sb_agents(tenant_id,agent_id,region,draft,published,updated_by)
                VALUES(%s,%s,%s,%s,%s,'initial-migration') ON CONFLICT DO NOTHING''',
                (tenant, region.lower(), region, Jsonb(config), Jsonb(config)))


def dto(row):
    return {**row, 'id': row['agent_id'], 'effective': deepcopy(row['published']),
            'config_version': 'agent:' + row['agent_id'] + ':' + str(row['published_revision']) + ':' + digest(row['published'])[:16],
            'field_sources': {key: {'source': 'published_agent', 'revision': row['published_revision']} for key in row['published']},
            'field_effects': FIELD_EFFECTS}


def list_agents(actor):
    with connect() as conn:
        rows = conn.execute('SELECT * FROM sb_agents WHERE tenant_id=%s AND region=ANY(%s) ORDER BY region,agent_id',
                             (actor.tenant_id, sorted(actor.regions))).fetchall()
    return [dto(r) for r in rows]


def get_agent(actor, agent_id):
    with connect() as conn:
        row = conn.execute('SELECT * FROM sb_agents WHERE tenant_id=%s AND agent_id=%s', (actor.tenant_id, agent_id)).fetchone()
    if not row:
        raise HTTPException(404, 'agent_not_found')
    actor.require_region(row['region'])
    return dto(row)


def save_agent(actor, agent_id, expected_revision, draft=None, publish=False):
    if not agent_id or len(agent_id) > 120 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-' for c in agent_id):
        raise HTTPException(422, 'agent_id_invalid')
    with connect() as conn:
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', (actor.tenant_id + ':agent:' + agent_id,))
        row = conn.execute('SELECT * FROM sb_agents WHERE tenant_id=%s AND agent_id=%s FOR UPDATE', (actor.tenant_id, agent_id)).fetchone()
        if row:
            actor.require_region(row['region'])
            if row['revision'] != expected_revision:
                raise HTTPException(409, 'agent_version_conflict')
        elif expected_revision != 0 or draft is None:
            raise HTTPException(409, 'agent_version_conflict')
        config = AgentConfig.model_validate(draft if draft is not None else row['draft']).model_dump()
        actor.require_region(config['region'])
        if row and row['region'] != config['region']:
            raise HTTPException(409, 'agent_region_immutable_create_another_agent')
        revision = expected_revision + 1
        if row:
            result = conn.execute('''UPDATE sb_agents SET revision=%s,draft=%s,
                published=CASE WHEN %s THEN %s ELSE published END,
                published_revision=CASE WHEN %s THEN %s ELSE published_revision END,
                updated_by=%s,updated_at=now() WHERE tenant_id=%s AND agent_id=%s RETURNING *''',
                (revision, Jsonb(config), publish, Jsonb(config), publish, revision, actor.subject, actor.tenant_id, agent_id)).fetchone()
        else:
            result = conn.execute('''INSERT INTO sb_agents(tenant_id,agent_id,region,revision,published_revision,draft,published,updated_by)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *''',
                (actor.tenant_id, agent_id, config['region'], revision, revision if publish else 0,
                 Jsonb(config), Jsonb(config if publish else {}), actor.subject)).fetchone()
        conn.execute('INSERT INTO sb_agent_audit(tenant_id,agent_id,revision,actor,operation,content_hash) VALUES(%s,%s,%s,%s,%s,%s)',
                     (actor.tenant_id, agent_id, revision, actor.subject, 'publish' if publish else 'save_draft', digest(config)))
        return dto(result)


def effective_agent(actor, customer):
    agent = get_agent(actor, customer.get('agent_id') or customer['region'].lower())
    published = agent['published']
    if agent['published_revision'] < 1 or not published.get('active'):
        raise HTTPException(409, 'agent_not_active_or_published')
    if published['region'] != customer['region']:
        raise HTTPException(409, 'agent_customer_region_mismatch')
    result = deepcopy(published)
    result.update(id=agent['agent_id'], name=published['name'], config_version=agent['config_version'],
                  field_sources=deepcopy(agent['field_sources']), brand_display=published['brand'],
                  catalog_scope_source='authenticated_domain_services')
    # Explicit customer language is a preference, not inferred from a customer code/name.
    if customer.get('language_explicit') and customer.get('language'):
        result['language'] = customer['language']
        result['field_sources']['language'] = {'source': 'explicit_customer_language', 'revision': customer['revision']}
    return result
