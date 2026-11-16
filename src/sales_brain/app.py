from __future__ import annotations
import asyncio
import os
from fastapi import Depends, HTTPException, Query
from pydantic import Field
from starlette.responses import JSONResponse
from ai_contracts import Actor, ServiceClient, ServiceError, require_scope, create_app as base_app
from . import agents, runs, runtime
from .agents import AgentConfig, StrictModel
from .core.agent_loop import BrainError
from .core.contracts import ContractError
from .core.diagnostics import safe_failure
from .provider import V21DeepSeekProvider, V21JsonModel
from .transport.errors import BrainError as ProviderError
from .db import connect


class PlanRequest(StrictModel):
    customer_ref: str = Field(min_length=1,max_length=200)
    event_ref: str | None = Field(default=None,max_length=240)
    request_id: str = Field(min_length=1,max_length=120)


class SaveAgent(StrictModel):
    expected_revision: int = Field(ge=0)
    draft: AgentConfig


class PublishAgent(StrictModel):
    expected_revision: int = Field(ge=1)


def transport():
    return V21DeepSeekProvider(private_dir=os.environ.get('BRAIN_PRIVATE_DIR') or os.environ.get('SA_MODEL_PRIVATE_DIR'))


def create_app():
    app = base_app('sales-brain')

    @app.exception_handler(BrainError)
    @app.exception_handler(ContractError)
    async def decision_error(request,error):
        status = 409 if error.code in {'context_changed','customer_inactive','event_not_current_customer_source'} else 422
        return JSONResponse({'error':{'code':error.code,'message':error.code}},status_code=status)

    @app.exception_handler(ProviderError)
    async def model_error(request,error):
        return JSONResponse({'error':{'code':error.code,'message':error.message}},status_code=error.status)

    @app.get('/v1/status')
    def status(actor: Actor = Depends(require_scope('read'))):
        health = transport().health()
        with connect() as conn:
            row = conn.execute('SELECT count(*) AS n FROM sb_agents WHERE tenant_id=%s AND region=ANY(%s)',(actor.tenant_id,sorted(actor.regions))).fetchone()
        return {'service':'sales-brain','status':'ready' if health['ready'] else 'model_not_ready',
            'model':health,'agents':row['n'],'schema_version':'2.1','execution_mode':'plan_only',
            'capabilities':['source_bound_intent','read_tool_loop','http_domain_tools','evidence_validation','multi_action_plans','published_agents'],
            'business_execution':False}

    @app.get('/v1/agents')
    def list_agents(actor: Actor = Depends(require_scope('read'))):
        return {'items':agents.list_agents(actor)}

    @app.get('/v1/agents/{agent_id}')
    def get_agent(agent_id:str,actor: Actor = Depends(require_scope('read'))):
        return agents.get_agent(actor,agent_id)

    @app.put('/v1/agents/{agent_id}')
    def save_agent(agent_id:str,data:SaveAgent,actor: Actor = Depends(require_scope('admin'))):
        return agents.save_agent(actor,agent_id,data.expected_revision,data.draft.model_dump())

    @app.post('/v1/agents/{agent_id}/publish')
    def publish_agent(agent_id:str,data:PublishAgent,actor: Actor = Depends(require_scope('admin'))):
        return agents.save_agent(actor,agent_id,data.expected_revision,publish=True)

    @app.get('/v1/config-version')
    def config_version(region:str,agent_id:str | None=Query(default=None,max_length=120),actor: Actor=Depends(require_scope('read'))):
        normalized=actor.require_region(region)
        row=agents.get_agent(actor,agent_id or normalized.lower())
        if row['region']!=normalized:
            raise HTTPException(409,'agent_customer_region_mismatch')
        return {'agent_id':row['agent_id'],'region':normalized,'config_version':row['config_version'],
                'published_revision':row['published_revision'],'active':bool(row['published'].get('active'))}

    @app.post('/v1/provider/probe')
    def probe(actor:Actor=Depends(require_scope('admin'))):
        # Explicit admin action, two anonymous format calls. Health never spends model tokens.
        return transport().probe_v21()

    @app.post('/v1/plan')
    async def plan(data:PlanRequest,actor:Actor=Depends(require_scope('execute'))):
        body=data.model_dump()
        run_id,prior=await asyncio.to_thread(runs.claim,actor,body)
        if prior is not None:
            if prior.get('status')=='failed':
                raise HTTPException(409,prior['error']['code'])
            return prior
        try:
            provider=transport()
            model_factory=getattr(app.state,'model_factory',None)
            model=model_factory() if model_factory else V21JsonModel(provider)
            if model_factory is None and not provider.health()['ready']:
                raise ProviderError('model_not_ready','模型尚未配置或尚未完成本版本匿名格式验证',503)
            async def mark(event):
                await asyncio.to_thread(runs.checkpoint,run_id,event)
            result=await runtime.plan(actor,data.customer_ref,data.event_ref,run_id,model,
                         client_factory=getattr(app.state,'client_factory',ServiceClient),checkpoint=mark)
            await asyncio.to_thread(runs.finish,run_id,result)
            return result
        except Exception as error:
            code=getattr(error,'code',None)
            if isinstance(error,HTTPException):
                code=error.detail if isinstance(error.detail,str) else 'request_rejected'
            await asyncio.to_thread(runs.finish,run_id,None,code or 'planning_failed',safe_failure(error))
            raise

    @app.get('/v1/runs/{run_id}')
    def read_run(run_id:str,actor:Actor=Depends(require_scope('read'))):
        return runs.read(actor,run_id)

    return app


app=create_app()
