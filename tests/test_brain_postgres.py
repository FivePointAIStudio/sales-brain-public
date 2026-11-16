from contextlib import contextmanager
import os
from uuid import uuid4
import pytest
import psycopg
from psycopg.rows import dict_row
from psycopg import sql
from ai_contracts import Actor
from fastapi import HTTPException
from sales_brain import db,agents,runs

pytestmark=pytest.mark.skipif(not os.environ.get('SA_TEST_DATABASE_URL'),reason='requires explicit isolated PostgreSQL test connection')


@pytest.fixture
def database(monkeypatch):
    url=os.environ['SA_TEST_DATABASE_URL'];schema='test_sb_'+uuid4().hex
    with psycopg.connect(url,autocommit=True) as conn:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    @contextmanager
    def connection():
        with psycopg.connect(url,row_factory=dict_row,options='-c search_path='+schema+' -c statement_timeout=15000') as conn:
            yield conn
    for module in (db,agents,runs):monkeypatch.setattr(module,'connect',connection)
    monkeypatch.setenv('SA_TENANT_ID','test-tenant');db.migrate()
    yield Actor('operator-test','test-tenant',frozenset({'admin'}),frozenset({'BKK','HN','HCM','IDK'}))
    with psycopg.connect(url,autocommit=True) as conn:
        conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


def test_agent_draft_does_not_change_runtime_until_publish_and_cas(database):
    actor=database;old=agents.get_agent(actor,'bkk')
    draft={**old['draft'],'public_identity':'新公开身份','tone':'简洁自然'}
    saved=agents.save_agent(actor,'bkk',old['revision'],draft)
    assert saved['effective']['public_identity']==old['effective']['public_identity']
    assert saved['config_version']==old['config_version']
    published=agents.save_agent(actor,'bkk',saved['revision'],publish=True)
    assert published['effective']['public_identity']=='新公开身份'
    assert published['config_version']!=old['config_version']
    with pytest.raises(HTTPException) as error:
        agents.save_agent(actor,'bkk',old['revision'],draft)
    assert error.value.status_code==409


def test_run_request_identity_and_durable_result(database):
    payload={'customer_ref':'c','event_ref':'e','request_id':'r'}
    run_id,cached=runs.claim(database,payload)
    assert cached is None
    with pytest.raises(HTTPException) as error:runs.claim(database,payload)
    assert error.value.detail=='planning_in_progress'
    result={'status':'ready','plan':{'region':'BKK'}}
    runs.finish(run_id,result)
    assert runs.claim(database,payload)==(run_id,result)
    with pytest.raises(HTTPException) as error:runs.claim(database,{**payload,'customer_ref':'another'})
    assert error.value.detail=='idempotency_conflict'


def test_failed_run_preserves_safe_checkpoint_history_without_retrying_model(database):
    payload={'customer_ref':'test-customer','event_ref':'test-event','request_id':'failed-interpretation'}
    run_id,_=runs.claim(database,payload)
    runs.checkpoint(run_id,{'stage':'loading_context'})
    diagnostic={'stage':'intent_resolution','code':'json_schema_violation',
                'errors':[{'validator':'required','field_path':['intents',0],'missing_fields':['source_quotes']}]}
    runs.checkpoint(run_id,{'stage':'intent_resolution_rejected','attempts':2,'diagnostics':diagnostic})
    runs.finish(run_id,None,'invalid_model_output',diagnostic)
    value=runs.read(database,run_id)
    assert value['status']=='failed'
    assert value['result']['last_checkpoint']['stage']=='intent_resolution_rejected'
    assert len(value['result']['checkpoints'])==2
    assert value['result']['failure']['diagnostics']==diagnostic
    with pytest.raises(HTTPException) as error:runs.claim(database,payload)
    assert error.value.detail=='invalid_model_output'
