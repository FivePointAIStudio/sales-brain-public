from uuid import uuid4
from fastapi import HTTPException
from psycopg.types.json import Jsonb
from ai_contracts import digest
from .db import connect


def claim(actor, payload):
    with connect() as conn:
        key = actor.tenant_id + ':' + actor.subject + ':' + payload['request_id']
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', (key,))
        old = conn.execute('SELECT *,lease_until<now() AS expired FROM sb_runs WHERE tenant_id=%s AND caller_ref=%s AND request_id=%s',
                           (actor.tenant_id, actor.subject, payload['request_id'])).fetchone()
        if old:
            if old['payload_hash'] != digest(payload):
                raise HTTPException(409,'idempotency_conflict')
            if old.get('region'):
                actor.require_region(old['region'])
            if old['status'] == 'ready':
                return old['run_id'], old['result']
            if old['status'] == 'failed':
                raise HTTPException(409, old['error_code'] or 'previous_plan_failed_new_request_required')
            if not old['expired']:
                raise HTTPException(409, 'planning_in_progress')
            # A process crash leaves a visibly failed decision, never a hidden automatic paid retry.
            conn.execute("UPDATE sb_runs SET status='failed',error_code='planner_lease_expired',updated_at=now() WHERE run_id=%s",(old['run_id'],))
            # Return after the transaction commits; do not roll back the recovery mark.
            return old['run_id'], {'status':'failed','error':{'code':'planner_lease_expired'}}
        run_id = 'brain:' + uuid4().hex
        conn.execute('''INSERT INTO sb_runs(run_id,tenant_id,caller_ref,request_id,customer_ref,event_ref,payload_hash)
            VALUES(%s,%s,%s,%s,%s,%s,%s)''',
            (run_id,actor.tenant_id,actor.subject,payload['request_id'],payload['customer_ref'],payload.get('event_ref'),digest(payload)))
        return run_id,None


def checkpoint(run_id, stage):
    with connect() as conn:
        conn.execute("""UPDATE sb_runs SET result=jsonb_set(
            jsonb_set(coalesce(result,'{}'::jsonb),'{last_checkpoint}',%s),
            '{checkpoints}',coalesce(result->'checkpoints','[]'::jsonb)||jsonb_build_array(%s::jsonb)),
            updated_at=now(),lease_until=now()+interval '5 minutes'
            WHERE run_id=%s AND status='running'""", (Jsonb(stage),Jsonb(stage),run_id))


def finish(run_id, result=None, code=None, diagnostics=None):
    with connect() as conn:
        if result is not None:
            conn.execute('''UPDATE sb_runs SET status='ready',result=%s,error_code=NULL,region=%s,updated_at=now()
                 WHERE run_id=%s AND status='running' ''',
                 (Jsonb(result),result.get('plan',{}).get('region'),run_id))
        else:
            # A rejected interpretation is still a diagnosable run. Preserve its
            # checkpoint history; never persist a raw prompt or rejected output.
            failure={'failure':{'code':code,'diagnostics':diagnostics or {}}}
            conn.execute('''UPDATE sb_runs SET status='failed',result=coalesce(result,'{}'::jsonb)||%s::jsonb,
                error_code=%s,updated_at=now() WHERE run_id=%s AND status='running' ''',
                (Jsonb(failure),code,run_id))


def read(actor, run_id):
    with connect() as conn:
        row = conn.execute('SELECT * FROM sb_runs WHERE tenant_id=%s AND run_id=%s',(actor.tenant_id,run_id)).fetchone()
    if not row:
        raise HTTPException(404,'brain_run_not_found')
    if row['region']:
        actor.require_region(row['region'])
    elif row['caller_ref'] != actor.subject and not actor.has_scope('admin'):
        raise HTTPException(403,'run_scope_forbidden')
    return {k:row[k] for k in ('run_id','customer_ref','event_ref','status','result','error_code','created_at','updated_at')}
