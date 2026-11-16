CREATE TABLE IF NOT EXISTS sb_agents (
 tenant_id text NOT NULL, agent_id text NOT NULL, region text NOT NULL CHECK(region IN ('BKK','HN','HCM','IDK')),
 revision bigint NOT NULL DEFAULT 1, published_revision bigint NOT NULL DEFAULT 1,
 draft jsonb NOT NULL, published jsonb NOT NULL,
 updated_by text NOT NULL, updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(tenant_id,agent_id)
);
CREATE TABLE IF NOT EXISTS sb_agent_audit (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, tenant_id text NOT NULL,agent_id text NOT NULL,
 revision bigint NOT NULL, actor text NOT NULL, operation text NOT NULL,
 content_hash text NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sb_runs (
 run_id text PRIMARY KEY, tenant_id text NOT NULL, caller_ref text NOT NULL, request_id text NOT NULL,
 customer_ref text NOT NULL, event_ref text, payload_hash text NOT NULL, region text,
 status text NOT NULL DEFAULT 'running', result jsonb, error_code text,
 lease_until timestamptz NOT NULL DEFAULT now()+interval '5 minutes',
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(tenant_id,caller_ref,request_id)
);
