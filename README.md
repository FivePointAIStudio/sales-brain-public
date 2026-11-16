# Sales Brain

Sales Brain is a Python service that turns an incoming sales event and authorized business context into a structured, evidence-backed action plan. It supports source-bound intent, bounded read-tool loops, explicit evidence and policy checks, and plan-only output. It does not execute business actions, send customer messages, or replace the caller’s authorization and execution checks.

## Capabilities and boundaries

- Validates intents against the current customer message and source references.
- Calls allowlisted domain-service read endpoints, with bounded requests and response sizes.
- Produces versioned action plans with evidence, dependencies, and execution conditions.
- Enforces checks around product identity, displayed-product references, customer language, and claims about completed actions.
- Exposes health and administrative endpoints behind bearer-token authentication and scopes.

The service requires your own business systems to provide customer context, conversation history, product catalog and matching, company knowledge, sales capabilities, quotations, presentation history, and any other enabled read tools. The trusted caller must perform current authorization and execute approved actions, then provide actual receipts before the plan may claim completion. No ERP, CRM, customer dataset, product catalog, credentials, hosted model, or production service is included.

## Install

Requires Python 3.11 or later. From this directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[test]'
```

Run the service with your own private deployment configuration:

```sh
uvicorn sales_brain.app:app --host 127.0.0.1 --port 8000
```

The configuration contract is represented by `examples/`. Replace every placeholder and configure secrets outside the repository. Set `SA_DATABASE_URL`, `SA_AUTH_TOKENS_FILE`, `SA_SERVICE_REGISTRY_FILE`, and `SA_TENANT_ID`; model credentials and provider settings belong in the private model configuration directory indicated by `BRAIN_PRIVATE_DIR`. The service registry must contain only trusted, TLS-protected endpoints. The included `ai_contracts` package is part of this distribution; no separately named `ai-contracts` PyPI package is required.

Apply the database schema only to a database you own and have explicitly configured:

```sh
sales-brain-migrate
```

## Offline tests

```sh
pytest -q
```

The PostgreSQL integration test is skipped unless `SA_TEST_DATABASE_URL` is explicitly provided. The test suite uses synthetic fixtures. Tests do not require a model provider or production services.

## Package data

Schema resources are installed inside `ai_contracts/schemas/`. Migration SQL and the optional admin UI module descriptor are installed under the `share/sales-brain-public/` data directory.
