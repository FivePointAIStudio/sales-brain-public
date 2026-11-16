# Security policy

Do not put API keys, bearer tokens, customer records, production URLs, or database credentials in source control, issue reports, or model prompts. Store credentials in a secret manager or owner-only files outside the repository. Use TLS for service connections and grant each caller only the required tenant, region, and scopes.

This service creates proposed plans. A trusted business system must independently validate current authorization, evidence, idempotency, and execution results before carrying out any action. Do not expose internal traces or evidence references to customers without an access policy.

For a vulnerability report, use the private security contact designated by the maintainers. Do not send vulnerability details through public issues, and do not include live credentials or customer data.
