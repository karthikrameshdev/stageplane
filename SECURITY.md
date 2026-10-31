# Security policy

Stageplane is an early research implementation for a trusted worker pool. Security
reports are welcome for the executor, workers, model distribution, dashboard,
and supported transports.

## Supported versions

| Version | Security maintenance |
| --- | --- |
| Latest `main` | Best-effort investigation and fixes |
| Older commits and historical snapshots | Upgrade to the latest `main`; no separate maintenance branch |

The project does not currently offer a fixed security response SLA or a production
support guarantee.

## Report a vulnerability privately

Email **karthik.ramesh.dev@gmail.com** with the subject
**`[SECURITY] Stageplane vulnerability report`**. This is the reporting channel;
GitHub private vulnerability reporting is not required to use it. Do not publish
exploit details in public issues or pull requests before coordinated disclosure.

Include, where possible:

- The affected commit, Python version, operating system, and deployment topology.
- A description of the impact and affected component.
- Minimal reproduction steps or a proof of concept using a tiny checkpoint.
- Relevant redacted logs, configuration, and a suggested fix, if available.
- Whether you would like public credit after a fix is available.

Examples include authentication bypass, unsafe artifact paths or imports,
unexpected outbound requests, transport authorization failures, and exposure of
session data. Never send live tokens, private keys, private prompts, or restricted
model weights in a report.

The maintainer will investigate reports on a best-effort basis, coordinate a fix
and disclosure with the reporter, and credit the reporter with their consent.
For ordinary bugs or resource-capacity failures, use the
[bug report form](https://github.com/karthikrameshdev/stageplane/issues/new?template=bug_report.yml).
