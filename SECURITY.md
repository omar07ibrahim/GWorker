# Security policy

## Supported code

GWorker has no published release yet. Security fixes target the current `main`
branch only.

| Version | Supported |
| --- | --- |
| `main` | Yes |
| Tagged releases | None published |

## Report a vulnerability privately

Please use [GitHub private vulnerability reporting](https://github.com/omar07ibrahim/GWorker/security/advisories/new).
Do not disclose a suspected vulnerability in a public issue, pull request,
discussion, screenshot, terminal transcript, or generated evidence bundle.

Include the smallest synthetic reproducer you can, the affected revision,
expected and observed behavior, and the security impact. Never attach a real
GWorker journal: objectives, timestamps, interruption reasons, and policy
reviews may be sensitive even when they do not contain explicit identity fields.

## Security boundaries

The supported storage hardening targets Linux/POSIX filesystems. GWorker checks
journal permissions, path substitution, canonical replay, and logical
consistency, but it does not encrypt journals and does not defend against a
hostile process already running as the same operating-system user. Device or
full-disk encryption remains necessary for data at rest.

Committed demos and visual evidence must use synthetic records, reject
host-specific paths and secret-shaped output, and remain reproducible from the
documented commands. A security report should preserve those same boundaries.

## Disclosure process

Reports are triaged through the private advisory. A coordinated fix should add
a regression test, update affected documentation or evidence, pass the complete
CI matrix, and be disclosed only after the fix is available. Repository
licensing and rights questions are tracked separately and are not security
reports.
