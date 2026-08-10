# Contributing to GWorker

GWorker is maintained as an auditable, local-first research system. Small,
reviewable changes with explicit claims and reproducible evidence are preferred.

The repository is intentionally unlicensed while Omar resolves the rights
status of the preserved 2023 upload. Public availability does not grant reuse
rights. External contributions are not currently solicited; open an issue
before preparing a substantial patch or contributing third-party material.

## Development setup

Use Linux/POSIX and one of the supported Python versions (3.11, 3.12, or 3.13):

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --no-input -e '.[dev]'
```

Run the same core checks as CI:

```bash
ruff check .
ruff format --check .
mypy src scripts/visuals/capture_cli_motion.py scripts/verify_distribution.py
python -m compileall -q src tests scripts
coverage run --branch -m unittest discover -s tests -v
coverage report --show-missing --fail-under=90
```

Run every committed evidence check:

```bash
PYTHONPATH=src python -B scripts/visuals/generate.py --check
PYTHONPATH=src python -B -m scripts.visuals.generate_offline_replay --check
PYTHONPATH=src python -B scripts/visuals/capture_terminal.py check
PYTHONPATH=src python -B scripts/visuals/capture_cli_motion.py check
```

## Evidence changes

Generated diagrams, terminal captures, charts, GIFs, posters, transcripts, and
their manifests are review evidence—not decoration.

- Use only deterministic synthetic fixtures; never use a personal journal,
  secret, hostname, absolute host path, or personal identifier.
- Do not hand-edit generated output or checksum records.
- Regenerate with the repository scripts and inspect the actual rendered
  artifacts at their original dimensions.
- Commit the source change, regenerated outputs, and updated manifests together.
- Keep claims within the boundaries recorded by each manifest. In particular,
  synthetic replay diagnostics are not human outcomes or locked-evaluation
  results.
- If CI uploads a private evidence candidate because drift was detected, review
  it before adoption; do not publish an unreviewed candidate.

The evidence workflow deliberately fails closed when committed bytes drift from
reproduction.

## Pull requests

Keep history linear and give each commit one meaningful responsibility. Describe
the user-visible or research effect, the tests run, evidence changed, privacy
impact, and any nonclaims. All commit authors and committers for maintainer work
must be Omar Ibrahim using the repository noreply address.

Security vulnerabilities belong in a
[private advisory](https://github.com/omar07ibrahim/GWorker/security/advisories/new),
not a public pull request.
