# Real CLI motion evidence

GWorker's motion bundle records the production CLI module directly from a
clean repository checkout. It does not invent a graphical timer. Four
separate Python processes open the same disposable private journal and
execute this fixed workflow:

1. make a seeded recommendation with no review history;
2. attach one explicit review to the stored decision;
3. reopen the journal and make a second seeded recommendation; and
4. reopen it again and verify every decision for the current policy.

The recorder gives each process a 120×40 pseudo-terminal for stdout, keeps
stderr on a separate pipe, closes stdin, disables the shell, applies a 30
second timeout, and limits each stream to 65,536 bytes. The workspace is mode
`0700`; the SQLite journal is mode `0600`; both remain under the repository's
ignored `.gworker/` runtime tree and the disposable workspace is removed
before evidence is published.

## Record, render, and verify

Install the pinned development tools first:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Only `record` starts GWorker processes. It requires a clean committed
worktree so the event document can bind the exact source commit and source
bytes:

```bash
PYTHONPATH=src .venv/bin/python -B \
  scripts/visuals/capture_cli_motion.py record
```

`render` regenerates the transcript, GIF, poster, and manifest from the
committed event document without starting the application:

```bash
PYTHONPATH=src .venv/bin/python -B \
  scripts/visuals/capture_cli_motion.py render
```

`check` is read-only and process-free. It validates the closed event schema,
source hashes, output inventory, exact transcript, eleven GIF frames, poster,
artifact hashes, and canonical manifest:

```bash
PYTHONPATH=src .venv/bin/python -B \
  scripts/visuals/capture_cli_motion.py check
```

## Published artifacts

The bundle under `docs/visuals/motion/` contains:

- `durable-policy-workflow.events.json`, the path-free process records,
  commands, exact stdout, PTY contract, storage modes, and source provenance;
- `durable-policy-workflow.txt`, the accessible canonical transcript;
- `durable-policy-workflow.gif`, eleven fixed 960×540 presentation frames;
- `durable-policy-workflow.png`, a static final-state poster; and
- `manifest.json`, the Pillow/FreeType identity, claim boundary, source
  records, and SHA-256/byte count for every non-manifest artifact.

Frame durations are deliberately fixed. They make the workflow readable and
do not represent process runtime or latency. The capture uses synthetic
identifiers and context, does not invoke the evaluator or publication runner,
does not estimate a human outcome, and does not claim that a graphical timer
exists.
