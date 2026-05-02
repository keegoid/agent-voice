# agent-voice Agent Instructions

Global Keegoid instructions still apply. Keep this file short; put durable
workflow detail in executable scripts or workspace playbooks.

## Branches, PRs, and Reviews

Use `~/keegoid/bin/agent-pr-flow` for agent-authored git/GitHub work.
Do not hand-roll branch sync, bot pushes, PR creation, labels, or independent
reviews unless the workflow script is unavailable and the operator accepts the
fallback.

Common flow:

```bash
agent-pr-flow begin --actor codex --branch <slug>
agent-pr-flow commit --actor codex --all --message "<message>"
agent-pr-flow publish --actor codex --title "<title>" --body-file <file>
agent-pr-flow after-merge --actor codex --pr <number> --delete-local
```

Actors are `codex`, `fig`, `trd`, `sub`, and `opn`. TRD/SUB/OPN are the
Paperclip company CEO identities when those agents produce code or PRs.

## Installed Runtime Sync

After a merged change affects installed runtime files, sync through the repo
script instead of manually copying into `~/.agent-voice`:

```bash
scripts/sync-installed --from main --test
```

The sync script runs the installer from the checked-out source, skips
interactive Codex config edits, writes `~/.agent-voice/install-manifest.json`,
and verifies `/v1/health` unless `--no-verify` is explicitly passed.
