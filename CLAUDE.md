<!-- GENERATED from AGENT_GUIDE.md. Do not edit directly. -->
<!-- Run ops/scripts/sync-agent-guides.sh after editing AGENT_GUIDE.md. -->

# agent-voice Agent Instructions

Global Keegoid instructions still apply (see `~/keegoid/AGENT_GUIDE.md`).
Keep this file short; put durable workflow detail in executable scripts or
workspace playbooks.

## Branches, PRs, and Reviews

Use `~/keegoid/ops/bin/agent-pr-flow` for agent-authored git/GitHub work.
Do not hand-roll branch sync, bot pushes, PR creation, labels, or independent
reviews unless the workflow script is unavailable and the operator accepts the
fallback. **Never run raw `git commit`, `git push`, `gh pr create`,
`gh pr review`, or `gh pr comment` for agent work** — the resulting commit
or PR will be attributed to the human operator instead of the bot.

Common flow:

```bash
agent-pr-flow begin --actor fig --branch <slug>
agent-pr-flow commit --actor fig --all --message "<message>"
agent-pr-flow publish --actor fig --title "<title>" --body-file <file>
agent-pr-flow after-merge --actor fig --pr <number> --delete-local
```

Actors are `codex`, `fig`, `cc`, `trd`, `sub`, and `opn`. PAI inherits a
Fig adapter from `~/keegoid/repos/fig/adapters/`, so PAI-authored work uses
`--actor fig`. `cc` is reviewer-only on codex-authored PRs.
TRD/SUB/OPN are the Paperclip company CEO identities when those agents
produce code or PRs.

## Installed Runtime Sync

After a merged change affects installed runtime files, sync through the repo
script instead of manually copying into `~/.agent-voice`:

```bash
scripts/sync-installed --from main --test
```

The sync script runs the installer from the checked-out source, skips
interactive Codex config edits, writes `~/.agent-voice/install-manifest.json`,
and verifies `/v1/health` unless `--no-verify` is explicitly passed.

## Adapter Sync

`AGENT_GUIDE.md` is canonical. `CLAUDE.md` and `AGENTS.md` are generated
adapters. After editing this file, regenerate them:

```bash
~/keegoid/ops/scripts/sync-agent-guides.sh
```

Do not edit `CLAUDE.md` or `AGENTS.md` directly.
