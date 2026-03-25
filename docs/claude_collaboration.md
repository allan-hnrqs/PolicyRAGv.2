# Claude Collaboration

This repo supports repeatable Codex <-> Claude peer consultation without
mixing machine-local session state into git.

## Durable Project Context

- `CLAUDE.md` is the committed project-level guidance for Claude.
- It carries the repo's shared collaboration rules, current architecture
  constraints, and branch discipline.
- Claude loads project `CLAUDE.md` on every session.

## Local Session State

- Live Claude session IDs are stored under `.claude/session_local/`.
- That directory is gitignored on purpose.
- This keeps repeated consultations resumable on one machine without leaking
  local runtime state into the repository.

## Consultation Wrapper

Use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/consult_claude.ps1 `
  -Prompt "What is the highest-value next move on this branch?"
```

Behavior:
- uses `claude-opus-4-6`
- uses `--effort max`
- resumes the same session by default for the named lane
- writes each reply to `.claude/session_local/transcripts/`

Optional inputs:
- `-PromptFile <path>`
- `-SessionName <name>`
- `-MaxBudgetUsd <amount>`
- `-RawJson`

## Why This Split

Official Claude Code docs support the same separation:
- project-scoped settings and `CLAUDE.md` are meant for team-shared context
- local scope is meant for personal or machine-specific state
- resumed sessions preserve prior conversation context
- subagents and background tasks are useful for parallel research, but only when
  the work is self-contained

## Current Use In This Repo

- Claude is used as a peer reviewer and architecture sounding board.
- Codex keeps final architectural judgment and integration responsibility.
- Promotion decisions still require repo-local evidence:
  - deterministic eval metrics
  - structured judge outputs
  - pairwise A/B checks where available
