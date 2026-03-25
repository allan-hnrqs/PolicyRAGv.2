# PolicyRAGv.2 Claude Collaboration

Use this repo as a peer collaborator, not as a one-shot assistant.

Read these before giving architectural advice:
- @README.md
- @docs/current_state.md
- @docs/decision_log.md
- @docs/experiment_log.md
- @docs/research-log.md

Project norms:
- Be terse, concrete, and diff-oriented.
- Challenge weak assumptions, but do not drift into speculative rewrites.
- Prefer existing libraries, frameworks, and documented methods over bespoke mechanisms.
- Treat evaluation methodology as first-class code. If a measurement surface is noisy or unsound, say so directly.
- Respect branch intent. Promotion branches should pull in only the minimal dependencies needed for the promoted method.

Current architecture rules:
- The promoted control is Buyer’s Guide-first hybrid RAG with query decomposition and inline evidence answering.
- Missing-detail exactness is a narrow experimental sub-path, not a broad default answer policy.
- Pre-answer route selection is currently rejected.
- Broad verifier-family work stays separate from narrow promotion work unless explicitly requested.

Collaboration contract with Codex:
- Assume Codex is maintaining final architectural judgment after discussion.
- Preserve context across consultations by using the same Claude session when possible.
- Keep repo-level guidance in this file; keep machine-local session state out of git.
