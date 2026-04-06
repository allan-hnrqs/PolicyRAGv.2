# Eval Authoring Guide

This document makes the evaluation-bank rebuild durable and handoff-friendly.
It exists so new cases are authored systematically rather than opportunistically.

## Principles

- Author cases from official source pages collected by this repo, not from old
  scraped artifacts.
- Write questions the way a buyer, procurement officer, or client advisor would
  actually ask them.
- Prefer operational decision questions over definitional trivia.
- Keep Buyer’s Guide pages as the primary source of truth unless the case is
  explicitly about support-source boundaries or supporting policy context.
- Treat abstention and source-boundary behavior as first-class quality targets.

## Required Case Fields

Each case must include:

- `id`
- `question`
- `primary_urls`
- `supporting_urls`
- `expected_primary_source_family`
- `restricted_source_valid`
- `open_browse_valid`
- `must_include_concepts`
- `should_avoid`
- `evaluation_focus`
- `required_claims`
- `reference_answer`
- `claim_evidence`

For new or rebuilt cases, `claim_evidence` should carry both:

- `evidence_doc_urls`
- `evidence_chunk_ids`

The URL anchors keep page-level evaluation readable. The chunk IDs are the
decisive-support anchors for chunk-level retrieval analysis and should point to
the snapshot chunk IDs that actually justify the claim.

Add these when appropriate:

- `forbidden_claims`
- `expect_abstain`
- `notes`
- `tags`

## Authoring Rules

- Every required claim should be traceable to one or more source pages.
- Every required claim should also be traceable to one or more decisive chunk
  IDs once the source snapshot is built.
- Required claims should be atomic enough that the judge can score them
  independently.
- Forbidden claims should target realistic failure modes, not arbitrary
  phrasing.
- `restricted_source_valid` means the claim set is fair for the indexed,
  snapshot-bound RAG lane.
- `open_browse_valid` means the claim set is also fair for an official-site
  browsing comparator that may fetch additional official pages.
- Do not mark a case as open-browse-valid if the only way to pass it is to stay
  blind to real official evidence outside the frozen source set.
- If a case is a source-boundary case, make the authority expectation explicit.
- If a case is an abstention case, make the missing detail concrete enough that
  fabrication would be obvious.
- Avoid tuning directly against frozen holdout or final-blind cases.

## Split Policy

- `dev` is for active tuning and rapid comparisons.
- `holdout` is for promotion checks after a change looks good on dev.
- `final` is sealed and should not be inspected during iteration.

## Good New-Case Families

- Buyer’s Guide-primary vs support-policy boundary questions
- navigation-to-detail questions
- workflow branches and deadline consequences
- authority/delegation threshold questions
- abstention/contact-detail questions
- cross-page synthesis where the Buyer’s Guide remains operationally primary
- open-browse-valid questions where an official-site comparator is expected to
  use extra official pages legitimately

## Promotion Bar

Adding cases is not enough. A new case batch is ready only when:

- fields validate cleanly
- claims are source-grounded
- the batch improves coverage of a documented gap
- the split assignment is justified in the manifest
