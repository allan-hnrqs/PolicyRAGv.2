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
- `must_include_concepts`
- `should_avoid`
- `evaluation_focus`
- `required_claims`
- `reference_answer`
- `claim_evidence`

Add these when appropriate:

- `forbidden_claims`
- `expect_abstain`
- `notes`
- `tags`

## Authoring Rules

- Every required claim should be traceable to one or more source pages.
- Required claims should be atomic enough that the judge can score them
  independently.
- Forbidden claims should target realistic failure modes, not arbitrary
  phrasing.
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

## Promotion Bar

Adding cases is not enough. A new case batch is ready only when:

- fields validate cleanly
- claims are source-grounded
- the batch improves coverage of a documented gap
- the split assignment is justified in the manifest
