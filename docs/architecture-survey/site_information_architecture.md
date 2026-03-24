# Source Site Information Architecture

## Purpose

Record what the actual source domain looks like so retrieval and source-topology
decisions are driven by the real information space, not just eval pressure.

## First-pass observations from the current reference manifest

Reference source:
- `DepartmentDefence-Winter2026-feat-retrieval-expensive-methods-eval/data/manifest.json`

Document-family counts:
- `buyers_guide`: `228`
- `buy_canadian_policy`: `4`
- `tbs_directive`: `1`

## Main Buyer’s Guide branches

Top first-level Buyer’s Guide branches by document count in the current
reference manifest:

- `plan`: `72`
- `manage`: `48`
- `approve`: `30`
- `create-solicitation`: `30`
- `receive-and-evaluate`: `16`
- `solicit`: `15`
- `negotiate`: `11`
- `award`: `5`

## Implications

1. Buyer’s Guide-first retrieval is the correct default.
The corpus is dominated by Buyer’s Guide content, and most operational buyer
questions are likely to be answerable there first.

2. Supporting sources should be triggered intentionally.
Buy Canadian policy and the TBS directive are sparse, high-authority supporting
sources rather than the main navigational surface.

3. Retrieval should preserve site structure.
The major branches reflect recognizable buyer workflows, so chunking and
metadata should preserve breadcrumb, heading path, lineage, and source-family
signals instead of flattening everything into undifferentiated text.

4. Evaluation should include source-boundary cases.
Because supporting sources are sparse but important, the eval bank should test:
- when Buyer’s Guide alone is enough
- when policy/directive pull-in is required
- when supporting-source overuse is a regression
