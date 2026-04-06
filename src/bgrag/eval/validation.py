"""Validation helpers for evaluation case assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bgrag.types import EvalCase


@dataclass(frozen=True)
class EvalValidationIssue:
    severity: str
    message: str


_SPLIT_FOLDERS = {"dev", "holdout", "parity", "generated", "final"}
_STRICT_CASE_FOLDERS = {"dev", "holdout", "parity", "generated"}


def infer_eval_split(path: Path) -> str | None:
    folder = path.parent.name.strip().lower()
    return folder if folder in _SPLIT_FOLDERS else None


def load_and_validate_eval_cases(path: Path) -> tuple[list[EvalCase], list[EvalValidationIssue]]:
    issues: list[EvalValidationIssue] = []
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    inferred_split = infer_eval_split(path)

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(EvalValidationIssue("error", f"{path}:{line_number}: invalid JSON: {exc}"))
            continue

        try:
            case = EvalCase.model_validate(payload)
        except Exception as exc:  # keep the script surface small
            issues.append(EvalValidationIssue("error", f"{path}:{line_number}: schema error: {exc}"))
            continue

        if inferred_split and case.split is None:
            case = case.model_copy(update={"split": inferred_split})
        elif inferred_split in {"dev", "holdout"} and case.split not in {None, inferred_split}:
            issues.append(
                EvalValidationIssue(
                    "error",
                    f"{path}:{line_number}: case {case.id} has split={case.split!r} but lives under {inferred_split!r}",
                )
            )

        if case.id in seen_ids:
            issues.append(EvalValidationIssue("error", f"{path}:{line_number}: duplicate case id {case.id!r}"))
        else:
            seen_ids.add(case.id)

        if inferred_split in _STRICT_CASE_FOLDERS:
            if not case.required_claims:
                issues.append(
                    EvalValidationIssue(
                        "error",
                        f"{path}:{line_number}: case {case.id} is missing required_claims on a scored eval surface",
                    )
                )
            if not case.reference_answer:
                issues.append(
                    EvalValidationIssue(
                        "error",
                        f"{path}:{line_number}: case {case.id} is missing reference_answer on a scored eval surface",
                    )
                )
            if not case.primary_urls:
                issues.append(
                    EvalValidationIssue(
                        "error",
                        f"{path}:{line_number}: case {case.id} is missing primary_urls on a scored eval surface",
                    )
                )

        if len(set(case.required_claims)) != len(case.required_claims):
            issues.append(
                EvalValidationIssue("error", f"{path}:{line_number}: case {case.id} has duplicate required_claims")
            )
        if len(set(case.forbidden_claims)) != len(case.forbidden_claims):
            issues.append(
                EvalValidationIssue("error", f"{path}:{line_number}: case {case.id} has duplicate forbidden_claims")
            )

        seen_claim_evidence_claims: set[str] = set()
        for item in case.claim_evidence:
            claim_text = item.claim.strip()
            if not claim_text:
                issues.append(
                    EvalValidationIssue(
                        "error",
                        f"{path}:{line_number}: case {case.id} has claim_evidence entry with blank claim text",
                    )
                )
            if not item.evidence_doc_urls and not item.evidence_doc_prefixes and not item.evidence_chunk_ids:
                issues.append(
                    EvalValidationIssue(
                        "error",
                        f"{path}:{line_number}: case {case.id} claim_evidence[{claim_text!r}] has no evidence anchors",
                    )
                )
            if claim_text in seen_claim_evidence_claims:
                issues.append(
                    EvalValidationIssue(
                        "warning",
                        f"{path}:{line_number}: case {case.id} repeats claim_evidence claim {claim_text!r}",
                    )
                )
            else:
                seen_claim_evidence_claims.add(claim_text)

        if case.claim_evidence and len(case.claim_evidence) != len(case.required_claims):
            issues.append(
                EvalValidationIssue(
                    "warning",
                    (
                        f"{path}:{line_number}: case {case.id} has {len(case.claim_evidence)} claim_evidence item(s) "
                        f"for {len(case.required_claims)} required_claim(s)"
                    ),
                )
            )

        if inferred_split == "final":
            if case.restricted_source_valid is None:
                issues.append(
                    EvalValidationIssue(
                        "warning",
                        f"{path}:{line_number}: case {case.id} is missing restricted_source_valid on acceptance surface",
                    )
                )
            if case.open_browse_valid is None:
                issues.append(
                    EvalValidationIssue(
                        "warning",
                        f"{path}:{line_number}: case {case.id} is missing open_browse_valid on acceptance surface",
                    )
                )

        cases.append(case)

    return cases, issues
