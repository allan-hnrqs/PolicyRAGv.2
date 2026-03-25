from bgrag.answering.strategies import (
    AnswerRewriteVerdictPayload,
    ContractSlotSelectionPayload,
    ContractSlotCoverageVerdictPayload,
    CitedStructuredAnswerContractPayload,
    MissingDetailExactnessVerdictPayload,
    CitedStructuredAnswerContract,
    StructuredAnswerSlotValue,
    _build_answer_rewrite_verdict_prompt,
    _build_contract_aware_answer_rewrite_verdict_prompt,
    _build_missing_detail_exactness_verdict_prompt,
    _build_contract_slot_selection_prompt,
    _build_contract_slot_coverage_verdict_prompt,
    StructuredAnswerSlotPayload,
    _build_answer_plan_prompt,
    _build_cited_structured_answer_contract_prompt,
    _build_compact_mode_aware_answer_plan_prompt,
    _build_compact_missing_detail_answer_prompt,
    _build_compact_navigation_answer_prompt,
    _build_compact_workflow_answer_prompt,
    _build_answer_repair_plan_prompt,
    _build_answer_revision_prompt,
    _build_contextual_missing_detail_prompt,
    _build_mode_aware_answer_plan_prompt,
    _build_navigation_answer_prompt,
    _collect_contract_citations,
    _normalize_cited_structured_answer_contract,
    _build_structured_answer_contract_prompt,
    _render_cited_structured_contract_answer,
    _build_mode_aware_planned_inline_evidence_prompt,
    _build_planned_inline_evidence_prompt,
    _build_query_guided_inline_evidence_prompt,
    _normalize_answer_plan,
    _normalize_answer_repair_plan,
    _normalize_answer_rewrite_verdict_payload,
    _normalize_missing_detail_exactness_verdict_payload,
    _normalize_contract_slot_selection_payload,
    _normalize_contract_slot_coverage_verdict_payload,
    _normalize_cited_structured_answer_contract_payload,
    _normalize_mode_aware_answer_plan,
    _normalize_structured_answer_contract,
    _prune_cited_structured_answer_contract,
    _looks_corrupted,
    _looks_like_missing_detail_abstention,
    _looks_quantitative_contract_slot,
    _select_compact_mode_aware_answer_route,
    _select_mode_aware_answer_route,
    _select_structured_contract_answer_route,
)
from bgrag.types import ChunkRecord, EvidenceBundle, SourceFamily


def _chunk(chunk_id: str, text: str = "text") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        canonical_url=f"https://example.com/{chunk_id}",
        title=chunk_id,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text=text,
        heading_path=["Root", chunk_id],
    )


def test_query_guided_prompt_includes_retrieval_aspects() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
        retrieval_queries=[
            "original question",
            "prerequisite requirement",
            "expiry consequence",
        ],
    )

    prompt = _build_query_guided_inline_evidence_prompt("original question", evidence)

    assert "Retrieved aspects to cover explicitly:" in prompt
    assert "- prerequisite requirement" in prompt
    assert "- expiry consequence" in prompt
    assert "- original question" not in prompt


def test_answer_plan_prompt_includes_aspects_and_json_shape() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
        retrieval_queries=[
            "original question",
            "branch condition",
        ],
    )

    prompt = _build_answer_plan_prompt("original question", evidence)

    assert '{"coverage_points":["concrete evidence-backed point"]}' in prompt
    assert "- branch condition" in prompt
    assert "- original question" not in prompt


def test_normalize_answer_plan_filters_duplicates() -> None:
    raw = """
    {
      "coverage_points": [
        "Must do X",
        "Must do X",
        "Then do Y"
      ]
    }
    """

    plan = _normalize_answer_plan(raw)

    assert plan == ["Must do X", "Then do Y"]


def test_planned_prompt_renders_checklist() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    prompt = _build_planned_inline_evidence_prompt(
        "original question",
        evidence,
        [
            "Must do X",
            "Then do Y",
        ],
    )

    assert "Coverage checklist:" in prompt
    assert "- Must do X" in prompt
    assert "- Then do Y" in prompt
    assert "If the question compares two or more mechanisms, options, or scenarios" in prompt
    assert "Do not end with an unfinished heading" in prompt


def test_mode_aware_plan_prompt_includes_schema_and_modes() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
        retrieval_queries=[
            "original question",
            "missing contact detail",
        ],
    )

    prompt = _build_mode_aware_answer_plan_prompt("original question", evidence)

    assert '"answer_mode":"workflow"' in prompt
    assert '- "missing_detail"' in prompt
    assert "missing contact detail" in prompt
    assert "- original question" not in prompt


def test_compact_mode_aware_plan_prompt_limits_to_indispensable_points() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
        retrieval_queries=[
            "original question",
            "branch condition",
        ],
    )

    prompt = _build_compact_mode_aware_answer_plan_prompt("original question", evidence)

    assert '"coverage_points":["indispensable evidence-backed point"]' in prompt
    assert "coverage_points must contain 2 to 6" in prompt
    assert "Do not include supported but optional background" in prompt
    assert "required file, documentation, notification, publication" in prompt
    assert "prefer the general method, page, directory, list, or locator" in prompt


def test_normalize_mode_aware_answer_plan_filters_duplicates() -> None:
    raw = """
    {
      "answer_mode": "workflow",
      "should_abstain": true,
      "abstain_reason": " Exact file name is not given. ",
      "coverage_points": [
        "Must do X",
        "Must do X",
        "Then do Y"
      ]
    }
    """

    plan = _normalize_mode_aware_answer_plan(raw)

    assert plan.answer_mode == "workflow"
    assert plan.should_abstain is True
    assert plan.abstain_reason == "Exact file name is not given."
    assert plan.coverage_points == ["Must do X", "Then do Y"]


def test_structured_answer_contract_prompt_includes_mode_specific_slots() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
        retrieval_queries=[
            "original question",
            "branch condition",
        ],
    )

    prompt = _build_structured_answer_contract_prompt("original question", evidence)

    assert '"slots":{"bottom_line":"","prerequisite_or_scope":""' in prompt
    assert "workflow: bottom_line, prerequisite_or_scope, general_rule, branch_if_all" in prompt
    assert "navigation: start_page, parent_stage, child_page_with_rule" in prompt
    assert "- branch condition" in prompt


def test_cited_structured_answer_contract_prompt_requests_slot_citations() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
        retrieval_queries=[
            "original question",
            "branch condition",
        ],
    )

    prompt = _build_cited_structured_answer_contract_prompt("original question", evidence)

    assert '"citation_chunk_ids":["chunk_id"]' in prompt
    assert "Every non-empty slot must include a short text field" in prompt
    assert "Do not invent slot content or citation IDs" in prompt


def test_normalize_structured_answer_contract_filters_unknown_slots() -> None:
    contract = _normalize_structured_answer_contract(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "slots": {
            "bottom_line": "Say the bottom line.",
            "prerequisite_or_scope": "Check whether the exception applies first.",
            "branch_if_some": "If only some agree, continue with those who extended or cancel.",
            "irrelevant": "should be ignored"
          }
        }
        """
    )

    assert contract.answer_mode == "workflow"
    assert contract.should_abstain is False
    assert contract.slots == {
        "bottom_line": "Say the bottom line.",
        "prerequisite_or_scope": "Check whether the exception applies first.",
        "branch_if_some": "If only some agree, continue with those who extended or cancel.",
    }


def test_normalize_cited_structured_answer_contract_filters_unknown_slots_and_ids() -> None:
    contract = _normalize_cited_structured_answer_contract(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "slots": {
            "bottom_line": {
              "text": "Say the bottom line.",
              "citation_chunk_ids": ["c1", "c1", "c2"]
            },
            "follow_on_requirement": {
              "text": "Document the file.",
              "citation_chunk_ids": ["c3"]
            },
            "irrelevant": {
              "text": "ignore me",
              "citation_chunk_ids": ["c9"]
            }
          }
        }
        """
    )

    assert contract.answer_mode == "workflow"
    assert contract.slots["bottom_line"].text == "Say the bottom line."
    assert contract.slots["bottom_line"].citation_chunk_ids == ["c1", "c2"]
    assert contract.slots["follow_on_requirement"].citation_chunk_ids == ["c3"]
    assert "irrelevant" not in contract.slots


def test_normalize_cited_structured_answer_contract_payload_filters_unknown_slots_and_ids() -> None:
    contract = _normalize_cited_structured_answer_contract_payload(
        CitedStructuredAnswerContractPayload(
            answer_mode="workflow",
            should_abstain=False,
            abstain_reason="",
            slots={
                "bottom_line": StructuredAnswerSlotPayload(
                    text=" Say the bottom line. ",
                    citation_chunk_ids=["c1", "c1", "c2"],
                ),
                "follow_on_requirement": StructuredAnswerSlotPayload(
                    text="Document the file.",
                    citation_chunk_ids=["c3"],
                ),
                "irrelevant": StructuredAnswerSlotPayload(
                    text="ignore me",
                    citation_chunk_ids=["c9"],
                ),
            },
        )
    )

    assert contract.answer_mode == "workflow"
    assert contract.slots["bottom_line"].text == "Say the bottom line."
    assert contract.slots["bottom_line"].citation_chunk_ids == ["c1", "c2"]
    assert contract.slots["follow_on_requirement"].citation_chunk_ids == ["c3"]
    assert "irrelevant" not in contract.slots


def test_render_cited_structured_contract_answer_and_collect_citations() -> None:
    evidence_chunks = [
        _chunk("c1"),
        _chunk("c2"),
        _chunk("c3"),
    ]
    contract = _normalize_cited_structured_answer_contract(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "slots": {
            "bottom_line": {
              "text": "Use the exemption only when the mandatory instrument does not address the obligation.",
              "citation_chunk_ids": ["c1"]
            },
            "required_document_or_input": {
              "text": "Keep the supporting declaration on file.",
              "citation_chunk_ids": ["c2"]
            },
            "follow_on_requirement": {
              "text": "Document the rationale in the file.",
              "citation_chunk_ids": ["c3"]
            }
          }
        }
        """
    )

    answer_text = _render_cited_structured_contract_answer(contract)
    citations = _collect_contract_citations(contract, evidence_chunks)

    assert "Use the exemption only when the mandatory instrument does not address the obligation. [c1]" in answer_text
    assert "- Keep the supporting declaration on file. [c2]" in answer_text
    assert "- Document the rationale in the file. [c3]" in answer_text
    assert [citation.chunk_id for citation in citations] == ["c1", "c2", "c3"]


def test_normalize_answer_repair_plan_sets_revision_when_points_exist() -> None:
    repair_plan = _normalize_answer_repair_plan(
        """
        {
          "needs_revision": false,
          "missing_supported_points": ["Add the timing rule."],
          "unsupported_or_overstated_points": []
        }
        """
    )

    assert repair_plan.needs_revision is True
    assert repair_plan.missing_supported_points == ["Add the timing rule."]
    assert repair_plan.unsupported_or_overstated_points == []


def test_answer_rewrite_verdict_prompt_demands_conservative_post_draft_decision() -> None:
    evidence = EvidenceBundle(
        query="question",
        retrieval_queries=["question", "deadline consequence"],
        packed_chunks=[_chunk("c1")],
    )

    prompt = _build_answer_rewrite_verdict_prompt(
        "question",
        evidence,
        "draft answer",
    )

    assert '"action":"keep"' in prompt
    assert '- "rewrite_structured_contract"' in prompt
    assert "Be conservative: if you are unsure, choose keep with low confidence." in prompt
    assert "Do not propose missing points or rewrite text yourself." in prompt
    assert "- deadline consequence" in prompt


def test_contract_aware_answer_rewrite_verdict_prompt_uses_structured_contract() -> None:
    evidence = EvidenceBundle(
        query="question",
        packed_chunks=[_chunk("c1")],
    )
    contract = _normalize_cited_structured_answer_contract(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "slots": {
            "bottom_line": {
              "text": "Continue with offers that accepted the extension or cancel.",
              "citation_chunk_ids": ["c1"]
            },
            "deadline_or_timing": {
              "text": "Request the extension at least three days before expiry.",
              "citation_chunk_ids": ["c1"]
            }
          }
        }
        """
    )

    prompt = _build_contract_aware_answer_rewrite_verdict_prompt(
        "question",
        evidence,
        contract,
        "draft answer",
    )

    assert "Structured answer contract:" in prompt
    assert "- bottom_line: Continue with offers that accepted the extension or cancel. [c1]" in prompt
    assert "Treat each populated contract slot as a supported answer obligation" in prompt
    assert "Do not propose missing points or rewrite text yourself." in prompt


def test_contract_slot_coverage_verdict_prompt_demands_slot_keys_only() -> None:
    evidence = EvidenceBundle(
        query="question",
        packed_chunks=[_chunk("c1")],
    )
    contract = _normalize_cited_structured_answer_contract(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "slots": {
            "bottom_line": {
              "text": "Continue with offers that accepted the extension or cancel.",
              "citation_chunk_ids": ["c1"]
            },
            "consequence": {
              "text": "If the solicitation is canceled, it must be reissued.",
              "citation_chunk_ids": ["c1"]
            }
          }
        }
        """
    )

    prompt = _build_contract_slot_coverage_verdict_prompt(
        "question",
        evidence,
        contract,
        "draft answer",
    )

    assert '{"confidence":"low","rationale":"short explanation","missing_or_weakened_slots":["slot_key"],"unsupported_detail_risk":false}' in prompt
    assert "Only use slot keys that are populated in the structured contract." in prompt
    assert "Be willing to mark substantive omissions" in prompt
    assert "mark supporting_rule missing when the draft only gives a locator" in prompt


def test_contract_slot_selection_prompt_demands_small_sufficient_keep_set() -> None:
    evidence = EvidenceBundle(
        query="question",
        packed_chunks=[_chunk("c1")],
    )
    contract = _normalize_cited_structured_answer_contract(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "slots": {
            "bottom_line": {
              "text": "Use the 25-day minimum.",
              "citation_chunk_ids": ["c1"]
            },
            "required_document_or_input": {
              "text": "Publish an NPP.",
              "citation_chunk_ids": ["c1"]
            }
          }
        }
        """
    )

    prompt = _build_contract_slot_selection_prompt("question", evidence, contract)

    assert '{"keep_slot_keys":["slot_key"],"rationale":"short explanation"}' in prompt
    assert "Drop populated slots that are only background" in prompt
    assert "Prefer the smallest sufficient keep set." in prompt


def test_missing_detail_exactness_prompt_targets_nearby_identifier_overstatement() -> None:
    evidence = EvidenceBundle(
        query="question",
        packed_chunks=[_chunk("c1")],
    )
    contract = _normalize_cited_structured_answer_contract(
        """
        {
          "answer_mode": "missing_detail",
          "should_abstain": true,
          "abstain_reason": "The exact form number is not provided.",
          "slots": {
            "exact_detail_status": {
              "text": "The exact form number is not provided.",
              "citation_chunk_ids": ["c1"]
            },
            "closest_supported_context": {
              "text": "ADM approval must be in writing and on file.",
              "citation_chunk_ids": ["c1"]
            }
          }
        }
        """
    )

    prompt = _build_missing_detail_exactness_verdict_prompt(
        "Which exact form number do I need?",
        evidence,
        contract,
        "The exact form number is not provided, but use PWGSC-TPSGC 1151-1.",
    )

    assert '"exact_detail_overstatement_risk":false' in prompt
    assert "nearby identifier, form, file name, template name, contact method" in prompt
    assert "Do not flag a draft merely for giving neutral closest-supported context" in prompt


def test_normalize_answer_rewrite_verdict_payload_filters_invalid_values() -> None:
    verdict = _normalize_answer_rewrite_verdict_payload(
        AnswerRewriteVerdictPayload(
            action="something_else",
            confidence="maybe",
            rationale="  Missing a branch consequence.  ",
            omission_risk=True,
            exact_detail_abstain_risk=False,
            unsupported_detail_risk=True,
        )
    )

    assert verdict.action == "keep"
    assert verdict.confidence == "low"
    assert verdict.rationale == "Missing a branch consequence."
    assert verdict.omission_risk is True
    assert verdict.unsupported_detail_risk is True


def test_normalize_contract_slot_coverage_verdict_payload_filters_to_allowed_populated_slots() -> None:
    verdict = _normalize_contract_slot_coverage_verdict_payload(
        ContractSlotCoverageVerdictPayload(
            confidence="maybe",
            rationale="  Missed branch and deadline.  ",
            missing_or_weakened_slots=["branch_if_some", "unknown_slot", "deadline_or_timing"],
            unsupported_detail_risk=True,
        ),
        answer_mode="workflow",
        populated_slot_keys={"branch_if_some", "deadline_or_timing"},
    )

    assert verdict.confidence == "low"
    assert verdict.rationale == "Missed branch and deadline."
    assert verdict.missing_or_weakened_slots == ["branch_if_some", "deadline_or_timing"]
    assert verdict.unsupported_detail_risk is True


def test_normalize_missing_detail_exactness_verdict_payload_filters_values() -> None:
    verdict = _normalize_missing_detail_exactness_verdict_payload(
        MissingDetailExactnessVerdictPayload(
            confidence="maybe",
            rationale="  Nearby form is presented too strongly.  ",
            exact_detail_overstatement_risk=True,
            offending_details=["PWGSC-TPSGC 1151-1", "PWGSC-TPSGC 1151-1"],
        )
    )

    assert verdict.confidence == "low"
    assert verdict.rationale == "Nearby form is presented too strongly."
    assert verdict.exact_detail_overstatement_risk is True
    assert verdict.offending_details == ["PWGSC-TPSGC 1151-1"]


def test_normalize_contract_slot_selection_payload_filters_to_allowed_populated_slots() -> None:
    selection = _normalize_contract_slot_selection_payload(
        ContractSlotSelectionPayload(
            keep_slot_keys=["bottom_line", "unknown_slot", "deadline_or_timing", "bottom_line"],
            rationale="  Keep the core rule and timing only.  ",
        ),
        answer_mode="workflow",
        populated_slot_keys={"bottom_line", "deadline_or_timing"},
    )

    assert selection.keep_slot_keys == ["bottom_line", "deadline_or_timing"]
    assert selection.rationale == "Keep the core rule and timing only."


def test_prune_cited_structured_answer_contract_keeps_selected_slots_in_mode_order() -> None:
    contract = CitedStructuredAnswerContract(
        answer_mode="workflow",
        should_abstain=False,
        abstain_reason="",
        slots={
            "bottom_line": StructuredAnswerSlotValue(text="Main rule.", citation_chunk_ids=["c1"]),
            "required_document_or_input": StructuredAnswerSlotValue(text="Publish an NPP.", citation_chunk_ids=["c2"]),
            "deadline_or_timing": StructuredAnswerSlotValue(text="Urgency can shorten the period.", citation_chunk_ids=["c3"]),
        },
    )

    pruned = _prune_cited_structured_answer_contract(
        contract,
        keep_slot_keys={"deadline_or_timing", "bottom_line"},
    )

    assert list(pruned.slots) == ["bottom_line", "deadline_or_timing"]
    assert "required_document_or_input" not in pruned.slots


def test_prune_cited_structured_answer_contract_falls_back_when_keep_set_is_empty() -> None:
    contract = CitedStructuredAnswerContract(
        answer_mode="missing_detail",
        should_abstain=True,
        abstain_reason="Not provided.",
        slots={
            "exact_detail_status": StructuredAnswerSlotValue(text="Not provided.", citation_chunk_ids=["c1"]),
            "page_or_location": StructuredAnswerSlotValue(text="Use the buyer portal page.", citation_chunk_ids=["c2"]),
        },
    )

    pruned = _prune_cited_structured_answer_contract(contract, keep_slot_keys=set())

    assert pruned == contract


def test_looks_like_missing_detail_abstention_detects_generic_abstain_language() -> None:
    assert _looks_like_missing_detail_abstention(
        "The evidence does not explicitly provide the exact email address."
    )
    assert _looks_like_missing_detail_abstention(
        "The exact form number is not provided in the directive."
    )
    assert not _looks_like_missing_detail_abstention(
        "Use the Schedule 3 template and send the notice to the listed group."
    )
    assert not _looks_like_missing_detail_abstention(
        "The exact form number is PWGSC-TPSGC 1151-1."
    )


def test_looks_corrupted_detects_repetitive_gibberish() -> None:
    corrupted = " ".join(["I", "ID", "I", "I", "ID", "I"] * 20)
    healthy = "This answer says the exact contact detail is not available and points the user to the relevant directory."

    assert _looks_corrupted(corrupted)
    assert not _looks_corrupted(healthy)


def test_looks_quantitative_contract_slot_detects_threshold_language() -> None:
    assert _looks_quantitative_contract_slot(
        "If limited tendering is used, the period can be less than 10 calendar days."
    )
    assert not _looks_quantitative_contract_slot(
        "Consult with management and legal services before providing feedback."
    )


def test_mode_aware_prompt_includes_abstain_and_supporting_source_instructions() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[
            _chunk("c1"),
            ChunkRecord(
                chunk_id="support1",
                doc_id="support1",
                canonical_url="https://example.com/support1",
                title="support1",
                source_family=SourceFamily.BUY_CANADIAN_POLICY,
                authority_rank=2,
                chunker_name="section_chunker",
                chunk_type="paragraph",
                text="support text",
                heading_path=["Root", "support1"],
            ),
        ],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "missing_detail",
          "should_abstain": true,
          "abstain_reason": "The exact identifier is not established by the evidence.",
          "coverage_points": ["State that the exact identifier is unavailable", "Provide the closest supported context"]
        }
        """
    )

    prompt = _build_mode_aware_planned_inline_evidence_prompt("original question", evidence, plan)

    assert "The plan determined that the answer should abstain" in prompt
    assert "Supporting-source evidence is present." in prompt
    assert "Do not infer or invent exact emails, direct contacts, form numbers, file names" in prompt


def test_contextual_missing_detail_prompt_keeps_abstention_and_context() -> None:
    evidence = EvidenceBundle(
        query="question",
        packed_chunks=[_chunk("c1", text="Use contract request forms PWGSC-TPSGC 1151-1 and PWGSC-TPSGC 1151-2.")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "missing_detail",
          "should_abstain": true,
          "abstain_reason": "The exact form number is unclear.",
          "coverage_points": ["PWGSC-TPSGC 1151-1 and PWGSC-TPSGC 1151-2 are referenced."]
        }
        """
    )

    prompt = _build_contextual_missing_detail_prompt("question", evidence, plan)

    assert "does not establish the exact requested detail" in prompt
    assert "closest supported context" in prompt
    assert "Do not present nearby identifiers or related forms as the exact requested detail" in prompt
    assert "PWGSC-TPSGC 1151-1 and PWGSC-TPSGC 1151-2 are referenced." in prompt


def test_selective_route_uses_mode_aware_prompt_for_navigation() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "navigation",
          "should_abstain": false,
          "abstain_reason": "",
          "coverage_points": ["Open the relevant page first."]
        }
        """
    )

    route = _select_mode_aware_answer_route("original question", evidence, plan)

    assert route.selected_path == "mode_aware_navigation"
    assert route.abstained is False
    assert "Start with the exact Buyer’s Guide section or path the user should open." in route.prompt
    assert "Include the direct page URL if the evidence provides it." in route.prompt


def test_navigation_answer_prompt_keeps_navigation_specific_constraints() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "navigation",
          "should_abstain": false,
          "abstain_reason": "",
          "coverage_points": ["Open the child page that contains the rule.", "Use the direct page URL."]
        }
        """
    )

    prompt = _build_navigation_answer_prompt("original question", evidence, plan)

    assert "The user is asking where in the Buyer's Guide to go." in prompt
    assert "Prefer a short answer with 2 to 4 bullets or short lines." in prompt
    assert "Do not drift into a long policy summary" in prompt
    assert "- Open the child page that contains the rule." in prompt


def test_compact_workflow_answer_prompt_forbids_meta_headings() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "coverage_points": ["State branch one", "State branch two"]
        }
        """
    )

    prompt = _build_compact_workflow_answer_prompt("original question", evidence, plan)

    assert "Start with one direct bottom-line sentence." in prompt
    assert "Do not use section labels such as Direct Answer" in prompt
    assert "Coverage checklist:" in prompt
    assert "make each bullet map to a checklist point" in prompt
    assert "required file, documentation, notification, publication" in prompt
    assert "Do not add penalties, disqualification outcomes, consultation steps" in prompt


def test_compact_navigation_prompt_keeps_short_navigation_shape() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "navigation",
          "should_abstain": false,
          "abstain_reason": "",
          "coverage_points": ["Open the child page that contains the rule."]
        }
        """
    )

    prompt = _build_compact_navigation_answer_prompt("original question", evidence, plan)

    assert "Use 2 to 4 short bullets or short lines only." in prompt
    assert "Do not add a policy summary." in prompt
    assert "Do not use meta headings" in prompt


def test_compact_missing_detail_prompt_prefers_locator_over_extra_contact_guidance() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "missing_detail",
          "should_abstain": true,
          "abstain_reason": "The exact email address is not established by the evidence.",
          "coverage_points": ["State that the exact email address is unavailable.", "Point the user to the directory or list named on the page."]
        }
        """
    )

    prompt = _build_compact_missing_detail_answer_prompt("original question", evidence, plan)

    assert "closest supported context from the checklist" in prompt
    assert "Prefer the general method, page path, directory, list, or locator" in prompt
    assert "Do not add contact-assistance or escalation guidance" in prompt


def test_answer_repair_plan_prompt_includes_checklist_and_draft_answer() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "coverage_points": ["State branch one", "State deadline"]
        }
        """
    )

    prompt = _build_answer_repair_plan_prompt(
        "original question",
        evidence,
        plan,
        "Draft answer text.",
    )

    assert "Coverage checklist:" in prompt
    assert "Draft answer:" in prompt
    assert "- State branch one" in prompt
    assert "missing_supported_points" in prompt


def test_answer_revision_prompt_includes_missing_and_unsupported_points() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "coverage_points": ["State branch one", "State deadline"]
        }
        """
    )
    repair_plan = _normalize_answer_repair_plan(
        """
        {
          "needs_revision": true,
          "missing_supported_points": ["Add the missed deadline."],
          "unsupported_or_overstated_points": ["Remove the unsupported exception."]
        }
        """
    )

    prompt = _build_answer_revision_prompt(
        "original question",
        evidence,
        plan,
        "Draft answer text.",
        repair_plan,
    )

    assert "Missing supported points to add:" in prompt
    assert "- Add the missed deadline." in prompt
    assert "Unsupported or overstated points to correct:" in prompt
    assert "- Remove the unsupported exception." in prompt


def test_selective_route_uses_planned_workflow_for_workflow_mode() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "coverage_points": ["State branch one", "State branch two"]
        }
        """
    )

    route = _select_mode_aware_answer_route("original question", evidence, plan)

    assert route.selected_path == "planned_workflow"
    assert route.abstained is False
    assert "Coverage checklist:" in route.prompt


def test_compact_selective_route_uses_compact_workflow_prompt() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "workflow",
          "should_abstain": false,
          "abstain_reason": "",
          "coverage_points": ["State branch one", "State branch two"]
        }
        """
    )

    route = _select_compact_mode_aware_answer_route("original question", evidence, plan)

    assert route.selected_path == "compact_workflow"
    assert route.abstained is False
    assert "Do not use section labels such as Direct Answer" in route.prompt




def test_selective_route_uses_contextual_missing_detail_prompt_for_abstention() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1", text="Contact the group using the information on the page.")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "missing_detail",
          "should_abstain": true,
          "abstain_reason": "The exact email address is not established by the evidence.",
          "coverage_points": ["The page names the group but does not provide direct contact details."]
        }
        """
    )

    route = _select_mode_aware_answer_route("original question", evidence, plan)

    assert route.selected_path == "contextual_missing_detail"
    assert route.abstained is True
    assert "does not establish the exact requested detail" in route.prompt
    assert "closest supported context" in route.prompt


def test_selective_route_keeps_contextual_missing_detail_path_when_identifiers_exist() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1", text="Use contract request forms PWGSC-TPSGC 1151-1 and PWGSC-TPSGC 1151-2.")],
    )
    plan = _normalize_mode_aware_answer_plan(
        """
        {
          "answer_mode": "missing_detail",
          "should_abstain": true,
          "abstain_reason": "The exact form number is unclear.",
          "coverage_points": ["PWGSC-TPSGC 1151-1 and PWGSC-TPSGC 1151-2 are referenced."]
        }
        """
    )

    route = _select_mode_aware_answer_route("original question", evidence, plan)

    assert route.selected_path == "contextual_missing_detail"
    assert route.abstained is True
    assert "Coverage checklist:" in route.prompt
    assert "closest supported context" in route.prompt


def test_structured_contract_route_uses_navigation_prompt() -> None:
    evidence = EvidenceBundle(
        query="original question",
        packed_chunks=[_chunk("c1")],
    )
    contract = _normalize_structured_answer_contract(
        """
        {
          "answer_mode": "navigation",
          "should_abstain": false,
          "abstain_reason": "",
          "slots": {
            "start_page": "Open the Receive offers page.",
            "direct_url": "https://example.com/path"
          }
        }
        """
    )

    route = _select_structured_contract_answer_route("original question", evidence, contract)

    assert route.selected_path == "contract_navigation"
    assert route.abstained is False
    assert "The user is asking where in the Buyer's Guide to go." in route.prompt
    assert "Structured answer contract:" in route.prompt
