from __future__ import annotations

import unittest

from nihaisha_kg import pdf_vector


def result(
    text: str,
    *,
    source_path: str = "/tmp/伤寒论.pdf",
    page: int = 68,
    paragraph_id: str = "p-1",
) -> dict[str, object]:
    return {
        "paragraph_id": paragraph_id,
        "source_path": source_path,
        "title": f"{source_path.rsplit('/', 1)[-1]} p{page}",
        "page_start": page,
        "page_end": page,
        "text": text,
        "matched_knowledge_units": [],
    }


class AnswerQualityTests(unittest.TestCase):
    def test_source_lookup_answer_uses_retrieved_formula_and_location_only(self) -> None:
        answer = pdf_vector.synthesize_pdf_rag_answer(
            "桂枝汤的出处在哪本书哪一页？",
            [result("太阳中风，汗出恶风，桂枝汤主之。")],
        )

        self.assertEqual(answer["intent"], "source_lookup")
        self.assertIn("桂枝汤", answer["answer"])
        self.assertIn("伤寒论.pdf p68", answer["answer"])
        self.assertNotIn("木香饼", answer["answer"])
        self.assertNotIn("热熨", answer["answer"])

    def test_clinical_answer_uses_query_focus_without_fixed_diarrhea_checklist(self) -> None:
        answer = pdf_vector.synthesize_pdf_rag_answer(
            "患者咳嗽、怕冷、无汗，课程有哪些相关线索？",
            [result("恶寒无汗而喘，麻黄汤主之。", page=35)],
        )

        self.assertEqual(answer["intent"], "clinical")
        self.assertTrue(any(clue in answer["answer"] for clue in ("咳嗽", "怕冷", "无汗")))
        self.assertIn("麻黄汤", answer["answer"])
        self.assertNotIn("下利性质", answer["answer"])
        self.assertNotIn("心下痞满", answer["answer"])

    def test_no_results_never_emit_empty_citation_marker(self) -> None:
        answer = pdf_vector.synthesize_pdf_rag_answer("桂枝汤的出处在哪本书？", [])

        self.assertNotIn("证据见 。", answer["answer"])
        self.assertNotIn("证据见。", answer["answer"])
        self.assertEqual(answer["citations"], [])

    def test_detect_answer_intent_orders_specific_intents(self) -> None:
        for query in ("桂枝汤出处", "桂枝汤在哪本书", "桂枝汤哪一页", "桂枝汤原文"):
            with self.subTest(query=query):
                self.assertEqual(pdf_vector.detect_answer_intent(query), "source_lookup")
        for query in ("桂枝汤和麻黄汤的鉴别", "桂枝汤与麻黄汤比较", "二者有什么区别"):
            with self.subTest(query=query):
                self.assertEqual(pdf_vector.detect_answer_intent(query), "comparison")
        self.assertEqual(pdf_vector.detect_answer_intent("患者咳嗽、怕冷、无汗"), "clinical")
        self.assertEqual(pdf_vector.detect_answer_intent("古时候一钱是多少克？"), "dosage")

    def test_answer_anchor_terms_prefers_named_formula_and_excludes_task_words(self) -> None:
        anchors = pdf_vector.answer_anchor_terms("桂枝汤的出处在哪本书哪一页？")

        self.assertIn("桂枝汤", anchors)
        self.assertNotIn("出处", anchors)
        self.assertNotIn("哪本书", anchors)

    def test_reliable_source_anchors_exclude_generic_reference_phrases(self) -> None:
        query = "这种治法的原文在哪一段？"
        evidence = result("老师说明此法用于课程示例。")

        self.assertEqual(pdf_vector.reliable_source_anchors(query), [])
        self.assertEqual(
            pdf_vector.filter_results_for_intent(query, "source_lookup", [evidence]),
            [evidence],
        )

    def test_source_anchors_do_not_create_cross_boundary_symptom_terms(self) -> None:
        query = "咳嗽怕冷无汗出处"

        self.assertNotIn("汗出", pdf_vector.answer_anchor_terms(query))
        self.assertEqual(pdf_vector.reliable_source_anchors(query), [])

    def test_reliable_source_anchors_keep_explicit_named_entities(self) -> None:
        cases = {
            "桂枝汤出处": ["桂枝汤"],
            "木香饼热熨法原文": ["木香饼"],
            "一钱的原文": ["一钱"],
            "太阳病原文": ["太阳病"],
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(pdf_vector.reliable_source_anchors(query), expected)

    def test_source_filter_requires_primary_anchor_when_direct_evidence_exists(self) -> None:
        generic = result("用火温之，再以热熨法处理。", paragraph_id="p-generic")
        direct = result("木香饼（生地木香作饼），热熨贴之。", paragraph_id="p-direct")

        filtered = pdf_vector.filter_results_for_intent(
            "木香饼热熨法的出处在哪一段？",
            "source_lookup",
            [generic, direct],
        )

        self.assertEqual([item["paragraph_id"] for item in filtered], ["p-direct"])

    def test_source_filter_rejects_evidence_for_another_named_topic(self) -> None:
        unrelated = result("生地木香作饼，热熨贴之。", paragraph_id="p-unrelated")
        direct = result("太阳中风，桂枝汤主之。", paragraph_id="p-direct")

        filtered = pdf_vector.filter_results_for_intent(
            "桂枝汤的出处在哪本书？",
            "source_lookup",
            [unrelated, direct],
        )

        self.assertEqual([item["paragraph_id"] for item in filtered], ["p-direct"])

    def test_cough_followups_do_not_inject_unseen_gastrointestinal_clues(self) -> None:
        questions = pdf_vector.build_followup_questions(
            "患者咳嗽、怕冷、无汗",
            "clinical",
            [{"content": "恶寒无汗而喘，麻黄汤主之。"}],
            [{"evidence_quote": "恶寒无汗而喘，麻黄汤主之。"}],
        )
        joined = "\n".join(questions)

        for clue in ("下利", "恶心", "腹痛", "心下痞"):
            self.assertNotIn(clue, joined)

    def test_followups_are_empty_without_a_differentiating_clue(self) -> None:
        questions = pdf_vector.build_followup_questions(
            "患者想了解课程相关线索",
            "clinical",
            [{"content": "麻黄汤主之。"}],
            [{"evidence_quote": "麻黄汤主之。"}],
        )

        self.assertEqual(questions, [])

    def test_followups_do_not_add_absent_differentiation_or_safety_facts(self) -> None:
        questions = pdf_vector.build_followup_questions(
            "患者下利",
            "clinical",
            [],
            [{"subject": "某汤", "predicate": "主治", "object": "下利", "evidence_quote": "下利，某汤主之。"}],
        )
        joined = "\n".join(questions)

        self.assertIn("下利", joined)
        for absent in ("黄臭", "寒利", "完谷不化", "腹痛", "妊娠", "附子", "峻下"):
            self.assertNotIn(absent, joined)

    def test_followups_can_derive_a_question_from_guide_node_label(self) -> None:
        questions = pdf_vector.build_followup_questions(
            "患者情况待核对",
            "clinical",
            [{"label": "麻黄汤", "content": "", "path": "方证 > 麻黄汤"}],
            [],
        )

        self.assertTrue(any("麻黄汤" in question for question in questions))


if __name__ == "__main__":
    unittest.main()
