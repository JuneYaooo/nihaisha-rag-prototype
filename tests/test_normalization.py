from __future__ import annotations

import unittest

from nihaisha_kg.normalization import lexical_query_terms, normalize_query_text


class NormalizationTests(unittest.TestCase):
    def test_lexical_query_terms_extracts_recognized_domain_terms(self) -> None:
        terms = lexical_query_terms(
            "桂枝湯和麻黃湯的方證如何鑒別？",
            domain_terms=("桂枝汤", "麻黄汤", "方证", "鉴别"),
        )

        self.assertTrue({"桂枝汤", "麻黄汤", "方证", "鉴别"}.issubset(terms))
        self.assertNotIn("桂枝汤和麻黄汤的方证如何鉴别", terms)

    def test_lexical_query_terms_prioritizes_domain_terms_and_drops_scaffolding(self) -> None:
        terms = lexical_query_terms(
            "太阳病欲解时从什么时候到什么时候？",
            domain_terms=("太阳病", "欲解"),
        )

        self.assertEqual(terms[:2], ["太阳病", "欲解"])
        self.assertNotIn("什么时候", terms)

    def test_normalize_query_text_translates_traditional_query_characters(self) -> None:
        self.assertEqual(
            normalize_query_text("一錢、方證、發熱"),
            "一钱、方证、发热",
        )

    def test_lexical_terms_preserve_phrase_boundaries_and_remove_generic_glue(self) -> None:
        cases = (
            (
                "调和营卫的方证有哪些？",
                ("方证",),
                ["方证", "方證", "调和营卫", "調和營衛"],
            ),
            (
                "和解少阳如何理解？",
                (),
                ["和解少阳", "和解少陽", "理解"],
            ),
            (
                "从容脉是什么脉象？",
                (),
                ["从容脉", "從容脈", "脉象", "脈象"],
            ),
            (
                "手足厥冷应该用什么方？",
                (),
                ["手足厥冷"],
            ),
        )

        for query, domain_terms, expected in cases:
            with self.subTest(query=query):
                terms = lexical_query_terms(query, domain_terms=domain_terms)

                self.assertEqual(terms, expected)
                self.assertFalse(
                    any(
                        glue in term
                        for term in terms
                        for glue in (
                            "什么",
                            "哪些",
                            "这个",
                            "那个",
                            "问题",
                            "资料",
                            "内容",
                            "时候",
                        )
                    )
                )

    def test_lexical_terms_emit_normalized_and_traditional_domain_anchors(self) -> None:
        terms = lexical_query_terms(
            "太陽病的脈證有哪些？",
            domain_terms=("太阳病", "脉证", "太阳"),
        )

        self.assertEqual(terms, ["太阳病", "脉证", "太陽病", "脈證"])

    def test_overlapping_domain_term_requires_an_independent_occurrence(self) -> None:
        nested_only = lexical_query_terms("太阳病", domain_terms=("太阳病", "太阳"))
        independently_present = lexical_query_terms(
            "太阳病属于太阳证",
            domain_terms=("太阳病", "太阳"),
        )

        self.assertNotIn("太阳", nested_only)
        self.assertEqual(independently_present[:2], ["太阳病", "太阳"])

    def test_normalize_query_text_maps_reviewer_confirmed_traditional_characters(self) -> None:
        self.assertEqual(
            normalize_query_text(
                "脈隨兩陽陰時從麼書頁經衛營氣虛實風濕為歸屬開關體"
            ),
            "脉随两阳阴时从么书页经卫营气虚实风湿为归属开关体",
        )

    def test_zero_fallback_budget_preserves_only_unbounded_domain_variants(self) -> None:
        terms = lexical_query_terms(
            "方證和调和营卫",
            domain_terms=("方证",),
            max_fallback_terms=0,
        )

        self.assertEqual(terms, ["方证", "方證"])


if __name__ == "__main__":
    unittest.main()
