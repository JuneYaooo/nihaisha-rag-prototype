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
        self.assertEqual(normalize_query_text("一錢、方證、發熱"), "一钱、方证、发热")


if __name__ == "__main__":
    unittest.main()
