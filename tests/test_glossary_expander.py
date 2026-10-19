import unittest
import tempfile
import subprocess
import sys
from pathlib import Path

from core.engine.text_separator import find_terms_with_hashtable


class GlossaryExpanderTests(unittest.TestCase):
    def test_short_uppercase_acronym_keeps_original_case_only(self):
        from core.engine.glossary_expander import generate_term_variants

        variants = generate_term_variants("PLC")

        self.assertEqual({"PLC"}, variants)

    def test_short_title_case_word_is_not_overly_restricted(self):
        from core.engine.glossary_expander import generate_term_variants

        variants = generate_term_variants("Valve")

        self.assertIn("valve", variants)
        self.assertIn("VALVE", variants)
        self.assertIn("Valve", variants)

    def test_phrase_generates_case_and_separator_variants(self):
        from core.engine.glossary_expander import generate_term_variants

        variants = generate_term_variants("water flow regulator")

        self.assertIn("Water Flow Regulator", variants)
        self.assertIn("WATER FLOW REGULATOR", variants)
        self.assertIn("water-flow-regulator", variants)
        self.assertIn("waterflowregulator", variants)

    def test_y_plural_variant_is_generated_when_substring_would_miss(self):
        from core.engine.glossary_expander import generate_term_variants

        variants = generate_term_variants("safety policy")

        self.assertIn("safety policies", variants)
        self.assertNotIn("safety policys", variants)

    def test_hash_match_finds_controlled_variants(self):
        entries = [
            ("Safety-Policies", "安全策略"),
            ("Water-Flow-Regulator", "水流量调节器"),
        ]

        text = "Safety-Policies and the Water-Flow-Regulator are configured."

        self.assertEqual(
            [
                ("Water-Flow-Regulator", "水流量调节器"),
                ("Safety-Policies", "安全策略"),
            ],
            find_terms_with_hashtable(text, entries),
        )

    def test_hash_match_does_not_expand_terms_at_runtime(self):
        self.assertEqual(
            [],
            find_terms_with_hashtable("The valves are open.", [("Valve", "阀")]),
        )

    def test_short_latin_term_does_not_match_inside_longer_word(self):
        entries = [
            ("A", "Amp"),
            ("Airmould", "Airmould"),
        ]

        self.assertEqual(
            [("Airmould", "Airmould")],
            find_terms_with_hashtable("Airmould gas-assist technology", entries),
        )

    def test_expand_glossary_csv_preserves_columns_and_writes_variants(self):
        from core.engine.glossary_expander import expand_glossary_csv

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "terms.csv"
            output_path = Path(temp_dir) / "terms_expanded.csv"
            input_path.write_text(
                "en,zh,context\n"
                "safety policy,安全策略,manual\n"
                "PLC,可编程逻辑控制器,automation\n",
                encoding="utf-8-sig",
            )

            count = expand_glossary_csv(str(input_path), str(output_path))

            output = output_path.read_text(encoding="utf-8-sig")
            self.assertGreater(count, 2)
            self.assertIn("en,zh,context", output)
            self.assertIn("safety policies,安全策略,manual", output)
            self.assertIn("Safety-Policies,安全策略,manual", output)
            self.assertIn("PLC,可编程逻辑控制器,automation", output)
            self.assertNotIn("plc,可编程逻辑控制器,automation", output)

    def test_cli_expands_csv_in_place(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "glossary.csv"
            output_path = Path(temp_dir) / "glossary_expanded.csv"
            input_path.write_text(
                "en,zh\n"
                "water flow regulator,水流量调节器\n",
                encoding="utf-8-sig",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.expand_glossary",
                    str(input_path),
                    "-o",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(output_path.exists())
            self.assertIn("water-flow-regulator", output_path.read_text(encoding="utf-8-sig"))

    def test_expand_glossary_csv_supports_gbk_input(self):
        from core.engine.glossary_expander import expand_glossary_csv

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "glossary.csv"
            output_path = Path(temp_dir) / "glossary_expanded.csv"
            input_path.write_text(
                "en,zh\n"
                "water flow regulator,水流量调节器\n",
                encoding="gbk",
            )

            count = expand_glossary_csv(str(input_path), str(output_path))

            self.assertGreater(count, 0)
            self.assertIn("water-flow-regulator", output_path.read_text(encoding="utf-8-sig"))

    def test_gui_backend_expands_file_without_eager_pyqt_import(self):
        from tools.expand_glossary_gui import default_output_path, run_expansion

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "terms.csv"
            input_path.write_text(
                "en,zh\n"
                "safety policy,安全策略\n",
                encoding="utf-8-sig",
            )

            output_path = default_output_path(str(input_path))
            count = run_expansion(str(input_path), output_path, src_lang="en")

            self.assertTrue(output_path.endswith("_expanded.csv"))
            self.assertGreater(count, 0)
            self.assertIn("safety-policies", Path(output_path).read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
