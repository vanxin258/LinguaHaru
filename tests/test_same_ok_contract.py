# The {translation, same_ok} output contract: the model may declare an item as
# needing no translation, and the pipeline honours that only when the text
# itself looks like it (the safety net), then drops it from write-back.
import json
import os
import tempfile
import unittest
from pathlib import Path

from core.engine.translation_checker import (
    _plausible_same_ok,
    extract_translation_result,
    process_translation_results,
)


class SameOkPlausibilityTests(unittest.TestCase):
    """The safety net under the model's claim."""

    def test_accepts_machine_content(self):
        for text in ("100%", "3.2.1", "2026-10-16", "1,234.56", "GPT-4",
                     "AAEN000002", "v1.2.3", "N/A", "•••"):
            with self.subTest(text=text):
                self.assertTrue(_plausible_same_ok(text), text)

    def test_accepts_whole_string_urls_emails_paths(self):
        for text in ("https://example.com/a?b=1", "someone@example.com",
                     "C:\\docs\\report.docx", "report_final.docx"):
            with self.subTest(text=text):
                self.assertTrue(_plausible_same_ok(text), text)

    def test_rejects_prose(self):
        for text in ("Report type",
                     "Describe what happened, when it started, and what is known.",
                     "This is a sentence that a reader would expect translated.",
                     "信息安全事故报告表",
                     "Einleitung"):
            with self.subTest(text=text):
                self.assertFalse(_plausible_same_ok(text), text)

    def test_rejects_empty(self):
        self.assertFalse(_plausible_same_ok(""))
        self.assertFalse(_plausible_same_ok(None))


class ExtractTranslationResultTests(unittest.TestCase):
    def test_reads_both_shapes(self):
        payload = {
            "1": {"translation": "你好", "same_ok": False},
            "2": {"translation": "A-101", "same_ok": True},
            "3": "legacy plain string",
        }
        self.assertEqual(extract_translation_result(payload, "1"), ("你好", False))
        self.assertEqual(extract_translation_result(payload, "2"), ("A-101", True))
        self.assertEqual(extract_translation_result(payload, "3"), ("legacy plain string", False))

    def test_missing_and_null_do_not_raise(self):
        payload = {"1": None, "2": 42, "3": {"no_translation_key": "x"}}
        self.assertEqual(extract_translation_result(payload, "1"), ("", False))
        self.assertEqual(extract_translation_result(payload, "2"), ("42", False))
        self.assertEqual(extract_translation_result(payload, "3"), ("", False))
        self.assertEqual(extract_translation_result(payload, "missing"), ("", False))


class ProcessTranslationResultsTests(unittest.TestCase):
    def _run(self, original, translated, src_lang="en", dst_lang="zh"):
        """Returns (result map, result items, failed items)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split = [{"count_split": int(k), "value": v} for k, v in original.items()]
            (root / "src_split.json").write_text(json.dumps(split, ensure_ascii=False),
                                                 encoding="utf-8")
            (root / "result_split.json").write_text("[]", encoding="utf-8")
            (root / "failed.json").write_text("[]", encoding="utf-8")
            results = process_translation_results(
                json.dumps(original, ensure_ascii=False),
                json.dumps(translated, ensure_ascii=False),
                str(root / "src_split.json"),
                str(root / "result_split.json"),
                str(root / "failed.json"),
                src_lang, dst_lang,
            )
            items = json.loads((root / "result_split.json").read_text(encoding="utf-8"))
            failed = json.loads((root / "failed.json").read_text(encoding="utf-8"))
            return results, items, failed

    def test_honours_same_ok_for_an_identifier(self):
        results, items, failed = self._run(
            {"1": "AAEN000002"},
            {"1": {"translation": "AAEN000002", "same_ok": True}},
        )
        self.assertTrue(results, f"expected acceptance, failed={failed}")
        self.assertEqual(results["1"], "AAEN000002")
        self.assertTrue(items and items[0].get("same_ok") is True,
                        f"same_ok must survive into the result items: {items}")

    def test_rejects_same_ok_claimed_for_prose(self):
        """A wrong claim must degrade to the old behaviour (echo -> failed),
        never silently ship untranslated prose."""
        results, _items, failed = self._run(
            {"1": "Report type"},
            {"1": {"translation": "Report type", "same_ok": True}},
        )
        self.assertFalse(results.get("1"), results)
        self.assertTrue(failed, "the echoed prose should be queued for retry")

    def test_plain_string_shape_still_works(self):
        results, items, _failed = self._run(
            {"1": "Hello world"},
            {"1": "你好世界"},
        )
        self.assertEqual(results["1"], "你好世界")
        self.assertNotIn("same_ok", items[0])


class PromptContractTests(unittest.TestCase):
    """Every translation prompt must ask for the object shape, or the checker's
    same_ok handling silently never fires."""

    def test_all_prompts_request_same_ok(self):
        prompts_dir = Path(__file__).resolve().parents[1] / "config" / "prompts"
        prompts = [p for p in prompts_dir.glob("*.json")]
        self.assertTrue(prompts)
        for path in prompts:
            with self.subTest(prompt=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                system = data.get("system_prompt", "")
                self.assertIn("same_ok", system, f"{path.name} does not mention same_ok")
                self.assertIn("translation", system, f"{path.name} does not mention translation")


if __name__ == "__main__":
    unittest.main()
