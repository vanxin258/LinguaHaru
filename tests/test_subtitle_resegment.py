# LLM-driven subtitle resegmentation: the model names source indices, the code
# recomputes every timestamp, and the result is written through the normal SRT
# writer (each final cue as its own single-cue group).
import json
import os
import tempfile
import unittest
from pathlib import Path

from core.pipelines import subtitle_resegment as reseg
from core.pipelines.subtitle_translation_pipeline import (
    parse_srt_cues, write_translated_content_to_srt)

SRT = """1
00:00:00,000 --> 00:00:02,000
Hello there my

2
00:00:02,000 --> 00:00:04,000
friend how are you

3
00:00:06,000 --> 00:00:08,000
I am fine
"""


def write_srt(path, text=SRT):
    Path(path).write_text(text, encoding="utf-8")
    return path


class BuildTimedCuesTests(unittest.TestCase):
    def test_adds_absolute_timing_and_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            cues = parse_srt_cues(write_srt(os.path.join(tmp, "a.srt")))
            timed = reseg.build_timed_cues(cues)

        self.assertEqual([c["count_src"] for c in timed], [1, 2, 3])
        self.assertEqual(timed[0]["start_ms"], 0)
        self.assertEqual(timed[0]["end_ms"], 2000)
        self.assertEqual(timed[0]["duration_ms"], 2000)
        self.assertEqual(timed[1]["gap_prev_ms"], 0)     # back to back
        self.assertEqual(timed[1]["gap_next_ms"], 2000)  # the pause before cue 3
        self.assertEqual(timed[2]["gap_next_ms"], 0)     # last cue


class ValidateResegmentedCuesTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.raw = reseg.build_timed_cues(
                parse_srt_cues(write_srt(os.path.join(tmp, "a.srt"))))

    def _cues(self, *index_lists):
        return [{"cue_id": i + 1, "source_indices": list(idx),
                 "source_text": "text"} for i, idx in enumerate(index_lists)]

    def test_accepts_merge_and_split(self):
        # 1+2 merged, 3 split into two halves
        cues = self._cues([1, 2], [3], [3])
        self.assertEqual(reseg.validate_resegmented_cues(cues, self.raw), cues)

    def test_rejects_missing_coverage(self):
        with self.assertRaises(ValueError):
            reseg.validate_resegmented_cues(self._cues([1, 2]), self.raw)

    def test_rejects_non_contiguous(self):
        with self.assertRaises(ValueError):
            reseg.validate_resegmented_cues(self._cues([1], [3]), self.raw)

    def test_rejects_descending_ids(self):
        cues = [{"cue_id": 2, "source_indices": [1], "source_text": "a"},
                {"cue_id": 1, "source_indices": [2, 3], "source_text": "b"}]
        with self.assertRaises(ValueError):
            reseg.validate_resegmented_cues(cues, self.raw)

    def test_rejects_unknown_index(self):
        with self.assertRaises(ValueError):
            reseg.validate_resegmented_cues(self._cues([1, 2, 9]), self.raw)

    def test_rejects_empty_list(self):
        with self.assertRaises(ValueError):
            reseg.validate_resegmented_cues([], self.raw)


class AssignCueTimestampsTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.raw = reseg.build_timed_cues(
                parse_srt_cues(write_srt(os.path.join(tmp, "a.srt"))))

    def test_merged_cue_takes_the_outer_bounds(self):
        cues = [{"cue_id": 1, "source_indices": [1, 2], "source_text": "Hello there my friend how are you"},
                {"cue_id": 2, "source_indices": [3], "source_text": "I am fine"}]
        timed = reseg.assign_cue_timestamps(cues, self.raw)
        self.assertEqual((timed[0]["start_time"], timed[0]["end_time"]),
                         ("00:00:00,000", "00:00:04,000"))
        self.assertEqual((timed[1]["start_time"], timed[1]["end_time"]),
                         ("00:00:06,000", "00:00:08,000"))

    def test_split_divides_the_span_and_stays_ordered(self):
        cues = [{"cue_id": 1, "source_indices": [1], "source_text": "Hello there my"},
                {"cue_id": 2, "source_indices": [1], "source_text": "friend how are you"},
                {"cue_id": 3, "source_indices": [2, 3], "source_text": "rest"}]
        timed = reseg.assign_cue_timestamps(cues, self.raw)
        # First two share source 1 (0-2000ms) and divide it by text length.
        self.assertEqual(timed[0]["start_time"], "00:00:00,000")
        self.assertGreater(timed[0]["end_ms"], timed[0]["start_ms"])
        self.assertEqual(timed[0]["end_ms"], timed[1]["start_ms"])
        self.assertEqual(timed[1]["end_time"], "00:00:02,000")
        # And every cue stays strictly ordered and non-overlapping.
        for earlier, later in zip(timed, timed[1:]):
            self.assertLessEqual(earlier["end_ms"], later["start_ms"])

    def test_untouched_cues_keep_their_timing(self):
        cues = [{"cue_id": i + 1, "source_indices": [i + 1], "source_text": f"cue {i}"}
                for i in range(3)]
        timed = reseg.assign_cue_timestamps(cues, self.raw)
        self.assertEqual([(c["start_time"], c["end_time"]) for c in timed],
                         [("00:00:00,000", "00:00:02,000"),
                          ("00:00:02,000", "00:00:04,000"),
                          ("00:00:06,000", "00:00:08,000")])


class WriteThroughTests(unittest.TestCase):
    """The whole point: resegmented cues reach the SRT with the new boundaries
    and timings, via the ordinary write-back path."""

    def test_resegmented_srt_is_written_with_new_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_srt(os.path.join(tmp, "movie.srt"))
            raw = reseg.build_timed_cues(parse_srt_cues(src))

            # Model merges 1+2 into one cue and leaves 3 alone.
            cues = [
                {"cue_id": 1, "source_indices": [1, 2],
                 "source_text": "Hello there my friend how are you"},
                {"cue_id": 2, "source_indices": [3], "source_text": "I am fine"},
            ]
            reseg.validate_resegmented_cues(cues, raw)
            items = reseg.to_source_items(reseg.assign_cue_timestamps(cues, raw))

            src_json = os.path.join(tmp, "src.json")
            Path(src_json).write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            translated = os.path.join(tmp, "translated.json")
            Path(translated).write_text(json.dumps(
                [{"count_src": 1, "translated": "你好，我的朋友，你好吗"},
                 {"count_src": 2, "translated": "我很好"}], ensure_ascii=False), encoding="utf-8")

            out = write_translated_content_to_srt(
                src, src_json, translated, os.path.join(tmp, "result"),
                src_lang="en", dst_lang="zh")

            text = Path(out).read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:04,000", text)
            self.assertIn("00:00:06,000 --> 00:00:08,000", text)
            self.assertIn("你好，我的朋友，你好吗", text)
            self.assertIn("我很好", text)
            # Two cues, not the original three.
            self.assertEqual(text.count("-->"), 2)


if __name__ == "__main__":
    unittest.main()
