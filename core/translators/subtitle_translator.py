import json
import os

from core.engine.base_translator import DocumentTranslator
from core.engine.translation_checker import clean_json
from core.engine.windowed_pass import (
    build_sliding_windows, merge_windowed_results, run_windowed_pass)
from core.llm.llm_wrapper import translate_text
from core.load_prompt import load_prompt
from core.log_config import app_logger
from core.pipelines import subtitle_resegment as reseg
from core.pipelines.subtitle_translation_pipeline import (
    extract_srt_content_to_json, parse_srt_cues, write_translated_content_to_srt)

RESEGMENT_PROMPT = "subtitle_resegment"
DEFAULT_BATCH_SIZE = 10
DEFAULT_CONTEXT_OVERLAP = 2


class SubtitlesTranslator(DocumentTranslator):
    """SRT translator. bilingual_mode puts translation + original in each cue.

    subtitle_resegment_mode additionally asks the model where the cue boundaries
    should be before translating — see core.pipelines.subtitle_resegment. Long
    files are resegmented in overlapping windows so one bad batch cannot spoil
    the whole file.
    """

    def __init__(self, *args, bilingual_mode=False, subtitle_resegment_mode=False,
                 subtitle_resegment_batch_size=None,
                 subtitle_resegment_context_overlap=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bilingual_mode = bilingual_mode
        self.subtitle_resegment_mode = subtitle_resegment_mode
        self.subtitle_resegment_batch_size = (
            subtitle_resegment_batch_size or DEFAULT_BATCH_SIZE)
        self.subtitle_resegment_context_overlap = (
            subtitle_resegment_context_overlap or DEFAULT_CONTEXT_OVERLAP)

    def extract_content_to_json(self, progress_callback=None):
        if not self.subtitle_resegment_mode:
            return extract_srt_content_to_json(self.input_file_path, self.temp_dir)
        try:
            return self._extract_resegmented(progress_callback)
        except Exception as exc:  # noqa: BLE001
            # Resegmentation is an enhancement; a failure must fall back to the
            # original boundaries rather than lose the file.
            app_logger.warning(
                f"Subtitle resegment failed ({exc}); keeping the original cue boundaries")
            return extract_srt_content_to_json(self.input_file_path, self.temp_dir)

    def write_translated_json_to_file(self, json_path, translated_json_path, progress_callback=None):
        self._written_output_path = write_translated_content_to_srt(
            self.input_file_path, json_path, translated_json_path,
            self.result_dir, self.src_lang, self.dst_lang,
            bilingual_mode=self.bilingual_mode)

    # -- resegmentation ----------------------------------------------------

    def _extract_resegmented(self, progress_callback=None):
        raw = reseg.build_timed_cues(parse_srt_cues(self.input_file_path))
        if not raw:
            raise ValueError("no cues parsed from the subtitle")

        windows = build_sliding_windows(
            raw,
            window_size=self.subtitle_resegment_batch_size,
            # context_overlap counts cues on each side of the committed range
            overlap_size=self.subtitle_resegment_context_overlap * 2,
        )
        app_logger.info(
            f"Subtitle resegment: {len(raw)} cues in {len(windows)} windows "
            f"(batch={self.subtitle_resegment_batch_size}, "
            f"context={self.subtitle_resegment_context_overlap}/side)")

        window_results = run_windowed_pass(
            windows,
            self._resegment_window,
            max_workers=self.num_threads,
            max_retries=self.max_retries,
            progress_callback=progress_callback,
            logger=app_logger,
            check_stop=self.check_for_stop,
        )
        cues = merge_windowed_results(
            raw, windows, window_results,
            item_id_getter=lambda item: item["count_src"],
            result_indices_getter=lambda cue: cue["source_indices"],
            fallback_builder=self._default_cue,
        )
        cues = reseg.assign_cue_timestamps(cues, raw)
        reseg.log_summary(raw, cues)

        items = reseg.to_source_items(cues)
        temp_folder = os.path.join(
            self.temp_dir, os.path.splitext(os.path.basename(self.input_file_path))[0])
        os.makedirs(temp_folder, exist_ok=True)
        json_path = os.path.join(temp_folder, "src.json")
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(items, json_file, ensure_ascii=False, indent=4)
        return json_path

    @staticmethod
    def _default_cue(raw_segment, cue_id):
        """One source cue kept as-is — the fallback for a window that failed."""
        return {
            "cue_id": cue_id,
            "source_indices": [raw_segment["count_src"]],
            "source_text": raw_segment["value"],
        }

    def _resegment_window(self, window):
        system_prompt, user_prompt, previous_prompt, previous_default, glossary_prompt = load_prompt(
            self.src_lang, self.dst_lang, prompt_code_override=RESEGMENT_PROMPT)
        payload = json.dumps(reseg.resegment_payload(window.items), ensure_ascii=False)

        result, success, token_usage = translate_text(
            payload, previous_default, self.model, self.use_online, self.api_key,
            system_prompt, user_prompt, previous_prompt, glossary_prompt,
            check_stop_callback=self.check_for_stop)
        self._add_token_usage(token_usage)
        if not success or not result:
            raise ValueError("resegment pass returned nothing")

        cues = reseg.normalize_resegmented_cue_ids(json.loads(clean_json(result)))
        for cue in cues:
            if not str(cue.get("source_text", "")).strip():
                cue["source_text"] = reseg.join_source_text(
                    window.items, cue.get("source_indices") or [])
        reseg.validate_resegmented_cues(cues, window.items)
        return cues
