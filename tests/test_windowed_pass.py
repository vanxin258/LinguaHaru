import pytest

from core.engine.windowed_pass import (
    build_sliding_windows,
    merge_windowed_results,
    run_windowed_pass,
)


def _raw_items(count):
    return [{"count_src": index, "value": f"line {index}"} for index in range(1, count + 1)]


def _cue(*indices):
    return {
        "cue_id": 1,
        "source_indices": list(indices),
        "source_text": " ".join(f"line {index}" for index in indices),
    }


def _fallback(item, cue_id):
    return {
        "cue_id": cue_id,
        "source_indices": [item["count_src"]],
        "source_text": item["value"],
    }


def test_build_sliding_windows_assigns_commit_ranges_without_gaps():
    windows = build_sliding_windows(_raw_items(125), window_size=60, overlap_size=6)

    assert [(window.start, window.end) for window in windows] == [(0, 60), (54, 114), (108, 125)]
    assert [(window.commit_start, window.commit_end) for window in windows] == [(0, 57), (57, 111), (111, 125)]

    committed_positions = [
        position
        for window in windows
        for position in range(window.commit_start, window.commit_end)
    ]
    assert committed_positions == list(range(125))


def test_merge_uses_candidate_pool_to_fill_boundary_gaps():
    raw_items = _raw_items(8)
    windows = build_sliding_windows(raw_items, window_size=6, overlap_size=2)
    window_results = [
        [_cue(1, 2, 3, 4), _cue(5, 6)],
        [_cue(5, 6), _cue(7), _cue(8)],
    ]

    merged = merge_windowed_results(
        raw_items,
        windows,
        window_results,
        item_id_getter=lambda item: item["count_src"],
        result_indices_getter=lambda cue: cue["source_indices"],
        fallback_builder=_fallback,
    )

    assert [cue["source_indices"] for cue in merged] == [[1, 2, 3, 4], [5, 6], [7], [8]]
    assert [cue["cue_id"] for cue in merged] == [1, 2, 3, 4]


def test_merge_falls_back_for_uncovered_source_items():
    raw_items = _raw_items(4)
    windows = build_sliding_windows(raw_items, window_size=4, overlap_size=0)
    window_results = [
        [
            _cue(1),
            _cue(4),
        ]
    ]

    merged = merge_windowed_results(
        raw_items,
        windows,
        window_results,
        item_id_getter=lambda item: item["count_src"],
        result_indices_getter=lambda cue: cue["source_indices"],
        fallback_builder=_fallback,
    )

    assert [cue["source_indices"] for cue in merged] == [[1], [2], [3], [4]]
    assert [cue["source_text"] for cue in merged] == ["line 1", "line 2", "line 3", "line 4"]


def test_window_size_must_be_larger_than_overlap():
    with pytest.raises(ValueError, match="larger than overlap"):
        build_sliding_windows(_raw_items(5), window_size=5, overlap_size=5)


def test_run_windowed_pass_retries_failures_and_preserves_window_order():
    windows = build_sliding_windows(_raw_items(8), window_size=4, overlap_size=0)
    attempts = {window.index: 0 for window in windows}

    def process_window(window):
        attempts[window.index] += 1
        if window.index == 2 and attempts[window.index] == 1:
            raise ValueError("temporary failure")
        return [item["count_src"] for item in window.items]

    results = run_windowed_pass(
        windows,
        process_window,
        max_workers=2,
        max_retries=1,
        progress_message_builder=lambda completed, total: f"{completed}/{total}",
        retry_sleep_seconds=0,
    )

    assert results == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert attempts == {1: 1, 2: 2}
