from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class SlidingWindow:
    index: int
    start: int
    end: int
    commit_start: int
    commit_end: int
    items: list


def build_sliding_windows(items, window_size, overlap_size):
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap_size < 0:
        raise ValueError("overlap_size must not be negative")
    if window_size <= overlap_size:
        raise ValueError("window_size must be larger than overlap_size")

    item_count = len(items)
    if item_count == 0:
        return []

    stride = window_size - overlap_size
    contexts = []
    start = 0
    while start < item_count:
        end = min(item_count, start + window_size)
        contexts.append((start, end))
        if end == item_count:
            break
        start += stride

    boundaries = []
    for current, next_context in zip(contexts, contexts[1:]):
        overlap_start = next_context[0]
        overlap_end = current[1]
        overlap_length = max(0, overlap_end - overlap_start)
        boundaries.append(overlap_start + overlap_length // 2)

    windows = []
    for index, (start, end) in enumerate(contexts):
        commit_start = 0 if index == 0 else boundaries[index - 1]
        commit_end = item_count if index == len(contexts) - 1 else boundaries[index]
        windows.append(
            SlidingWindow(
                index=index + 1,
                start=start,
                end=end,
                commit_start=commit_start,
                commit_end=commit_end,
                items=items[start:end],
            )
        )
    return windows


def merge_windowed_results(
    raw_items,
    windows,
    window_results,
    item_id_getter,
    result_indices_getter,
    fallback_builder,
):
    id_to_position = {item_id_getter(item): index for index, item in enumerate(raw_items)}
    candidates = _collect_candidates(windows, window_results, id_to_position, result_indices_getter)

    selected = []
    covered_positions = set()
    for candidate in candidates:
        if candidate["fully_in_commit"]:
            selected.append(candidate["result"])
            covered_positions.update(candidate["positions"])

    for gap in _find_position_gaps(len(raw_items), covered_positions):
        gap_positions = set(gap)
        exact_candidate = _best_exact_gap_candidate(candidates, gap_positions)
        if exact_candidate:
            selected.append(exact_candidate["result"])
            covered_positions.update(exact_candidate["positions"])
            continue

        for result in _build_gap_results(
            gap,
            raw_items,
            candidates,
            fallback_builder,
        ):
            selected.append(result)
            covered_positions.update(id_to_position[index] for index in result_indices_getter(result))

    selected.sort(key=lambda result: min(id_to_position[index] for index in result_indices_getter(result)))
    return _renumber_results(selected)


def run_windowed_pass(
    windows,
    process_window,
    max_workers=1,
    max_retries=0,
    progress_callback=None,
    progress_message_builder=None,
    logger=None,
    check_stop=None,
    retry_sleep_seconds=1,
):
    if not windows:
        return []

    worker_count = max(1, int(max_workers or 1))
    retry_count = max(0, int(max_retries or 0))
    results = [None] * len(windows)

    def log_warning(message):
        if logger is not None and hasattr(logger, "warning"):
            logger.warning(message)

    def run_one(window):
        attempts = retry_count + 1
        last_error = None
        for attempt in range(1, attempts + 1):
            if check_stop:
                check_stop()
            try:
                return process_window(window)
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                log_warning(
                    f"Windowed pass window {window.index} failed "
                    f"(attempt {attempt}/{attempts}): {exc}"
                )
                if retry_sleep_seconds:
                    time.sleep(retry_sleep_seconds)
        raise last_error

    total = len(windows)
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_position = {
            executor.submit(run_one, window): position
            for position, window in enumerate(windows)
        }
        for future in as_completed(future_to_position):
            if check_stop:
                check_stop()
            position = future_to_position[future]
            results[position] = future.result()
            completed += 1
            if progress_callback is not None:
                message = None
                if progress_message_builder is not None:
                    message = progress_message_builder(completed, total)
                progress_callback(completed / total, message)

    return results


def _collect_candidates(windows, window_results, id_to_position, result_indices_getter):
    candidates = []
    for window, results in zip(windows, window_results):
        commit_positions = set(range(window.commit_start, window.commit_end))
        for result in results:
            positions = [
                id_to_position[index]
                for index in result_indices_getter(result)
                if index in id_to_position
            ]
            if not positions:
                continue
            position_set = set(positions)
            candidates.append(
                {
                    "result": result,
                    "positions": position_set,
                    "first_position": min(position_set),
                    "window_index": window.index,
                    "commit_overlap": len(position_set & commit_positions),
                    "fully_in_commit": position_set <= commit_positions,
                }
            )
    return candidates


def _find_position_gaps(item_count, covered_positions):
    gaps = []
    current_gap = []
    for position in range(item_count):
        if position in covered_positions:
            if current_gap:
                gaps.append(current_gap)
                current_gap = []
        else:
            current_gap.append(position)
    if current_gap:
        gaps.append(current_gap)
    return gaps


def _best_exact_gap_candidate(candidates, gap_positions):
    exact_candidates = [
        candidate
        for candidate in candidates
        if candidate["positions"] == gap_positions
    ]
    if not exact_candidates:
        return None
    return max(
        exact_candidates,
        key=lambda candidate: (
            candidate["commit_overlap"],
            -candidate["window_index"],
        ),
    )


def _build_gap_results(gap, raw_items, candidates, fallback_builder):
    gap_positions = set(gap)
    results = []
    local_covered = set()
    for position in gap:
        if position in local_covered:
            continue
        candidate = _best_candidate_starting_at(candidates, position, gap_positions, local_covered)
        if candidate:
            results.append(candidate["result"])
            local_covered.update(candidate["positions"])
            continue
        results.append(fallback_builder(raw_items[position], len(results) + 1))
        local_covered.add(position)
    return results


def _best_candidate_starting_at(candidates, position, gap_positions, local_covered):
    matching = [
        candidate
        for candidate in candidates
        if candidate["first_position"] == position
        and candidate["positions"] <= gap_positions
        and not candidate["positions"] & local_covered
    ]
    if not matching:
        return None
    return max(
        matching,
        key=lambda candidate: (
            len(candidate["positions"]),
            candidate["commit_overlap"],
            -candidate["window_index"],
        ),
    )


def _renumber_results(results):
    renumbered = []
    for cue_id, result in enumerate(results, start=1):
        item = deepcopy(result)
        item["cue_id"] = cue_id
        renumbered.append(item)
    return renumbered
