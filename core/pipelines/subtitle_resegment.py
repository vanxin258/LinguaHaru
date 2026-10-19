"""LLM-driven subtitle resegmentation.

Upstream groups cues into sentence units heuristically (punctuation + pause) and
splits the translation back over the SAME cues, so cue boundaries never move.
Captions that came out of speech recognition are cut at fixed intervals, though,
and often hold half a sentence or two unrelated ones — no amount of re-splitting
fixes a boundary that is in the wrong place.

This module asks the model where the boundaries should be instead: it merges
half-sentences and splits run-ons, naming only source indices. Timings are then
recomputed here — the model never emits a timestamp, so it cannot invent one.
"""
from core.log_config import app_logger
from core.pipelines.subtitle_groups import flat, join_texts, seconds


def _ms(timestamp):
    """'00:01:02,345' -> 62345."""
    return int(round(seconds(timestamp) * 1000))


def _format(ms):
    ms = max(0, int(round(ms)))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_timed_cues(cues):
    """Attach the timing metadata the model is given and the timings are rebuilt
    from: absolute start/end in ms plus the gap to each neighbour."""
    timed = []
    for index, cue in enumerate(cues, start=1):
        start_ms, end_ms = _ms(cue["start_time"]), _ms(cue["end_time"])
        timed.append({
            "count_src": index,
            "value": cue["value"],
            "extra_time": cue.get("extra_time", ""),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": max(0, end_ms - start_ms),
        })
    for index, cue in enumerate(timed):
        previous = timed[index - 1] if index else None
        following = timed[index + 1] if index + 1 < len(timed) else None
        cue["gap_prev_ms"] = cue["start_ms"] - previous["end_ms"] if previous else 0
        cue["gap_next_ms"] = following["start_ms"] - cue["end_ms"] if following else 0
    return timed


def normalize_resegmented_cue_ids(cues):
    """Renumber cue_id 1..N in order, so a model that reuses or skips ids still
    produces a usable list."""
    return [dict(cue, cue_id=index) for index, cue in enumerate(cues, start=1)]


def validate_resegmented_cues(cues, raw_segments):
    """Raise ValueError unless the model's cues are usable as-is.

    The model's grouping is the whole point of the pass, so anything that would
    lose or reorder source text is rejected outright and the caller falls back to
    the un-resegmented extraction rather than shipping a truncated subtitle.
    """
    if not isinstance(cues, list) or not cues:
        raise ValueError("resegmented cues must be a non-empty list")

    valid_indices = {item["count_src"] for item in raw_segments}
    covered = set()
    previous_id = 0

    for cue in cues:
        if not isinstance(cue, dict):
            raise ValueError("each cue must be an object")

        cue_id = cue.get("cue_id")
        if not isinstance(cue_id, int) or cue_id <= previous_id:
            raise ValueError("cue_id must be strictly increasing integers")
        previous_id = cue_id

        indices = cue.get("source_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("source_indices must be a non-empty list")
        if any(not isinstance(i, int) or i not in valid_indices for i in indices):
            raise ValueError("source_indices contains an unknown source index")
        if indices != sorted(indices):
            raise ValueError("source_indices must be ascending")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("source_indices must be contiguous")
        if not str(cue.get("source_text", "")).strip():
            raise ValueError("source_text must not be empty")

        covered.update(indices)

    if covered != valid_indices:
        missing = sorted(valid_indices - covered)
        raise ValueError(f"resegmented cues must cover every source index (missing {missing})")
    return cues


def assign_cue_timestamps(cues, raw_segments):
    """Give every cue a timing, derived only from the source cues it covers.

    A cue that spans several sources takes their outer bounds. Several cues that
    share the same sources (a split) divide that span by text length, so a cue
    with more words stays on screen longer. Every timestamp is recomputed here —
    nothing the model said about timing is trusted, because it is not asked.
    """
    by_index = {item["count_src"]: item for item in raw_segments}
    timed = [dict(cue) for cue in cues]

    for cluster in _cluster_by_shared_sources(timed):
        covered = sorted({index for cue in cluster for index in cue["source_indices"]})
        start_ms = by_index[covered[0]]["start_ms"]
        end_ms = by_index[covered[-1]]["end_ms"]
        total = max(1, end_ms - start_ms)

        if len(cluster) == 1:
            cluster[0]["start_ms"], cluster[0]["end_ms"] = start_ms, end_ms
            continue

        weights = [max(1, len(flat(cue.get("source_text", "")))) for cue in cluster]
        total_weight = sum(weights)
        cursor = start_ms
        for position, cue in enumerate(cluster):
            if position == len(cluster) - 1:
                cue_end = end_ms
            else:
                share = round(total * (weights[position] / total_weight))
                cue_end = min(end_ms, max(cursor + 1, cursor + share))
            cue["start_ms"], cue["end_ms"] = cursor, cue_end
            cursor = cue_end

    for cue in timed:
        cue["start_time"] = _format(cue["start_ms"])
        cue["end_time"] = _format(cue["end_ms"])
    return timed


def _cluster_by_shared_sources(cues):
    """Consecutive cues that touch the same sources — the runs a split divides
    and a merge unifies. Different runs keep their own source bounds."""
    clusters, current, current_indices = [], [], set()
    for cue in cues:
        indices = set(cue["source_indices"])
        if current and current_indices.isdisjoint(indices):
            clusters.append(current)
            current, current_indices = [], set()
        current.append(cue)
        current_indices.update(indices)
    if current:
        clusters.append(current)
    return clusters


def to_source_items(cues):
    """The extraction shape the write-back path already understands.

    Each final cue is emitted as its own single-cue group, so upstream's writer
    writes it 1:1 with the timings computed above and needs no change at all.
    """
    items = []
    for index, cue in enumerate(cues, start=1):
        value = str(cue.get("source_text", ""))
        items.append({
            "count_src": index,
            "start_time": cue["start_time"],
            "end_time": cue["end_time"],
            "extra_time": cue.get("extra_time", ""),
            "value": value,
            "cues": [{
                "start_time": cue["start_time"],
                "end_time": cue["end_time"],
                "extra_time": cue.get("extra_time", ""),
                "value": value,
            }],
        })
    return items


def resegment_payload(raw_segments):
    """The JSON handed to the model: source text plus the timing it needs to
    judge whether a merge or split reads well."""
    return [
        {
            "index": cue["count_src"],
            "text": cue["value"],
            "duration_ms": cue["duration_ms"],
            "gap_prev_ms": cue["gap_prev_ms"],
            "gap_next_ms": cue["gap_next_ms"],
        }
        for cue in raw_segments
    ]


def join_source_text(raw_segments, indices):
    """Fallback text for a cue the model returned without usable source_text."""
    by_index = {item["count_src"]: item for item in raw_segments}
    return join_texts([flat(by_index[i]["value"]) for i in indices if i in by_index])


def log_summary(raw_segments, cues):
    """How much the boundaries actually moved — the pass is a no-op otherwise."""
    merged = sum(1 for cue in cues if len(cue["source_indices"]) > 1)
    split = sum(
        1 for cue in cues
        if any(other is not cue and set(cue["source_indices"]) & set(other["source_indices"])
               for other in cues)
    )
    app_logger.info(
        f"Subtitle resegment: {len(raw_segments)} source cues -> {len(cues)} cues "
        f"({merged} merged, {split} sharing a source)"
    )
