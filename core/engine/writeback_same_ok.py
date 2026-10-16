from core.log_config import app_logger


def is_same_ok_item(item):
    """Return True when a translated item should be left untouched during writeback."""
    return isinstance(item, dict) and bool(item.get("same_ok", False))


def default_translation_key(item):
    """Return the common LinguaHaru writeback key for translated result items."""
    return item.get("id", item.get("count_src"))


def build_writeback_translation_map(
    translated_data,
    key_getter=None,
    value_field="translated",
    stringify_keys=True,
    skip_same_ok=True,
):
    """
    Build a writeback translation map, omitting same_ok items by default.

    same_ok segments are already known to be intentionally unchanged, so writing
    them back can only disturb the original file structure.
    """
    key_getter = key_getter or default_translation_key
    translations = {}
    skipped_count = 0

    for item in translated_data:
        if not isinstance(item, dict):
            continue

        if skip_same_ok and is_same_ok_item(item):
            skipped_count += 1
            continue

        key = key_getter(item)
        if key is None or value_field not in item:
            continue

        if stringify_keys:
            key = str(key)
        translations[key] = item[value_field]

    if skipped_count:
        app_logger.info(f"Skipped {skipped_count} same_ok segments during writeback")

    return translations


def build_same_ok_key_set(translated_data, key_getter=None, stringify_keys=True):
    """Return writeback keys whose translated items are marked same_ok."""
    key_getter = key_getter or default_translation_key
    keys = set()

    for item in translated_data:
        if not is_same_ok_item(item):
            continue

        key = key_getter(item)
        if key is None:
            continue

        if stringify_keys:
            key = str(key)
        keys.add(key)

    return keys
