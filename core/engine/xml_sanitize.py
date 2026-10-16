"""XML-safety sanitizing for translated text.

The LLM occasionally returns characters that are not legal in XML 1.0 — a raw
C0 control such as ``\\x03``, a lone surrogate. Storing them is harmless, but
write-back assigns them to lxml element text and serializes the part, which
raises ``ValueError: All strings must be XML compatible: Unicode or ASCII, no
NULL bytes or control characters`` and fails the whole document *after* all the
translation work has already been paid for.

Stripping them at the boundary is the same treatment the Excel writer already
gives cell values (``excel_translation_pipeline._safe_cell_value``); this module
generalizes it so every format is covered, including the ones whose writers
build XML with lxml (Word, PPT, EPUB, ODT, HTML, Excel's chart/drawing parts).

Design notes
------------
* Legal XML 1.0 characters are ``#x9 | #xA | #xD | [#x20-#xD7FF] |
  [#xE000-#xFFFD] | [#x10000-#x10FFFF]``; the pattern below is the complement,
  so tab/newline/carriage-return survive. Private-use codepoints (the U+F000-U+F0FF
  Wingdings glyphs, the U+E000-U+E001 mask sentinels) are legal and kept.
* Characters are **removed**, not replaced with a placeholder: a control
  character carries no meaning worth preserving.
* Sanitizing is idempotent, so it is safe to run on both freshly produced
  translations and ones re-read from disk in continue mode.
"""

import json
import os
import re

# The class is assembled from code points rather than written as one literal so
# that no invisible non-characters (U+FFFE/U+FFFF) or lone surrogates end up in
# this source file, where editors and diffs handle them badly.
_INVALID_CODEPOINTS = (
    list(range(0x00, 0x09))          # C0 controls except \t
    + [0x0B, 0x0C]                   # ... except \n
    + list(range(0x0E, 0x20))        # ... except \r
    + list(range(0xD800, 0xE000))    # surrogates: unencodable on their own
    + [0xFFFE, 0xFFFF]               # XML non-characters
)
INVALID_XML_CHARS = re.compile("[" + "".join(map(chr, _INVALID_CODEPOINTS)) + "]")


def strip_xml_invalid(text):
    """Remove XML-incompatible characters from `text`.

    Non-strings pass through unchanged so callers can hand over any JSON value.
    """
    if not isinstance(text, str):
        return text
    return INVALID_XML_CHARS.sub("", text)


def strip_xml_invalid_in_json(json_file_path, field="translated"):
    """Strip XML-incompatible characters from `field` of every item of a
    translation JSON file, in place. Returns the number of items changed.

    Rewrites the file only when something actually changed, atomically (tmp +
    os.replace), mirroring ``DocumentTranslator._apply_text_rules``.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        value = item.get(field)
        if isinstance(value, str):
            cleaned = INVALID_XML_CHARS.sub("", value)
            if cleaned != value:
                item[field] = cleaned
                changed += 1

    if changed:
        tmp = json_file_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(tmp, json_file_path)

    return changed
