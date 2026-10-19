import csv
import re
from collections import OrderedDict
from pathlib import Path


SHORT_ACRONYM_MAX_LEN = 4
SEPARATOR_PATTERN = re.compile(r"(?<=[A-Za-z0-9])[\s_-]+(?=[A-Za-z0-9])")
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
CONSONANT_Y_PATTERN = re.compile(r"[^aeiou]y$", re.IGNORECASE)
PREFERRED_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030", "big5", "latin1", "shift-jis", "cp949"]


def _clean_term(term):
    if term is None:
        return ""
    return re.sub(r"\s+", " ", str(term).strip())


def _read_csv_with_encodings(path):
    last_error = None
    for encoding in PREFERRED_ENCODINGS:
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                rows = list(reader)
            return fieldnames, rows
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise ValueError(f"Unable to read CSV file: {path}")


def load_spacy_model(model_name="en_core_web_sm", model_path=None):
    """Load a spaCy pipeline lazily so GUI packaging stays optional."""
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is not installed. Install spaCy to use model-based glossary expansion.") from exc

    candidate_path = Path(model_path) if model_path else None
    if candidate_path and candidate_path.exists():
        return spacy.load(str(candidate_path))

    try:
        return spacy.load(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load spaCy model '{model_name}'. "
            "Bundle the model with the exe or provide a valid local model path."
        ) from exc


def _is_short_uppercase_acronym(term):
    compact = _clean_term(term)
    if not compact or SEPARATOR_PATTERN.search(compact):
        return False
    if len(compact) > SHORT_ACRONYM_MAX_LEN:
        return False
    return bool(re.search(r"[A-Z]", compact)) and compact.upper() == compact


def _case_variants(value):
    if not value:
        return set()
    return {value, value.lower(), value.upper(), value.title()}


def _phrase_parts(term):
    normalized = SEPARATOR_PATTERN.sub(" ", _clean_term(term))
    return [part for part in normalized.split(" ") if part]


def _separator_variants(parts):
    if not parts:
        return set()
    if len(parts) == 1:
        return {parts[0]}

    return {
        " ".join(parts),
        "-".join(parts),
        "".join(parts),
    }


def _split_last_word(value):
    matches = list(WORD_PATTERN.finditer(value))
    if not matches:
        return None
    match = matches[-1]
    return value[: match.start()], match.group(0), value[match.end() :]


def _pluralize_non_substring_noun(noun):
    if not noun or not re.search(r"[A-Za-z]", noun):
        return ""

    lower = noun.lower()
    plural = ""

    if CONSONANT_Y_PATTERN.search(lower):
        plural = noun[:-1] + ("IES" if noun.isupper() else "ies")
    elif lower.endswith("fe"):
        plural = noun[:-2] + ("VES" if noun.isupper() else "ves")
    elif lower.endswith("f"):
        plural = noun[:-1] + ("VES" if noun.isupper() else "ves")
    elif lower.endswith(("s", "x", "z", "ch", "sh")):
        plural = noun + ("ES" if noun.isupper() else "es")

    if plural and noun.lower() not in plural.lower():
        return plural
    return ""


def _plural_variants(value):
    last_word = _split_last_word(value)
    if not last_word:
        return set()

    prefix, noun, suffix = last_word
    plural = _pluralize_non_substring_noun(noun)
    if not plural:
        return set()
    return {f"{prefix}{plural}{suffix}"}


def _generate_term_variants_rule_based(term):
    cleaned = _clean_term(term)
    if not cleaned:
        return set()

    if _is_short_uppercase_acronym(cleaned):
        return {cleaned}

    bases = {cleaned}
    parts = _phrase_parts(cleaned)
    bases.update(_separator_variants(parts))

    for base in list(bases):
        bases.update(_plural_variants(base))

    expanded = set()
    for base in bases:
        expanded.update(_case_variants(base))

    return {_clean_term(variant) for variant in expanded if _clean_term(variant)}


def _generate_term_variants_spacy(term, nlp):
    cleaned = _clean_term(term)
    if not cleaned:
        return set()

    if _is_short_uppercase_acronym(cleaned):
        return {cleaned}

    variants = {cleaned}
    variants.update(_generate_term_variants_rule_based(cleaned))

    try:
        doc = nlp(cleaned)
    except Exception:
        return variants

    lemmas = []
    noun_token = None
    for token in doc:
        token_text = _clean_term(token.text)
        if not token_text:
            continue
        lemma = _clean_term(getattr(token, "lemma_", "") or token_text)
        if lemma and lemma != "-PRON-":
            lemmas.append(lemma)
        else:
            lemmas.append(token_text)

        if noun_token is None:
            pos = getattr(token, "pos_", "")
            tag = getattr(token, "tag_", "")
            if pos in {"NOUN", "PROPN"} or tag.startswith("NN"):
                noun_token = token

    base_form = _clean_term(" ".join(lemmas))
    if base_form:
        variants.add(base_form)
        variants.update(_case_variants(base_form))
        variants.update(_separator_variants(_phrase_parts(base_form)))

    if noun_token is not None and getattr(noun_token, "pos_", "") != "PROPN":
        noun_lemma = _clean_term(getattr(noun_token, "lemma_", "") or noun_token.text)
        plural = _pluralize_non_substring_noun(noun_lemma)
        if plural and plural.lower() != noun_lemma.lower():
            plural_parts = list(lemmas)
            noun_index = None
            for idx, token in enumerate(doc):
                if token is noun_token:
                    noun_index = idx
                    break
            if noun_index is not None and noun_index < len(plural_parts):
                plural_parts[noun_index] = plural
                plural_variant = _clean_term(" ".join(plural_parts))
                if plural_variant:
                    variants.add(plural_variant)
                    variants.update(_case_variants(plural_variant))
                    variants.update(_separator_variants(_phrase_parts(plural_variant)))

    return {_clean_term(variant) for variant in variants if _clean_term(variant)}


def generate_term_variants(term):
    """Generate controlled English term variants for glossary recall."""
    return _generate_term_variants_rule_based(term)


def expand_glossary_entries(glossary_entries, nlp=None):
    """Expand (source, target) glossary entries while preserving first-hit order."""
    expanded = OrderedDict()

    sorted_entries = sorted(
        ((_clean_term(src), dst) for src, dst in glossary_entries),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for source, target in sorted_entries:
        if not source or not target:
            continue
        for variant in sorted(generate_term_variants(source) if nlp is None else _generate_term_variants_spacy(source, nlp), key=len, reverse=True):
            expanded.setdefault(variant, target)

    return list(expanded.items())


def expand_glossary_csv(input_file, output_file, src_lang="en", nlp=None):
    """Expand a CSV glossary file by replacing source terms with variants."""
    fieldnames, source_rows = _read_csv_with_encodings(input_file)
    if not fieldnames:
        raise ValueError("Glossary CSV is missing a header row")
    if src_lang not in fieldnames:
        raise ValueError(f"Glossary CSV is missing source column: {src_lang}")

    rows = []
    seen = set()
    for row in source_rows:
        term = _clean_term(row.get(src_lang, ""))
        if not term:
            continue
        variant_source = _generate_term_variants_spacy(term, nlp) if nlp is not None else generate_term_variants(term)
        for variant in sorted(variant_source, key=len, reverse=True):
            output_row = dict(row)
            output_row[src_lang] = variant
            key = tuple(output_row.get(field, "") for field in fieldnames)
            if key in seen:
                continue
            seen.add(key)
            rows.append(output_row)

    with open(output_file, "w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
