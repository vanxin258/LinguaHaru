"""真实 API 验收:用真实模型跑一遍自定义功能,检验模型的实际行为。

密钥不写进代码,也不进版本库 —— 从桌面版安装目录的 DPAPI 密钥文件解密读取
(仅 Windows,且绑定当前用户)。

用法:
    python tests/acceptance/run_real_api.py same-ok     # 免翻译标记
    python tests/acceptance/run_real_api.py resegment   # 字幕 LLM 断句
    python tests/acceptance/run_real_api.py all

环境变量:
    LINGUAHARU_KEY_FILE   密钥文件路径(默认桌面版安装目录)
    LINGUAHARU_MODEL      接口配置名(默认 qwen)
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_KEY_FILE = r"D:\LinguaHaru-desktop\data\keys\qwen.json"
FIXTURES = REPO_ROOT / "tests" / "acceptance" / "fixtures"
WORK = REPO_ROOT / "tests" / "acceptance" / "_real"


def load_api_key():
    path = os.environ.get("LINGUAHARU_KEY_FILE", DEFAULT_KEY_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"找不到密钥文件:{path}(用 LINGUAHARU_KEY_FILE 指定)")
    from core.api_keys import _dpapi_unprotect
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    key = _dpapi_unprotect(payload.get("api_key_enc", "")) or payload.get("api_key", "")
    if not key:
        raise SystemExit(f"{path} 里的密钥解不开(DPAPI 绑定创建它的 Windows 用户)")
    return key


def build_translator(model, api_key, src, dst, file_path, **extra):
    from core import backend
    ext = os.path.splitext(file_path)[1]
    work = WORK / Path(file_path).stem
    temp_dir, result_dir = work / "temp", work / "result"
    for d in (temp_dir, result_dir):
        d.mkdir(parents=True, exist_ok=True)

    translator_class = backend.get_translator_class(ext, file_path=file_path, **extra)
    if translator_class is None:
        raise SystemExit(f"没有可用的翻译器:{ext}")
    return translator_class(
        str(file_path), model, True, api_key, src, dst, False,
        max_token=backend.max_token_for_model(model),
        max_retries=backend.max_retries_for_model(model),
        thread_count=backend.thread_count_for_mode(True, model),
        glossary_path=None, temp_dir=str(temp_dir), result_dir=str(result_dir),
        session_lang="en", log_dir=str(result_dir),
    )


def translated_items(translator):
    path = Path(translator.result_json_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------- same_ok --
def check_same_ok(model, api_key, src="en", dst="zh", doc_name=None):
    """Does the model actually use the object contract, and does the pipeline
    honour it without shipping untranslated prose?"""
    docs = sorted(FIXTURES.glob("*.docx"), key=lambda p: p.stat().st_size)
    if not docs:
        raise SystemExit(f"没有测试文档,请放进 {FIXTURES}")
    if doc_name:
        docs = [d for d in docs if d.name == doc_name] or docs
    # Smallest first: a real API run costs money and time per item.
    doc = docs[0]
    print(f"文档:{doc.name} ({doc.stat().st_size // 1024} KB)  {src} -> {dst}\n")

    translator = build_translator(model, api_key, src, dst, doc)
    out, _missing = translator.process(doc.stem, doc.suffix)
    items = translated_items(translator)

    marked = [i for i in items if i.get("same_ok")]
    print(f"结果条目 {len(items)} 条,其中 same_ok 标记 {len(marked)} 条")
    for item in marked[:15]:
        print(f"  [{item.get('count_src')}] {str(item.get('original'))[:70]!r}")
    if len(marked) > 15:
        print(f"  ...另有 {len(marked) - 15} 条")

    from core.engine.translation_checker import _plausible_same_ok
    implausible = [i for i in marked if not _plausible_same_ok(i.get("original"))]
    print(f"\n安全网复检:标记中 {len(implausible)} 条不像该保留的"
          f"({'通过' if not implausible else '异常'})")
    for item in implausible[:5]:
        print(f"  可疑: {str(item.get('original'))[:70]!r}")

    print(f"\n输出:{out}")
    return len(items), len(marked), len(implausible)


# ------------------------------------------------------------- resegment --
def build_asr_style_srt(source_srt, target, cuts_per_cue=3, max_cues=None):
    """Re-cut a known-good SRT into ASR-style fixed-ish chunks, so the correct
    answer is known and the model can be scored against it."""
    from core.pipelines.subtitle_translation_pipeline import parse_srt_cues
    from core.pipelines.subtitle_groups import seconds

    def stamp(ms):
        ms = max(0, int(round(ms)))
        h, rest = divmod(ms, 3_600_000)
        m, rest = divmod(rest, 60_000)
        s, ms = divmod(rest, 1_000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines, index, truth = [], 0, []
    for cue in parse_srt_cues(str(source_srt)):
        if max_cues is not None and len(truth) >= max_cues:
            break
        text = cue["value"].replace("␊", " ").replace("␍", " ")
        words = text.split()
        start_ms = int(seconds(cue["start_time"]) * 1000)
        end_ms = int(seconds(cue["end_time"]) * 1000)
        if len(words) < cuts_per_cue * 2:
            chunks = [text]
        else:
            size = max(1, len(words) // cuts_per_cue)
            chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
        span = (end_ms - start_ms) / len(chunks)
        indices = []
        for n, chunk in enumerate(chunks):
            index += 1
            indices.append(index)
            lines.append(f"{index}\n{stamp(start_ms + n * span)} --> "
                         f"{stamp(start_ms + (n + 1) * span)}\n{chunk}\n\n")
        truth.append(indices)

    Path(target).write_text("".join(lines), encoding="utf-8")
    return len(truth), index, truth


def check_resegment(model, api_key, src="en", dst="zh", max_cues=40):
    """Feed an ASR-style SRT in and see whether the model puts the boundaries
    back where they belong — the known-good cue grouping is the answer key."""
    good = REPO_ROOT / "result" / "8月25日 1_en2en_resegmented.srt"
    if not good.exists():
        raise SystemExit(f"找不到参考字幕:{good}")

    work = WORK / "resegment"
    work.mkdir(parents=True, exist_ok=True)
    asr_srt = work / "asr_style.srt"
    groups, cue_count, truth = build_asr_style_srt(
        good, asr_srt, max_cues=max_cues)
    print(f"参考字幕取前 {groups} 条 -> 切成 ASR 风格 {cue_count} 条")
    print(f"输入:{asr_srt}\n")

    from core import backend
    previous = backend.get_config("subtitle_resegment_mode", False)
    backend.set_config("subtitle_resegment_mode", True)
    try:
        translator = build_translator(model, api_key, src, dst, asr_srt)
        out, _missing = translator.process(asr_srt.stem, asr_srt.suffix)
    finally:
        backend.set_config("subtitle_resegment_mode", previous)

    from core.pipelines.subtitle_groups import seconds
    from core.pipelines.subtitle_translation_pipeline import parse_srt_cues
    result_cues = parse_srt_cues(str(out))
    print(f"断句后 {len(result_cues)} 条(输入 {cue_count} 条,参考 {groups} 条)")

    for cue in result_cues[:12]:
        print(f"  {cue['start_time']} --> {cue['end_time']}  "
              f"{cue['value'].replace('␊', ' ')[:58]!r}")
    if len(result_cues) > 12:
        print(f"  ...另有 {len(result_cues) - 12} 条")

    ordered = all(
        seconds(a["end_time"]) <= seconds(b["start_time"])
        for a, b in zip(result_cues, result_cues[1:])
    )
    print(f"\n时间轴严格有序不重叠:{'是' if ordered else '否'}")

    # Score against the answer key by BOUNDARY, not text: the output is the
    # translation, so its words differ by design. The reference's cue timings are
    # what a correct resegmentation should reproduce.
    def flat(text):
        return re.sub(r"\s+", " ", text.replace("␊", " ").replace("␍", " ")).strip()

    reference = parse_srt_cues(str(good))[:groups]
    produced = {(c["start_time"], c["end_time"]) for c in result_cues}
    matched = sum(1 for c in reference
                  if (c["start_time"], c["end_time"]) in produced)
    print(f"与参考答案边界吻合:{matched}/{len(reference)}")
    if matched < len(reference):
        for cue in reference:
            if (cue["start_time"], cue["end_time"]) not in produced:
                print(f"  边界不同: 参考 {cue['start_time']} --> {cue['end_time']}  "
                      f"{flat(cue['value'])[:44]!r}")
                break

    print(f"\n输出:{out}")
    return cue_count, len(result_cues), groups, ordered, matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["same-ok", "resegment", "all"])
    parser.add_argument("--doc", help="指定 fixtures 里的文档名(默认取最小的)")
    parser.add_argument("--keep", action="store_true",
                        help="保留上次的中间文件(默认每次清空)")
    args = parser.parse_args()

    model = os.environ.get("LINGUAHARU_MODEL", "qwen")
    config_path = REPO_ROOT / "config" / "api_config" / f"{model}.json"
    if not config_path.exists():
        available = sorted(p.stem for p in (REPO_ROOT / "config" / "api_config").glob("*.json"))
        raise SystemExit(
            f"找不到接口配置 {config_path}\n"
            f"用 LINGUAHARU_MODEL 指定,可选:{', '.join(available)}")
    api_key = load_api_key()
    print(f"接口:{model}  密钥:已从桌面版解密({len(api_key)} 字符)\n")

    if not args.keep:
        shutil.rmtree(WORK, ignore_errors=True)

    if args.what in ("same-ok", "all"):
        print("=" * 60, "\nsame_ok 免翻译标记\n", "=" * 60, sep="")
        check_same_ok(model, api_key, doc_name=args.doc)
        print()

    if args.what in ("resegment", "all"):
        print("=" * 60, "\n字幕 LLM 断句\n", "=" * 60, sep="")
        check_resegment(model, api_key)


if __name__ == "__main__":
    main()
