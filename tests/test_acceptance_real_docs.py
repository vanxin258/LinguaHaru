# 真实文档验收:提取 → 假翻译 → 写回,检查结构保真与写回完整性。
#
# 不调用 LLM:用 [T] 前缀冒充译文,只验证管线有没有丢结构、有没有把内容漏掉。
# 源文档放在 tests/acceptance/fixtures/(公司资料,已 gitignore)。
#
# 跑法:
#   python tests/test_acceptance_real_docs.py          # 打印报告
#   python -m pytest tests/test_acceptance_real_docs.py
import os
import re
import sys
import zipfile
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import json  # noqa: E402

from lxml import etree  # noqa: E402

from tests.corpus_common import CHECKS, check, fake_translate  # noqa: E402

FIXTURES = os.path.join(REPO_ROOT, "tests", "acceptance", "fixtures")
WORK = os.path.join(REPO_ROOT, "tests", "acceptance", "_work")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"

# 结构计数:这些在任何情况下都不该变
STRUCTURE_TAGS = ("tbl", "txbxContent", "drawing", "hyperlink", "instrText",
                  "footnoteReference", "endnoteReference", "bookmarkStart")


def parse(path):
    """每个 XML 部件解析成 lxml 树。"""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(".xml") and name.startswith("word/"):
                out[name] = etree.fromstring(z.read(name))
    return out


def count_all(trees, tag):
    return sum(len(t.findall(f".//{{{W}}}{tag}")) for t in trees.values())


def all_text(trees):
    """所有 word/*.xml 部件里的文字(去空白),正文+页眉页脚+脚注+文本框。"""
    chunks = []
    for tree in trees.values():
        for node in tree.iter():
            if node.tag == f"{{{W}}}t":
                chunks.append(node.text or "")
            elif node.tag == f"{{{W}}}tab":
                chunks.append(" ")
    return re.sub(r"[\s␊␍]+", "", "".join(chunks))


PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}|\[formula_\d+\]")


def strip_placeholders(text):
    """去掉占位符标记:写回后它们会还原成真实 XML,不会以字面形式出现。"""
    return PLACEHOLDER.sub("", text)


def field_codes(trees):
    return Counter(
        (n.text or "").strip()
        for t in trees.values()
        for n in t.iter(f"{{{W}}}instrText")
    )


def drawing_anchors(trees):
    return Counter(
        (n.get("name") or n.get("id") or "")
        for t in trees.values()
        for n in t.iter(f"{{{WP}}}docPr")
    )


def external_rels(path):
    out = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".rels"):
                continue
            for rel in etree.fromstring(z.read(name)):
                if rel.get("TargetMode") == "External":
                    out.append(rel.get("Target"))
    return sorted(out)


def char_format_profile(trees):
    """粗体/斜体各覆盖多少字符(参考指标,不作断言)。"""
    total = bold = italic = 0
    for t in trees.values():
        for r in t.iter(f"{{{W}}}r"):
            txt = "".join((n.text or "") for n in r.iter(f"{{{W}}}t"))
            if not txt:
                continue
            rpr = r.find(f"{{{W}}}rPr")
            total += len(txt)
            if rpr is not None and rpr.find(f"{{{W}}}b") is not None:
                bold += len(txt)
            if rpr is not None and rpr.find(f"{{{W}}}i") is not None:
                italic += len(txt)
    return total, bold, italic


def docx_fixtures():
    if not os.path.isdir(FIXTURES):
        return []
    return sorted(
        os.path.join(FIXTURES, n)
        for n in os.listdir(FIXTURES)
        if n.lower().endswith(".docx") and not n.startswith("~$")
    )


def run_one(src):
    from core.pipelines.word_translation_pipeline import (
        extract_word_content_to_json,
        write_translated_content_to_word,
    )
    name = os.path.basename(src)
    temp = os.path.join(WORK, name, "temp")
    result = os.path.join(WORK, name, "result")
    os.makedirs(temp, exist_ok=True)
    os.makedirs(result, exist_ok=True)

    src_json = extract_word_content_to_json(src, temp)
    dst_json = fake_translate(src_json)
    out = write_translated_content_to_word(
        src, src_json, dst_json, temp, result,
        bilingual_mode=False, src_lang="en", dst_lang="zh",
    )
    return out, src_json


def compare(src, out, src_json):
    a, b = parse(src), parse(out)

    # 1. 结构计数必须完全一致
    for tag in STRUCTURE_TAGS:
        na, nb = count_all(a, tag), count_all(b, tag)
        check(f"{tag} 数量一致 ({na})", na == nb, f"{na} -> {nb}")

    # 2. 域代码内容一致
    fa, fb = field_codes(a), field_codes(b)
    if fa or fb:
        lost = fa - fb
        check("域代码未丢失", not lost, f"丢失 {dict(lost)}" if lost else "")

    # 3. 图片/图形锚点一致
    da, db = drawing_anchors(a), drawing_anchors(b)
    if da or db:
        check("图片锚点一致", da == db, f"{da} -> {db}")

    # 4. 超链接目标一致
    ta, tb = external_rels(src), external_rels(out)
    if ta or tb:
        check("超链接目标一致", ta == tb, f"丢失 {sorted(set(ta) - set(tb))}")

    # 5. 每个提取项都写回了(核心:验证写回没漏内容)
    out_text = all_text(b)
    items = json.load(open(src_json, encoding="utf-8"))
    missing = []
    for item in items:
        value = strip_placeholders(str(item.get("value", "")))
        value = re.sub(r"[\s␊␍]+", "", value)
        if not value:
            continue
        if "[T]" + value not in out_text:
            missing.append(str(item.get("value", ""))[:60])
    check(f"提取项全部写回 ({len(items)} 条)", not missing,
          f"{len(missing)} 条未写回,例如 {missing[:3]}" if missing else "")

    # 6. 提取覆盖率(参考值,不作断言)
    src_text = all_text(a)
    covered = sum(
        len(re.sub(r"[\s␊␍]+", "", strip_placeholders(str(i.get("value", "")))))
        for i in items
    )
    print(f"    提取覆盖率约 {min(covered * 100 // max(len(src_text), 1), 999)}%"
          f" ({covered}/{len(src_text)} 字符)")

    t, bold, it = char_format_profile(b)
    print(f"    输出字符 {t},粗体 {bold},斜体 {it}(参考)")


def main():
    fixtures = docx_fixtures()
    if not fixtures:
        print(f"没有找到测试文档。请把 .docx 放进 {FIXTURES}")
        return 0

    print(f"验收文档 {len(fixtures)} 份\n")
    for src in fixtures:
        print(f"== {os.path.basename(src)}")
        try:
            out, src_json = run_one(src)
        except Exception as exc:  # noqa: BLE001
            check("写回完成", False, f"{type(exc).__name__}: {exc}")
            print()
            continue
        compare(src, out, src_json)
        print()

    failed = [n for n, ok in CHECKS if not ok]
    print(f"合计 {len(CHECKS)} 项检查,失败 {len(failed)} 项")
    for n in failed:
        print(f"  失败: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
