"""PDF -> clean text -> structure-aware chunks.

What the data looks like (and why the chunker is built this way)
-----------------------------------------------------------------
Every FinBase PDF has the same skeleton:
  * Sections 1-3 (sometimes up to 6): the real policy text.
  * Sections 4-20: ~17 near-identical filler sections (only the numbers change).
  * Section 21/22: fee tables / exclusions.
  * Section 23: 100 FAQ entries that repeat ~10 unique questions, each followed
    by boilerplate metadata bullets.
Naive fixed-size chunking would index ~100 near-duplicate chunks per document,
which floods the top-k with copies of the same text and pushes real policy
clauses out. So we:
  1. fix the PDF's rupee glyph (it is extracted as a black square),
  2. drop the table of contents,
  3. chunk by section / sub-section with a context header,
  4. keep ONE copy of repeated filler sections (and record where it repeats),
  5. turn each FAQ into one Q+A chunk and de-duplicate repeated questions.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config

RUPEE_GLYPH = "\u25a0"  # the PDF font has no rupee glyph; text extraction yields "■"

HEADING_RE = re.compile(
    r"^Section\s+(\d+(?:\.\d+)?|[A-Z]+-EXT-\d+):\s*(.+?)\s*$", re.M
)
FAQ_SPLIT_RE = re.compile(r"(?m)^(Q\d{3}):\s*")
CASE_LABEL_RE = re.compile(r"\(([A-Za-z][A-Za-z\s]*?)\s*\d+\)\s*\n")
REF_RE = re.compile(r"\[Reference[^\]]*\]", re.S)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_code: str
    section: str
    section_title: str
    kind: str  # policy | table | faq | boilerplate
    text: str  # what the LLM / user sees
    embed_text: str  # what gets embedded + BM25-indexed (header + core content)
    meta: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        if self.kind == "faq":
            ref = self.meta.get("clause_ref", "")
            return f"{self.doc_title} — FAQ {self.meta.get('faq_ids', [''])[0]} {ref}".strip()
        return f"{self.doc_title} — Section {self.section}: {self.section_title}"


# ----------------------------------------------------------------------------
# Loading + cleaning
# ----------------------------------------------------------------------------
def load_pdf_text(path: Path) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def clean_text(raw: str) -> str:
    # pdfplumber exposes the bullet glyph as "(cid:127)" and the rupee sign as "■"
    text = raw.replace("(cid:127)", "•").replace(RUPEE_GLYPH, "₹").replace("\f", "\n")
    # In ZapfDingbats the rupee placeholder square is the letter "n": "n1,00,000" -> "₹1,00,000"
    text = re.sub(r"(?<![A-Za-z0-9])n(?=\d)", "₹", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # markdown bold leaked from source
    lines = []
    for line in text.splitlines():
        line = line.rstrip()
        if re.match(r"^\s*•\s*\[Section", line):  # table-of-contents entries
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def doc_metadata(text: str, path: Path) -> tuple[str, str]:
    title = next((l.strip() for l in text.splitlines() if l.strip()), path.stem)
    # titles can wrap onto the next line ("... Cardholder\nAgreement")
    m = re.search(r"Document Code:\s*(\S+)", text)
    code = m.group(1) if m else ""
    first_block = text.split("Document Code:")[0].strip().splitlines()
    title = " ".join(l.strip() for l in first_block if l.strip()) or title
    return title, code


# ----------------------------------------------------------------------------
# Section splitting
# ----------------------------------------------------------------------------
def split_sections(text: str) -> list[dict]:
    matches = list(HEADING_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(
            {"num": m.group(1), "title": m.group(2), "body": text[m.end():end].strip()}
        )
    return sections


def is_boilerplate(sec: dict) -> bool:
    """Filler sections: '... Standards 7' with number == section number, or EXT annexes."""
    if "-EXT-" in sec["num"]:
        return True
    m = re.search(r"\s(\d+)$", sec["title"])
    return bool(m and m.group(1) == sec["num"])


def normalise_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text)).strip().lower()


def _pack_lines(body: str, max_chars: int) -> list[str]:
    """Greedy line packing so tables / bullet lists are never cut mid-line."""
    if len(body) <= max_chars:
        return [body]
    parts, cur = [], ""
    for line in body.splitlines():
        if cur and len(cur) + len(line) + 1 > max_chars:
            parts.append(cur.strip())
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


# ----------------------------------------------------------------------------
# FAQ parsing
# ----------------------------------------------------------------------------
def parse_faq(body: str) -> list[dict]:
    parts = FAQ_SPLIT_RE.split(body)
    items = []
    for qid, rest in zip(parts[1::2], parts[2::2]):
        rest = rest.strip()
        m = CASE_LABEL_RE.search(rest)
        if not m:
            continue
        question = re.sub(r"\s+", " ", rest[: m.start()]).strip()
        after = rest[m.end():]
        ans_lines, meta_lines, in_meta = [], [], False
        for line in after.splitlines():
            if line.strip().startswith("•"):
                in_meta = True
            (meta_lines if in_meta else ans_lines).append(line)
        answer_raw = " ".join(l.strip() for l in ans_lines).strip()
        ref = REF_RE.search(answer_raw)
        clause_ref = ""
        if ref:
            clause_ref = re.sub(r"\s+", " ", ref.group(0))
            answer_raw = answer_raw.replace(ref.group(0), "").strip()
            answer_raw = re.sub(r"\s+\.$", ".", answer_raw)
        items.append(
            {
                "faq_id": qid,
                "question": question,
                "answer": re.sub(r"\s+", " ", answer_raw),
                "clause_ref": clause_ref,
                "extra": "\n".join(l.strip() for l in meta_lines if l.strip()),
            }
        )
    return items


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def build_chunks_for_doc(path: Path) -> list[Chunk]:
    text = clean_text(load_pdf_text(path))
    title, code = doc_metadata(text, path)
    doc_id = path.stem
    sections = split_sections(text)
    chunks: list[Chunk] = []
    seen_boiler: dict[str, Chunk] = {}
    seen_faq: dict[str, Chunk] = {}

    for sec in sections:
        num, stitle, body = sec["num"], sec["title"], sec["body"]
        header = f"{title} — Section {num}: {stitle}"

        if "FAQ" in stitle:
            for item in parse_faq(body):
                key = normalise_for_dedup(item["question"])
                if key in seen_faq:
                    seen_faq[key].meta["faq_ids"].append(item["faq_id"])
                    continue
                core = f"Q: {item['question']}\nA: {item['answer']}"
                ch = Chunk(
                    chunk_id=f"{doc_id}-faq-{item['faq_id']}",
                    doc_id=doc_id, doc_title=title, doc_code=code,
                    section=num, section_title=stitle, kind="faq",
                    text=core + (f"\nAdditional notes:\n{item['extra']}" if item["extra"] else ""),
                    embed_text=f"{title} FAQ\n{core}",
                    meta={"faq_ids": [item["faq_id"]], "clause_ref": item["clause_ref"]},
                )
                seen_faq[key] = ch
                chunks.append(ch)
            continue

        if not body:
            continue  # pure heading (e.g. "Section 6: Prepayment" followed by 6.1/6.2)

        if is_boilerplate(sec):
            key = normalise_for_dedup(body)
            if key in seen_boiler:
                seen_boiler[key].meta.setdefault("repeats_in", []).append(num)
                continue
            ch = Chunk(
                chunk_id=f"{doc_id}-s{num}", doc_id=doc_id, doc_title=title,
                doc_code=code, section=num, section_title=stitle, kind="boilerplate",
                text=body, embed_text=f"{header}\n{body}", meta={"repeats_in": []},
            )
            seen_boiler[key] = ch
            chunks.append(ch)
            continue

        kind = "table" if re.search(r"Schedule|Matrix", stitle) else "policy"
        limit = config.MAX_CHUNK_CHARS * (2 if kind == "table" else 1)
        for i, piece in enumerate(_pack_lines(body, limit)):
            suffix = f"-p{i + 1}" if i else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-s{num}{suffix}", doc_id=doc_id, doc_title=title,
                    doc_code=code, section=num, section_title=stitle, kind=kind,
                    text=piece, embed_text=f"{header}\n{piece}",
                )
            )
    return chunks


def build_all(raw_dir: Path = config.RAW_DIR, out: Path = config.CHUNKS_PATH) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        chunks.extend(build_chunks_for_doc(pdf))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1))
    return chunks


def load_chunks(path: Path = config.CHUNKS_PATH) -> list[Chunk]:
    if not path.exists():
        return build_all(out=path)
    return [Chunk(**d) for d in json.loads(path.read_text())]


if __name__ == "__main__":
    cs = build_all()
    from collections import Counter

    print(f"{len(cs)} chunks:", dict(Counter(c.kind for c in cs)))
