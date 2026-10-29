"""Split statute text into retrievable chunks.

Every chunk carries doc, section, part and status so retrieval can filter on
them. Provisos become their own chunks: the flagship case turns on the second
proviso to s.53(1), which a main-clause-only chunk would never surface.

Input is ``pdftotext -layout`` output. Layout mode keeps the section numbers at
line start but injects marginalia (running heads, gazette footers, page
numbers) into the middle of body text. Those lines are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contracts.analyst_result import Chunk, ProvisionStatus

# --- marginalia -----------------------------------------------------------
# Lines that are page furniture, not statute.
_MARGINALIA = [
    re.compile(r"^\s*\d{1,3}\s+THE GAZETTE OF INDIA"),   # page header
    re.compile(r"^\s*\[PART II"),                        # page header tail
    re.compile(r"^\s*\d{1,3}\s*$"),                      # bare page number
    re.compile(r"^\s*SEC\.\s*\d+\]"),                    # section running head
]

# --- structure ------------------------------------------------------------
# "53. (1) Gratuity shall be payable..." or "2. Definitions.—"
# Indent varies with layout, so it is bounded loosely. The body must start with
# a capital letter or a sub-section marker; that rejects footnote markers
# ("1. Ins. by Act 26 of 1984") and list items.
_SECTION_START = re.compile(
    r"^\s*(\d{1,3}[A-Z]?)\.\s+(?:\((\d+)\)\s+)?([A-Z(].*)$"
)

# Same, but the section number sits after a left-margin running head:
# "Continuous         54. For the purposes of this Chapter,—"
# The left fragment must be a short running head (no digits, <= 20 chars) so
# footnote columns like "45 of 1955." do not register as section starts.
_SECTION_START_MARGIN = re.compile(
    r"^(?P<head>[^\d]{1,20}?)\s{5,}(?P<num>\d{1,3}[A-Z]?)\.\s+(?:\((?P<sub>\d+)\)\s+)?(?P<body>[A-Z(].*)$"
)

# Footnote / amendment-history lines that look like numbered paragraphs.
_FOOTNOTE = re.compile(
    r"^\s*\d+\.\s+(Ins\.|Subs\.|Omitted|Added|Repealed|Rep\.|The words|For\s+section)"
)

# Any proviso opener, including "Provided also that" and wrapped continuations.
_PROVISO_START = re.compile(r"^\s{15,}Provided\b")

# "Explanation 1.—" / "Explanation.—"
_EXPLANATION_START = re.compile(r"^\s{0,20}Explanation\s*\d*\.\s*—")

# Sub-section start, e.g. "(2) For every completed year"
_SUBSECTION_START = re.compile(r"^\s*\((\d+)\)\s")


@dataclass
class RawSection:
    """A section's full text, before splitting into parts."""

    number: str
    heading: str
    lines: list[str]


def _is_marginalia(line: str) -> bool:
    return any(p.match(line) for p in _MARGINALIA)


# --- right-margin column ---------------------------------------------------
# ``pdftotext -layout`` puts section headings ("Payment of gratuity.") in a
# right-hand column. Those words land mid-sentence when lines are joined,
# e.g. "termination of his employment Payment of after he has rendered...".
# The gap before the margin note is 5+ spaces; ordinary word spacing is one.
_MARGIN_GAP = re.compile(r"(?<=\S)(\s{5,})(\S.*)$")


def _strip_margin(line: str) -> str:
    """Drop the right-hand heading column from one layout line."""
    if not line.strip():
        return line
    m = _MARGIN_GAP.search(line)
    if m and len(m.group(2)) <= 40:
        # A section number after a left running head is statute, not marginalia:
        # "Continuous         54. For the purposes of this Chapter,—"
        if _SECTION_START.match(m.group(2)):
            return line
        return line[: m.start(1)].rstrip()
    return line


def _clean(text: str) -> str:
    """Collapse whitespace so embedded text reads as one passage."""
    return re.sub(r"\s+", " ", text).strip()


def extract_sections(text: str) -> list[RawSection]:
    """Cut the document at section-number boundaries."""
    sections: list[RawSection] = []
    current: RawSection | None = None

    for line in text.splitlines():
        if _is_marginalia(line):
            continue
        line = _strip_margin(line)
        if not line.strip():
            continue
        m = _SECTION_START.match(line)
        number: str | None = None
        body: str | None = None
        if m:
            number, body = m.group(1), m.group(3)
        else:
            mm = _SECTION_START_MARGIN.match(line)
            # Only trust a margin heading whose number is plausible as the
            # next section (monotonically greater than what we have seen).
            if mm:
                cand = mm.group("num")
                prev = int(re.sub(r"[A-Z]", "", sections[-1].number)) if sections else 0
                if int(re.sub(r"[A-Z]", "", cand)) > prev:
                    number, body = cand, mm.group("body")
                    line = f"{number}. {body}"
                    m = mm
        # A section start is a numbered line whose number is plausible and
        # followed by prose. Guards against list items like "(a)" and dates.
        if m and number is not None and not _PROVISO_START.match(line):
            heading = body.split(".—")[0].split(".—")[0][:80]
            current = RawSection(number=number, heading=_clean(heading), lines=[line])
            sections.append(current)
        elif current is not None:
            current.lines.append(line)

    return sections


def _split_parts(lines: list[str]) -> list[tuple[str, str]]:
    """Split one section into (part_name, text) pairs.

    ``main`` holds everything before the first proviso or explanation.
    Provisos are numbered in order of appearance, 1-based. Explanations are
    separate parts because they define terms used by the main clause.
    """
    parts: list[tuple[str, list[str]]] = []
    current_name = "main"
    current_lines: list[str] = []
    proviso_n = 0
    explanation_n = 0

    for line in lines:
        if _PROVISO_START.match(line):
            if current_lines:
                parts.append((current_name, current_lines))
            proviso_n += 1
            current_name = f"proviso_{proviso_n}"
            current_lines = [line]
        elif _EXPLANATION_START.match(line):
            if current_lines:
                parts.append((current_name, current_lines))
            explanation_n += 1
            current_name = f"explanation_{explanation_n}" if explanation_n > 1 else "explanation"
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        parts.append((current_name, current_lines))

    return [(name, _clean(" ".join(ls))) for name, ls in parts if _clean(" ".join(ls))]


def chunk_document(
    text: str,
    doc: str,
    status: ProvisionStatus,
) -> list[Chunk]:
    """Turn one document's text into ``Chunk`` objects.

    Section numbers repeat in schedule rows and footnote lists, so the first
    occurrence keeps ``s<n>`` and later ones get ``s<n>_r<k>``. The section
    field itself stays the bare number: it is what retrieval filters on.
    """
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for section in extract_sections(text):
        seen[section.number] = seen.get(section.number, 0) + 1
        repeat = seen[section.number]
        suffix = "" if repeat == 1 else f"_r{repeat}"
        for part, body in _split_parts(section.lines):
            chunk_id = f"{doc}-s{section.number}{suffix}-{part}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc=doc,
                    section=section.number,
                    part=part,
                    status=status,
                    text=body,
                )
            )
    return chunks