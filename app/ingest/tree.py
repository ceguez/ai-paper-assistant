"""Proxy-Pointer core: Markdown -> skeleton tree -> sized nodes -> breadcrumbed pointers.

Pure Python, no model calls. Each node is stored WHOLE; only its small pointer is embedded.
At query time a pointer hit is expanded back to the full node (see agent/tools.read_section).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
CAPTION = re.compile(r"^\s*(?:\*\*)?(Figure|Fig\.|Table)\s*(\d+)[.:]?(?:\*\*)?\s*(.*)", re.I)
NUMBERED = re.compile(r"^((?:\d+\.)*\d+|[IVXLC]+(?:\.[A-Z])?)\.?\s+")
NOISE_TITLES = {"acknowledgements", "acknowledgments", "funding", "author contributions",
                "conflict of interest", "competing interests", "references", "bibliography"}


def count_tokens(text: str) -> int:
    """Cheap estimate. In production pass the model's tokenizer (llama.cpp /tokenize)."""
    return max(1, int(len(text.split()) * 1.33))


PAGE = re.compile(r"^\s*<!--\s*page:(\d+)\s*-->\s*$")
# Parsers often emit section titles as bold lines instead of headings: "**3.2 Training**".
BOLD_SECTION = re.compile(r"^\s*\*\*((?:\d+\.)*\d+\.?\s+[A-Z][^*]{1,80})\*\*\s*$")


@dataclass
class Node:
    node_id: str
    paper_id: str
    parent_id: str | None
    kind: str            # "section" | "table" | "figure" | "root"
    level: int
    ord: int
    title: str
    breadcrumb: str
    text: str = ""
    embeddable: bool = True
    page_start: int | None = None
    page_end: int | None = None
    children: list["Node"] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)


def _nid(paper_id: str, *parts: object) -> str:
    return hashlib.sha1("|".join(map(str, (paper_id, *parts))).encode()).hexdigest()[:16]


def _heading(line: str) -> tuple[int, str] | None:
    if m := HEADING.match(line):
        return len(m.group(1)), m.group(2).replace("**", "").strip()
    if m := BOLD_SECTION.match(line):
        title = m.group(1).strip()
        number = title.split()[0].rstrip(".")
        return 1 + number.count(".") + 1, title  # "3" -> level 2, "3.2" -> level 3
    return None


def build_tree(markdown: str, paper_id: str, paper_title: str = "") -> Node:
    """Parse Markdown headings into a hierarchy; orphan text attaches to the nearest heading.

    `<!-- page:N -->` markers set page_start/page_end on each node. A document with no
    headings at all (some parsers, some PDFs) falls back to one section per page.
    """
    lines = markdown.splitlines()
    if not any(_heading(ln) for ln in lines):
        lines = [f"{ln}\n## Page {m.group(1)}" if (m := PAGE.match(ln)) else ln for ln in lines]
        lines = "\n".join(lines).splitlines()

    root = Node(_nid(paper_id, "root"), paper_id, None, "root", 0, 0, paper_title or "Paper", paper_title or "Paper")
    stack: list[Node] = [root]
    counter, page = 0, None
    buf: list[str] = []

    def flush():
        if buf:
            stack[-1].text = (stack[-1].text + "\n" + "\n".join(buf)).strip()
            buf.clear()

    for line in lines:
        if m := PAGE.match(line):
            page = int(m.group(1))
            continue
        h = _heading(line)
        if not h:
            buf.append(line)
            if line.strip() and page is not None:
                node = stack[-1]
                node.page_start = node.page_start or page
                node.page_end = page
            continue
        flush()
        level, title = h
        # The first H1 is usually the paper title: make it the root, not a section.
        if level == 1 and counter == 0 and (not paper_title or title.lower() == paper_title.lower()):
            root.title = root.breadcrumb = paper_title or title
            root.page_start = root.page_end = page
            continue
        while stack[-1].level >= level and stack[-1] is not root:
            stack.pop()
        parent = stack[-1]
        counter += 1
        node = Node(_nid(paper_id, counter, title), paper_id, parent.node_id, "section", level, counter,
                    title, f"{parent.breadcrumb} > {title}", page_start=page, page_end=page)
        node.embeddable = NUMBERED.sub("", title).lower() not in NOISE_TITLES
        parent.children.append(node)
        stack.append(node)
    flush()
    if count_tokens(root.text) > 12:  # abstract/author block before the first heading: keep it indexed
        front = Node(_nid(paper_id, "front"), paper_id, root.node_id, "section", 2, 0, "Front matter",
                     f"{root.breadcrumb} > Front matter", root.text, page_start=root.page_start or 1,
                     page_end=root.page_end or root.page_start or 1)
        root.children.insert(0, front)
        root.text = ""
    _split_elements(root)
    return root


def _split_elements(node: Node) -> None:
    """Pull tables and figure captions out into their own child nodes."""
    kept, extra, i = [], [], 0
    lines = node.text.splitlines()
    while i < len(lines):
        line = lines[i]
        cap = CAPTION.match(line)
        if line.lstrip().startswith("|"):            # Markdown table block
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i]); i += 1
            caption = kept.pop() if kept and CAPTION.match(kept[-1]) else ""
            extra.append(("table", caption.strip() or "Table", "\n".join(block)))
            continue
        if cap and cap.group(1).lower().startswith("fig"):
            extra.append(("figure", line.strip(), line.strip()))
        else:
            kept.append(line)
        i += 1
    node.text = "\n".join(kept).strip()
    for j, (kind, caption, body) in enumerate(extra):
        node.children.append(Node(_nid(node.paper_id, node.node_id, kind, j), node.paper_id, node.node_id, kind,
                                  node.level + 1, 10_000 + j, caption[:120], f"{node.breadcrumb} > {caption[:80]}",
                                  body if kind == "table" else caption,
                                  page_start=node.page_start, page_end=node.page_end))
    for child in node.children:
        if child.kind == "section":
            _split_elements(child)


def size_nodes(root: Node, max_tokens: int = 1500, min_tokens: int = 60) -> None:
    """Split oversized sections at paragraph boundaries; fold tiny leaf sections into their parent."""
    for child in list(root.children):
        size_nodes(child, max_tokens, min_tokens)
    if root.kind == "section" and root.token_count > max_tokens:
        paras, parts, cur = [p for p in root.text.split("\n\n") if p.strip()], [], ""
        for p in paras:
            if cur and count_tokens(cur + p) > max_tokens:
                parts.append(cur); cur = ""
            cur = f"{cur}\n\n{p}" if cur else p
        parts.append(cur)
        if len(parts) > 1:
            root.text = ""
            for k, part in enumerate(parts):
                root.children.insert(k, Node(_nid(root.paper_id, root.node_id, "part", k), root.paper_id,
                                             root.node_id, "section", root.level + 1, -1000 + k,
                                             f"{root.title} (part {k + 1}/{len(parts)})",
                                             f"{root.breadcrumb} (part {k + 1}/{len(parts)})", part,
                                             page_start=root.page_start, page_end=root.page_end))
    for child in list(root.children):
        if child.kind == "section" and not child.children and child.token_count < min_tokens and root.kind != "root":
            root.text = f"{root.text}\n\n### {child.title}\n{child.text}".strip()
            root.children.remove(child)


def pointer_text(node: Node, summary: str | None = None, max_words: int = 60) -> str:
    """The proxy: breadcrumb + lead sentences (+ optional summary). This is what gets embedded."""
    if node.kind in ("table", "figure"):
        return f"[{node.kind}] {node.breadcrumb}"
    lead = " ".join(node.text.split()[:max_words])
    return f"{node.breadcrumb}\n{summary + chr(10) if summary else ''}{lead}"


def walk(node: Node):
    yield node
    for c in node.children:
        yield from walk(c)
