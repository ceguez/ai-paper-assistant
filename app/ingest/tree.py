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
    children: list["Node"] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)


def _nid(paper_id: str, *parts: object) -> str:
    return hashlib.sha1("|".join(map(str, (paper_id, *parts))).encode()).hexdigest()[:16]


def build_tree(markdown: str, paper_id: str, paper_title: str = "") -> Node:
    """Parse Markdown headings into a hierarchy; orphan text attaches to the nearest heading."""
    root = Node(_nid(paper_id, "root"), paper_id, None, "root", 0, 0, paper_title or "Paper", paper_title or "Paper")
    stack: list[Node] = [root]
    counter = 0
    buf: list[str] = []

    def flush():
        if buf:
            stack[-1].text = (stack[-1].text + "\n" + "\n".join(buf)).strip()
            buf.clear()

    for line in markdown.splitlines():
        m = HEADING.match(line)
        if not m:
            buf.append(line)
            continue
        flush()
        level, title = len(m.group(1)), m.group(2).strip()
        # The first H1 is usually the paper title: make it the root, not a section.
        if level == 1 and counter == 0 and (not paper_title or title.lower() == paper_title.lower()):
            root.title = root.breadcrumb = paper_title or title
            continue
        while stack[-1].level >= level and stack[-1] is not root:
            stack.pop()
        parent = stack[-1]
        counter += 1
        node = Node(_nid(paper_id, counter, title), paper_id, parent.node_id, "section", level, counter,
                    title, f"{parent.breadcrumb} > {title}")
        node.embeddable = NUMBERED.sub("", title).lower() not in NOISE_TITLES
        parent.children.append(node)
        stack.append(node)
    flush()
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
                                  body if kind == "table" else caption))
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
                                             root.node_id, "section", root.level + 1, k,
                                             f"{root.title} (part {k + 1}/{len(parts)})",
                                             f"{root.breadcrumb} (part {k + 1}/{len(parts)})", part))
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
