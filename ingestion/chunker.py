"""
Markdown policy document chunker.

Splits structured Markdown files into semantically coherent chunks at the H3
subsection level, preserving H1/H2 breadcrumb context in each chunk's content.
An H3 section longer than MAX_CHUNK_CHARS is further split on paragraph
boundaries with overlap, so a long section in a real (not hand-authored)
document doesn't become one oversized chunk. Outputs a single JSON file
suitable for downstream embedding and retrieval.

Usage:
    uv run chunker data/policies
    uv run chunker data/policies --output data/output
    uv run chunker data/policies --output data/output --output-file my-chunks.json
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Chunks stay under this size before overlap is added back; sized so a chunk
# plus a couple of neighbours still fits comfortably in a retrieval prompt.
MAX_CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    source: str
    doc_title: str
    section: str
    heading: str
    content: str


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def extract_doc_title(text: str) -> str:
    """Return the H1 title of a Markdown document, or 'Unknown' if absent."""
    match = re.search(r"^# (.+)", text, re.MULTILINE)
    return match.group(1).strip() if match else "Unknown"


def _split_long_content(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs into pieces <= max_chars, repeating the tail of each
    piece at the start of the next so nearby facts aren't split across a
    retrieval boundary with no shared context."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            pieces.append(current)
            current = f"{current[-overlap:]}\n\n{para}"
    if current:
        pieces.append(current)
    return pieces


def chunk_document(text: str, doc_title: str, source_file: str) -> list[Chunk]:
    """
    Split a single Markdown document into H3-level chunks.

    Each chunk's content is prefixed with a breadcrumb in the form:
        [<doc_title> > <H2 section>]
    so that retrieval context is self-contained without needing metadata lookups.
    """
    chunks: list[Chunk] = []
    current_h2 = ""

    # Split on any H1/H2/H3 boundary while keeping the heading with its body
    parts = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(r"^## ", part):
            current_h2 = part.splitlines()[0].lstrip("#").strip()

        elif re.match(r"^### ", part):
            heading = part.splitlines()[0].lstrip("#").strip()
            content = f"[{doc_title} > {current_h2}]\n\n{part}"
            pieces = _split_long_content(content, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
            for i, piece in enumerate(pieces):
                chunk_id = (
                    f"{source_file}::{heading}"
                    if len(pieces) == 1
                    else (f"{source_file}::{heading}::{i}")
                )
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        source=source_file,
                        doc_title=doc_title,
                        section=current_h2,
                        heading=heading,
                        content=piece,
                    )
                )

    return chunks


def chunk_directory(input_dir: Path) -> list[Chunk]:
    """Chunk all Markdown files found in *input_dir*."""
    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No Markdown files found in '{input_dir}'.")

    all_chunks: list[Chunk] = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        title = extract_doc_title(text)
        doc_chunks = chunk_document(text, title, path.name)
        all_chunks.extend(doc_chunks)
        print(f"  {path.name:<40} {len(doc_chunks):>3} chunks")

    return all_chunks


def write_output(chunks: list[Chunk], output_path: Path) -> None:
    """Serialise chunks to JSON and write to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(c) for c in chunks]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chunker",
        description="Chunk Markdown policy documents into H3-level JSON fragments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        metavar="INPUT_DIR",
        type=Path,
        help="Directory containing Markdown (.md) files to chunk.",
    )
    parser.add_argument(
        "--output",
        metavar="OUTPUT_DIR",
        type=Path,
        default=Path("data/output"),
        help="Directory to write the output file (default: data/output).",
    )
    parser.add_argument(
        "--output-file",
        metavar="FILENAME",
        default="document-chunks.json",
        help="Output filename (default: document-chunks.json).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir: Path = args.input
    output_path: Path = args.output / args.output_file

    if not input_dir.is_dir():
        parser.error(f"INPUT_DIR '{input_dir}' does not exist or is not a directory.")

    print(f"Chunking documents in '{input_dir}' ...\n")
    try:
        chunks = chunk_directory(input_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    write_output(chunks, output_path)

    print(f"\n{len(chunks)} total chunks written to '{output_path}'")


# ---------------------------------------------------------------------------
# Self-check (ponytail: run directly with `python -m ingestion.chunker --selftest`)
# ---------------------------------------------------------------------------


def _selftest() -> None:
    short_doc = "# Title\n\n## Section\n\n### Heading\n\nShort body."
    chunks = chunk_document(short_doc, "Title", "short.md")
    assert len(chunks) == 1
    assert chunks[0].id == "short.md::Heading"
    assert "Short body." in chunks[0].content

    long_body = "\n\n".join(
        f"Paragraph {i} with some real sentence content." for i in range(80)
    )
    long_doc = f"# Title\n\n## Section\n\n### Heading\n\n{long_body}"
    long_chunks = chunk_document(long_doc, "Title", "long.md")
    assert len(long_chunks) > 1, "expected the long section to be split"
    assert all(
        len(c.content) <= MAX_CHUNK_CHARS + CHUNK_OVERLAP_CHARS for c in long_chunks
    )
    assert all(c.heading == "Heading" for c in long_chunks)
    ids = [c.id for c in long_chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    for a, b in zip(long_chunks, long_chunks[1:]):
        assert a.content[-50:-25] in b.content, (
            "overlap should carry into the next piece"
        )

    print(f"chunker self-test OK ({len(long_chunks)} pieces from one long section)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
