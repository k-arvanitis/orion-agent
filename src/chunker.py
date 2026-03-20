import json
import re
from pathlib import Path


def chunk_document(text: str, doc_title: str, source_file: str) -> list[dict]:
    chunks = []
    current_h2 = ""

    parts = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^## ", part):
            current_h2 = part.splitlines()[0].lstrip("# ").strip()
        elif re.match(r"^### ", part):
            heading = part.splitlines()[0].lstrip("# ").strip()
            content = f"[{doc_title} > {current_h2}]\n\n{part}"
            chunks.append(
                {
                    "id": f"{source_file}::{heading}",
                    "source": source_file,
                    "doc_title": doc_title,
                    "section": current_h2,
                    "heading": heading,
                    "content": content,
                }
            )

    return chunks


def extract_doc_title(text: str) -> str:
    match = re.search(r"^# (.+)", text, re.MULTILINE)
    return match.group(1).strip() if match else "Unknown"


def chunk_all(data_dir: str = "data/policies", output_file: str = "chunks.json") -> list[dict]:
    all_chunks = []

    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = extract_doc_title(text)
        chunks = chunk_document(text, title, path.name)
        all_chunks.extend(chunks)
        print(f"{path.name}: {len(chunks)} chunks")

    Path(output_file).write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal: {len(all_chunks)} chunks → {output_file}")
    return all_chunks


if __name__ == "__main__":
    chunk_all()
