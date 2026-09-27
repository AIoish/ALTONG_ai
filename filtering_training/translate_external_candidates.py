"""Batch-translate selected public synthetic notifications for Korean review.

The translated file is a Git-ignored intermediate, never a labeled training set.
Source priority is deliberately excluded. Source text stays for comparison.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from filtering_training.select_external_candidates import OUTPUT_PATH as SOURCE_PATH

OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "external" / "notifai" / "translated_candidates.jsonl"
MODEL_NAME = "facebook/m2m100_418M"


def translate(source: Path, output: Path, limit: int, batch_size: int, model_name: str) -> dict:
    if limit < 1 or batch_size < 1:
        raise ValueError("limit and batch_size must be positive")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()][:limit]
    if not rows:
        raise ValueError("source contains no records")
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "en"
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    started = time.perf_counter()
    result = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        # Translate titles and bodies separately to preserve the JSON fields.
        texts = [value for row in batch for value in (row["title"], row["body"])]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=160).to(device)
        with torch.inference_mode():
            tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("ko"), max_length=96, num_beams=1)
        translated = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        for index, row in enumerate(batch):
            result.append({
                "source_id": row["source_id"], "app": row["app"],
                "source_title": row["title"], "source_body": row["body"],
                "title": translated[2 * index].strip(), "body": translated[2 * index + 1].strip(),
                "source_folder": row["source_folder"],
            })
        if len(result) == len(rows) or (offset // batch_size) % 20 == 0:
            print(json.dumps({"translated": len(result), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in result:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": model_name, "translated_count": len(result),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "status": "machine-translated draft; no ALTONG context or labels",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    print(json.dumps(translate(args.source, args.output, args.limit, args.batch_size, args.model), ensure_ascii=False))


if __name__ == "__main__":
    main()