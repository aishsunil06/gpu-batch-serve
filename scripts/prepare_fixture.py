#!/usr/bin/env python
"""Tokenize the workload once, offline, into a fixture every arm shares.

Tokenization is removed from the request path on purpose. It is CPU work
that happens before any batching decision, identical in every arm, and
leaving it in would add a constant per-request cost that shrinks the
measured difference between arms for a reason that has nothing to do with
serving.

Pre-tokenizing offline also lets a Go client participate without needing a
tokenizer implementation, and guarantees every arm and every language
sends byte-identical input.

Usage:
    python scripts/prepare_fixture.py
    python scripts/prepare_fixture.py --size 1024 --sequence-length 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpu_batch_serve.bench.workload import build_workload  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "bench" / "fixtures" / "workload_tokens.json"
    )
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    texts = build_workload(args.size, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Padded to a fixed length, not to the longest in a dynamic batch. A
    # fixed shape is what lets Triton's dynamic batcher coalesce arbitrary
    # requests without ragged batching, and it puts every arm on identical
    # footing. The cost is real compute spent on padding for short queries,
    # which is noted in the findings rather than hidden.
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.sequence_length,
        return_tensors="np",
    )

    records = []
    for index in range(len(texts)):
        records.append(
            {
                "input_ids": encoded["input_ids"][index].astype(int).tolist(),
                "attention_mask": encoded["attention_mask"][index].astype(int).tolist(),
                "token_type_ids": encoded["token_type_ids"][index].astype(int).tolist()
                if "token_type_ids" in encoded
                else [0] * args.sequence_length,
            }
        )

    real_tokens = [int(sum(r["attention_mask"])) for r in records]
    payload = {
        "meta": {
            "model_id": args.model_id,
            "size": len(records),
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "real_tokens_min": min(real_tokens),
            "real_tokens_max": max(real_tokens),
            "real_tokens_mean": round(sum(real_tokens) / len(real_tokens), 2),
            "padding_fraction": round(
                1 - (sum(real_tokens) / (len(records) * args.sequence_length)), 4
            ),
        },
        "requests": records,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.1f} MB)")
    print(
        f"  {len(records)} requests, sequence length {args.sequence_length}, "
        f"real tokens {payload['meta']['real_tokens_min']}-"
        f"{payload['meta']['real_tokens_max']} "
        f"(mean {payload['meta']['real_tokens_mean']})"
    )
    print(
        f"  {payload['meta']['padding_fraction']:.1%} of every forward pass is padding"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
