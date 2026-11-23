#!/usr/bin/env python
"""Export the encoder, with pooling fused into the graph, to ONNX.

Pooling belongs inside the graph for a reason specific to this comparison.
If the exported model stopped at ``last_hidden_state``, every arm would
return a 64x384 tensor, roughly 98 KB serialised, against the 1.5 KB of a
pooled vector. The comparison would then be dominated by response encoding
rather than by serving, and Triton would be penalised for nothing it did.

Fusing pooling also means all three arms execute byte-identical graphs, so
any difference between them is attributable to batching policy, transport
and runtime, which is the whole question.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --sequence-length 128 --opset 17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# torch.onnx prints progress with emoji. On a Windows console defaulting to
# cp1252 that raises UnicodeEncodeError *after* a successful export, which
# looks exactly like an export failure. Force UTF-8 before importing torch.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class PooledEncoder(nn.Module):
    """Encoder plus mean pooling plus L2 normalisation, as one graph.

    The pooling masks padding tokens, so an embedding never depends on the
    length of whatever else shared its batch. Without that masking every
    measurement in this repository would be batch-size dependent and the
    experiment would be measuring its own artefact.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        self.encoder.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return nn.functional.normalize(pooled, p=2, dim=1)


def export(
    model_id: str,
    out_path: Path,
    sequence_length: int,
    opset: int,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # A previous export may have written weights to a sidecar file. This
    # export is self-contained, and a stale sidecar is both dead weight on
    # disk and a file Triton will try to interpret if it is left behind.
    sidecar = out_path.with_name(out_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    model = PooledEncoder(model_id)
    model.eval()

    # A batch of 2 in the sample input, not 1: exporting from a batch of 1
    # lets the tracer bake a size-1 dimension into shapes that should stay
    # dynamic, which then fails the moment Triton forms a real batch.
    sample = {
        "input_ids": torch.ones(2, sequence_length, dtype=torch.int64),
        "attention_mask": torch.ones(2, sequence_length, dtype=torch.int64),
        "token_type_ids": torch.zeros(2, sequence_length, dtype=torch.int64),
    }

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
            str(out_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["embedding"],
            # Only the batch axis is dynamic. Sequence length is fixed on
            # purpose: a fixed shape is what lets Triton's dynamic batcher
            # coalesce arbitrary requests without ragged-batching support,
            # and it keeps all three arms on the same footing.
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "token_type_ids": {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            # The dynamo exporter in torch 2.13 emits a graph with a
            # duplicated output name for this model, which onnxruntime
            # rejects at load. The legacy tracer handles dynamic_axes and
            # output_names correctly here, so pin it explicitly rather than
            # depend on whichever path the installed torch defaults to.
            dynamo=False,
        )

    return out_path


def verify(model_id: str, onnx_path: Path, sequence_length: int) -> None:
    """Check the exported graph against Torch on real text.

    An export that silently loses the pooling, or bakes in a fixed batch
    size, still produces a loadable file. The only way to know the graph is
    the model is to compare outputs, and the only way to know the batch
    axis is dynamic is to run a batch size the export never saw.
    """
    import numpy as np
    import onnxruntime as ort

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    texts = [
        "how do i serve a model on a gpu",
        "what is the best way to reduce tail latency",
        "why does batching help throughput",
    ]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )

    torch_model = PooledEncoder(model_id)
    torch_model.eval()
    with torch.inference_mode():
        expected = torch_model(
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded["token_type_ids"],
        ).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["embedding"],
        {
            "input_ids": encoded["input_ids"].numpy(),
            "attention_mask": encoded["attention_mask"].numpy(),
            "token_type_ids": encoded["token_type_ids"].numpy(),
        },
    )[0]

    max_diff = float(np.abs(expected - actual).max())
    norms = np.linalg.norm(actual, axis=1)

    print(f"  batch of {len(texts)}: max abs diff vs torch = {max_diff:.2e}")
    print(f"  output L2 norms = {np.round(norms, 5).tolist()}")

    if max_diff > 1e-4:
        raise SystemExit(f"export mismatch: max abs diff {max_diff:.2e} exceeds 1e-4")
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise SystemExit("export does not produce normalised vectors")

    # A batch size the export never saw, proving the batch axis is dynamic.
    unseen = session.run(
        ["embedding"],
        {
            "input_ids": np.ones((7, sequence_length), dtype=np.int64),
            "attention_mask": np.ones((7, sequence_length), dtype=np.int64),
            "token_type_ids": np.zeros((7, sequence_length), dtype=np.int64),
        },
    )[0]
    if unseen.shape != (7, actual.shape[1]):
        raise SystemExit(f"batch axis is not dynamic: got {unseen.shape}")
    print(f"  batch of 7 (never exported): shape {unseen.shape}, batch axis is dynamic")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "serving"
        / "triton"
        / "model_repository"
        / "minilm_pooled"
        / "1"
        / "model.onnx",
    )
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args(argv)

    print(f"exporting {args.model_id} at sequence length {args.sequence_length}")
    path = export(args.model_id, args.out, args.sequence_length, args.opset)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.1f} MB)")

    if not args.skip_verify:
        print("verifying against torch:")
        verify(args.model_id, path, args.sequence_length)
        print("export verified")

    return 0


if __name__ == "__main__":
    sys.exit(main())
