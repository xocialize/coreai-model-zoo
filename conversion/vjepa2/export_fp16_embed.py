# Community port — export V-JEPA 2 (SSv2) to Core AI, fp16, TWO outputs: logits + pooled embedding.
"""Variant of `export_fp16.py` for ForgeOptimizerKit's §6.3 planner-hint seam.

WHY A VARIANT EXISTS. The shipped bundle returns `logits [1,174]` only. Forge does not use those
labels — `mlxengine-forge/Docs/VJEPA2-HINT-ADAPTER.md` is explicit ("labels are useless, the
embedding is the signal"): the SSv2 checkpoint ships no label names, its vocabulary is hand-object
actions, and it is semantically wrong for signage. The planner's actual signal is the pooled
**[1,1024]** router embedding that `ForgeCore.GraphicProbe` is fitted over. So this export exposes
both outputs; everything else — the RoPE overlay, fp16, the externalize-drop — is unchanged.

THE GATE IS DUAL, and the second half is the point:
  1. `logits` vs the existing oracle  → proves nothing regressed relative to the verified bundle.
  2. `embedding` vs a torch reference → proves the NEW output is the vector the probe consumes.
Parity on logits does not by itself certify the embedding. logits = classifier(pooled) through a
single Linear, so a high logit cosine is strong evidence — but `GraphicProbe.evaluate` takes a
1024-dot-product near a decision boundary, and "strong evidence" is not a measurement. Gate #2 is
held tighter than #1 for exactly that reason.

Writes to `artifacts/vjepa2_ssv2_fp16_embed/` — a SEPARATE directory, so a failed run here can
never clobber the published, verified single-output bundle.

Run:  python3 export_fp16_embed.py
Prereq: `reference_run.py` once, for oracle_input.npy / oracle_logits.npy.
"""
import os, shutil, asyncio, inspect, numpy as np, torch
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import coreai_models.export.macos as _macos
from coreai_models.export.macos import export_to_coreai
_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]

from transformers import VJEPA2ForVideoClassification
import transformers.models.vjepa2.modeling_vjepa2 as _vj

REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
OUT_DIR = Path(HERE) / "artifacts" / "vjepa2_ssv2_fp16_embed"
AIM_NAME = "vjepa2_ssv2_embed_fp16.aimodel"

# Gate bars. Logits mirror the original export's bar; the embedding is held tighter because it is
# the input to a linear probe whose class flips at the boundary.
LOGIT_COS_BAR = 0.99
EMBED_COS_BAR = 0.999


def _rotate_queries_or_keys(x, pos):
    # Reference copy MINUS the no-op `squeeze(-1)` on a size-D/2 dim (torch ignores it; the Core AI
    # converter maps it to ShrinkDims which requires size 1 and fails). Math unchanged.
    # Identical to export_fp16.py — the overlay is mandatory for ANY V-JEPA2 export, not incidental
    # to the single-output one.
    B, num_heads, N, D = x.size()
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega
    emb_sin = freq.sin().repeat(1, 1, 1, 2)
    emb_cos = freq.cos().repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


_vj.rotate_queries_or_keys = _rotate_queries_or_keys


# ---------------------------------------------------------------------------------------------
# Submodule resolution — discovered and VERIFIED, never assumed.
#
# The single-output export could wrap the whole model and take `.logits`, so it never had to know
# the internal layout. Reaching `pooled` does. Rather than hardcode attribute names against one
# transformers version, resolve by candidate list and then PROVE the recomposition reproduces the
# model's own logits. If the names drift, this fails loudly at author time with the real structure
# printed — instead of shipping a bundle whose "embedding" is some other tensor.
# ---------------------------------------------------------------------------------------------
BACKBONE_NAMES = ("vjepa2", "model", "vjepa2_model", "backbone", "encoder")
POOLER_NAMES = ("pooler", "attentive_pooler", "classifier_pooler")
CLASSIFIER_NAMES = ("classifier", "head", "class_head", "score")


def _first_attr(obj, names, what):
    for n in names:
        mod = getattr(obj, n, None)
        if mod is not None and isinstance(mod, torch.nn.Module):
            print(f"[resolve] {what} -> .{n} ({mod.__class__.__name__})", flush=True)
            return mod
    children = [f"{n} ({m.__class__.__name__})" for n, m in obj.named_children()]
    raise RuntimeError(
        f"could not resolve the {what} among {names}.\n"
        f"  actual children: {children}\n"
        f"  Fix: add the real attribute name to the *_NAMES list above and re-run — the "
        f"verification step below will confirm the choice."
    )


def _hidden(out):
    """Backbone output -> last hidden state. Branches on Python type, so it resolves at trace
    time; it is not a data-dependent condition and does not enter the exported graph."""
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class VJEPA2ExportWithEmbedding(torch.nn.Module):
    """`pixel_values_videos [1,16,3,256,256]` -> (`logits [1,174]`, `embedding [1,1024]`).

    Output ORDER is load-bearing: it pairs with `output_names=("logits", "embedding")` below, and
    the harness reads results by name — but a swap here would silently rename both tensors.
    The gate catches it (an embedding scored against oracle logits cannot reach 0.999).

    🔑 `skip_predictor=True` IS NOT OPTIONAL, and it is invisible to the gate. The reference
    `VJEPA2ForVideoClassification.forward` passes it (modeling_vjepa2.py:1197); `VJEPA2Model`
    defaults it to **False**, in which case it runs the entire JEPA *predictor* — a second
    transformer stack — and files the result under a SEPARATE output field. `last_hidden_state`
    is `sequence_output` either way (modeling_vjepa2.py:1091/1118), so omitting the flag changes
    no number: the recomposition check below still passes, the logits still match the oracle, and
    the bundle is simply much larger and slower for a stack nothing reads. A numeric gate cannot
    see this class of defect — only reading the reference can.
    """

    def __init__(self, backbone, pooler, classifier, skip_predictor: bool):
        super().__init__()
        self.backbone = backbone
        self.pooler = pooler
        self.classifier = classifier
        # Plain bool read at trace time — a Python branch, not a graph condition.
        self.skip_predictor = skip_predictor

    def forward(self, pixel_values_videos):
        if self.skip_predictor:
            out = self.backbone(pixel_values_videos=pixel_values_videos, skip_predictor=True)
        else:
            out = self.backbone(pixel_values_videos=pixel_values_videos)
        h = _hidden(out)
        pooled = self.pooler(h)
        # The attentive pooler may emit [B, 1, D] (one learned query) or [B, D]. Flatten to
        # [B, D] so the published contract is one fixed rank regardless of the version's shape.
        pooled = pooled.reshape(pooled.shape[0], -1)
        logits = self.classifier(pooled)
        return logits, pooled


def build():
    model = VJEPA2ForVideoClassification.from_pretrained(REPO, dtype=torch.float32).eval()

    px = torch.from_numpy(np.load(os.path.join(HERE, "oracle_input.npy")))           # [1,16,3,256,256]
    ref_logits = torch.from_numpy(np.load(os.path.join(HERE, "oracle_logits.npy")))  # [1,174]

    backbone = _first_attr(model, BACKBONE_NAMES, "backbone")
    pooler = _first_attr(model, POOLER_NAMES, "attentive pooler")
    classifier = _first_attr(model, CLASSIFIER_NAMES, "classifier")

    # Detected rather than assumed, same discipline as the names above: on a transformers that
    # renamed or dropped the flag we fall back to the default path (correct numbers, fat graph)
    # and SAY SO, instead of dying on an unexpected kwarg.
    skip = "skip_predictor" in inspect.signature(backbone.forward).parameters
    print(f"[resolve] backbone skip_predictor supported={skip}"
          + ("" if skip else "  ⚠️ predictor stack will be traced into the graph — see class docstring"),
          flush=True)

    w = VJEPA2ExportWithEmbedding(backbone, pooler, classifier, skip_predictor=skip).eval()

    # --- Verification: does the recomposition reproduce the model's own logits? -----------------
    # This is what makes the resolution above trustworthy. If `pooled` were the wrong tensor, the
    # classifier applied to it could not reproduce `.logits`.
    with torch.inference_mode():
        ref_out = model(pixel_values_videos=px).logits
        logits, pooled = w(px)

    cos_recompose = torch.nn.functional.cosine_similarity(
        logits.reshape(-1), ref_out.reshape(-1), dim=0).item()
    max_abs = (logits - ref_out).abs().max().item()
    print(f"[verify] recomposed logits vs model.logits  cos={cos_recompose:.8f}  max|Δ|={max_abs:.3e}",
          flush=True)
    if cos_recompose < 0.99999:
        raise RuntimeError(
            "recomposition does not reproduce the model's logits — the resolved pooler/classifier "
            "are not the real ones. Do NOT export; fix the *_NAMES resolution first.")

    if pooled.shape[0] != 1 or pooled.dim() != 2:
        raise RuntimeError(f"expected pooled [1, D], got {tuple(pooled.shape)}")
    print(f"[verify] embedding shape {tuple(pooled.shape)}  "
          f"(GraphicProbe.weights.count must equal {pooled.shape[1]})", flush=True)

    cos_oracle = torch.nn.functional.cosine_similarity(
        logits.reshape(-1), ref_logits.reshape(-1), dim=0).item()
    print(f"[gate] torch wrapper vs oracle logits cos={cos_oracle:.6f}", flush=True)

    # Persist the fp32 torch embedding as the reference for gate #2 — and for any FUTURE re-gate,
    # which then needs neither torch nor a network.
    ref_embedding = pooled.float().cpu().numpy()
    np.save(os.path.join(HERE, "oracle_embedding.npy"), ref_embedding)
    print(f"[gate] saved oracle_embedding.npy {ref_embedding.shape}", flush=True)

    # --- fp16 export (precision follows traced dtype) -------------------------------------------
    w = w.half()
    ref_inputs = {"pixel_values_videos": px.half()}
    print("[export] export_to_coreai (fp16, 2 outputs)…", flush=True)
    prog = export_to_coreai(w, ref_inputs, dynamic_shapes=None,
                            input_names=("pixel_values_videos",),
                            output_names=("logits", "embedding"),
                            state_names=None)
    print("[export] EXPORT OK ✅", flush=True)
    prog.optimize()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    aim = OUT_DIR / AIM_NAME

    import coreai.runtime as rt
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    print("[export] saved", aim, flush=True)
    return aim, px, ref_logits.numpy(), ref_embedding


async def gate(aim, px, ref_logits, ref_embedding):
    import coreai.runtime as rt

    def cos(a, b):
        a, b = a.ravel(), b.ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    inp = rt.NDArray(np.ascontiguousarray(px.numpy().astype(np.float16)))
    out = await fn(inputs={"pixel_values_videos": inp})

    missing = [k for k in ("logits", "embedding") if k not in out]
    if missing:
        raise RuntimeError(f"bundle is missing output(s) {missing}; got {list(out.keys())}")

    lg = out["logits"].numpy().astype(np.float32)
    emb = out["embedding"].numpy().astype(np.float32)

    c_log = cos(lg, ref_logits)
    t_ref = np.argsort(-ref_logits.ravel())[:5]
    t_eng = np.argsort(-lg.ravel())[:5]
    print(f"[gate] LOGITS    cos={c_log:.6f}  top5 ref={t_ref.tolist()} eng={t_eng.tolist()}", flush=True)

    c_emb = cos(emb, ref_embedding)
    rel_l2 = float(np.linalg.norm(emb.ravel() - ref_embedding.ravel()) /
                   (np.linalg.norm(ref_embedding.ravel()) + 1e-9))
    print(f"[gate] EMBEDDING cos={c_emb:.6f}  relL2={rel_l2:.3e}  shape={emb.shape}", flush=True)

    ok_log = c_log > LOGIT_COS_BAR and t_ref[0] == t_eng[0]
    ok_emb = c_emb > EMBED_COS_BAR and emb.shape[-1] == ref_embedding.shape[-1]
    print(f"[gate] logits {'PASS ✅' if ok_log else 'FAIL ❌'} (bar {LOGIT_COS_BAR}, top1 match)", flush=True)
    print(f"[gate] embed  {'PASS ✅' if ok_emb else 'FAIL ❌'} (bar {EMBED_COS_BAR}, dim match)", flush=True)
    print("[gate] OVERALL PASS ✅" if (ok_log and ok_emb) else "[gate] OVERALL FAIL ❌", flush=True)

    if ok_log and ok_emb:
        print(f"\nNext: point the calibration harness at this bundle —\n"
              f"  python3 /Volumes/Satechi/Development/mlxengine-forge/Tools/hintcal/collect_coreai.py \\\n"
              f"      --model {aim}", flush=True)


if __name__ == "__main__":
    aim, px, ref_logits, ref_embedding = build()
    asyncio.run(gate(aim, px, ref_logits, ref_embedding))
