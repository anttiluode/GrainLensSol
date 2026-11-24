# GrainLensSol — numeric grain of a frozen model

Companion to [RakeisuusClaude](https://github.com/anttiluode/RakeisuusClaude) (semantic grain). Question: does progressively rounding a frozen model's *activations* preferentially remove fine distinctions, or merely damage computation? The two branches should compare identical items, questions, checkpoint and intervention sites before making a joint claim.

**Status: implementation and preregistered pilot, no measured Qwen results yet.** No weight edits, training, speedup or new interpretability method claimed.

## Windows / RTX 3060

Clone this and [Qwenimagination](https://github.com/anttiluode/Qwenimagination) as sibling directories. In your existing CUDA PyTorch environment:

```bat
python -m pip install -r requirements.txt
python grain.py --self-test
python grain.py --quick --out runs/quick.json
python grain.py --out runs/gate1.json
python summarize.py runs/gate1.json
```

Default Qwen3-1.7B, FP16 CUDA (FP32 CPU), unquantized weights, non-thinking chat template. Use `--model` for a local checkpoint, `--revision` to pin a Hugging Face revision, `--qimag` for the upstream checkout. `--no-j` skips the Jacobian arm, not the numeric controls. `--quick` is a plumbing/pilot subset, not confirmation. The first real run may download weights. Record the exact checkpoint and upstream checkout; the two projects must agree before comparison.

## Measurement

The identical item prefix is followed by a broad, intermediate, or fine two-choice question. Perturb **only the item-name tokens** at one intermediate layer per run; the question and other positions are untouched. The clean source activation is causally independent of the later question. Other positions and earlier layers can carry redundant information, so a null effect does not prove the distinction is absent. Fine questions include literal identity recall; these are not a pure hierarchy test.

Arms: native symmetric rounding, full-space orthogonal rotated rounding, rounding in the orthonormal span of independently calibrated J-lens gradients, and an orthogonal rotation *within the same J span*. J-space here means that measured subspace, not a complete set of semantic axes. The orthogonal complement is untouched in the J arms. All calculations use float32 before converting back to the model dtype. Bit labels 2/3/4/6/8 refer to signed symmetric quantizer levels (one unused code); this is fake quantization, not a memory-saving kernel. Each arm has a per-token equal-L2 random displacement control (the J arms also have a matched random displacement within their span), a half-residue restoration and equal-norm random half-restoration, and exact restoration as a plumbing check.

Record answer margins, forced-choice accuracy, full-vocabulary next-token KL from clean, perturbation norm after dtype conversion, and item/question/layer/seed identifiers. Do not choose a layer or bit setting by held-out outcomes. Compare against the random arm at the same actual perturbation energy, not merely the same bit label. See PROTOCOL.md.

The J implementation imports the existing Qwenimagination lens code rather than assuming the Qwen2.5 lens transfers to Qwen3. It recalibrates on independent text and a small broad vocabulary. `--self-test` tests numerical operations without downloading a model.

## Lineage

Sihti (core + residue), Qwenimagination (J-lens), NumberHell (coarse versus fine), GraphNNSol (quantized storage and controls), GeneralInstrument (discriminating measurements).
