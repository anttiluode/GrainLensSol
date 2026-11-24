# November 24 audit

Neon identified calibration label imbalance, intervention bypasses and literal-copy confounding. All three are substantive limitations. V1 is kept as a pipeline pilot and raw quantization reference. V2 changes the protocol, so its results are not a continuation of V1's preregistration. See PROTOCOL_V2.md.

The cross-repo architecture (coarse router, owned fine slots, abstaining seam) is a useful integration hypothesis, not an established improvement: hierarchical indexing, isolated storage, confidence gating and provenance-aware updates all have ordinary baselines. Test fixed memory and compute budgets, retrieval accuracy, overwrite interference, source separation, and rejection under ambiguous routing. Do not use coarse state as sole storage for independently writable details.

Genealogy's operator-defines-persistence motif motivates the native residual-update study: see the design in PROTOCOL_V2.md. A telescoping identity is bookkeeping, not evidence that a layer update is an independently recoverable concept.
