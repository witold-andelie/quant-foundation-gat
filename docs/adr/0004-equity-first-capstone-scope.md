# Capstone scope: equity single-track end-to-end, energy deferred

The minimum credible capstone result is one complete, leakage-safe, beats-the-
baseline chain on the **equity track** — graph -> GAT -> composite -> four gates
-> Streamlit comparison. The energy track is documented as a framework
extensibility story; its *island* alphas are wired into the unified seam
(`LegacyEnergyProvider`, 2026-06-11) but its *relational* layer
(`edges_energy`, energy GAT, hourly-return label) is deferred, not built.

> Update (2026-06-11): the `LegacyEnergyProvider` shim is implemented (the 8
> imperative energy alphas exposed as `Factor`s, memoised, faithfulness- and
> memoisation-tested) — there are no `NotImplementedError` stubs left in
> `src/`. Only the energy *relational* track remains deferred per below.

## Considered Options

Doing both tracks half-way vs. one track end-to-end. We chose one end-to-end.

## Consequences

- Persuasiveness comes from a single complete chain, not two half-finished ones.
- Energy carries real correctness traps that a half-build makes dangerous: its
  label needs a different formula (hourly `(next-cur)/abs(cur).clip(20)`) and `k`
  is in hours, not days. A mis-aligned energy label would cast doubt on the
  equity results too. Better absent than half-correct.
- Deferred to energy phase: hourly-return label variant, bidding-zone expansion
  (~15-30 zones + ENTSO-E ingestion), ADR-0001's energy edges
  (`edges_energy` interconnector topology), and the typed graph universe (#4),
  which mainly serves energy's heterogeneous nodes/edges.
- The seam code (Topology, Propagator, Factor/provider, training primitives) is
  already track-agnostic, so the energy phase is additive, not a rewrite.
