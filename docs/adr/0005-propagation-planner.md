# ADR-0005: Propagation methods and planner rules

- **Status**: accepted
- **Date**: 2026-10-16
- **Plan step**: P4.6

## Context

LUMOS propagates fields from any grid to any grid (REQ-PROP-1/2) and must choose the numerical method
automatically, log the choice and allow forcing it (REQ-PROP-4). Each method is accurate only in a domain set
by the sampling of the field, the distance and the output grid; a wrong choice gives aliased or approximate
fields without any error message. The plan asked for "transfer function, impulse response, two steps" Fresnel
methods and for scaled or shifted angular spectra (MFT or CZT).

## Decision

**Methods.** Six kernels, all pure functions of backend arrays (ADR-0002), applied to every polarisation
component, mode and wavelength:

| Name | Output grid | Domain |
|------|-------------|--------|
| `rs` | any | everywhere (direct sum, cost `N_in N_out`) |
| `fraunhofer` | any | far field, focal plane of a lens, Fraunhofer regime |
| `fresnel_tf` | input grid | Fresnel regime, `dx ≳ λz/L` |
| `fresnel_ir` | input grid | Fresnel regime, `dx ≲ λz/L` |
| `fresnel_mft` | any | Fresnel regime, chirp sampled over the field support |
| `asm` | any | spectrum inside the Matsushima band limit |

The "two-step" Fresnel method of the plan is replaced by `fresnel_mft`, a one-step Fresnel transform evaluated
with a matrix Fourier transform: it gives any output sampling and centre directly, which is what the two-step
scheme was meant to provide. Scaled and shifted angular spectra use an inverse matrix Fourier transform
rather than a chirp-z transform: same result, simpler code, and the cost is dominated by the FFT for the grid
sizes of interest.

**Planner.** The planner measures the spatial support and the spectral support of the actual field (boxes
holding `1 - 1e-6` of the power) and applies, for every wavelength, these rules in order:

1. `distance=None` → `fraunhofer`;
2. `z < 0` → `asm` (exactly invertible);
3. spectral support inside the band limit and output inside the padded window → `asm`;
4. neglected Fresnel phase `k ρ⁴ / 8z³ ≤ 0.01` rad → `fresnel_mft` if the chirped field is sampled and the
   output frequencies avoid the periodic replicas, otherwise `fresnel_ir` / `fresnel_tf` on the input grid
   (split at the critical distance `N dx² / λ`);
5. input quadratic phase `k ρ_in² / 2z ≤ 0.05` rad → `fraunhofer`;
6. otherwise `rs`, with a warning above `1e10` operations.

A wavecube or bundle uses one method for all its channels: the most conservative per-wavelength choice. The
decision and its reason are logged on `lumos.propagation` and available without propagating through
`lumos.propagation.plan()`. An explicit `method=` bypasses the planner.

## Alternatives considered

- Choosing from the grids only (sampling criteria such as `z_c = N dx² / λ`): simpler, but blind to the
  actual bandwidth of the field; a smooth beam is propagated exactly by the angular spectrum far beyond `z_c`,
  while a sharp aperture is not.
- Always using the angular spectrum with large padding: exact but memory-hungry at long distances, and it
  cannot reach the far field.
- Chirp-z transforms for scaled outputs: equivalent results, more code; can be added later as a kernel.

## Consequences

- Every rule is checked by the validation suite (P4.1) and by Hypothesis batteries against analytic solutions
  (P4.6), which demonstrate a valid choice in all tested cases.
- Supports are computed with an FFT of up to 16 representative fields on the host: the planner runs outside
  compiled code. Compiled pipelines should fix the method once (from `plan()`) and call the kernel directly.
- New methods register with `lumos.propagation.register_method`; the planner must be extended explicitly.
