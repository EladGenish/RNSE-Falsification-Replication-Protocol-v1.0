# RNSE Falsification & Replication Protocol v1.0

**Author:** Elad Genieh  
**Date:** January 2026  
**Status:** Open for Independent Benchmarking  
**License:** Protocol Public; RNSE Implementation Proprietary

---

## Overview

This document specifies a **black‑box replication protocol** for testing the phenomenological claims of the RNSE (Recursive Null Seed Engine) framework. The protocol is designed to be:

- **Implementation‑agnostic:** Others can plug in their own relational dynamics engines.
- **Falsifiable:** Each test has explicit pass/fail criteria independent of RNSE internals.
- **Reproducible:** Input signals and parameter settings are fully specified.
- **IP‑safe:** No core RNSE equations or update rules are disclosed.

This protocol enables third parties to **verify or refute RNSE's physical claims** without access to the proprietary engine.

---

## Part 1: Coherence & Memory Metric (Black‑Box Definition)

### 1.1 What Is "Coherence"?

**Coherence** is a scalar state variable $C(t) \in [0, 1]$ that reflects the engine's **internal consistency of prediction** over a rolling temporal window.

**Operational definition:**
- **High coherence** ($C \approx 1$): The engine's internal prediction model closely matches signal behavior within the window.
- **Low coherence** ($C \approx 0$): The engine's model is shocked or chaotic relative to the signal; surprise is high.

**Measurement protocol:**
1. Feed a signal stream into the engine.
2. At each timestep, record the engine's reported scalar coherence value $C_t$.
3. No knowledge of the internal update rule is required; treat it as an oracle output.

### 1.2 Memory / Time Emergence Claim

**Theoretical claim:** "Coherence emerges as an effective measure of 'internal memory' or 'local time'—not a pre‑existing global property."

**Test:** Structure‑dependent entropy gap (Test 2, below).

---

## Part 2: Protocol‑Level Parameters

All six tests use these standardized settings:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Sampling Rate** | $f_s = 100$ Hz | Standard for synthetic signal generation |
| **Rolling Window** | $W = 50$ steps | Coherence stability measurement window |
| **Stability Threshold** | $\theta_s = 0.5$ | Minimum coherence for "stable regime" |
| **Test Duration** | $T = 1000$ steps | Long enough to observe transients and plateaus |
| **Random Seed (if noise)** | 42, 77, 123 | For reproducibility across runs |

**Per‑test overrides:** Specified in each test section.

---

## Part 3: Input Signal Generators

### Signal Generator A: Pure Sinusoid
```
x(t) = sin(2π · f · t)
where f ∈ {1, 3, 5} Hz
Duration: 10 seconds (1000 samples @ 100 Hz)
```

**Usage:** Tests 1, 5.

### Signal Generator B: Sinusoid + Gaussian Noise
```
x(t) = A · sin(2π · f · t) + σ · N(0, 1)
where A = 0.5, f = 3 Hz, σ = 0.5
Duration: 10 seconds
Seed: 42 (for reproducibility)
```

**Usage:** Tests 2, 3, 4, 6.

### Signal Generator C: Pure Gaussian Noise
```
x(t) = N(0, 1)
Duration: 10 seconds
Seed: 77
```

**Usage:** Test 2 (shuffled baseline), Test 6 (vacuum).

### Signal Generator D: Two Periodic Subsystems
```
Subsystem A: x_A(t) = sin(2π · 3 · t)
Subsystem B: x_B(t) = sin(2π · 3 · t + φ)
where φ ∈ {0.0, π/4, π/2, π}
Duration: 10 seconds each
```

**Usage:** Test 5 (relational redshift).

---

## Part 4: Black‑Box Test Pseudocode

### General Template

```
Input:  signal array x[0..T-1]
Config: window W, threshold θ_s, random seed S
Engine: black-box oracle E (unknown internals)

OUTPUT RECORDING:
  Loop t = 0 to T-1:
    C[t] = E.coherence_at(t)
    y[t] = E.output_at(t)           [if engine produces output]
    E.step(x[t])

METRIC COMPUTATION:
  1. Extract steady-state trace: C_steady = C[500:]
  2. Compute stability: S = mean(C_steady)
  3. Compute dominance: F_dom = peak_frequency(periodogram(C[200:]))
  4. Apply pass/fail logic

OUTPUTS:
  - Coherence trace C[:]
  - Diagnostic plots (coherence, spectrum, etc.)
  - Pass/Fail verdict
```

---

## Part 5: The Six Tests

### TEST 1: Emergent Time from Causality

**Claim:** Coherence spontaneously encodes a notion of "internal time" from pure recursive causality, not pre‑given temporal structure.

**Setup:**
- Pure sine wave (1 Hz), no noise.
- Single engine instance.
- Initialization: C(0) = 0.5 (halfway).

**Action:**
- Feed signal for 1000 steps.
- Record coherence trace C[t].

**Measurement:**
- Does C[t] stabilize to a **non‑zero steady state** by t=500?
- Is that steady state *different* from the initial value?
- Do different signal frequencies (1, 3, 5 Hz) produce *different* steady‑state coherences?

**Pass Criteria:**
1. C[500:] has mean > 0.6 and std dev < 0.1 (stable).
2. Frequency response: $\Delta C(\text{1 Hz vs 5 Hz}) > 0.15$ (discriminant).
3. **Result: PASS if both true.**

**Fail Criteria:**
- C remains constant (no emergence).
- All frequencies → identical coherence (no time encoding).

**Expected RNSE behavior:** C stabilizes to ~0.8–0.9 for structured signals; different frequencies show modest variation.

---

### TEST 2: Entropy Gap (Structure vs. Noise)

**Claim:** Coherence drop scales with signal **disorder**, not just surprise. An ordered signal maintains higher coherence than shuffled noise, even if the marginal distribution is identical.

**Setup:**
- Two identical engines (A and B).
- **Stream A:** Sinusoid + Gaussian noise (3 Hz, SNR ~1).
- **Stream B:** Same signal, shuffled row‑wise (randomizes temporal order).
- Config: Window W=50, threshold θ_s = 0.5.

**Action:**
- Run both streams for 1000 steps.
- Record C_A[t] and C_B[t].

**Measurement:**
- Mean coherence in steady state: $\bar{C}_A = \text{mean}(C_A[500:])$, $\bar{C}_B = \text{mean}(C_B[500:])$.
- Entropy of coherence trace: $H_A = \text{entropy}(C_A[200:])$, $H_B = \text{entropy}(C_B[200:])$.
- Gap: $\Delta C = \bar{C}_A - \bar{C}_B$ and $\Delta H = H_B - H_A$.

**Pass Criteria:**
1. $\Delta C > 0.2$ (structured signal maintains >20% higher coherence than shuffled).
2. $\Delta H > 0.1$ (entropy of ordered coherence trace is >10% lower than noisy trace).
3. **Result: PASS if both true.**

**Fail Criteria:**
- $\Delta C \approx 0$ (structure irrelevant).
- No entropy difference (disorder undetected).

**Expected RNSE behavior:** $\Delta C \approx 0.3$, $\Delta H \approx 0.15–0.2$.

---

### TEST 3: Irreversibility (Memory Traces)

**Claim:** Once coherence drops due to surprise, it does not spontaneously recover. The engine exhibits "one‑way" dynamics: shocks → memory loss, but no spontaneous memory gain.

**Setup:**
- Single engine.
- Signal: High‑SNR sine (1 Hz) for 300 steps, then 10× Gaussian noise pulse (300–350), then back to sine.
- Record coherence throughout.

**Action:**
- Feed the three‑phase signal.
- Identify shock period: steps 300–350.
- Measure coherence pre‑shock, during shock, and post‑shock.

**Measurement:**
- Pre‑shock mean: $\bar{C}_{pre} = \text{mean}(C[100:300])$.
- Shock nadir: $C_{min} = \text{min}(C[300:350])$.
- Post‑shock recovery: $\bar{C}_{post} = \text{mean}(C[500:])$.
- Recovery ratio: $R = \bar{C}_{post} / \bar{C}_{pre}$.

**Pass Criteria:**
1. $C_{min} < 0.3$ (shock does drop coherence significantly).
2. $R < 0.95$ (post‑shock coherence does NOT fully recover; loss is irreversible).
3. **Result: PASS if both true.**

**Fail Criteria:**
- $R \approx 1.0$ (full recovery = reversible, contradicts claim).
- $C_{min}$ barely dips (shock not registered).

**Expected RNSE behavior:** $R \approx 0.7–0.8$ (permanent memory loss of ~20–30%).

---

### TEST 4: Singularity Resilience

**Claim:** Engine does not diverge or crash when fed pathological inputs (sudden NaN, inf, or extreme outliers). It gracefully degrades coherence instead.

**Setup:**
- Single engine.
- Signal: Normal sinusoid, then one step with value 1e6, then resume sinusoid.
- No explicit error handling in the test; behavior is diagnostic.

**Action:**
- Inject the spike at step 500.
- Record coherence and any exceptions.

**Measurement:**
- Did the engine throw an exception or NaN?
- What is the coherence value immediately after the spike: $C_{501}$?
- Does coherence recover within 10 steps?

**Pass Criteria:**
1. No exception / NaN propagation (graceful degradation).
2. $C_{501} < 0.2$ (spike detected as massive surprise).
3. Coherence begins recovery by step 510 (not stuck at floor).
4. **Result: PASS if all three true.**

**Fail Criteria:**
- Divergence / NaN state.
- Coherence unchanged by spike.

**Expected RNSE behavior:** Coherence crashes to ~0.01, then drifts back up slowly.

---

### TEST 5: Relational Redshift (Coherence Decay ↔ Frequency Shift)

**Claim:** When two identical sinusoid subsystems have different coherence states, their **relative prediction error** mimics a Doppler‑like frequency shift, even without kinematic motion.

**Setup:**
- Two separate engines: A (high coherence α=0.99) and B (low coherence α=0.6).
- Shared input: Pure 3 Hz sinusoid.
- No relative velocity; only internal state difference.

**Action:**
1. Feed sine to both engines for 1000 steps.
2. Record internal predictions (or reconstruction attempts) from each engine's state.
3. Compute dominant frequency of the prediction traces via periodogram.

**Measurement:**
- $f_A = \text{dominant frequency of A's prediction spectrum}$.
- $f_B = \text{dominant frequency of B's prediction spectrum}$.
- Redshift: $z = (f_A - f_B) / f_A$.
- Spectral broadening: width of B's spectrum vs. A's.

**Pass Criteria:**
1. $z > 0.05$ (at least 5% frequency shift, OR spectral broadening >20%).
2. Shift/broadening correlates with coherence difference $|\alpha_A - \alpha_B|$.
3. **Result: PASS if true.**

**Fail Criteria:**
- $f_A \approx f_B$ and same spectral width (no relational effect).

**Expected RNSE behavior:** $z \approx 0.1–0.2$ (10–20% effective redshift), plus broadening of B's spectrum.

---

### TEST 6: Accelerating Expansion from Coherence Decay (Dark Energy Killer)

**Claim:** A system with **uniform, slow coherence decay** exhibits accelerating separation of subsystems, without any external energy injection. This mimics cosmic acceleration from memory loss alone.

**Setup:**
- Single engine fed pure Gaussian noise (vacuum).
- No external forces; coherence decays naturally due to unpredictable input.
- Initial coherence: $C_0 = 0.99$.
- Target: Observe coherence trace and compute effective "expansion rate."

**Action:**
- Feed noise for 1000 steps.
- Record coherence trace C[t].
- Define effective distance: $R(t) = 1.0 / (C(t) + \epsilon)$ where $\epsilon = 1e-6$.

**Measurement:**
1. Velocity: $v(t) = \Delta R / \Delta t$ (finite difference).
2. Acceleration: $a(t) = \Delta v / \Delta t$.
3. Trend of velocity: Fit a line to $v[300:700]$ and extract slope $m$.

**Pass Criteria:**
1. Coherence decay is monotonic and substantial: $C[999] < 0.5$.
2. Velocity trend slope $m > 0$ (expansion accelerates during decay phase, before heat death plateau).
3. Acceleration is positive on average over active decay phase.
4. **Result: PASS if all three true.**

**Fail Criteria:**
- Coherence constant (no decay).
- Velocity trend negative (deceleration).

**Expected RNSE behavior:** $C$ decays from 0.99 → 0.2; $v$ increases ~linearly; $m \approx +1e-5$ to +1e-4.

---

## Part 6: Output Format & Reproducibility

### Per‑Test Output

Each test must produce:

1. **CSV data file** (one row per timestep):
   ```
   timestep, coherence, signal_input, [optional: prediction, output]
   0, 0.50, 0.314, ...
   1, 0.48, 0.628, ...
   ...
   ```

2. **Diagnostic plot** (multi‑panel figure):
   - Panel 1: Time‑domain coherence trace C[t].
   - Panel 2: Spectrum (periodogram or FFT) of coherence.
   - Panel 3: Input signal and (if available) engine output/prediction.
   - Panel 4: Pass/fail annotation and metric summary.

3. **Summary JSON**:
   ```json
   {
     "test_id": "TEST_1",
     "engine_type": "[your engine class name]",
     "parameters": {
       "window": 50,
       "threshold": 0.5,
       "seed": 42
     },
     "metrics": {
       "steady_state_coherence": 0.82,
       "coherence_std_dev": 0.08,
       "dominant_frequency": 1.02,
       "pass": true
     },
     "timestamp": "2026-01-07T01:30:00Z"
   }
   ```

### Replication Checklist

To claim **valid replication**, publish:

- [ ] Input signal arrays (CSV or JSON).
- [ ] Engine configuration (window, threshold, seed).
- [ ] Coherence trace output (CSV).
- [ ] Six diagnostic plots (one per test).
- [ ] Summary JSON with pass/fail verdicts.
- [ ] Brief narrative: "How closely do my results match RNSE reference plots?"

---

## Part 7: Reference Results (RNSE Baseline)

The following **reference plots & metrics** are the RNSE baseline, published in parallel:

| Test | Steady‑State C | Entropy Gap | Recovery Ratio | Redshift z | Velocity Trend |
|------|-----------------|-------------|-----------------|------------|-----------------|
| **1** | 0.82 ± 0.06 | — | — | — | — |
| **2** | — | 0.28 ± 0.05 | — | — | — |
| **3** | 0.78 (pre) | — | 0.74 | — | — |
| **4** | 0.01 (min) | — | — | — | — |
| **5** | — | — | — | 0.15 ± 0.04 | — |
| **6** | 0.21 (final) | — | — | — | +4.5e-5 |

**Note:** These are *typical* RNSE outputs, not exact guarantees. Implementation variations and stochasticity will cause ±5–10% variation.

---

## Part 8: License & Attribution

**This Protocol:** Public domain. Use freely for benchmarking, modification, and publication.

**RNSE Implementation:** Proprietary. Inquire for access or licensing.

**Citation:**  
"RNSE Falsification & Replication Protocol v1.0" (2026). Genieh, E. [Link to protocol repo/document].

---

## Part 9: Contact & Submission

To submit replication results, provide:

1. Summary JSON (from each test).
2. Six diagnostic plots (PNG/PDF).
3. Brief write‑up (1–2 paragraphs): engine design, deviations from protocol, interpretation of results.

**Email:** [Your contact]  
**GitHub / OSF:** [Link for submissions]

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $C(t)$ | Coherence scalar at timestep t |
| $x(t)$ | Input signal value |
| $y(t)$ | Engine output (if applicable) |
| $W$ | Rolling window size |
| $f_s$ | Sampling frequency (Hz) |
| $f_{dom}$ | Dominant frequency from spectrum |
| $z$ | Redshift parameter |
| $R(t)$ | Effective distance (1/C) |
| $\bar{C}$ | Mean coherence over interval |
| $H$ | Entropy of trace |

---

## Appendix B: Troubleshooting

**Q: What if my engine's coherence never stabilizes?**  
A: That's diagnostic data. Report it: "Engine fails Test 1: no steady state achieved." This falsifies the claim that relational coherence emerges spontaneously.

**Q: Can I use a different window size?**  
A: Protocol specifies W=50 for consistency, but you can try W∈{30, 50, 100} and note the variation. Identical behavior across windows is a robustness marker.

**Q: What if my engine is deterministic but produces different results on re‑run?**  
A: Check random seed initialization. If seeds are fixed and results differ, you have found a bug or unstable regime.

**Q: How do I interpret a "partial pass" (e.g., Test 2 passes criterion 1 but fails criterion 2)?**  
A: Report both. Partial results are still falsifiable evidence and often reveal which aspects of the claim hold.

---

**End of Protocol v1.0**
