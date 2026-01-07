# RNSE-Falsification-Replication-Protocol-v1.0
A complete, IP‑safe benchmark suite for RNSE that defines six falsifiable tests, standardised signals, and black‑box metrics so any engine can be evaluated against RNSE’s emergent coherence, entropy, irreversibility, redshift, and dark‑energy claims.
# RNSE Falsification & Replication Protocol — Complete Benchmark Suite

**Status:** Open for Independent Benchmarking (Jan 2026)  
**Proprietary:** RNSE engine internals; Open: Test protocols & methodology  
**Author:** Elad Genieh

---

## What This Is

A **black-box replication harness** for testing the empirical claims of the RNSE (Recursive Null Seed Engine) framework. 

This suite allows **any researcher** to:
- ✅ Plug in their own relational dynamics engine (not just RNSE).
- ✅ Run six falsifiable tests with explicit pass/fail criteria.
- ✅ Reproduce published results or generate counter-evidence.
- ✅ Keep RNSE's proprietary internals sealed while proving/disproving its physics claims.

**Key principle:** The *science* (testable claims) is public. The *implementation* (engine code) is proprietary until funding/IP protection is in place.

---

## Files in This Suite

| File | Purpose |
|------|---------|
| **RNSE_Replication_Protocol.md** | Complete protocol specification (8 parts). Defines all tests, signals, parameters, metrics, pass/fail criteria, and expected RNSE behavior. |
| **rnse_blackbox_harness.py** | Executable test runner. Generates signals, runs an engine black-box, computes metrics, plots results, exports JSON/CSV. |
| **README.md** (this file) | Quick-start guide and overview. |

---

## Quick Start

### 1. Understand the Protocol

Read **RNSE_Replication_Protocol.md**. Key sections:
- **Part 1:** What is "coherence"? (Black-box operational definition—no internals revealed.)
- **Part 2:** Standardized parameters (window size, seed, duration).
- **Part 3:** Input signal generators (sinusoids, noise, shocks, phase offsets).
- **Part 4:** Black-box pseudocode template (how to run a test).
- **Part 5:** The six tests with explicit metrics and pass/fail thresholds.
- **Part 6:** Output format (CSV, JSON, PNG).
- **Part 7:** RNSE reference results (baseline to compare against).
- **Part 8–9:** Citation, contact, appendix.

### 2. Prepare Your Engine

If testing RNSE:
```python
from rnse_universe_core import RecursiveNode, EngineConfig

class RNSEBlackBoxWrapper(BlackBoxEngine):
    def __init__(self):
        self.cfg = EngineConfig(tau=0.1, rho=0.01, alpha=0.995)
        self.node = RecursiveNode(config=self.cfg, seed=42)
    
    def step(self, signal_value):
        self.node.step(signal_value)
    
    def get_coherence(self):
        return self.node.C
```

If testing your own engine:
```python
class YourEngineBlackBoxWrapper(BlackBoxEngine):
    def __init__(self):
        self.engine = YourEngine(...)
    
    def step(self, signal_value):
        self.engine.process(signal_value)
    
    def get_coherence(self):
        # Return a [0,1] scalar representing internal state stability.
        return self.engine.stability_metric()
```

### 3. Run the Harness

```bash
python rnse_blackbox_harness.py --engine rnse_wrapper --output ./results/
```

**Output:**
```
./results/
├── TEST_1_emergent_time.png
├── TEST_1_data.csv
├── TEST_2_entropy_gap.png
├── TEST_2_data.csv
├── ... (Tests 3–6 similarly)
└── summary.json
```

### 4. Interpret Results

Open **summary.json**:
```json
{
  "timestamp": "2026-01-07T01:30:00Z",
  "protocol_version": "1.0",
  "tests": {
    "TEST_1": {
      "passed": true,
      "steady_state_coherence": 0.82,
      "frequency_discriminance": 0.18
    },
    "TEST_2": {
      "passed": true,
      "coherence_gap": 0.28,
      "entropy_gap": 0.15
    },
    ...
  },
  "overall_pass": true
}
```

### 5. Compare Against RNSE Baseline

See **Protocol Part 7** for expected RNSE metrics. If your results are **within ±5–10%** of baseline, congratulations—your implementation likely captures similar relational dynamics.

If results **deviate significantly**:
- ✅ You've found a regime difference (interesting!).
- ✅ You've discovered a flaw in the protocol (please report).
- ❌ Your engine doesn't implement the underlying physics (falsification).

---

## The Six Tests

### TEST 1: Emergent Time
**Claim:** Pure coherence, not pre-given temporal structure, encodes "internal time."  
**Method:** Does coherence stabilize differently for different frequencies?  
**Pass:** Stable coherence + frequency discrimination.

### TEST 2: Entropy Gap
**Claim:** Coherence responds to signal *order*, not just marginal distribution.  
**Method:** Compare structured signal vs. shuffled noise.  
**Pass:** >20% coherence gap + >10% entropy gap.

### TEST 3: Irreversibility
**Claim:** Memory loss is one-way; no spontaneous recovery.  
**Method:** Inject a noise shock, measure recovery ratio.  
**Pass:** Coherence drops on shock + doesn't fully recover.

### TEST 4: Singularity Resilience
**Claim:** Engine gracefully degrades on pathological inputs (no divergence).  
**Method:** Inject a 1e6 spike, check for NaN/Inf.  
**Pass:** No exception + coherence recovers over 10 steps.

### TEST 5: Relational Redshift
**Claim:** Different coherence states mimic Doppler shift without motion.  
**Method:** Two engines, same signal, different internal states → frequency shift.  
**Pass:** >5% frequency shift OR >20% spectral broadening.

### TEST 6: Dark Energy
**Claim:** Coherence decay alone produces accelerating expansion.  
**Method:** Feed pure noise, measure distance $R \sim 1/C$ and its acceleration.  
**Pass:** Coherence decays + velocity trend positive.

---

## For Replicators

If you want to **publish your replication** or benchmark another engine:

1. **Run the harness** on your engine.
2. **Collect:**
   - `summary.json` (metrics & pass/fail verdicts).
   - Six PNG plots (one per test).
   - Brief narrative (~200 words):
     - Engine name, design, key parameters.
     - Deviations from protocol (if any).
     - Interpretation (e.g., "Engine X captures emergent time but not relational redshift").
3. **Submit to:**
   - [RNSE GitHub Issues](https://github.com/...)
   - [OSF Repository](https://osf.io/...)
   - Direct email: [elad contact]

**Attribution:** Cite this protocol as:
> Genieh, E. (2026). RNSE Falsification & Replication Protocol v1.0. [URL]

---

## Important Notes

### IP Safety
- **Closed:** RNSE engine code, coherence update equations, learning rules, geometry penalty.
- **Open:** This protocol, input signals, test definitions, diagnostic metrics, baseline results.

**Why?** Keeps RNSE scientifically falsifiable while protecting proprietary implementation until IP/funding is secure.

### Extensibility
The protocol is **modular**. You can:
- Add new tests (specify input, metric, pass/fail clearly).
- Modify parameter sets (document deviations).
- Test on real data (financial time series, cosmic data, etc.).

Just keep the **black-box interface** clean: `engine.step(x)` → `engine.get_coherence()`.

### Limitations
- Protocol assumes a **scalar coherence** metric. Engines with vector or complex states need adaptation.
- Tests are synthetic. Real-world data may show different statistics.
- Falsification is probabilistic; use multiple runs (seeds) to confirm.

---

## Troubleshooting

**Q: My engine fails TEST 1. What does that mean?**
A: Your engine doesn't exhibit frequency-dependent coherence. It may be:
  - Frequency-agnostic (not encoding time).
  - Unstable (coherence never settles).
  - Misscaled (coherence always near 0 or 1).
  
Check your coherence metric definition against Protocol Part 1.

**Q: Can I modify the window size?**
A: Yes, but document it. Protocol specifies W=50 for consistency. Try W∈{30, 50, 100} and compare results.

**Q: What if my results are halfway between "pass" and "fail"?**
A: Report it honestly. Partial results are valid falsification evidence. (E.g., "TEST 2 passes coherence gap but fails entropy gap" → coherence might be distribution-sensitive, not order-sensitive.)

**Q: Can I use real data instead of synthetic signals?**
A: Absolutely. Just document the signal source and ensure you compute the same metrics (coherence trace + periodogram + entropy).

---

## References & Further Reading

- RNSE Physics Framework: [Link to paper / arXiv]
- Relational Quantum Mechanics: Rovelli (1996) et al.
- Dynamical Systems & Chaos: Strogatz "Nonlinear Dynamics and Chaos"
- Signal Processing: Oppenheim & Schafer "Discrete-Time Signal Processing"

---

## License

**Protocol & Harness:** MIT or Creative Commons (choose below)  
**RNSE Implementation:** Proprietary (contact author for licensing)
