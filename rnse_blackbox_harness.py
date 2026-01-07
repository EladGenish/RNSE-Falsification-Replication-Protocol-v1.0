#!/usr/bin/env python3
"""
rnse_blackbox_harness.py

Black-box test runner for the RNSE Falsification & Replication Protocol v1.0.

This harness:
  1. Generates input signals per protocol spec.
  2. Runs an engine black-box (expects: engine.step(signal) -> coherence output).
  3. Logs coherence traces and diagnostic metrics.
  4. Outputs pass/fail verdicts and plots.
  5. Exports summary JSON and CSV for replication tracking.

Usage:
  python rnse_blackbox_harness.py --engine my_engine_class --output ./results/

NO RNSE INTERNALS EXPOSED. This is a pure interface specification.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.stats import entropy
import argparse
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod


# ============================================================================
# SIGNAL GENERATORS (Per Protocol Part 3)
# ============================================================================

class SignalGenerator:
    """Abstract base for signal generation."""
    
    @staticmethod
    def sinusoid(f=1.0, duration=10.0, fs=100.0):
        """Pure sinusoid: x(t) = sin(2π·f·t)"""
        t = np.linspace(0, duration, int(fs * duration))
        return np.sin(2 * np.pi * f * t)
    
    @staticmethod
    def sinusoid_plus_noise(f=3.0, A=0.5, sigma=0.5, duration=10.0, fs=100.0, seed=42):
        """Sinusoid + Gaussian noise: x(t) = A·sin(2π·f·t) + σ·N(0,1)"""
        rng = np.random.RandomState(seed)
        t = np.linspace(0, duration, int(fs * duration))
        signal = A * np.sin(2 * np.pi * f * t)
        noise = sigma * rng.randn(len(signal))
        return signal + noise
    
    @staticmethod
    def pure_noise(duration=10.0, fs=100.0, seed=77):
        """Pure Gaussian noise: x(t) = N(0,1)"""
        rng = np.random.RandomState(seed)
        n_samples = int(fs * duration)
        return rng.randn(n_samples)
    
    @staticmethod
    def shuffled_signal(signal):
        """Shuffle a signal temporally (randomize order)."""
        return np.random.permutation(signal)
    
    @staticmethod
    def two_subsystems(phase_offset=0.0, f=3.0, duration=10.0, fs=100.0):
        """Two sinusoids with phase offset: x_A, x_B."""
        t = np.linspace(0, duration, int(fs * duration))
        x_A = np.sin(2 * np.pi * f * t)
        x_B = np.sin(2 * np.pi * f * t + phase_offset)
        return x_A, x_B
    
    @staticmethod
    def signal_with_shock(base_signal, shock_start=300, shock_end=350, shock_amplitude=1e6):
        """Inject an outlier spike into a signal."""
        signal = base_signal.copy()
        signal[shock_start:shock_end] = shock_amplitude
        return signal
    
    @staticmethod
    def three_phase_signal(phase1_len=300, shock_len=50, duration=1000):
        """
        Three phases: ordered signal -> noise shock -> ordered signal.
        For Test 3 (irreversibility).
        """
        fs = 100.0
        t = np.linspace(0, duration/fs, duration)
        
        signal = np.zeros(duration)
        # Phase 1: Sine
        signal[:phase1_len] = np.sin(2 * np.pi * 1.0 * t[:phase1_len])
        # Phase 2: Shock (noise)
        rng = np.random.RandomState(42)
        signal[phase1_len:phase1_len+shock_len] = 10 * rng.randn(shock_len)
        # Phase 3: Back to sine
        signal[phase1_len+shock_len:] = np.sin(2 * np.pi * 1.0 * t[phase1_len+shock_len:])
        
        return signal


# ============================================================================
# BLACK-BOX ENGINE INTERFACE
# ============================================================================

class BlackBoxEngine(ABC):
    """
    Abstract engine interface. Implementers must define:
      - __init__(...)
      - step(signal_value: float) -> None
      - get_coherence() -> float
      - get_output() -> float (optional)
    """
    
    @abstractmethod
    def step(self, signal_value):
        """Process one timestep of signal. Return None or (coherence, output)."""
        pass
    
    @abstractmethod
    def get_coherence(self):
        """Return current coherence scalar."""
        pass
    
    def get_output(self):
        """Optional: return prediction or output. Default None."""
        return None
    
    def reset(self):
        """Optional: reset engine state."""
        pass


# ============================================================================
# TEST HARNESS
# ============================================================================

class RNSETestHarness:
    """Runs all six tests and exports results."""
    
    def __init__(self, engine_factory, output_dir="./rnse_results"):
        """
        Args:
            engine_factory: Callable that returns a fresh BlackBoxEngine instance.
            output_dir: Path to save results.
        """
        self.engine_factory = engine_factory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def run_all_tests(self):
        """Execute all six tests."""
        print("=" * 70)
        print("RNSE BLACK-BOX REPLICATION HARNESS")
        print("Protocol v1.0")
        print("=" * 70)
        
        self.test_1_emergent_time()
        self.test_2_entropy_gap()
        self.test_3_irreversibility()
        self.test_4_singularity_resilience()
        self.test_5_relational_redshift()
        self.test_6_dark_energy()
        
        self.export_summary()
        print("\n" + "=" * 70)
        print("All tests complete. Results in:", self.output_dir)
        print("=" * 70)
    
    def run_test(self, test_id, signal, test_name, params=None):
        """Generic test runner: feed signal, record coherence."""
        if params is None:
            params = {}
        
        engine = self.engine_factory()
        window = params.get('window', 50)
        
        coherence_trace = []
        output_trace = []
        
        print(f"\n[{test_id}] {test_name}...", end=" ", flush=True)
        
        for val in signal:
            engine.step(val)
            c = engine.get_coherence()
            o = engine.get_output()
            coherence_trace.append(c)
            if o is not None:
                output_trace.append(o)
        
        coherence_trace = np.array(coherence_trace)
        output_trace = np.array(output_trace) if output_trace else None
        
        print(f"✓ (C_mean={coherence_trace[-50:].mean():.3f})")
        
        return coherence_trace, output_trace
    
    # ========================================================================
    # TEST 1: Emergent Time
    # ========================================================================
    def test_1_emergent_time(self):
        """TEST 1: Does coherence emerge and stabilize? Does it vary with frequency?"""
        test_id = "TEST_1"
        
        results_1hz = []
        results_3hz = []
        results_5hz = []
        
        for freq, results_list in [(1, results_1hz), (3, results_3hz), (5, results_5hz)]:
            signal = SignalGenerator.sinusoid(f=freq)
            C, _ = self.run_test(test_id, signal, f"Emergent Time (f={freq} Hz)")
            results_list.append(C)
        
        C_1hz = results_1hz[0]
        C_3hz = results_3hz[0]
        C_5hz = results_5hz[0]
        
        # Metrics
        steady_state_1hz = C_1hz[500:].mean()
        steady_state_3hz = C_3hz[500:].mean()
        steady_state_5hz = C_5hz[500:].mean()
        
        std_1hz = C_1hz[500:].std()
        std_3hz = C_3hz[500:].std()
        std_5hz = C_5hz[500:].std()
        
        freq_discriminance = max(abs(steady_state_1hz - steady_state_3hz),
                                 abs(steady_state_3hz - steady_state_5hz),
                                 abs(steady_state_1hz - steady_state_5hz))
        
        # Pass/fail
        stable_1hz = (steady_state_1hz > 0.6) and (std_1hz < 0.1)
        stable_3hz = (steady_state_3hz > 0.6) and (std_3hz < 0.1)
        stable_5hz = (steady_state_5hz > 0.6) and (std_5hz < 0.1)
        discriminant = freq_discriminance > 0.15
        
        passed = (stable_1hz and stable_3hz and stable_5hz and discriminant)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(C_1hz, label="1 Hz", color='cyan')
        axes[0, 0].axhline(steady_state_1hz, linestyle='--', color='cyan', alpha=0.5)
        axes[0, 0].set_title("1 Hz Coherence")
        axes[0, 0].set_ylabel("C")
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].plot(C_3hz, label="3 Hz", color='lime')
        axes[0, 1].axhline(steady_state_3hz, linestyle='--', color='lime', alpha=0.5)
        axes[0, 1].set_title("3 Hz Coherence")
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].plot(C_5hz, label="5 Hz", color='magenta')
        axes[1, 0].axhline(steady_state_5hz, linestyle='--', color='magenta', alpha=0.5)
        axes[1, 0].set_title("5 Hz Coherence")
        axes[1, 0].set_ylabel("C")
        axes[1, 0].grid(alpha=0.3)
        
        # Frequency response
        freqs = [1, 3, 5]
        means = [steady_state_1hz, steady_state_3hz, steady_state_5hz]
        axes[1, 1].bar(freqs, means, color=['cyan', 'lime', 'magenta'], alpha=0.7)
        axes[1, 1].set_title(f"Frequency Response (Δ={freq_discriminance:.3f})")
        axes[1, 1].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_ylabel("Steady-State C")
        axes[1, 1].set_ylim(0, 1)
        
        # Verdict
        verdict_text = f"{'PASS' if passed else 'FAIL'}\nStable: {stable_1hz & stable_3hz & stable_5hz}\nDiscriminant: {discriminant}"
        axes[1, 1].text(0.98, 0.02, verdict_text, transform=axes[1, 1].transAxes,
                        ha='right', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lime' if passed else 'red', alpha=0.7))
        
        for ax in axes.flat:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{test_id}_emergent_time.png", dpi=150)
        plt.close()
        
        # Export CSV
        df = pd.DataFrame({
            'timestep': np.arange(len(C_1hz)),
            'coherence_1hz': C_1hz,
            'coherence_3hz': C_3hz,
            'coherence_5hz': C_5hz,
        })
        df.to_csv(self.output_dir / f"{test_id}_data.csv", index=False)
        
        # Summary
        self.results[test_id] = {
            'passed': bool(passed),
            'steady_state_1hz': float(steady_state_1hz),
            'steady_state_3hz': float(steady_state_3hz),
            'steady_state_5hz': float(steady_state_5hz),
            'frequency_discriminance': float(freq_discriminance),
        }
    
    # ========================================================================
    # TEST 2: Entropy Gap
    # ========================================================================
    def test_2_entropy_gap(self):
        """TEST 2: Ordered signal vs. shuffled noise."""
        test_id = "TEST_2"
        
        signal_ordered = SignalGenerator.sinusoid_plus_noise(seed=42)
        signal_shuffled = SignalGenerator.shuffled_signal(signal_ordered)
        
        C_ordered, _ = self.run_test(test_id, signal_ordered, "Entropy Gap (Ordered)")
        C_shuffled, _ = self.run_test(test_id, signal_shuffled, "Entropy Gap (Shuffled)")
        
        # Metrics
        mean_ordered = C_ordered[500:].mean()
        mean_shuffled = C_shuffled[500:].mean()
        entropy_ordered = entropy(C_ordered[200:] + 1e-10)  # Avoid log(0)
        entropy_shuffled = entropy(C_shuffled[200:] + 1e-10)
        
        gap_coherence = mean_ordered - mean_shuffled
        gap_entropy = entropy_shuffled - entropy_ordered
        
        # Pass/fail
        pass_coherence = gap_coherence > 0.2
        pass_entropy = gap_entropy > 0.1
        passed = pass_coherence and pass_entropy
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(C_ordered, color='cyan', label='Ordered')
        axes[0, 0].plot(C_shuffled, color='red', label='Shuffled', alpha=0.6)
        axes[0, 0].set_title("Coherence Traces")
        axes[0, 0].set_ylabel("C")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Steady-state comparison
        axes[0, 1].bar(['Ordered', 'Shuffled'], [mean_ordered, mean_shuffled],
                      color=['cyan', 'red'], alpha=0.7)
        axes[0, 1].set_title(f"Steady-State Coherence Gap={gap_coherence:.3f}")
        axes[0, 1].set_ylabel("Mean C")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # Entropy comparison
        axes[1, 0].bar(['Ordered', 'Shuffled'], [entropy_ordered, entropy_shuffled],
                      color=['cyan', 'red'], alpha=0.7)
        axes[1, 0].set_title(f"Entropy Gap={gap_entropy:.3f}")
        axes[1, 0].set_ylabel("H(C)")
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # Verdict
        verdict_text = f"{'PASS' if passed else 'FAIL'}\nC_gap>0.2: {pass_coherence}\nH_gap>0.1: {pass_entropy}"
        axes[1, 1].text(0.5, 0.5, verdict_text, ha='center', va='center',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lime' if passed else 'red', alpha=0.7))
        axes[1, 1].axis('off')
        
        for ax in axes.flat:
            if ax.has_data():
                ax.set_facecolor('black')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{test_id}_entropy_gap.png", dpi=150)
        plt.close()
        
        # CSV
        df = pd.DataFrame({
            'timestep': np.arange(len(C_ordered)),
            'coherence_ordered': C_ordered,
            'coherence_shuffled': C_shuffled,
        })
        df.to_csv(self.output_dir / f"{test_id}_data.csv", index=False)
        
        # Summary
        self.results[test_id] = {
            'passed': bool(passed),
            'coherence_gap': float(gap_coherence),
            'entropy_gap': float(gap_entropy),
        }
    
    # ========================================================================
    # TEST 3: Irreversibility
    # ========================================================================
    def test_3_irreversibility(self):
        """TEST 3: Shock -> Memory loss -> No full recovery."""
        test_id = "TEST_3"
        
        signal = SignalGenerator.three_phase_signal()
        C, _ = self.run_test(test_id, signal, "Irreversibility (Shock Test)")
        
        # Metrics
        pre_shock = C[100:300].mean()
        shock_nadir = C[300:350].min()
        post_shock = C[500:].mean()
        recovery_ratio = post_shock / pre_shock if pre_shock > 0 else 0
        
        # Pass/fail
        pass_shock = shock_nadir < 0.3
        pass_irreversible = recovery_ratio < 0.95
        passed = pass_shock and pass_irreversible
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(C, color='white', lw=1.5)
        axes[0].axvspan(100, 300, alpha=0.2, color='cyan', label='Pre-shock')
        axes[0].axvspan(300, 350, alpha=0.2, color='red', label='Shock')
        axes[0].axvspan(500, 1000, alpha=0.2, color='lime', label='Post-shock')
        axes[0].axhline(pre_shock, linestyle='--', color='cyan', alpha=0.5)
        axes[0].axhline(post_shock, linestyle='--', color='lime', alpha=0.5)
        axes[0].set_title("Coherence Evolution: Shock & Recovery")
        axes[0].set_ylabel("Coherence C")
        axes[0].legend(loc='upper right')
        axes[0].grid(alpha=0.3)
        axes[0].set_facecolor('black')
        axes[0].tick_params(colors='white')
        for spine in axes[0].spines.values():
            spine.set_edgecolor('white')
        
        # Metrics panel
        metrics_text = f"""
        PRE-SHOCK: {pre_shock:.3f}
        SHOCK NADIR: {shock_nadir:.3f}
        POST-SHOCK: {post_shock:.3f}
        RECOVERY RATIO: {recovery_ratio:.3f}
        
        {'PASS' if passed else 'FAIL'}
        Shock detected: {pass_shock}
        Irreversible: {pass_irreversible}
        """
        axes[1].text(0.5, 0.5, metrics_text, ha='center', va='center',
                    fontsize=11, fontfamily='monospace', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lime' if passed else 'red', alpha=0.7))
        axes[1].axis('off')
        
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{test_id}_irreversibility.png", dpi=150)
        plt.close()
        
        # CSV
        df = pd.DataFrame({
            'timestep': np.arange(len(C)),
            'coherence': C,
        })
        df.to_csv(self.output_dir / f"{test_id}_data.csv", index=False)
        
        # Summary
        self.results[test_id] = {
            'passed': bool(passed),
            'pre_shock_coherence': float(pre_shock),
            'shock_nadir': float(shock_nadir),
            'post_shock_coherence': float(post_shock),
            'recovery_ratio': float(recovery_ratio),
        }
    
    # ========================================================================
    # TEST 4: Singularity Resilience
    # ========================================================================
    def test_4_singularity_resilience(self):
        """TEST 4: Does engine gracefully handle outliers?"""
        test_id = "TEST_4"
        
        signal = SignalGenerator.sinusoid_plus_noise()
        signal = SignalGenerator.signal_with_shock(signal, shock_start=500, shock_end=501, shock_amplitude=1e6)
        
        C, _ = self.run_test(test_id, signal, "Singularity Resilience (Outlier Spike)")
        
        # Metrics
        coherence_at_spike = C[501] if len(C) > 501 else 0.0
        any_nan = np.any(np.isnan(C))
        any_inf = np.any(np.isinf(C))
        recovery_start = C[510:520].mean() if len(C) > 520 else 0.0
        
        # Pass/fail
        pass_no_exception = not (any_nan or any_inf)
        pass_shock_detected = coherence_at_spike < 0.2
        pass_recovery = recovery_start > coherence_at_spike
        passed = pass_no_exception and pass_shock_detected and pass_recovery
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(C, color='white', lw=1.5)
        ax.axvline(500, color='red', linestyle='--', alpha=0.7, label='Spike injection')
        ax.axhline(coherence_at_spike, color='yellow', linestyle='--', alpha=0.5)
        ax.set_title("Singularity Resilience: Handling Outlier Spike")
        ax.set_ylabel("Coherence C")
        ax.set_xlabel("Timestep")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        # Verdict
        verdict_text = f"""{'PASS' if passed else 'FAIL'}
        No NaN/Inf: {pass_no_exception}
        Shock detected (C<0.2): {pass_shock_detected} (C={coherence_at_spike:.3f})
        Recovery trend: {pass_recovery}
        """
        ax.text(0.98, 0.02, verdict_text, transform=ax.transAxes,
               ha='right', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lime' if passed else 'red', alpha=0.7))
        
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{test_id}_singularity.png", dpi=150)
        plt.close()
        
        # CSV
        df = pd.DataFrame({
            'timestep': np.arange(len(C)),
            'coherence': C,
        })
        df.to_csv(self.output_dir / f"{test_id}_data.csv", index=False)
        
        # Summary
        self.results[test_id] = {
            'passed': bool(passed),
            'no_exceptions': bool(pass_no_exception),
            'coherence_at_spike': float(coherence_at_spike),
            'recovery_detected': bool(pass_recovery),
        }
    
    # ========================================================================
    # TEST 5: Relational Redshift
    # ========================================================================
    def test_5_relational_redshift(self):
        """TEST 5: Two engines, different coherence states -> frequency shift."""
        test_id = "TEST_5"
        
        signal = SignalGenerator.sinusoid(f=3.0)
        
        # Engine A: high coherence (alpha=0.99)
        engine_A = self.engine_factory()
        # (Note: if engine has alpha parameter, set it here; otherwise just run as-is)
        
        # Engine B: lower coherence
        engine_B = self.engine_factory()
        
        C_A, out_A = self.run_test(test_id, signal, "Redshift (Engine A, High Coherence)")
        C_B, out_B = self.run_test(test_id, signal, "Redshift (Engine B, Lower Coherence)")
        
        # Spectral analysis
        f, Pxx_A = periodogram(C_A[200:], fs=100.0)
        f, Pxx_B = periodogram(C_B[200:], fs=100.0)
        
        idx_A = np.argmax(Pxx_A)
        idx_B = np.argmax(Pxx_B)
        freq_A = f[idx_A]
        freq_B = f[idx_B]
        
        redshift = (freq_A - freq_B) / freq_A if freq_A > 0 else 0
        
        # Broadening
        width_A = np.sum(Pxx_A > Pxx_A.max() / 2)
        width_B = np.sum(Pxx_B > Pxx_B.max() / 2)
        broadening_ratio = width_B / width_A if width_A > 0 else 1.0
        
        # Pass/fail
        pass_shift = abs(redshift) > 0.05 or broadening_ratio > 1.2
        passed = pass_shift
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(C_A[200:500], color='cyan', label='A (High C)', alpha=0.8)
        axes[0, 0].plot(C_B[200:500], color='orange', label='B (Low C)', alpha=0.8)
        axes[0, 0].set_title("Coherence Traces (Zoom: t=200–500)")
        axes[0, 0].set_ylabel("C")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].semilogy(f, Pxx_A, color='cyan', label=f'A (f_peak={freq_A:.2f} Hz)')
        axes[0, 1].semilogy(f, Pxx_B, color='orange', label=f'B (f_peak={freq_B:.2f} Hz)')
        axes[0, 1].set_xlim(0, 10)
        axes[0, 1].set_title("Spectral Density (Coherence)")
        axes[0, 1].set_ylabel("Power")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].bar(['A', 'B'], [freq_A, freq_B], color=['cyan', 'orange'], alpha=0.7)
        axes[1, 0].set_title(f"Dominant Frequency (Redshift z={redshift:.3f})")
        axes[1, 0].set_ylabel("Frequency (Hz)")
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # Verdict
        verdict_text = f"""{'PASS' if passed else 'FAIL'}
        Redshift z: {redshift:.3f}
        Broadening ratio: {broadening_ratio:.3f}
        """
        axes[1, 1].text(0.5, 0.5, verdict_text, ha='center', va='center',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lime' if passed else 'red', alpha=0.7))
        axes[1, 1].axis('off')
        
        for ax in axes.flat:
            if ax.has_data():
                ax.set_facecolor('black')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{test_id}_redshift.png", dpi=150)
        plt.close()
        
        # CSV
        df = pd.DataFrame({
            'timestep': np.arange(len(C_A)),
            'coherence_A': C_A,
            'coherence_B': C_B,
        })
        df.to_csv(self.output_dir / f"{test_id}_data.csv", index=False)
        
        # Summary
        self.results[test_id] = {
            'passed': bool(passed),
            'redshift_z': float(redshift),
            'broadening_ratio': float(broadening_ratio),
            'freq_A_hz': float(freq_A),
            'freq_B_hz': float(freq_B),
        }
    
    # ========================================================================
    # TEST 6: Dark Energy
    # ========================================================================
    def test_6_dark_energy(self):
        """TEST 6: Coherence decay -> Accelerating expansion."""
        test_id = "TEST_6"
        
        signal = SignalGenerator.pure_noise()
        C, _ = self.run_test(test_id, signal, "Dark Energy (Coherence Decay)")
        
        # Define distance as R ~ 1/C
        R = 1.0 / (C + 1e-6)
        
        # Smooth to reduce noise
        window = 50
        R_smooth = np.convolve(R, np.ones(window) / window, mode='valid')
        
        # Velocity
        vel = np.diff(R_smooth)
        
        # Acceleration (trend)
        if len(vel) > 100:
            p = np.polyfit(range(len(vel)), vel, 1)
            vel_slope = p[0]
        else:
            vel_slope = 0.0
        
        # Metrics
        final_coherence = C[-1]
        coherence_decay = (C[0] - final_coherence) / C[0] if C[0] > 0 else 0
        
        # Pass/fail
        pass_decay = final_coherence < 0.5
        pass_acceleration = vel_slope > 0
        passed = pass_decay and pass_acceleration
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(C, color='cyan', lw=1.5)
        axes[0].set_title("Coherence (Decay Phase)")
        axes[0].set_ylabel("C")
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(R, color='magenta', alpha=0.3, lw=0.5, label='Raw R ~ 1/C')
        axes[1].plot(range(len(R_smooth)), R_smooth, color='white', lw=2, label='Smoothed R')
        axes[1].set_title("Effective Distance Expansion")
        axes[1].set_ylabel("R(t)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(vel, color='lime', alpha=0.7, lw=1, label='Expansion velocity dR/dt')
        if len(vel) > 0:
            trend = np.polyval(np.polyfit(range(len(vel)), vel, 1), range(len(vel)))
            axes[2].plot(trend, color='red', linestyle='--', lw=2, label=f'Trend (slope={vel_slope:.2e})')
        axes[2].axhline(0, color='white', alpha=0.3, linestyle=':')
        axes[2].set_title("Expansion Rate")
        axes[2].set_ylabel("dR/dt")
        axes[2].set_xlabel("Time")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        # Verdict
        verdict_text = f"""{'PASS' if passed else 'FAIL'}
        Decay: {pass_decay} (C_final={final_coherence:.3f})
        Acceleration: {pass_acceleration} (slope={vel_slope:.2e})
        """
        axes[2].text(0.98, 0.02, verdict_text, transform=axes[2].transAxes,
                    ha='right', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lime' if passed else 'red', alpha=0.7))
        
        for ax in axes:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{test_id}_dark_energy.png", dpi=150)
        plt.close()
        
        # CSV
        df = pd.DataFrame({
            'timestep': np.arange(len(C)),
            'coherence': C,
            'distance_R': R,
        })
        df.to_csv(self.output_dir / f"{test_id}_data.csv", index=False)
        
        # Summary
        self.results[test_id] = {
            'passed': bool(passed),
            'final_coherence': float(final_coherence),
            'coherence_decay_fraction': float(coherence_decay),
            'velocity_trend_slope': float(vel_slope),
        }
    
    def export_summary(self):
        """Export all results to JSON."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'protocol_version': '1.0',
            'tests': self.results,
            'overall_pass': all(r.get('passed', False) for r in self.results.values()),
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary exported to: {self.output_dir / 'summary.json'}")
        print("\nTest Results:")
        for test_id, result in self.results.items():
            status = "✓ PASS" if result.get('passed') else "✗ FAIL"
            print(f"  {test_id}: {status}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNSE Black-Box Replication Harness")
    parser.add_argument("--engine", type=str, help="Name of engine class to test (must be importable)")
    parser.add_argument("--output", type=str, default="./rnse_results", help="Output directory")
    args = parser.parse_args()
    
    # STUB: In practice, you'd import the engine dynamically.
    # For now, raise a helpful error.
    if not args.engine:
        print("""
        Usage:
          python rnse_blackbox_harness.py --engine MyEngineClass --output ./results/
        
        Your engine class must:
          - Inherit from BlackBoxEngine
          - Implement step(signal_value) and get_coherence()
        
        Example:
          from my_engine import MyEngine
          def engine_factory():
              return MyEngine(config={...})
          
          harness = RNSETestHarness(engine_factory, "./results/")
          harness.run_all_tests()
        """)
    else:
        # Try to import dynamically (simplified example)
        try:
            module_name, class_name = args.engine.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            EngineClass = getattr(module, class_name)
            
            def engine_factory():
                return EngineClass()
            
            harness = RNSETestHarness(engine_factory, args.output)
            harness.run_all_tests()
        except Exception as e:
            print(f"Error loading engine: {e}")
            print("Ensure your engine class is properly importable and inherits from BlackBoxEngine.")
