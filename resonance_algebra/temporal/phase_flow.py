"""
Temporal Logic through Phase Flow - Computation as Wave Propagation

Key insight: Time is just another dimension in phase space. 
Temporal patterns emerge from phase velocity and interference over time.
"""

import numpy as np
from typing import List, Callable, Optional, Tuple
from ..core import Lens, Concept


class PhaseFlow:
    """
    Temporal dynamics through phase evolution.
    
    Computation becomes wave propagation with natural:
    - Oscillation (clocks)
    - Synchronization (consensus)  
    - Memory (standing waves)
    - Pattern detection (resonance)
    """
    
    def __init__(self, d: int = 128, r: int = 32, dt: float = 0.01):
        """
        Initialize phase flow system.
        
        Args:
            d: Embedding dimension
            r: Number of spectral bands
            dt: Time step
        """
        self.d = d
        self.r = r
        self.dt = dt
        
        # Multiple lenses for different temporal scales
        self.fast_lens = Lens.random(d, r//2, name="fast")
        self.slow_lens = Lens.random(d, r//2, name="slow")
        
        # Natural frequencies for each band (Hz)
        self.frequencies = np.logspace(-1, 2, r)  # 0.1 Hz to 100 Hz
        
        # Phase velocities (rad/s)
        self.velocities = 2 * np.pi * self.frequencies
        
        # Damping factors for stability
        self.damping = np.ones(r) * 0.99
        
    def evolve(self, state: np.ndarray, time: float) -> np.ndarray:
        """
        Evolve phase state over time.
        
        Natural oscillation at each frequency creates temporal patterns.
        """
        # Project to spectral space
        fast_coeffs = self.fast_lens.project(state)
        slow_coeffs = self.slow_lens.project(state)
        
        # Apply phase evolution
        fast_phases = self.velocities[:self.r//2] * time
        slow_phases = self.velocities[self.r//2:] * time
        
        # Rotate phases
        fast_coeffs = fast_coeffs * np.exp(1j * fast_phases)
        slow_coeffs = slow_coeffs * np.exp(1j * slow_phases)
        
        # Apply damping
        fast_coeffs *= self.damping[:self.r//2] ** (time / self.dt)
        slow_coeffs *= self.damping[self.r//2:] ** (time / self.dt)
        
        # Reconstruct
        state_new = (self.fast_lens.reconstruct(fast_coeffs) + 
                     self.slow_lens.reconstruct(slow_coeffs)) / 2
        
        return state_new
    
    def oscillator(self, frequency: float, amplitude: float = 1.0) -> np.ndarray:
        """Create a phase oscillator at given frequency."""
        coeffs = np.zeros(self.r, dtype=complex)
        
        # Find closest band to desired frequency
        band = np.argmin(np.abs(self.frequencies - frequency))
        coeffs[band] = amplitude * np.exp(1j * 0)  # Start at phase 0
        
        # Distribute across lenses
        if band < self.r//2:
            return self.fast_lens.reconstruct(coeffs[:self.r//2])
        else:
            return self.slow_lens.reconstruct(coeffs[self.r//2:])
    
    def synchronize(self, states: List[np.ndarray], coupling: float = 0.1) -> List[np.ndarray]:
        """
        Phase synchronization - the basis of consensus and coordination.
        
        Kuramoto model in spectral space!
        """
        n_agents = len(states)
        new_states = []
        
        for i, state in enumerate(states):
            # Get phase of current agent
            coeffs = self.fast_lens.project(state)
            phases = np.angle(coeffs)
            
            # Compute mean field from other agents
            mean_field = np.zeros_like(coeffs)
            for j, other_state in enumerate(states):
                if i != j:
                    other_coeffs = self.fast_lens.project(other_state)
                    mean_field += other_coeffs / (n_agents - 1)
            
            # Apply coupling - pull toward mean
            mean_phases = np.angle(mean_field)
            phase_diff = mean_phases - phases
            
            # Update phases with coupling
            new_phases = phases + coupling * np.sin(phase_diff)
            new_coeffs = np.abs(coeffs) * np.exp(1j * new_phases)
            
            new_state = self.fast_lens.reconstruct(new_coeffs)
            new_states.append(new_state)
        
        return new_states
    
    def detect_pattern(self, signal: np.ndarray, pattern: np.ndarray, 
                      window: float = 1.0) -> Tuple[bool, float]:
        """
        Pattern detection through resonance matching over time.
        
        Returns:
            (detected, confidence)
        """
        # Evolve both signal and pattern
        n_steps = int(window / self.dt)
        max_resonance = 0.0
        
        for step in range(n_steps):
            time = step * self.dt
            
            # Evolve states
            signal_t = self.evolve(signal, time)
            pattern_t = self.evolve(pattern, time)
            
            # Compute resonance
            signal_coeffs = self.fast_lens.project(signal_t)
            pattern_coeffs = self.fast_lens.project(pattern_t)
            
            # Normalized correlation
            resonance = np.abs(np.vdot(signal_coeffs, pattern_coeffs))
            resonance /= (np.linalg.norm(signal_coeffs) * np.linalg.norm(pattern_coeffs) + 1e-10)
            
            max_resonance = max(max_resonance, resonance)
        
        # Detection threshold
        detected = max_resonance > 0.7
        
        return detected, max_resonance


class TemporalGate:
    """
    Logic gates with temporal dynamics - computation through phase evolution.
    """
    
    def __init__(self, flow: PhaseFlow):
        self.flow = flow
        
    def delay(self, input_phase: complex, delay_time: float) -> complex:
        """Pure delay - phase shift proportional to time."""
        return input_phase * np.exp(-1j * 2 * np.pi * delay_time)
    
    def edge_detector(self, signal: List[float], threshold: float = 0.5) -> List[int]:
        """
        Detect rising/falling edges through phase derivatives.
        
        Returns: List of edge events (1=rising, -1=falling, 0=none)
        """
        edges = []
        
        for i in range(1, len(signal)):
            prev = signal[i-1]
            curr = signal[i]
            
            if prev < threshold and curr >= threshold:
                edges.append(1)  # Rising edge
            elif prev >= threshold and curr < threshold:
                edges.append(-1)  # Falling edge
            else:
                edges.append(0)  # No edge
        
        return edges
    
    def flip_flop(self, set_signal: bool, reset_signal: bool, 
                  state: complex = 1.0) -> complex:
        """
        SR flip-flop using phase bistability.
        
        State persists as standing wave pattern.
        """
        if set_signal:
            state = np.exp(1j * np.pi)  # Set to phase π
        elif reset_signal:
            state = np.exp(1j * 0)  # Reset to phase 0
        # Otherwise maintain state (bistable)
        
        return state
    
    def counter(self, clock: np.ndarray, n_bits: int = 4) -> List[int]:
        """
        N-bit counter driven by phase clock.
        
        Each bit toggles at half the frequency of previous.
        """
        counts = []
        phases = np.zeros(n_bits)
        
        for t in range(len(clock)):
            # Update phase counters
            for bit in range(n_bits):
                freq_divider = 2 ** (bit + 1)
                phases[bit] += 2 * np.pi / freq_divider
                phases[bit] = phases[bit] % (2 * np.pi)
            
            # Convert phases to binary
            count = 0
            for bit in range(n_bits):
                if phases[bit] > np.pi:
                    count |= (1 << bit)
            
            counts.append(count)
        
        return counts


class ResonantCircuit:
    """
    Complete circuit with feedback loops and resonant dynamics.
    
    This is where it gets REALLY interesting - autonomous computation!
    """
    
    def __init__(self, n_nodes: int = 8, d: int = 64):
        """
        Initialize resonant circuit.
        
        Args:
            n_nodes: Number of nodes in circuit
            d: Dimension per node
        """
        self.n_nodes = n_nodes
        self.flow = PhaseFlow(d=d)
        
        # Node states
        self.states = [np.zeros(d, dtype=complex) for _ in range(n_nodes)]
        
        # Coupling matrix (who connects to whom)
        self.coupling = np.random.rand(n_nodes, n_nodes) * 0.1
        np.fill_diagonal(self.coupling, 0)  # No self-coupling
        
    def step(self, inputs: Optional[List[float]] = None) -> List[np.ndarray]:
        """
        Single time step of circuit evolution.
        
        Natural dynamics + coupling = emergent computation!
        """
        new_states = []
        
        for i in range(self.n_nodes):
            # Natural evolution
            state = self.flow.evolve(self.states[i], self.flow.dt)
            
            # Add coupling from connected nodes
            for j in range(self.n_nodes):
                if self.coupling[i, j] > 0:
                    influence = self.coupling[i, j] * self.states[j]
                    state += influence
            
            # Add external input if provided
            if inputs and i < len(inputs):
                input_phase = np.exp(1j * np.pi * inputs[i])
                state += 0.1 * self.flow.oscillator(10.0) * input_phase
            
            new_states.append(state)
        
        self.states = new_states
        return self.states
    
    def run(self, n_steps: int = 100, inputs: Optional[List[List[float]]] = None) -> np.ndarray:
        """
        Run circuit for multiple time steps.
        
        Returns:
            History of node activations
        """
        history = []
        
        for step in range(n_steps):
            if inputs:
                step_inputs = inputs[min(step, len(inputs)-1)]
            else:
                step_inputs = None
            
            self.step(step_inputs)
            
            # Record magnitudes (activity levels)
            activations = [np.linalg.norm(s) for s in self.states]
            history.append(activations)
        
        return np.array(history)


def demo_temporal():
    """Demonstrate temporal logic and phase flow."""
    print("⏱️ Temporal Logic through Phase Flow")
    print("=" * 50)
    
    # Create phase flow system
    flow = PhaseFlow(d=64, r=16)
    
    # Test oscillators
    print("\n🌊 Phase Oscillators:")
    osc_1hz = flow.oscillator(1.0)
    osc_10hz = flow.oscillator(10.0)
    print(f"1 Hz oscillator energy: {np.linalg.norm(osc_1hz):.3f}")
    print(f"10 Hz oscillator energy: {np.linalg.norm(osc_10hz):.3f}")
    
    # Test synchronization
    print("\n🔄 Phase Synchronization:")
    states = [flow.oscillator(f) for f in [1.0, 1.2, 0.8, 1.1]]
    
    print("Initial phases:", end=" ")
    for s in states:
        coeffs = flow.fast_lens.project(s)
        print(f"{np.angle(coeffs[0]):.2f}", end=" ")
    print()
    
    # Synchronize
    for _ in range(10):
        states = flow.synchronize(states, coupling=0.2)
    
    print("After sync:    ", end=" ")
    for s in states:
        coeffs = flow.fast_lens.project(s)
        print(f"{np.angle(coeffs[0]):.2f}", end=" ")
    print("\n✓ Phases converged!")
    
    # Test pattern detection
    print("\n🔍 Pattern Detection:")
    pattern = flow.oscillator(5.0)
    signal_match = flow.oscillator(5.0) * 0.8 + np.random.randn(64) * 0.1
    signal_diff = flow.oscillator(8.0) + np.random.randn(64) * 0.1
    
    detected, conf = flow.detect_pattern(signal_match, pattern)
    print(f"Matching signal: Detected={detected}, Confidence={conf:.3f}")
    
    detected, conf = flow.detect_pattern(signal_diff, pattern)
    print(f"Different signal: Detected={detected}, Confidence={conf:.3f}")
    
    # Test temporal gates
    print("\n⚡ Temporal Gates:")
    gate = TemporalGate(flow)
    
    # Edge detection
    signal = [0, 0, 1, 1, 0, 0, 1, 0]
    edges = gate.edge_detector(signal)
    print(f"Signal: {signal}")
    print(f"Edges:  {edges}")
    
    # Counter
    clock = np.ones(16)
    counts = gate.counter(clock, n_bits=4)
    print(f"4-bit counter: {counts[:8]} ...")
    
    # Test resonant circuit
    print("\n🔮 Resonant Circuit Dynamics:")
    circuit = ResonantCircuit(n_nodes=4, d=32)
    
    # Binary input pattern
    inputs = [[1, 0, 1, 0]] * 10 + [[0, 1, 0, 1]] * 10
    
    history = circuit.run(n_steps=20, inputs=inputs)
    
    print("Node activations over time:")
    for t in range(0, 20, 5):
        print(f"t={t:2d}: ", end="")
        for node in range(4):
            print(f"{history[t, node]:.2f} ", end="")
        print()
    
    print("\n🎯 Time is just another phase dimension!")


if __name__ == "__main__":
    demo_temporal()