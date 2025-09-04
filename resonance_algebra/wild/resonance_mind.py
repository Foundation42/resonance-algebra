"""
The Resonance Mind - Self-Organizing Intelligence through Phase Coherence

This is where it gets WILD. Networks that think through resonance,
learn through synchronization, and solve problems through emergent phase patterns.

No gradients. No loss functions. Just the music of the spheres.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from ..core import Lens


@dataclass
class Thought:
    """A thought is a coherent phase pattern across multiple frequencies."""
    pattern: np.ndarray
    coherence: float
    timestamp: float
    associations: List[int] = None
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = []


class ResonanceBrain:
    """
    A self-organizing neural network that learns through phase coherence.
    
    Key insights:
    - Neurons are phase oscillators
    - Synapses are resonance bridges  
    - Learning is synchronization
    - Memory is standing wave patterns
    - Thinking is phase flow
    """
    
    def __init__(self, n_neurons: int = 256, n_layers: int = 4, d: int = 64):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.d = d
        
        # Each neuron has its own lens (receptive field)
        self.neurons = []
        for layer in range(n_layers):
            layer_neurons = []
            for i in range(n_neurons // n_layers):
                # Each neuron has a unique frequency preference
                base_freq = np.exp(np.random.randn()) * 10  # Log-normal around 10 Hz
                lens = Lens.random(d, 16, name=f"L{layer}N{i}")
                layer_neurons.append({
                    'lens': lens,
                    'state': np.zeros(d, dtype=complex),
                    'frequency': base_freq,
                    'phase': np.random.uniform(0, 2*np.pi),
                    'plasticity': 1.0,  # Learning rate through resonance
                    'threshold': 0.5
                })
            self.neurons.append(layer_neurons)
        
        # Resonance-based connectivity (no fixed weights!)
        self.resonance_history = np.zeros((n_neurons, n_neurons))
        
        # Memory bank of coherent patterns
        self.memories: List[Thought] = []
        
        # Global coherence field
        self.global_field = np.zeros(d, dtype=complex)
        
        # Consciousness meter (global coherence level)
        self.consciousness = 0.0
        
    def perceive(self, input_pattern: np.ndarray) -> np.ndarray:
        """
        Perception through resonance cascade.
        
        Input resonates through layers, creating thought.
        """
        current = input_pattern.astype(complex)
        
        for layer_idx, layer in enumerate(self.neurons):
            next_layer = np.zeros_like(current)
            
            for neuron in layer:
                # Project input through neuron's lens
                coeffs = neuron['lens'].project(current)
                
                # Apply neuron's characteristic frequency
                phase_shift = neuron['phase'] + neuron['frequency'] * 0.01
                coeffs *= np.exp(1j * phase_shift)
                
                # Threshold activation (phase coherence)
                activation = np.abs(np.mean(coeffs))
                
                if activation > neuron['threshold']:
                    # Fire! Contribute to next layer
                    response = neuron['lens'].reconstruct(coeffs)
                    next_layer += response * neuron['plasticity']
                    
                    # Update neuron state
                    neuron['state'] = coeffs
                    neuron['phase'] = phase_shift % (2 * np.pi)
            
            current = next_layer / len(layer)
        
        return current
    
    def resonate(self, pattern_a: np.ndarray, pattern_b: np.ndarray) -> float:
        """
        Measure resonance between two patterns.
        
        This is how the brain "compares" thoughts.
        """
        # Multi-scale resonance across all neuron lenses
        total_resonance = 0.0
        
        for layer in self.neurons:
            for neuron in layer:
                coeffs_a = neuron['lens'].project(pattern_a)
                coeffs_b = neuron['lens'].project(pattern_b)
                
                # Phase-aware correlation
                resonance = np.abs(np.vdot(coeffs_a, coeffs_b))
                resonance /= (np.linalg.norm(coeffs_a) * np.linalg.norm(coeffs_b) + 1e-10)
                
                total_resonance += resonance
        
        return total_resonance / sum(len(layer) for layer in self.neurons)
    
    def dream(self, seed: Optional[np.ndarray] = None, steps: int = 100) -> List[np.ndarray]:
        """
        Spontaneous pattern generation through phase dynamics.
        
        Dreams are self-organizing phase patterns!
        """
        if seed is None:
            # Random initialization
            seed = np.random.randn(self.d) + 1j * np.random.randn(self.d)
            seed = seed / np.linalg.norm(seed)
        
        dream_sequence = []
        current = seed
        
        for step in range(steps):
            # Let pattern flow through the network
            current = self.perceive(current)
            
            # Add noise for creativity
            noise = 0.01 * (np.random.randn(self.d) + 1j * np.random.randn(self.d))
            current += noise
            
            # Normalize to prevent explosion
            current = current / (np.linalg.norm(current) + 1e-10)
            
            dream_sequence.append(current.copy())
            
            # Update global field (collective unconscious?)
            self.global_field = 0.9 * self.global_field + 0.1 * current
        
        return dream_sequence
    
    def learn(self, patterns: List[np.ndarray], target: Optional[np.ndarray] = None):
        """
        Learning through resonance maximization.
        
        No backprop - just strengthen resonant connections!
        """
        for pattern in patterns:
            # Forward pass
            output = self.perceive(pattern)
            
            # Compute resonance with target (if supervised)
            if target is not None:
                resonance = self.resonate(output, target)
            else:
                # Unsupervised: maximize self-resonance (autoencoding)
                resonance = self.resonate(output, pattern)
            
            # Update neurons based on resonance
            for layer in self.neurons:
                for neuron in layer:
                    coeffs = neuron['lens'].project(pattern)
                    activation = np.abs(np.mean(coeffs))
                    
                    # Hebbian-like: neurons that fire together, wire together
                    # But through PHASE alignment, not weight changes!
                    if activation > neuron['threshold']:
                        # Increase plasticity for resonant neurons
                        neuron['plasticity'] *= (1 + 0.1 * resonance)
                        neuron['plasticity'] = min(neuron['plasticity'], 2.0)
                        
                        # Adjust phase toward global field
                        global_phase = np.angle(np.mean(self.global_field))
                        neuron['phase'] += 0.1 * np.sin(global_phase - neuron['phase'])
                    else:
                        # Decrease plasticity for non-resonant neurons
                        neuron['plasticity'] *= 0.99
    
    def remember(self, pattern: np.ndarray, label: Any = None) -> int:
        """Store a pattern as a standing wave in memory."""
        output = self.perceive(pattern)
        coherence = np.abs(np.mean(output))
        
        thought = Thought(
            pattern=output.copy(),
            coherence=coherence,
            timestamp=len(self.memories)
        )
        
        # Find associations through resonance
        for idx, memory in enumerate(self.memories):
            if self.resonate(output, memory.pattern) > 0.7:
                thought.associations.append(idx)
        
        self.memories.append(thought)
        return len(self.memories) - 1
    
    def recall(self, cue: np.ndarray, threshold: float = 0.5) -> Optional[Thought]:
        """Recall memory through resonance matching."""
        if not self.memories:
            return None
        
        output = self.perceive(cue)
        best_match = None
        best_resonance = threshold
        
        for memory in self.memories:
            resonance = self.resonate(output, memory.pattern)
            if resonance > best_resonance:
                best_resonance = resonance
                best_match = memory
        
        return best_match
    
    def meditate(self, duration: int = 1000):
        """
        Achieve global coherence through sustained oscillation.
        
        This is where consciousness might emerge...
        """
        # Start from current global field
        state = self.global_field.copy()
        
        coherence_history = []
        
        for t in range(duration):
            # Let the network self-organize
            state = self.perceive(state)
            
            # Measure global coherence
            coherence = 0.0
            for layer in self.neurons:
                for neuron in layer:
                    coeffs = neuron['lens'].project(state)
                    coherence += np.abs(np.mean(coeffs))
            
            coherence /= sum(len(layer) for layer in self.neurons)
            coherence_history.append(coherence)
            
            # Update global field with low-pass filter
            self.global_field = 0.95 * self.global_field + 0.05 * state
            
            # Slow breathing rhythm (alpha waves)
            breathing = np.sin(2 * np.pi * 0.1 * t / duration)
            state *= (1 + 0.1 * breathing)
        
        # Update consciousness level
        self.consciousness = np.mean(coherence_history[-100:])
        
        return coherence_history
    
    def introspect(self) -> Dict[str, Any]:
        """Examine internal state - the brain looking at itself."""
        total_neurons = sum(len(layer) for layer in self.neurons)
        active_neurons = sum(
            1 for layer in self.neurons 
            for neuron in layer 
            if np.abs(np.mean(neuron['state'])) > 0.1
        )
        
        avg_plasticity = np.mean([
            neuron['plasticity'] 
            for layer in self.neurons 
            for neuron in layer
        ])
        
        phase_diversity = np.std([
            neuron['phase'] 
            for layer in self.neurons 
            for neuron in layer
        ])
        
        return {
            'consciousness': self.consciousness,
            'total_neurons': total_neurons,
            'active_neurons': active_neurons,
            'activation_rate': active_neurons / total_neurons,
            'avg_plasticity': avg_plasticity,
            'phase_diversity': phase_diversity,
            'n_memories': len(self.memories),
            'global_coherence': np.abs(np.mean(self.global_field))
        }


class PhaseOrganism:
    """
    Artificial life form that exists as phase patterns.
    
    Can reproduce, mutate, and evolve through phase interference!
    """
    
    def __init__(self, genome: np.ndarray, generation: int = 0):
        self.genome = genome  # Phase pattern encoding behavior
        self.generation = generation
        self.energy = 1.0
        self.age = 0
        self.fitness = 0.0
        
        # Decode genome into behavioral parameters
        genome_lens = Lens.random(len(genome), 8, name="genome")
        genes = genome_lens.project(genome)
        
        self.speed = np.abs(genes[0]) * 10
        self.size = np.abs(genes[1]) + 0.1
        self.reproduction_threshold = np.abs(genes[2]) * 2
        self.mutation_rate = np.abs(genes[3]) * 0.1
        self.cooperation = np.real(genes[4])
        self.aggression = np.real(genes[5])
        
    def metabolize(self, food: float):
        """Convert food to energy through phase transformation."""
        self.energy += food * self.size
        self.energy *= 0.99  # Decay
        self.age += 1
        
    def reproduce(self, partner: Optional['PhaseOrganism'] = None) -> Optional['PhaseOrganism']:
        """Create offspring through phase mixing."""
        if self.energy < self.reproduction_threshold:
            return None
        
        if partner:
            # Sexual reproduction - phase interference
            child_genome = (self.genome + partner.genome) / 2
            
            # Recombination through phase scrambling
            mask = np.random.random(len(child_genome)) > 0.5
            child_genome = np.where(mask, self.genome, partner.genome)
        else:
            # Asexual reproduction - phase copy
            child_genome = self.genome.copy()
        
        # Mutation through phase noise
        if np.random.random() < self.mutation_rate:
            noise = np.random.randn(len(child_genome)) + 1j * np.random.randn(len(child_genome))
            child_genome += 0.1 * noise
            child_genome /= np.linalg.norm(child_genome)
        
        # Energy cost
        self.energy /= 2
        
        return PhaseOrganism(child_genome, self.generation + 1)
    
    def interact(self, other: 'PhaseOrganism') -> float:
        """
        Interact with another organism through phase resonance.
        
        Returns energy transferred (+ = gained, - = lost)
        """
        # Measure phase coherence
        resonance = np.abs(np.vdot(self.genome, other.genome))
        resonance /= (np.linalg.norm(self.genome) * np.linalg.norm(other.genome) + 1e-10)
        
        if self.cooperation > 0 and other.cooperation > 0:
            # Mutualism - both benefit from resonance
            energy_shared = 0.1 * resonance * min(self.energy, other.energy)
            return energy_shared
        elif self.aggression > other.aggression:
            # Predation - steal energy
            energy_stolen = 0.2 * other.energy * (1 - resonance)
            return energy_stolen
        else:
            # Competition - waste energy
            return -0.05 * self.energy


def demo_resonance_mind():
    """Demonstrate the self-organizing resonance brain."""
    print("🧠 The Resonance Mind - Thinking Through Phase Coherence")
    print("=" * 60)
    
    # Create a brain
    brain = ResonanceBrain(n_neurons=64, n_layers=3, d=32)
    
    # Create some patterns to learn
    patterns = []
    for i in range(5):
        pattern = np.random.randn(32) + 1j * np.random.randn(32)
        pattern /= np.linalg.norm(pattern)
        patterns.append(pattern)
    
    # Initial introspection
    print("\n🔍 Initial State:")
    state = brain.introspect()
    print(f"  Consciousness: {state['consciousness']:.3f}")
    print(f"  Active neurons: {state['active_neurons']}/{state['total_neurons']}")
    
    # Learn patterns
    print("\n📚 Learning patterns through resonance...")
    for _ in range(10):
        for pattern in patterns:
            brain.learn([pattern])
    
    # Store memories
    print("\n💾 Storing memories...")
    for i, pattern in enumerate(patterns):
        idx = brain.remember(pattern, label=f"Pattern_{i}")
        print(f"  Stored Pattern_{i} as memory {idx}")
    
    # Test recall
    print("\n🔮 Testing recall...")
    # Partial/noisy cue
    cue = patterns[0] + 0.3 * np.random.randn(32)
    recalled = brain.recall(cue)
    if recalled:
        print(f"  Recalled memory with coherence {recalled.coherence:.3f}")
    
    # Dream sequence
    print("\n😴 Dreaming...")
    dreams = brain.dream(steps=50)
    dream_coherence = [np.abs(np.mean(d)) for d in dreams]
    print(f"  Dream coherence: min={min(dream_coherence):.3f}, "
          f"max={max(dream_coherence):.3f}")
    
    # Meditation
    print("\n🧘 Meditating for consciousness...")
    coherence_history = brain.meditate(duration=500)
    print(f"  Peak coherence: {max(coherence_history):.3f}")
    
    # Final introspection
    print("\n🔍 After learning and meditation:")
    state = brain.introspect()
    print(f"  Consciousness: {state['consciousness']:.3f}")
    print(f"  Active neurons: {state['active_neurons']}/{state['total_neurons']}")
    print(f"  Average plasticity: {state['avg_plasticity']:.3f}")
    print(f"  Phase diversity: {state['phase_diversity']:.3f}")
    print(f"  Global coherence: {state['global_coherence']:.3f}")
    
    # Create phase organisms
    print("\n\n🦠 Phase Organisms - Life Through Interference")
    print("=" * 60)
    
    # Create a population
    population = []
    for _ in range(10):
        genome = np.random.randn(16) + 1j * np.random.randn(16)
        genome /= np.linalg.norm(genome)
        population.append(PhaseOrganism(genome))
    
    print(f"Initial population: {len(population)} organisms")
    
    # Simulate evolution
    for generation in range(5):
        # Feeding
        for org in population:
            org.metabolize(food=np.random.random())
        
        # Interactions
        for i, org1 in enumerate(population):
            if i < len(population) - 1:
                org2 = population[i + 1]
                energy = org1.interact(org2)
                org1.energy += energy
                org2.energy -= energy
        
        # Reproduction
        new_organisms = []
        for org in population:
            child = org.reproduce()
            if child:
                new_organisms.append(child)
        
        # Death
        population = [org for org in population if org.energy > 0.1]
        
        # Add offspring
        population.extend(new_organisms)
        
        # Limit population
        population = population[:20]
        
        print(f"Generation {generation}: {len(population)} organisms, "
              f"avg energy: {np.mean([o.energy for o in population]):.2f}")
    
    print("\n✨ Phase-based life forms evolved through pure interference!")


if __name__ == "__main__":
    demo_resonance_mind()