#!/usr/bin/env python3
"""
Create beautiful, publication-quality figures for Resonance Algebra
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Arrow, FancyArrow
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

def create_spectral_lens_diagram():
    """Create diagram showing spectral decomposition through lenses"""
    fig = plt.figure(figsize=(14, 8))
    
    # Create main axis
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Input vector
    ax.arrow(0.5, 3, 1.5, 0, head_width=0.2, head_length=0.1, 
             fc='#2E86AB', ec='#2E86AB', linewidth=2)
    ax.text(1.25, 3.5, 'Input\nEmbedding', ha='center', fontsize=10, fontweight='bold')
    
    # Lens (prism-like)
    lens_x = [3, 3, 4, 4]
    lens_y = [1, 5, 5.5, 0.5]
    lens = patches.Polygon(list(zip(lens_x, lens_y)), 
                          facecolor='#A23B72', alpha=0.3, 
                          edgecolor='#A23B72', linewidth=2)
    ax.add_patch(lens)
    ax.text(3.5, 3, 'Spectral\nLens', ha='center', fontsize=10, 
            fontweight='bold', color='#A23B72')
    
    # Spectral bands (rainbow)
    colors_spectrum = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', 
                       '#0000FF', '#4B0082', '#9400D3']
    band_labels = ['Low freq', '', '', 'Mid', '', '', 'High freq']
    
    for i, (color, label) in enumerate(zip(colors_spectrum, band_labels)):
        y_pos = 0.5 + i * 0.7
        # Wavy line for each frequency
        x = np.linspace(4.5, 8, 100)
        freq = (i + 1) * 2
        y = y_pos + 0.1 * np.sin(freq * x)
        ax.plot(x, y, color=color, linewidth=2, alpha=0.8)
        
        if label:
            ax.text(8.2, y_pos, label, fontsize=9, color=color)
    
    # Output (reconstructed)
    ax.arrow(8.5, 3, 1, 0, head_width=0.2, head_length=0.1,
             fc='#F18F01', ec='#F18F01', linewidth=2)
    ax.text(9.5, 3.5, 'Spectral\nCoefficients', ha='center', fontsize=10, fontweight='bold')
    
    # Title
    ax.text(5, 5.8, 'Spectral Decomposition Through Lenses', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Annotations
    ax.annotate('', xy=(4, 3), xytext=(3, 3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.text(2, 2.5, 'Projection: Π(x) = B^T x', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('resonance_algebra/figures/spectral_lens.png', dpi=300, bbox_inches='tight')
    print("Created: spectral_lens.png")

def create_resonance_vs_cosine():
    """Create comparison of cosine similarity vs resonance matching"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cosine similarity (left)
    ax1.set_title('Traditional: Cosine Similarity', fontsize=12, fontweight='bold')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    
    # Draw vectors
    vec1 = [1.5, 0.5]
    vec2 = [0.8, 1.2]
    
    ax1.arrow(0, 0, vec1[0], vec1[1], head_width=0.1, head_length=0.1,
             fc='blue', ec='blue', linewidth=2, label='Vector A')
    ax1.arrow(0, 0, vec2[0], vec2[1], head_width=0.1, head_length=0.1,
             fc='red', ec='red', linewidth=2, label='Vector B')
    
    # Show angle
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    wedge = Wedge((0, 0), 0.5, 0, np.degrees(angle), 
                  facecolor='yellow', alpha=0.3, edgecolor='black', linewidth=1)
    ax1.add_patch(wedge)
    ax1.text(0.3, 0.1, f'θ={np.degrees(angle):.1f}°', fontsize=10)
    
    # Grid
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.legend()
    
    # Single similarity value
    similarity = np.cos(angle)
    ax1.text(0, -1.5, f'Similarity = cos(θ) = {similarity:.3f}', 
            fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="lightgray"))
    
    # Resonance matching (right)
    ax2.set_title('Resonance: Spectral Overlap', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    
    # Draw spectra
    freqs = np.linspace(1, 9, 100)
    
    # Spectrum A
    spec_a = (0.3 * np.exp(-(freqs-3)**2/0.5) + 
              0.5 * np.exp(-(freqs-5)**2/0.3) +
              0.2 * np.exp(-(freqs-7)**2/0.4))
    
    # Spectrum B
    spec_b = (0.2 * np.exp(-(freqs-2)**2/0.4) + 
              0.4 * np.exp(-(freqs-5)**2/0.5) +
              0.3 * np.exp(-(freqs-8)**2/0.3))
    
    ax2.fill_between(freqs, 2.5, 2.5 + spec_a, alpha=0.5, color='blue', label='Concept A')
    ax2.fill_between(freqs, 2.5, 2.5 + spec_b, alpha=0.5, color='red', label='Concept B')
    
    # Overlap
    overlap = np.minimum(spec_a, spec_b)
    ax2.fill_between(freqs, 2.5, 2.5 + overlap, alpha=0.7, color='purple', 
                     label='Resonance')
    
    # Frequency bands
    for i in range(1, 10):
        ax2.axvline(x=i, color='gray', alpha=0.2, linestyle='--')
    
    ax2.set_xlabel('Frequency Bands', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.legend()
    
    # Multiple resonance values
    ax2.text(5, 0.5, 'Band-specific resonances:\nLow: 0.82  Mid: 0.95  High: 0.61', 
            fontsize=10, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.suptitle('From Single Angle to Rich Spectral Matching', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('resonance_algebra/figures/resonance_vs_cosine.png', dpi=300, bbox_inches='tight')
    print("Created: resonance_vs_cosine.png")

def create_phase_flow_visualization():
    """Create temporal dynamics through phase flow"""
    fig = plt.figure(figsize=(14, 6))
    
    # Time evolution subplot
    ax1 = fig.add_subplot(121)
    ax1.set_title('Phase Evolution Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (ms)', fontsize=10)
    ax1.set_ylabel('Phase (radians)', fontsize=10)
    
    t = np.linspace(0, 100, 500)
    
    # Multiple oscillators with different frequencies
    for i, (freq, label, color) in enumerate([
        (1, 'Memory (1 Hz)', '#E74C3C'),
        (4, 'Theta (4 Hz)', '#3498DB'),
        (10, 'Alpha (10 Hz)', '#2ECC71'),
        (40, 'Gamma (40 Hz)', '#F39C12')
    ]):
        phase = 2 * np.pi * freq * t / 1000  # Convert to Hz
        signal = np.sin(phase) * np.exp(-t/200)  # Damped oscillation
        ax1.plot(t, signal + i*2, color=color, linewidth=2, label=label)
    
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Synchronization subplot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Phase Synchronization', fontsize=12, fontweight='bold')
    
    # Create spiral showing synchronization
    theta = np.linspace(0, 4 * np.pi, 100)
    
    # Before sync (dispersed)
    for i in range(5):
        phase_offset = i * 2 * np.pi / 5
        x = np.cos(theta + phase_offset) * (1 - theta/(4*np.pi))
        y = np.sin(theta + phase_offset) * (1 - theta/(4*np.pi))
        z = theta / (4 * np.pi)
        ax2.plot(x[:30], y[:30], z[:30], alpha=0.5, linewidth=1, color='red')
    
    # After sync (coherent)
    for i in range(5):
        phase_offset = i * 0.1  # Much smaller offset
        x = np.cos(theta + phase_offset) * (1 - theta/(4*np.pi))
        y = np.sin(theta + phase_offset) * (1 - theta/(4*np.pi))
        z = theta / (4 * np.pi)
        ax2.plot(x[70:], y[70:], z[70:], alpha=0.8, linewidth=2, color='green')
    
    # Transition
    ax2.plot([0], [0], [0.5], 'ko', markersize=8)
    ax2.text2D(0.05, 0.95, "Desync → Sync", transform=ax2.transAxes, 
               fontsize=10, verticalalignment='top')
    
    ax2.set_xlabel('Real', fontsize=9)
    ax2.set_ylabel('Imaginary', fontsize=9)
    ax2.set_zlabel('Time', fontsize=9)
    ax2.view_init(elev=20, azim=45)
    
    plt.suptitle('Temporal Dynamics Through Phase Flow', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('resonance_algebra/figures/phase_flow.png', dpi=300, bbox_inches='tight')
    print("Created: phase_flow.png")

def create_emergence_hierarchy():
    """Create hierarchy showing emergence from phase geometry"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    levels = [
        ("Phase Space\ne^(iφ)", "#1A1A2E", 0, "Foundation"),
        ("Interference\nPatterns", "#16213E", 1, ""),
        ("Logic Gates\nXOR, AND, OR", "#0F3460", 2, "Symbolic"),
        ("Arithmetic\nALU", "#533483", 3, ""),
        ("Memory\nStanding Waves", "#C74B50", 4, "Persistent"),
        ("Temporal\nOscillations", "#D49B54", 5, "Dynamic"),
        ("Learning\nSynchronization", "#87A922", 6, "Adaptive"),
        ("Intelligence\nResonance", "#3E8E7E", 7, "Emergent"),
        ("Consciousness\nGlobal Coherence", "#FFD700", 8, "Meta")
    ]
    
    # Draw pyramid
    for i, (name, color, level, annotation) in enumerate(levels):
        width = 10 - level * 0.9
        x_center = 6
        
        # Main block
        rect = FancyBboxPatch((x_center - width/2, level), width, 0.9,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        
        # Text
        ax.text(x_center, level + 0.45, name, ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')
        
        # Side annotation
        if annotation:
            ax.text(x_center + width/2 + 0.5, level + 0.45, annotation,
                   ha='left', va='center', fontsize=9, 
                   style='italic', color=color)
        
        # Emergence arrow
        if i < len(levels) - 1:
            ax.annotate('', xy=(x_center, level + 0.9), 
                       xytext=(x_center, level + 1),
                       arrowprops=dict(arrowstyle='->', lw=2, 
                                     color='white', alpha=0.5))
    
    # Side annotations
    ax.text(0.5, 4, '← No Training', fontsize=12, rotation=90, 
           va='center', fontweight='bold', color='#C74B50')
    ax.text(11.5, 4, 'No Backprop →', fontsize=12, rotation=270, 
           va='center', fontweight='bold', color='#3E8E7E')
    
    # Title
    ax.text(6, 9.5, 'The Emergence Hierarchy', fontsize=16, 
           fontweight='bold', ha='center')
    ax.text(6, 9.1, 'All Computation from Phase Geometry', fontsize=11, 
           ha='center', style='italic', color='gray')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('resonance_algebra/figures/emergence_hierarchy.png', dpi=300, bbox_inches='tight')
    print("Created: emergence_hierarchy.png")

if __name__ == "__main__":
    print("Creating beautiful figures for Resonance Algebra...")
    print("-" * 50)
    
    create_spectral_lens_diagram()
    create_resonance_vs_cosine()
    create_phase_flow_visualization()
    create_emergence_hierarchy()
    
    print("-" * 50)
    print("All figures created successfully!")
    print("\nFigures created:")
    print("1. spectral_lens.png - Spectral decomposition concept")
    print("2. resonance_vs_cosine.png - Paradigm comparison")
    print("3. phase_flow.png - Temporal dynamics")
    print("4. emergence_hierarchy.png - Complete emergence stack")
    print("\nReady for publication! 🎨")