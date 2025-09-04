#!/usr/bin/env python3
"""
Create publication-quality figures for the Resonance Algebra article
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_xor_phase_diagram():
    """Create the XOR truth table as phase interference patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('XOR as Phase Interference', fontsize=16, fontweight='bold')
    
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    outputs = [0, 1, 1, 0]
    
    for idx, ((a, b), out) in enumerate(zip(inputs, outputs)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Phase values
        phase_a = np.pi if a else 0
        phase_b = np.pi if b else 0
        phase_diff = phase_a - phase_b
        
        # Create unit circle
        circle = Circle((0, 0), 1, fill=False, edgecolor='gray', linewidth=1, linestyle='--')
        ax.add_patch(circle)
        
        # Plot phase vectors
        ax.arrow(0, 0, np.cos(phase_a), np.sin(phase_a), 
                head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2,
                label=f'φ₁={int(np.degrees(phase_a))}°')
        ax.arrow(0, 0, np.cos(phase_b), np.sin(phase_b),
                head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2,
                label=f'φ₂={int(np.degrees(phase_b))}°')
        
        # Show interference
        result_phase = np.cos(phase_diff)
        color = 'green' if out == 0 else 'orange'
        ax.arrow(0, 0, result_phase, 0,
                head_width=0.15, head_length=0.15, fc=color, ec=color, linewidth=3,
                alpha=0.7, label=f'Δφ={int(np.degrees(phase_diff))}°')
        
        # Formatting
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Input: ({a},{b}) → Output: {out}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        
        # Add phase labels
        ax.text(1.2, -1.3, f'XOR = {"1" if out else "0"}', 
               fontsize=14, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig('resonance_algebra/figures/xor_phase_interference.png', dpi=150, bbox_inches='tight')
    print("Created: xor_phase_interference.png")

def create_resonance_stack_diagram():
    """Create diagram showing the complete computational stack"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define layers
    layers = [
        ("Phase Geometry", "#FF6B6B", 0),
        ("Logic Gates", "#4ECDC4", 1),
        ("Arithmetic", "#45B7D1", 2),
        ("Memory", "#96CEB4", 3),
        ("Temporal Flow", "#FFEAA7", 4),
        ("Neural Networks", "#DDA0DD", 5),
        ("Intelligence", "#98D8C8", 6),
        ("Consciousness", "#F7DC6F", 7)
    ]
    
    # Draw layers
    for name, color, level in layers:
        rect = FancyBboxPatch((0.5, level), 8, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black',
                              linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(4.5, level + 0.4, name, ha='center', va='center',
               fontsize=12, fontweight='bold')
    
    # Add emergence arrows
    for i in range(len(layers) - 1):
        ax.arrow(9, i + 0.5, 0.5, 0.8, head_width=0.2, head_length=0.1,
                fc='gray', ec='gray', alpha=0.5)
        
    # Add descriptions
    descriptions = [
        "e^(iφ) operations",
        "XOR, AND, OR via interference",
        "ALU through phase accumulation",
        "Standing waves",
        "Oscillation & synchronization",
        "Resonance learning",
        "Self-organization",
        "Global coherence"
    ]
    
    for desc, level in zip(descriptions, range(len(layers))):
        ax.text(10.5, level + 0.4, desc, ha='left', va='center',
               fontsize=10, style='italic', color='#555')
    
    # Title and formatting
    ax.set_title('The Resonance Algebra Stack', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 8)
    ax.axis('off')
    
    # Add side note
    ax.text(6, -0.3, "All emerge from phase geometry - no training required!",
           ha='center', fontsize=11, style='italic', color='#333')
    
    plt.tight_layout()
    plt.savefig('resonance_algebra/figures/resonance_stack.png', dpi=150, bbox_inches='tight')
    print("Created: resonance_stack.png")

def create_phase_vs_backprop():
    """Create comparison diagram: Phase algebra vs backpropagation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Traditional backprop
    ax1.set_title('Traditional Neural Network', fontsize=14, fontweight='bold')
    
    # Draw network
    layers_x = [1, 3, 5]
    nodes_y = [[2, 3], [1, 2, 3, 4], [2, 3]]
    
    # Forward pass
    for l in range(len(layers_x) - 1):
        for n1 in nodes_y[l]:
            for n2 in nodes_y[l+1]:
                ax1.plot([layers_x[l], layers_x[l+1]], [n1, n2], 
                        'b-', alpha=0.3, linewidth=1)
    
    # Nodes
    for x, nodes in zip(layers_x, nodes_y):
        for y in nodes:
            ax1.scatter(x, y, s=200, c='blue', edgecolor='black', linewidth=2, zorder=5)
    
    # Backprop arrows
    for i in range(1000):
        if i % 100 == 0:
            alpha = 0.1 + (i / 2000)
            ax1.arrow(5.5, 2.5, -4.5, 0, head_width=0.2, head_length=0.2,
                     fc='red', ec='red', alpha=alpha, linewidth=1)
            ax1.text(6, 2.5 - i*0.001, f'Iteration {i}', fontsize=8, alpha=alpha)
    
    ax1.text(3, 0.5, '1000+ iterations\nGradient descent\nWeight updates',
            ha='center', fontsize=10, color='red')
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 5)
    ax1.axis('off')
    
    # Resonance approach
    ax2.set_title('Resonance Algebra', fontsize=14, fontweight='bold')
    
    # Draw phase circles
    phases = [0, np.pi]
    for i, phase in enumerate(phases):
        x = 2 + i * 2
        circle = Circle((x, 3), 0.8, fill=False, edgecolor='purple', linewidth=2)
        ax2.add_patch(circle)
        
        # Phase vector
        ax2.arrow(x, 3, 0.7*np.cos(phase), 0.7*np.sin(phase),
                 head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)
        ax2.text(x, 1.8, f'φ={int(np.degrees(phase))}°', ha='center', fontsize=10)
    
    # Interference pattern
    x = np.linspace(1.5, 4.5, 100)
    y = 3 + 0.3 * np.sin(10*x) * np.exp(-0.5*(x-3)**2)
    ax2.plot(x, y, 'g-', linewidth=2, alpha=0.7, label='Interference')
    
    # Result
    ax2.scatter(3, 3, s=300, c='green', marker='*', edgecolor='black', 
               linewidth=2, zorder=5, label='Instant result')
    
    ax2.text(3, 1, 'ZERO iterations\nPhase interference\nNo weights',
            ha='center', fontsize=10, color='green', fontweight='bold')
    
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    
    plt.suptitle('Paradigm Shift: From Iteration to Instantaneous', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('resonance_algebra/figures/phase_vs_backprop.png', dpi=150, bbox_inches='tight')
    print("Created: phase_vs_backprop.png")

if __name__ == "__main__":
    print("Creating figures for Resonance Algebra article...")
    create_xor_phase_diagram()
    create_resonance_stack_diagram()
    create_phase_vs_backprop()
    print("\nAll figures created successfully!")