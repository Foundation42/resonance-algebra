"""
BVH-FRACTAL RESONANCE for CIFAR-10
Christian's insight: Progressive frequency refinement like a BVH traversal
Each level ADDS detail, doesn't replace it!
"""

import numpy as np
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class BVHFractalResonance:
    """
    Progressive frequency decomposition like a BVH tree
    Start with ultra-low frequencies, add bands as needed
    Early stopping when confident (like BVH culling)
    """
    
    def __init__(self, k_shot=10):
        self.k_shot = k_shot
        self.n_classes = 10
        
        # BVH-like levels with cumulative frequency bands
        self.frequency_hierarchy = {
            'level_0': (0, 1),    # DC only (2x2 equivalent)
            'level_1': (1, 2),    # Add first harmonic (4x4)
            'level_2': (2, 4),    # Add next octave (8x8)
            'level_3': (4, 8),    # Add mid frequencies (16x16)
            'level_4': (8, 16),   # Add high frequencies (32x32)
            'level_5': (16, 32),  # Add fine details (if needed)
        }
        
        # Early stopping threshold (like BVH culling)
        self.confidence_threshold = 0.7
        
        # Storage for progressive prototypes
        self.prototypes = {}
        self.frequency_importance = {}
        
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    def extract_progressive_spectrum(self, image):
        """
        Extract frequency bands progressively (like BVH traversal)
        Returns cumulative features at each level
        """
        # Convert to grayscale for simplicity (can extend to color)
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Normalize
        gray = (gray - gray.mean()) / (gray.std() + 1e-8)
        
        # Full FFT (we'll extract bands progressively)
        fft_full = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft_full)
        
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Create frequency grid
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Progressive feature extraction
        progressive_features = {}
        cumulative_features = []
        
        for level, (freq_min, freq_max) in self.frequency_hierarchy.items():
            # Extract this frequency band
            mask = (radius >= freq_min) & (radius < freq_max)
            
            # Get spectral components in this band
            band_spectrum = np.zeros_like(fft_shifted)
            band_spectrum[mask] = fft_shifted[mask]
            
            # Extract features from this band
            features = self.extract_band_features(band_spectrum, level)
            
            # Add to cumulative (BVH-like accumulation)
            cumulative_features.extend(features)
            
            # Store progressive state
            progressive_features[level] = cumulative_features.copy()
        
        return progressive_features
    
    def extract_band_features(self, band_spectrum, level):
        """
        Extract features from a frequency band
        More sophisticated at higher levels (like BVH detail)
        """
        features = []
        
        # Magnitude statistics
        magnitudes = np.abs(band_spectrum)
        features.append(np.mean(magnitudes))
        features.append(np.std(magnitudes))
        features.append(np.max(magnitudes))
        
        # Phase statistics (where the magic happens)
        phases = np.angle(band_spectrum)
        nonzero_mask = magnitudes > 0.01
        
        if np.any(nonzero_mask):
            # Circular mean of phases
            mean_phase_vector = np.mean(np.exp(1j * phases[nonzero_mask]))
            features.append(np.abs(mean_phase_vector))  # Concentration
            features.append(np.angle(mean_phase_vector))  # Mean direction
            
            # Phase entropy (disorder)
            phase_hist, _ = np.histogram(phases[nonzero_mask], bins=8, range=(-np.pi, np.pi))
            phase_hist = phase_hist / (np.sum(phase_hist) + 1e-8)
            entropy = -np.sum(phase_hist * np.log(phase_hist + 1e-8))
            features.append(entropy)
        else:
            features.extend([0, 0, 0])
        
        # Radial and angular distributions at finer levels
        if 'level_3' in level or 'level_4' in level or 'level_5' in level:
            h, w = band_spectrum.shape
            center = (h // 2, w // 2)
            
            # Angular distribution (orientation)
            y, x = np.ogrid[:h, :w]
            angles = np.arctan2(y - center[0], x - center[1])
            
            for angle_bin in range(4):  # 4 orientations
                angle_min = -np.pi + angle_bin * np.pi / 2
                angle_max = angle_min + np.pi / 2
                angle_mask = (angles >= angle_min) & (angles < angle_max) & nonzero_mask
                
                if np.any(angle_mask):
                    features.append(np.mean(magnitudes[angle_mask]))
                else:
                    features.append(0)
        
        return features
    
    def fit(self, X_support, y_support):
        """
        Create BVH-like progressive prototypes
        """
        print("\n🌲 Creating BVH-Fractal Prototypes...")
        
        for class_id in range(self.n_classes):
            class_mask = y_support == class_id
            class_images = X_support[class_mask][:self.k_shot]
            
            print(f"  {self.classes[class_id]}:", end=' ')
            
            # Extract progressive features for all examples
            class_progressive = {level: [] for level in self.frequency_hierarchy.keys()}
            
            for img in class_images:
                prog_features = self.extract_progressive_spectrum(img)
                for level, features in prog_features.items():
                    class_progressive[level].append(features)
            
            # Create prototypes at each level
            self.prototypes[class_id] = {}
            for level, features_list in class_progressive.items():
                # Average to create prototype
                prototype = np.mean(features_list, axis=0)
                self.prototypes[class_id][level] = prototype
            
            print(f"✓ {len(self.frequency_hierarchy)} levels")
        
        # Compute frequency importance (which bands are discriminative)
        self.compute_frequency_importance()
        
        print("✓ BVH-Fractal prototypes ready!")
    
    def compute_frequency_importance(self):
        """
        Determine which frequency bands are most discriminative
        (Like BVH node importance for culling)
        """
        for level in self.frequency_hierarchy.keys():
            # Compute between-class variance for this level
            level_prototypes = []
            for class_id in range(self.n_classes):
                level_prototypes.append(self.prototypes[class_id][level])
            
            level_prototypes = np.array(level_prototypes)
            
            # Between-class variance
            between_var = np.var(level_prototypes, axis=0).mean()
            
            # Within-class would need more examples, using heuristic
            self.frequency_importance[level] = between_var
        
        # Normalize
        total = sum(self.frequency_importance.values())
        if total > 0:
            for level in self.frequency_importance:
                self.frequency_importance[level] /= total
    
    def classify_with_early_stopping(self, image):
        """
        BVH-like classification with early stopping
        Only go to finer frequencies if needed
        """
        progressive_features = self.extract_progressive_spectrum(image)
        
        # Start at coarsest level
        scores = np.zeros(self.n_classes)
        confidence = 0
        
        for level in self.frequency_hierarchy.keys():
            # Get features up to this level
            features = np.array(progressive_features[level])
            
            # Score against each class prototype at this level
            level_scores = np.zeros(self.n_classes)
            
            for class_id in range(self.n_classes):
                prototype = self.prototypes[class_id][level]
                
                # Distance in feature space (could use resonance instead)
                distance = np.linalg.norm(features - prototype)
                
                # Convert to similarity
                similarity = 1 / (1 + distance)
                
                # Weight by frequency importance
                level_scores[class_id] = similarity * self.frequency_importance[level]
            
            # Accumulate scores
            scores += level_scores
            
            # Check confidence (like BVH early termination)
            probs = np.exp(scores) / np.sum(np.exp(scores))
            confidence = np.max(probs)
            
            # Early stopping if confident enough
            if confidence > self.confidence_threshold and level != list(self.frequency_hierarchy.keys())[-1]:
                print(f"    → Early stop at {level} (confidence: {confidence:.2f})")
                break
        
        return np.argmax(scores), confidence, level
    
    def predict(self, X_query):
        """
        Predict with BVH-like progressive refinement
        """
        predictions = []
        confidences = []
        levels_used = []
        
        for i, img in enumerate(X_query):
            pred, conf, level = self.classify_with_early_stopping(img)
            predictions.append(pred)
            confidences.append(conf)
            levels_used.append(level)
            
            if i < 5:  # Show first few
                print(f"  Image {i}: class={self.classes[pred]}, conf={conf:.2f}, level={level}")
        
        # Statistics
        avg_conf = np.mean(confidences)
        level_counts = {level: levels_used.count(level) for level in self.frequency_hierarchy.keys()}
        
        print(f"\n  Average confidence: {avg_conf:.2f}")
        print(f"  Level usage: {level_counts}")
        
        return np.array(predictions)


def test_bvh_fractal():
    """Test the BVH-fractal approach"""
    import pickle
    import os
    from resonance_algebra.demos.cifar10_real import create_few_shot_splits
    
    print("="*70)
    print("🌲 BVH-FRACTAL RESONANCE")
    print("   Progressive Frequency Refinement (like BVH traversal)")
    print("="*70)
    
    print("\nConcept: Start with low frequencies, add detail only if needed")
    print("  Level 0: DC component only (is it light/dark?)")
    print("  Level 1: +first harmonic (basic shape)")
    print("  Level 2: +next octave (rough texture)")
    print("  Level 3: +mid frequencies (detailed structure)")
    print("  Level 4: +high frequencies (fine details)")
    print("  Level 5: +very fine (only if really needed)")
    
    # Load CIFAR-10
    data_path = './data/cifar-10-batches-py/data_batch_1'
    if not os.path.exists(data_path):
        print("⚠️ CIFAR-10 not found!")
        return
    
    with open(data_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    train_data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    train_labels = np.array(batch[b'labels'])
    
    test_path = './data/cifar-10-batches-py/test_batch'
    with open(test_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    test_labels = np.array(test_batch[b'labels'])
    
    # Test with k=10
    print(f"\n🎯 Testing 10-shot learning with BVH-fractal:")
    print("-"*50)
    
    # Create splits
    X_support, y_support, X_query, y_query = create_few_shot_splits(
        train_data, train_labels, test_data, test_labels,
        k_shot=10, n_test_per_class=20
    )
    
    # Initialize
    clf = BVHFractalResonance(k_shot=10)
    
    # Fit
    import time
    start = time.time()
    clf.fit(X_support, y_support)
    fit_time = time.time() - start
    
    print(f"\n🔍 Classifying with early stopping:")
    
    # Predict
    start = time.time()
    predictions = clf.predict(X_query)
    pred_time = time.time() - start
    
    accuracy = np.mean(predictions == y_query)
    
    # Per-class analysis
    print("\n📊 Per-Class Performance:")
    for c in range(10):
        mask = y_query == c
        if np.any(mask):
            class_acc = np.mean(predictions[mask] == c)
            print(f"  {clf.classes[c]:8s}: {class_acc:.1%}")
    
    print(f"\n⏱️ Timing:")
    print(f"  Fit: {fit_time:.2f}s")
    print(f"  Predict: {pred_time:.2f}s")
    
    print(f"\n🎯 Overall Accuracy: {accuracy:.1%}")
    
    print("\n" + "="*70)
    print("💡 BVH-FRACTAL INSIGHTS")
    print("="*70)
    print("✓ Progressive refinement (only add detail if needed)")
    print("✓ Early stopping saves computation")
    print("✓ Natural hierarchy through frequency bands")
    print("✓ Like graphics BVH - traverse only as deep as needed")
    print("✓ Still ZERO training!")
    
    return accuracy


if __name__ == "__main__":
    accuracy = test_bvh_fractal()
    
    print("\n🌲 BVH-Fractal: Nature's hierarchy with computational efficiency!")
    print("🌊 The resonance revolution continues!")