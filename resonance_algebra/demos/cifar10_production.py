"""
SPECTRAL PRODUCTION RULES FOR CIFAR-10
Instead of averaging exemplars, each one becomes a production rule:
"If input resonates with me, output my target spectrum scaled by resonance amplitude"
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')


class SpectralProductionClassifier:
    """
    Revolutionary approach: Each exemplar is a production rule
    Input → Resonance → Scaled Target Spectrum → Classification
    """
    
    def __init__(self, fft_size=64, n_bands=5):
        self.fft_size = fft_size
        self.n_bands = n_bands
        self.n_classes = 10
        
        # Each exemplar will have:
        # - source_spectrum: its frequency representation
        # - target_spectrum: the ideal output for its class
        self.productions = []
        
        # Target spectra for each class (learned or designed)
        self.target_spectra = {}
        
        # Frequency bands
        self.setup_bands()
        
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        print("🧬 Spectral Production Classifier initialized")
        print("  Each exemplar is a production rule, not averaged!")
    
    def setup_bands(self):
        """Create orthogonal frequency bands"""
        max_radius = self.fft_size // 2
        band_width = max_radius / (self.n_bands + 1)
        
        self.bands = []
        for i in range(self.n_bands):
            r_min = i * band_width * 1.1
            r_max = (i + 1) * band_width
            self.bands.append((r_min, r_max))
    
    def extract_spectrum(self, image):
        """Extract frequency spectrum from image"""
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        all_features = []
        
        for c in range(3):
            channel = image[:, :, c]
            
            # Normalize
            channel = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)
            
            # Window
            window = np.outer(np.hanning(32), np.hanning(32))
            windowed = channel * window
            
            # FFT
            padded = np.pad(windowed, 16, mode='constant')
            fft = np.fft.fft2(padded)
            fft_shifted = np.fft.fftshift(fft)
            
            # Power-law scaling
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)
            magnitude_scaled = np.power(magnitude + 1e-10, 0.3)
            
            # Complex spectrum
            fft_complex = magnitude_scaled * np.exp(1j * phase)
            
            # Extract band features
            h, w = fft_complex.shape
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            for band_idx, (r_min, r_max) in enumerate(self.bands):
                mask = (radius >= r_min) & (radius < r_max)
                
                if np.any(mask):
                    band_complex = fft_complex[mask]
                    
                    # Rich features preserving phase
                    features = [
                        np.mean(np.abs(band_complex)),
                        np.std(np.abs(band_complex)),
                        np.max(np.abs(band_complex)),
                        np.mean(np.real(band_complex)),
                        np.mean(np.imag(band_complex)),
                        np.abs(np.mean(band_complex)),
                        np.angle(np.mean(band_complex)),
                    ]
                    
                    # Phase coherence
                    phases = np.angle(band_complex)
                    coherence = np.abs(np.mean(np.exp(1j * phases)))
                    features.append(coherence)
                    
                else:
                    features = [0] * 8
                
                all_features.extend(features)
        
        return np.array(all_features)
    
    def create_target_spectra(self, X_support=None, y_support=None):
        """
        Create target spectra for each class
        Either from class examples or orthogonal construction
        """
        print("\n🎯 Creating target spectra...")
        
        spectrum_dim = len(self.extract_spectrum(np.zeros((32, 32, 3))))
        
        if X_support is not None and y_support is not None:
            # Learn targets from actual class patterns
            print("  Learning from class exemplars...")
            
            for class_id in range(self.n_classes):
                class_mask = y_support == class_id
                class_images = X_support[class_mask][:5]  # Use first 5 for target
                
                if len(class_images) > 0:
                    # Extract spectra and find common pattern
                    class_spectra = []
                    for img in class_images:
                        spectrum = self.extract_spectrum(img)
                        class_spectra.append(spectrum)
                    
                    # Target is the average magnitude with coherent phase
                    class_spectra = np.array(class_spectra)
                    
                    # Complex average preserves phase relationships
                    target = np.mean(class_spectra, axis=0)
                    
                    # Enhance discriminative features
                    # Amplify bands that vary less within class
                    std_per_feature = np.std(class_spectra, axis=0)
                    weight = 1.0 / (std_per_feature + 0.1)  # Inverse variance weighting
                    target = target * weight
                    
                    # Normalize
                    target = target / (np.linalg.norm(target) + 1e-8)
                    self.target_spectra[class_id] = target
                    
                    print(f"  {self.classes[class_id]}: Learned from {len(class_images)} examples")
        else:
            # Fallback to orthogonal construction
            print("  Using orthogonal construction...")
            
            for class_id in range(self.n_classes):
                # Create orthogonal target using different phase patterns
                target = np.zeros(spectrum_dim, dtype=complex)
                
                # Each class gets a unique phase signature
                np.random.seed(1000 + class_id)
                
                # Distribute energy across bands differently for each class
                for i in range(spectrum_dim):
                    # Deterministic but class-specific pattern
                    if i % (class_id + 1) == 0:
                        phase = 2 * np.pi * class_id / self.n_classes
                        amplitude = 1.0 / np.sqrt(spectrum_dim)
                        target[i] = amplitude * np.exp(1j * phase)
                
                # Normalize
                target = target / (np.linalg.norm(target) + 1e-8)
                self.target_spectra[class_id] = target
                
                print(f"  {self.classes[class_id]}: Phase signature at {class_id * 36}°")
    
    def fit(self, X_support, y_support, k_shot=10):
        """
        Create production rules from support set
        Each exemplar becomes: source_spectrum → target_spectrum
        """
        print(f"\n📚 Creating {k_shot}-shot production rules...")
        
        # First create target spectra from the support set
        self.create_target_spectra(X_support, y_support)
        
        # Clear previous productions
        self.productions = []
        
        for class_id in range(self.n_classes):
            class_mask = y_support == class_id
            class_images = X_support[class_mask][:k_shot]
            
            print(f"  {self.classes[class_id]}:", end=' ')
            
            for img in class_images:
                # Extract source spectrum
                source_spectrum = self.extract_spectrum(img)
                
                # Create production rule
                production = {
                    'source': source_spectrum,
                    'target': self.target_spectra[class_id],
                    'class': class_id,
                    'strength': 1.0  # Could be learned
                }
                
                self.productions.append(production)
            
            print(f"✓ {len(class_images)} rules")
        
        print(f"\n📊 Total production rules: {len(self.productions)}")
    
    def compute_resonance(self, spectrum1, spectrum2):
        """
        Compute resonance between two spectra
        This determines the activation strength of the production rule
        """
        if len(spectrum1) != len(spectrum2):
            return 0
        
        # Normalize
        if np.linalg.norm(spectrum1) > 0 and np.linalg.norm(spectrum2) > 0:
            # Combine magnitude correlation and phase coherence
            
            # 1. Magnitude correlation
            mag_corr = np.abs(np.corrcoef(
                np.abs(spectrum1), np.abs(spectrum2)
            )[0, 1])
            
            # 2. Phase coherence (for complex features)
            # Only compute for complex parts
            complex_mask = np.abs(np.imag(spectrum1)) > 1e-10
            if np.any(complex_mask):
                phase_diff = np.angle(spectrum1[complex_mask]) - np.angle(spectrum2[complex_mask])
                phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
            else:
                # Fallback to cosine similarity for real features
                phase_coherence = np.dot(spectrum1.real, spectrum2.real) / (
                    np.linalg.norm(spectrum1.real) * np.linalg.norm(spectrum2.real) + 1e-8
                )
                phase_coherence = max(0, phase_coherence)  # Keep positive
            
            # 3. Combined resonance (weighted average)
            resonance = 0.7 * mag_corr + 0.3 * phase_coherence
            
            return resonance
        return 0
    
    def predict(self, X_query, mode='winner_take_all', k=5):
        """
        Classify using spectral production:
        
        Mode 'winner_take_all': Like a child learning
        - Find which exemplars resonate most
        - They output their class at FULL strength (100% certain)
        
        Mode 'weighted_sum': Previous approach
        - Scale outputs by resonance strength
        """
        predictions = []
        
        for img in X_query:
            query_spectrum = self.extract_spectrum(img)
            
            if mode == 'winner_take_all':
                # Find top-k resonating productions
                resonances = []
                for production in self.productions:
                    resonance = self.compute_resonance(
                        query_spectrum, production['source']
                    )
                    resonances.append({
                        'resonance': resonance,
                        'class': production['class']
                    })
                
                # Sort by resonance
                resonances.sort(key=lambda x: x['resonance'], reverse=True)
                
                # Vote from top-k (each votes at full strength)
                class_votes = np.zeros(self.n_classes)
                for i in range(min(k, len(resonances))):
                    if resonances[i]['resonance'] > 0:
                        # Full strength vote (like mom saying "dog" with certainty)
                        class_votes[resonances[i]['class']] += 1.0
                
                predictions.append(np.argmax(class_votes))
                
            else:  # weighted_sum (original approach)
                # Accumulate scaled target spectra
                output_spectra = {c: np.zeros_like(self.target_spectra[0]) 
                                for c in range(self.n_classes)}
                
                # Apply each production rule
                for production in self.productions:
                    resonance = self.compute_resonance(
                        query_spectrum, production['source']
                    )
                    
                    if resonance > 0:
                        scaled_output = resonance * production['target']
                        output_spectra[production['class']] += scaled_output
                
                # Find class with strongest output
                class_scores = []
                for class_id in range(self.n_classes):
                    score = np.linalg.norm(output_spectra[class_id])
                    class_scores.append(score)
                
                predictions.append(np.argmax(class_scores))
        
        return np.array(predictions)
    
    def predict_with_lerp(self, X_query, lerp_factor=0.5):
        """
        Enhanced prediction with spectral morphing (lerp/slerp)
        Interpolate between top resonating productions
        """
        predictions = []
        
        for img in X_query:
            query_spectrum = self.extract_spectrum(img)
            
            # Find top resonating productions for each class
            class_resonances = {c: [] for c in range(self.n_classes)}
            
            for production in self.productions:
                resonance = self.compute_resonance(
                    query_spectrum, production['source']
                )
                class_resonances[production['class']].append({
                    'resonance': resonance,
                    'target': production['target']
                })
            
            # For each class, lerp between top productions
            class_scores = []
            for class_id in range(self.n_classes):
                if not class_resonances[class_id]:
                    class_scores.append(0)
                    continue
                
                # Sort by resonance
                sorted_productions = sorted(
                    class_resonances[class_id], 
                    key=lambda x: x['resonance'],
                    reverse=True
                )
                
                # Lerp between top 2 if available
                if len(sorted_productions) >= 2:
                    r1, t1 = sorted_productions[0]['resonance'], sorted_productions[0]['target']
                    r2, t2 = sorted_productions[1]['resonance'], sorted_productions[1]['target']
                    
                    # Weighted lerp based on resonance
                    weight = r1 / (r1 + r2 + 1e-8)
                    lerped_target = weight * t1 + (1 - weight) * t2
                    combined_resonance = (r1 + r2) / 2
                    
                    score = combined_resonance * np.linalg.norm(lerped_target)
                else:
                    r1, t1 = sorted_productions[0]['resonance'], sorted_productions[0]['target']
                    score = r1 * np.linalg.norm(t1)
                
                class_scores.append(score)
            
            predictions.append(np.argmax(class_scores))
        
        return np.array(predictions)


def test_spectral_production():
    """Test the spectral production approach"""
    import pickle
    import os
    from resonance_algebra.demos.cifar10_real import create_few_shot_splits
    
    print("="*70)
    print("🧬 SPECTRAL PRODUCTION RULES - A New Paradigm")
    print("="*70)
    print("\nEach exemplar is a production rule, not averaged!")
    print("Input resonance triggers scaled output spectra\n")
    
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
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    # Create splits
    X_support, y_support, X_query, y_query = create_few_shot_splits(
        train_data, train_labels, test_data, test_labels,
        k_shot=10, n_test_per_class=30
    )
    
    # Test production approach
    classifier = SpectralProductionClassifier()
    
    # Fit
    start = time.time()
    classifier.fit(X_support, y_support, k_shot=10)
    fit_time = time.time() - start
    
    # Test both modes
    print("\n🔮 Mode 1: Winner-Take-All (like child learning)...")
    start = time.time()
    predictions_wta = classifier.predict(X_query, mode='winner_take_all', k=5)
    pred_time = time.time() - start
    
    accuracy_wta = np.mean(predictions_wta == y_query)
    
    print("\n📊 Winner-Take-All Results (k=5 nearest):")
    for c in range(10):
        mask = y_query == c
        if np.any(mask):
            class_acc = np.mean(predictions_wta[mask] == c)
            print(f"  {classifier.classes[c]:8s}: {class_acc:.1%}")
    
    print(f"\n🎯 Accuracy (Winner-Take-All): {accuracy_wta:.1%}")
    
    # Compare with weighted sum
    print("\n🔮 Mode 2: Weighted Sum (previous approach)...")
    predictions_weighted = classifier.predict(X_query, mode='weighted_sum')
    accuracy_weighted = np.mean(predictions_weighted == y_query)
    
    print(f"🎯 Accuracy (Weighted Sum): {accuracy_weighted:.1%}")
    
    # Try with lerp
    print("\n🔄 Testing with spectral morphing (lerp)...")
    predictions_lerp = classifier.predict_with_lerp(X_query)
    accuracy_lerp = np.mean(predictions_lerp == y_query)
    
    print(f"🎯 Accuracy with lerp: {accuracy_lerp:.1%}")
    
    print(f"\n⏱️ Timing:")
    print(f"  Fit: {fit_time:.2f}s")
    print(f"  Predict: {pred_time:.2f}s ({pred_time/len(X_query)*1000:.1f}ms per image)")
    
    # Compare with different k values
    print("\n📈 Testing different k-shot values (Winner-Take-All mode)...")
    k_values = [1, 5, 10, 20]
    for k in k_values:
        classifier_k = SpectralProductionClassifier()
        classifier_k.fit(X_support, y_support, k_shot=k)
        predictions_k = classifier_k.predict(X_query, mode='winner_take_all', k=5)
        accuracy_k = np.mean(predictions_k == y_query)
        print(f"  k={k:2d} exemplars: {accuracy_k:.1%}")
    
    print("\n💡 KEY INSIGHTS:")
    print("  • Like a child: Each labeled example outputs its class with 100% certainty")
    print("  • Resonance determines WHICH productions fire, not their strength")
    print("  • Winner-take-all: Top k most similar vote at full strength")
    print("  • Much more biologically plausible than averaging!")
    
    return classifier, accuracy_wta


if __name__ == "__main__":
    classifier, accuracy = test_spectral_production()
    
    print("\n🧬 Spectral production rules - activation without gradients!")
    print("🌊 The resonance revolution continues!")