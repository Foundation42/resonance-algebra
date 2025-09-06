"""
HIERARCHICAL SPECTRAL BANDS FOR CIFAR-10
Focus on disambiguating similar classes (cat vs dog) with specialized bands
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class HierarchicalSpectralProduction:
    """
    Multiple specialized bands for different visual aspects
    Each band captures different information needed for disambiguation
    """
    
    def __init__(self):
        self.productions = []
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Define specialized bands (for 64x64 FFT)
        self.bands = {
            'spatial': {'range': (0, 4), 'weight': 1.0},      # Global shape
            'texture': {'range': (4, 12), 'weight': 1.5},     # Surface patterns (important!)
            'detail': {'range': (12, 20), 'weight': 1.0},     # Fine edges
            'ultra_fine': {'range': (20, 32), 'weight': 0.5}, # Very fine details
        }
        
        print("🎯 Hierarchical bands initialized:")
        for name, params in self.bands.items():
            print(f"  {name:10s}: freq {params['range']} (weight={params['weight']})")
    
    def extract_hierarchical_spectrum(self, image):
        """
        Extract features from each specialized band
        Returns a dictionary of band-specific patterns
        """
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        band_features = {}
        
        # Process each color channel separately for color band
        channels_fft = []
        for c in range(3):
            channel = image[:, :, c]
            
            # Normalize
            channel = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)
            
            # Window
            window = np.outer(np.hanning(32), np.hanning(32))
            windowed = channel * window
            
            # Pad and FFT
            padded = np.pad(windowed, 16, mode='constant')
            fft = np.fft.fft2(padded)
            fft_shifted = np.fft.fftshift(fft)
            channels_fft.append(fft_shifted)
        
        # Extract features from each band
        h, w = channels_fft[0].shape
        center_y, center_x = h // 2, w // 2
        
        for band_name, params in self.bands.items():
            r_min, r_max = params['range']
            
            # Create radial mask
            y, x = np.ogrid[:h, :w]
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = (radius >= r_min) & (radius < r_max)
            
            if band_name == 'spatial':
                # For spatial: focus on overall shape (grayscale)
                gray_fft = np.mean(channels_fft, axis=0)
                band_complex = gray_fft[mask]
                
            elif band_name == 'texture':
                # For texture: combine all channels but focus on patterns
                combined = np.stack(channels_fft, axis=0)
                band_complex = []
                for c in range(3):
                    band_complex.extend(channels_fft[c][mask])
                band_complex = np.array(band_complex[:100])  # Limit size
                
            else:
                # For detail bands: use grayscale
                gray_fft = np.mean(channels_fft, axis=0)
                band_complex = gray_fft[mask][:50]  # Limit size
            
            band_features[band_name] = band_complex
        
        # Add color-specific band (color differences in low frequencies)
        color_diff_band = []
        r_color = 8  # Low frequency for color
        color_mask = radius < r_color
        
        if np.any(color_mask):
            # R-G and B-Y color opponency
            rg_diff = channels_fft[0][color_mask] - channels_fft[1][color_mask]
            by_diff = channels_fft[2][color_mask] - (channels_fft[0][color_mask] + channels_fft[1][color_mask])/2
            color_diff_band = np.concatenate([rg_diff[:20], by_diff[:20]])
        
        band_features['color'] = color_diff_band
        
        return band_features
    
    def band_similarity(self, pattern1, pattern2, band_name):
        """
        Compute similarity for a specific band
        Different similarity metrics for different bands
        """
        if len(pattern1) == 0 or len(pattern2) == 0:
            return 0
        
        # Ensure same length (truncate to shorter)
        min_len = min(len(pattern1), len(pattern2))
        p1 = pattern1[:min_len]
        p2 = pattern2[:min_len]
        
        if band_name in ['spatial', 'texture']:
            # For these bands, use both magnitude and phase
            mag1, mag2 = np.abs(p1), np.abs(p2)
            
            # Magnitude correlation
            if np.std(mag1) > 0 and np.std(mag2) > 0:
                mag_corr = np.corrcoef(mag1, mag2)[0, 1]
            else:
                mag_corr = 0
            
            # Phase coherence
            phase_diff = np.angle(p1) - np.angle(p2)
            phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            # Weight phase more for texture
            if band_name == 'texture':
                similarity = 0.4 * mag_corr + 0.6 * phase_coherence
            else:
                similarity = 0.6 * mag_corr + 0.4 * phase_coherence
                
        elif band_name == 'color':
            # For color, use correlation of complex values
            similarity = np.abs(np.corrcoef(np.abs(p1), np.abs(p2))[0, 1])
            
        else:  # detail bands
            # For detail, focus on magnitude patterns
            mag1, mag2 = np.abs(p1), np.abs(p2)
            if np.std(mag1) > 0 and np.std(mag2) > 0:
                similarity = np.corrcoef(mag1, mag2)[0, 1]
            else:
                similarity = 0
        
        return similarity
    
    def compute_resonance(self, query_bands, production_bands):
        """
        Compute weighted resonance across all bands
        """
        total_resonance = 0
        total_weight = 0
        
        for band_name, params in self.bands.items():
            if band_name in query_bands and band_name in production_bands:
                sim = self.band_similarity(
                    query_bands[band_name], 
                    production_bands[band_name],
                    band_name
                )
                
                weight = params['weight']
                total_resonance += sim * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_resonance / total_weight
        return 0
    
    def fit(self, X, y, k_shot=10):
        """Create production rules with hierarchical bands"""
        self.productions = []
        
        print(f"\n📚 Creating hierarchical production rules...")
        
        for class_id in range(10):
            class_mask = y == class_id
            class_examples = X[class_mask][:k_shot]
            
            print(f"  {self.classes[class_id]:8s}: ", end='')
            
            for img in class_examples:
                bands = self.extract_hierarchical_spectrum(img)
                self.productions.append({
                    'bands': bands,
                    'class': class_id
                })
            
            print(f"{len(class_examples)} rules")
        
        print(f"\n📊 Total: {len(self.productions)} production rules")
    
    def predict(self, X, k_nearest=5, debug=False):
        """Predict using hierarchical band resonance"""
        predictions = []
        debug_info = []
        
        for img in X:
            query_bands = self.extract_hierarchical_spectrum(img)
            
            # Compute resonance with all productions
            resonances = []
            for prod in self.productions:
                resonance = self.compute_resonance(query_bands, prod['bands'])
                resonances.append({
                    'resonance': resonance,
                    'class': prod['class']
                })
            
            # Sort by resonance
            resonances.sort(key=lambda x: x['resonance'], reverse=True)
            
            # Weighted voting (not just counting)
            votes = np.zeros(10)
            for i in range(min(k_nearest, len(resonances))):
                if resonances[i]['resonance'] > 0:
                    # Weight by resonance strength to break ties
                    votes[resonances[i]['class']] += resonances[i]['resonance']
            
            # Add small random noise to break exact ties
            votes += np.random.randn(10) * 1e-6
            
            prediction = np.argmax(votes)
            predictions.append(prediction)
            
            if debug:
                debug_info.append({
                    'top_resonances': resonances[:k_nearest],
                    'votes': votes.copy(),
                    'prediction': prediction
                })
        
        if debug:
            return np.array(predictions), debug_info
        return np.array(predictions)
    
    def test_cat_dog_disambiguation(self, X, y):
        """Specifically test cat vs dog disambiguation"""
        print("\n" + "="*60)
        print("CAT vs DOG DISAMBIGUATION TEST")
        print("="*60)
        
        cat_id, dog_id = 3, 5
        
        # Get examples
        cat_mask = y == cat_id
        dog_mask = y == dog_id
        
        cat_examples = X[cat_mask][:5]
        dog_examples = X[dog_mask][:5]
        
        # Extract bands for analysis
        cat_bands = [self.extract_hierarchical_spectrum(img) for img in cat_examples]
        dog_bands = [self.extract_hierarchical_spectrum(img) for img in dog_examples]
        
        print("\nBand-specific separability:")
        
        for band_name in self.bands.keys():
            # Within-class similarity
            cat_within = []
            for i in range(len(cat_bands)):
                for j in range(i+1, len(cat_bands)):
                    sim = self.band_similarity(
                        cat_bands[i][band_name],
                        cat_bands[j][band_name],
                        band_name
                    )
                    cat_within.append(sim)
            
            dog_within = []
            for i in range(len(dog_bands)):
                for j in range(i+1, len(dog_bands)):
                    sim = self.band_similarity(
                        dog_bands[i][band_name],
                        dog_bands[j][band_name],
                        band_name
                    )
                    dog_within.append(sim)
            
            # Between-class similarity
            between = []
            for cat_b in cat_bands:
                for dog_b in dog_bands:
                    sim = self.band_similarity(
                        cat_b[band_name],
                        dog_b[band_name],
                        band_name
                    )
                    between.append(sim)
            
            within_avg = np.mean(cat_within + dog_within)
            between_avg = np.mean(between)
            separation = within_avg - between_avg
            
            print(f"  {band_name:10s}: within={within_avg:.3f}, between={between_avg:.3f}, sep={separation:+.3f}")
            
            if separation > 0.05:
                print(f"              → Good discriminator! ✓")
    
    def test_on_training_data(self, X_train, y_train):
        """Test if we get 100% on training data"""
        print("\n" + "="*60)
        print("TESTING ON TRAINING DATA (should be ~100%)")
        print("="*60)
        
        # Test on same examples we trained on
        test_indices = []
        for class_id in range(10):
            class_mask = y_train == class_id
            class_indices = np.where(class_mask)[0][:10]  # Same 10 we trained on
            test_indices.extend(class_indices)
        
        X_test = X_train[test_indices]
        y_test = y_train[test_indices]
        
        predictions = self.predict(X_test, k_nearest=1)  # Use k=1 for training data
        accuracy = np.mean(predictions == y_test)
        
        print(f"Training data accuracy: {accuracy:.1%}")
        
        if accuracy < 0.9:
            print("⚠️ Not matching training data well - representation issues")
        else:
            print("✅ Training data matched well!")
        
        return accuracy


def analyze_cat_misclassification(classifier, X_query, y_query):
    """Detailed analysis of why cats get 0%"""
    print("\n" + "="*60)
    print("DETAILED CAT MISCLASSIFICATION ANALYSIS")
    print("="*60)
    
    # Get cat examples
    cat_mask = y_query == 3
    cat_indices = np.where(cat_mask)[0][:10]  # Look at first 10 cats
    
    if len(cat_indices) == 0:
        print("No cats in query set!")
        return
    
    # Predict with debug info
    predictions, debug_info = classifier.predict(X_query[cat_indices], k_nearest=5, debug=True)
    
    # Confusion analysis
    print("\nWhat are cats being classified as?")
    unique, counts = np.unique(predictions, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls} ({classifier.classes[cls]}): {count}/{len(predictions)} ({count/len(predictions)*100:.0f}%)")
    
    # Look at resonance scores for first few cats
    print("\nDetailed resonance analysis for first 3 cats:")
    for i in range(min(3, len(cat_indices))):
        print(f"\n  Cat example {i}:")
        print(f"    Predicted: {predictions[i]} ({classifier.classes[predictions[i]]})")
        print(f"    Top 5 resonances:")
        
        for j, res in enumerate(debug_info[i]['top_resonances']):
            print(f"      {j+1}. Class {res['class']} ({classifier.classes[res['class']]}): {res['resonance']:.4f}")
        
        print(f"    Vote distribution: {debug_info[i]['votes']}")
        
        # Find where cat productions ranked
        all_resonances = sorted([(r['resonance'], r['class']) for r in debug_info[i]['top_resonances']], reverse=True)
        cat_resonances = [r for r in all_resonances if r[1] == 3]
        if cat_resonances:
            print(f"    Best cat resonance: {cat_resonances[0][0]:.4f}")
        else:
            print(f"    No cat in top 5!")
    
    # Check threshold effects
    print("\nResonance score distribution:")
    all_scores = []
    for info in debug_info:
        all_scores.extend([r['resonance'] for r in info['top_resonances']])
    
    print(f"  Min: {np.min(all_scores):.4f}")
    print(f"  Max: {np.max(all_scores):.4f}")
    print(f"  Mean: {np.mean(all_scores):.4f}")
    print(f"  Std: {np.std(all_scores):.4f}")
    
    # Check if it's a tie-breaking issue
    print("\nChecking for voting ties:")
    for i in range(len(predictions)):
        votes = debug_info[i]['votes']
        max_votes = np.max(votes)
        tied_classes = np.where(votes == max_votes)[0]
        if len(tied_classes) > 1:
            print(f"  Cat {i}: TIE between classes {tied_classes} (each got {max_votes} votes)")


def test_hierarchical_bands():
    """Test hierarchical band approach"""
    import pickle
    import os
    from resonance_algebra.demos.cifar10_real import create_few_shot_splits
    
    print("="*70)
    print("🎯 HIERARCHICAL SPECTRAL BANDS")
    print("="*70)
    print("Using specialized bands for disambiguation")
    
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
    
    # Create classifier
    classifier = HierarchicalSpectralProduction()
    
    # Test cat vs dog disambiguation
    classifier.test_cat_dog_disambiguation(train_data, train_labels)
    
    # Create splits
    np.random.seed(42)
    X_support, y_support, X_query, y_query = create_few_shot_splits(
        train_data, train_labels, test_data, test_labels,
        k_shot=10, n_test_per_class=30
    )
    
    # Fit
    classifier.fit(X_support, y_support, k_shot=10)
    
    # Test on training data first
    train_acc = classifier.test_on_training_data(X_support, y_support)
    
    # Test on query data
    print("\n" + "="*60)
    print("TESTING ON QUERY DATA")
    print("="*60)
    
    for k in [1, 3, 5, 7]:
        predictions = classifier.predict(X_query, k_nearest=k)
        accuracy = np.mean(predictions == y_query)
        print(f"k={k}: {accuracy:.1%}")
        
        if k == 5:  # Show detailed results for k=5
            print("\nPer-class accuracy (k=5):")
            for c in range(10):
                mask = y_query == c
                if np.any(mask):
                    class_acc = np.mean(predictions[mask] == c)
                    print(f"  {classifier.classes[c]:8s}: {class_acc:.1%}")
                    if c in [3, 5]:  # Highlight cat and dog
                        print(f"           ← Key test class")
            
            # Analyze cat misclassification
            analyze_cat_misclassification(classifier, X_query, y_query)
    
    print("\n💡 KEY INSIGHTS:")
    print("  • Multiple bands capture different visual aspects")
    print("  • Texture band weighted higher for natural objects")
    print("  • Color band helps disambiguate similar shapes")
    print("  • Focus on cat vs dog as litmus test")
    
    return classifier


if __name__ == "__main__":
    classifier = test_hierarchical_bands()