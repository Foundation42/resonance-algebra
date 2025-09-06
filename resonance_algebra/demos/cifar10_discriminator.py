"""
DISCRIMINATOR NEURONS FOR RESONANCE ALGEBRA
Specialized binary classifiers that learn to distinguish specific pairs
When main classifier is uncertain, discriminators provide decisive vote
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class DiscriminatorNeuron:
    """
    A specialized neuron that learns to distinguish between two specific classes
    It learns what spectral differences matter for this specific discrimination
    """
    
    def __init__(self, class_a, class_b, name_a="A", name_b="B"):
        self.class_a = class_a
        self.class_b = class_b
        self.name_a = name_a
        self.name_b = name_b
        
        # Will learn which frequencies discriminate best
        self.discriminative_bands = []
        self.production_a = None  # Spectral pattern that means "class A"
        self.production_b = None  # Spectral pattern that means "class B"
        
    def extract_spectrum(self, image):
        """Extract frequency spectrum"""
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        gray = np.mean(image, axis=2)
        gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-8)
        
        window = np.outer(np.hanning(32), np.hanning(32))
        windowed = gray * window
        
        fft = np.fft.fft2(windowed, s=(32, 32))
        fft_shifted = np.fft.fftshift(fft)
        
        return fft_shifted
    
    def learn_discrimination(self, X_a, X_b):
        """
        Learn what spectral patterns distinguish A from B
        This is the "training" but it's just finding differences, not gradient descent
        """
        print(f"  Learning {self.name_a} vs {self.name_b} discrimination...")
        
        # Extract spectra for both classes
        spectra_a = [self.extract_spectrum(img) for img in X_a]
        spectra_b = [self.extract_spectrum(img) for img in X_b]
        
        # Find frequencies with maximum separation
        h, w = spectra_a[0].shape
        center_y, center_x = h // 2, w // 2
        
        # Analyze different frequency regions
        best_regions = []
        
        for r_inner in range(0, 15, 3):
            r_outer = r_inner + 5
            
            y, x = np.ogrid[:h, :w]
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = (radius >= r_inner) & (radius < r_outer)
            
            if not np.any(mask):
                continue
            
            # Extract band for all examples
            band_a = [np.abs(s[mask]) for s in spectra_a]
            band_b = [np.abs(s[mask]) for s in spectra_b]
            
            # Compute separation in this band
            mean_a = np.mean(band_a, axis=0)
            mean_b = np.mean(band_b, axis=0)
            
            # Find most discriminative frequencies in this band
            diff = np.abs(mean_a - mean_b)
            separation_score = np.mean(diff)
            
            best_regions.append({
                'r_inner': r_inner,
                'r_outer': r_outer,
                'mask': mask,
                'separation': separation_score,
                'pattern_a': np.mean([s[mask] for s in spectra_a], axis=0),
                'pattern_b': np.mean([s[mask] for s in spectra_b], axis=0)
            })
        
        # Keep top 3 most discriminative bands
        best_regions.sort(key=lambda x: x['separation'], reverse=True)
        self.discriminative_bands = best_regions[:3]
        
        # Create production rules (spectral patterns that indicate each class)
        # These are difference patterns, not averages
        self.production_a = {}
        self.production_b = {}
        
        for i, band in enumerate(self.discriminative_bands):
            # The production is the characteristic pattern for each class
            self.production_a[i] = band['pattern_a']
            self.production_b[i] = band['pattern_b']
            
            print(f"    Band {i}: r={band['r_inner']}-{band['r_outer']}, sep={band['separation']:.3f}")
    
    def discriminate(self, image):
        """
        Apply discrimination: which class does this image belong to?
        Returns (class, confidence)
        """
        spectrum = self.extract_spectrum(image)
        
        score_a = 0
        score_b = 0
        
        for i, band in enumerate(self.discriminative_bands):
            # Extract band from query
            query_band = spectrum[band['mask']]
            
            # Compare to production patterns
            if len(query_band) > 0:
                # Correlation with class A pattern
                if np.std(np.abs(query_band)) > 0 and np.std(np.abs(self.production_a[i])) > 0:
                    corr_a = np.corrcoef(np.abs(query_band), np.abs(self.production_a[i]))[0, 1]
                else:
                    corr_a = 0
                
                # Correlation with class B pattern  
                if np.std(np.abs(query_band)) > 0 and np.std(np.abs(self.production_b[i])) > 0:
                    corr_b = np.corrcoef(np.abs(query_band), np.abs(self.production_b[i]))[0, 1]
                else:
                    corr_b = 0
                
                # Weight by band importance
                weight = band['separation']
                score_a += corr_a * weight
                score_b += corr_b * weight
        
        # Normalize scores
        total = score_a + score_b
        if total > 0:
            confidence = abs(score_a - score_b) / total
        else:
            confidence = 0
        
        if score_a > score_b:
            return self.class_a, confidence
        else:
            return self.class_b, confidence


class ResonanceWithDiscriminators:
    """
    Main classifier with specialized discriminator neurons
    When confused, asks specialists for help
    """
    
    def __init__(self):
        self.productions = []
        self.discriminators = {}
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    def extract_simple_spectrum(self, image):
        """Simple global spectrum for main classifier"""
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        gray = np.mean(image, axis=2)
        gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-8)
        
        window = np.outer(np.hanning(32), np.hanning(32))
        windowed = gray * window
        
        fft = np.fft.fft2(windowed, s=(32, 32))
        fft_shifted = np.fft.fftshift(fft)
        
        # Simple features
        magnitude = np.abs(fft_shifted)
        features = []
        
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        for r in [4, 8, 12, 16]:
            y, x = np.ogrid[:h, :w]
            mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) < r
            features.append(np.mean(magnitude[mask]))
            features.append(np.std(magnitude[mask]))
        
        return np.array(features)
    
    def fit(self, X, y, k_shot=10):
        """
        Fit main classifier and create discriminators for confusable pairs
        """
        print("\n📚 Training resonance classifier with discriminators...")
        
        # First, create main productions
        self.productions = []
        
        for class_id in range(10):
            class_mask = y == class_id
            class_examples = X[class_mask][:k_shot]
            
            for img in class_examples:
                spectrum = self.extract_simple_spectrum(img)
                self.productions.append({
                    'spectrum': spectrum,
                    'class': class_id
                })
        
        print(f"  Created {len(self.productions)} main productions")
        
        # Identify confusable pairs by testing on training data
        print("\n🔍 Identifying confusable pairs...")
        
        confusion_matrix = np.zeros((10, 10))
        
        for class_id in range(10):
            class_mask = y == class_id
            class_examples = X[class_mask][k_shot:k_shot+5]  # Different examples
            
            for img in class_examples:
                # Simple prediction
                query_spectrum = self.extract_simple_spectrum(img)
                
                best_score = -1
                best_class = -1
                
                for prod in self.productions:
                    if np.std(query_spectrum) > 0 and np.std(prod['spectrum']) > 0:
                        score = np.corrcoef(query_spectrum, prod['spectrum'])[0, 1]
                        if score > best_score:
                            best_score = score
                            best_class = prod['class']
                
                if best_class >= 0:
                    confusion_matrix[class_id, best_class] += 1
        
        # Create discriminators for top confusable pairs
        print("\n🧠 Creating specialized discriminators...")
        
        # Always create discriminators for known confusable pairs
        forced_pairs = [
            (3, 5),  # cat vs dog
            (3, 4),  # cat vs deer  
            (5, 7),  # dog vs horse
            (0, 1),  # plane vs car
            (0, 8),  # plane vs ship
            (1, 9),  # car vs truck
        ]
        
        threshold = 1  # Lower threshold
        
        for i in range(10):
            for j in range(i+1, 10):
                confusions = confusion_matrix[i, j] + confusion_matrix[j, i]
                
                # Force creation for known pairs or if threshold met
                if (i, j) in forced_pairs or confusions >= threshold:
                    print(f"\n  {self.classes[i]} ↔ {self.classes[j]}: {confusions:.0f} confusions")
                    
                    # Create discriminator
                    discriminator = DiscriminatorNeuron(
                        i, j, self.classes[i], self.classes[j]
                    )
                    
                    # Train it on examples
                    mask_i = y == i
                    mask_j = y == j
                    
                    X_i = X[mask_i][:k_shot]
                    X_j = X[mask_j][:k_shot]
                    
                    discriminator.learn_discrimination(X_i, X_j)
                    
                    # Store discriminator
                    self.discriminators[(i, j)] = discriminator
        
        print(f"\n✓ Created {len(self.discriminators)} discriminator neurons")
    
    def predict(self, X, use_discriminators=True):
        """
        Predict with option to use discriminators for close calls
        """
        predictions = []
        
        for img in X:
            query_spectrum = self.extract_simple_spectrum(img)
            
            # Get scores for all classes
            class_scores = np.zeros(10)
            class_counts = np.zeros(10)
            
            for prod in self.productions:
                if np.std(query_spectrum) > 0 and np.std(prod['spectrum']) > 0:
                    score = np.corrcoef(query_spectrum, prod['spectrum'])[0, 1]
                    class_scores[prod['class']] += score
                    class_counts[prod['class']] += 1
            
            # Average scores
            for c in range(10):
                if class_counts[c] > 0:
                    class_scores[c] /= class_counts[c]
            
            # Find top 2 classes
            sorted_indices = np.argsort(class_scores)[::-1]
            best_class = sorted_indices[0]
            second_class = sorted_indices[1]
            
            # Check if we need discrimination
            if use_discriminators:
                score_diff = class_scores[best_class] - class_scores[second_class]
                
                # If close call, use discriminator
                if score_diff < 0.1:  # Threshold for "uncertain"
                    # Check if we have a discriminator for this pair
                    pair = tuple(sorted([best_class, second_class]))
                    
                    if pair in self.discriminators:
                        # Ask the specialist!
                        specialist = self.discriminators[pair]
                        decision, confidence = specialist.discriminate(img)
                        
                        if confidence > 0.2:  # Trust the specialist if confident
                            best_class = decision
            
            predictions.append(best_class)
        
        return np.array(predictions)


def test_discriminator_neurons():
    """Test the discriminator neuron approach"""
    import pickle
    import os
    from resonance_algebra.demos.cifar10_real import create_few_shot_splits
    
    print("="*70)
    print("🧠 DISCRIMINATOR NEURONS FOR RESONANCE ALGEBRA")
    print("="*70)
    print("Specialized binary classifiers for confusable pairs")
    
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
    
    # Create splits
    np.random.seed(42)
    X_support, y_support, X_query, y_query = create_few_shot_splits(
        train_data, train_labels, test_data, test_labels,
        k_shot=10, n_test_per_class=30
    )
    
    # Create classifier with discriminators
    classifier = ResonanceWithDiscriminators()
    classifier.fit(X_support, y_support, k_shot=10)
    
    # Test without discriminators
    print("\n" + "="*60)
    print("TESTING WITHOUT DISCRIMINATORS")
    print("="*60)
    
    predictions_without = classifier.predict(X_query, use_discriminators=False)
    accuracy_without = np.mean(predictions_without == y_query)
    print(f"Overall accuracy: {accuracy_without:.1%}")
    
    # Test with discriminators
    print("\n" + "="*60)
    print("TESTING WITH DISCRIMINATORS")
    print("="*60)
    
    predictions_with = classifier.predict(X_query, use_discriminators=True)
    accuracy_with = np.mean(predictions_with == y_query)
    print(f"Overall accuracy: {accuracy_with:.1%}")
    
    print("\nPer-class accuracy:")
    for c in range(10):
        mask = y_query == c
        if np.any(mask):
            acc_without = np.mean(predictions_without[mask] == c)
            acc_with = np.mean(predictions_with[mask] == c)
            improvement = acc_with - acc_without
            
            print(f"  {classifier.classes[c]:8s}: {acc_without:.1%} → {acc_with:.1%}", end='')
            
            if improvement > 0.05:
                print(f" (+{improvement:.1%}) ✓")
            elif improvement < -0.05:
                print(f" ({improvement:.1%})")
            else:
                print()
    
    # Analyze cat-dog specifically
    cat_mask = y_query == 3
    dog_mask = y_query == 5
    
    if np.any(cat_mask) and np.any(dog_mask):
        print("\n🎯 Cat-Dog Analysis:")
        
        # Without discriminators
        cat_acc_without = np.mean(predictions_without[cat_mask] == 3)
        dog_acc_without = np.mean(predictions_without[dog_mask] == 5)
        
        # With discriminators
        cat_acc_with = np.mean(predictions_with[cat_mask] == 3)
        dog_acc_with = np.mean(predictions_with[dog_mask] == 5)
        
        print(f"  Cats: {cat_acc_without:.1%} → {cat_acc_with:.1%}")
        print(f"  Dogs: {dog_acc_without:.1%} → {dog_acc_with:.1%}")
        
        if (3, 5) in classifier.discriminators or (5, 3) in classifier.discriminators:
            print("  ✓ Cat-Dog discriminator is active!")
    
    print("\n💡 KEY INSIGHTS:")
    print("  • Discriminators specialize in specific confusions")
    print("  • They learn which frequencies matter for their pair")
    print("  • Main classifier defers to specialists when uncertain")
    print("  • Production rules + discriminators = adaptive system")
    
    return classifier


if __name__ == "__main__":
    classifier = test_discriminator_neurons()