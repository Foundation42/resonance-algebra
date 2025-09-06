"""
SPATIAL SPECTRAL PRODUCTION - Local patch analysis
Analyze different image regions to capture fine-grained details
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SpatialSpectralProduction:
    """
    Spatial sampling: Extract spectral signatures from local patches
    This captures details that get lost in global frequency analysis
    """
    
    def __init__(self):
        self.productions = []
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Define patch locations (overlapping for better coverage)
        self.patches = [
            # Center patches (most important)
            {'name': 'center', 'x': 8, 'y': 8, 'size': 16, 'weight': 1.5},
            
            # Quadrants (capture different parts)
            {'name': 'top_left', 'x': 0, 'y': 0, 'size': 16, 'weight': 1.0},
            {'name': 'top_right', 'x': 16, 'y': 0, 'size': 16, 'weight': 1.0},
            {'name': 'bottom_left', 'x': 0, 'y': 16, 'size': 16, 'weight': 1.0},
            {'name': 'bottom_right', 'x': 16, 'y': 16, 'size': 16, 'weight': 1.0},
            
            # Feature-specific regions
            {'name': 'top_center', 'x': 8, 'y': 0, 'size': 16, 'weight': 1.2},  # Head/ears
            {'name': 'middle_strip', 'x': 0, 'y': 12, 'size': 32, 'weight': 0.8},  # Body shape
        ]
        
        print("🔍 Spatial spectral production initialized")
        print(f"  {len(self.patches)} overlapping patches for local analysis")
    
    def extract_patch_spectrum(self, image, patch):
        """Extract spectrum from a specific image patch"""
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # Extract patch (handle boundaries)
        x, y = patch['x'], patch['y']
        size = patch['size']
        
        # Ensure we don't go out of bounds
        x_end = min(x + size, 32)
        y_end = min(y + size, 32)
        
        patch_img = image[y:y_end, x:x_end]
        
        # If patch is too small, pad it
        if patch_img.shape[0] < size or patch_img.shape[1] < size:
            padded = np.zeros((size, size, 3))
            padded[:patch_img.shape[0], :patch_img.shape[1]] = patch_img
            patch_img = padded
        
        # Get grayscale for simplicity
        gray = np.mean(patch_img, axis=2)
        
        # Normalize
        if np.std(gray) > 0:
            gray = (gray - np.mean(gray)) / np.std(gray)
        
        # Window to reduce artifacts
        window_size = min(gray.shape[0], gray.shape[1])
        window = np.outer(np.hanning(window_size), np.hanning(window_size))
        
        # Resize window if needed
        if window.shape != gray.shape:
            window = np.ones_like(gray)
        
        windowed = gray * window
        
        # FFT (smaller for patches)
        fft = np.fft.fft2(windowed, s=(16, 16))
        fft_shifted = np.fft.fftshift(fft)
        
        # Extract features
        features = []
        
        # Low frequencies (shape)
        low_freq = fft_shifted[6:10, 6:10].flatten()
        features.extend([np.abs(low_freq), np.angle(low_freq)])
        
        # Mid frequencies (texture)
        mid_freq = fft_shifted[4:6, 4:12].flatten()
        features.extend([np.abs(mid_freq), np.angle(mid_freq)])
        
        # Flatten and limit size
        features = np.concatenate(features)[:50]
        
        return features
    
    def extract_spatial_features(self, image):
        """Extract features from all patches"""
        patch_features = {}
        
        for patch in self.patches:
            features = self.extract_patch_spectrum(image, patch)
            patch_features[patch['name']] = features
        
        return patch_features
    
    def compute_patch_similarity(self, features1, features2):
        """Compare two patch feature sets"""
        if len(features1) != len(features2):
            # Pad shorter one
            max_len = max(len(features1), len(features2))
            f1 = np.pad(features1, (0, max_len - len(features1)))
            f2 = np.pad(features2, (0, max_len - len(features2)))
        else:
            f1, f2 = features1, features2
        
        # Correlation-based similarity
        if np.std(f1) > 0 and np.std(f2) > 0:
            return np.corrcoef(f1, f2)[0, 1]
        return 0
    
    def compute_spatial_resonance(self, query_patches, prod_patches):
        """Compute weighted resonance across all patches"""
        total_resonance = 0
        total_weight = 0
        
        patch_scores = []
        
        for patch in self.patches:
            name = patch['name']
            if name in query_patches and name in prod_patches:
                sim = self.compute_patch_similarity(
                    query_patches[name],
                    prod_patches[name]
                )
                
                weight = patch['weight']
                total_resonance += sim * weight
                total_weight += weight
                
                patch_scores.append((name, sim))
        
        if total_weight > 0:
            return total_resonance / total_weight, patch_scores
        return 0, []
    
    def fit(self, X, y, k_shot=10):
        """Create spatial production rules"""
        self.productions = []
        
        print(f"\n📚 Creating spatial production rules...")
        
        for class_id in range(10):
            class_mask = y == class_id
            class_examples = X[class_mask][:k_shot]
            
            print(f"  {self.classes[class_id]:8s}: ", end='')
            
            for img in class_examples:
                patches = self.extract_spatial_features(img)
                self.productions.append({
                    'patches': patches,
                    'class': class_id
                })
            
            print(f"{len(class_examples)} rules")
        
        print(f"\n📊 Total: {len(self.productions)} spatial production rules")
    
    def predict(self, X, k_nearest=5, debug=False):
        """Predict using spatial patch resonance"""
        predictions = []
        debug_info = []
        
        for img in X:
            query_patches = self.extract_spatial_features(img)
            
            # Compute resonance with all productions
            resonances = []
            for prod in self.productions:
                resonance, patch_scores = self.compute_spatial_resonance(
                    query_patches, prod['patches']
                )
                resonances.append({
                    'resonance': resonance,
                    'class': prod['class'],
                    'patch_scores': patch_scores
                })
            
            # Sort by resonance
            resonances.sort(key=lambda x: x['resonance'], reverse=True)
            
            # Weighted voting
            votes = np.zeros(10)
            for i in range(min(k_nearest, len(resonances))):
                if resonances[i]['resonance'] > 0:
                    votes[resonances[i]['class']] += resonances[i]['resonance']
            
            # Add tiny noise to break ties
            votes += np.random.randn(10) * 1e-8
            
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
    
    def analyze_cat_dog_patches(self, X, y):
        """Analyze which patches best distinguish cats from dogs"""
        print("\n" + "="*60)
        print("CAT vs DOG PATCH ANALYSIS")
        print("="*60)
        
        cat_id, dog_id = 3, 5
        
        # Get examples
        cat_mask = y == cat_id
        dog_mask = y == dog_id
        
        cat_examples = X[cat_mask][:5]
        dog_examples = X[dog_mask][:5]
        
        print("\nPatch-specific discrimination:")
        
        for patch in self.patches:
            name = patch['name']
            
            # Extract patch features
            cat_features = [self.extract_patch_spectrum(img, patch) for img in cat_examples]
            dog_features = [self.extract_patch_spectrum(img, patch) for img in dog_examples]
            
            # Within-class similarity
            cat_within = []
            for i in range(len(cat_features)):
                for j in range(i+1, len(cat_features)):
                    sim = self.compute_patch_similarity(cat_features[i], cat_features[j])
                    cat_within.append(sim)
            
            dog_within = []
            for i in range(len(dog_features)):
                for j in range(i+1, len(dog_features)):
                    sim = self.compute_patch_similarity(dog_features[i], dog_features[j])
                    dog_within.append(sim)
            
            # Between-class similarity
            between = []
            for cf in cat_features:
                for df in dog_features:
                    sim = self.compute_patch_similarity(cf, df)
                    between.append(sim)
            
            within_avg = np.mean(cat_within + dog_within) if (cat_within + dog_within) else 0
            between_avg = np.mean(between) if between else 0
            separation = within_avg - between_avg
            
            print(f"  {name:15s}: sep={separation:+.3f}", end='')
            
            if separation > 0.05:
                print(" ✓ Good discriminator!")
            elif separation > 0:
                print(" → Some discrimination")
            else:
                print()


def test_spatial_production():
    """Test spatial patch-based approach"""
    import pickle
    import os
    from resonance_algebra.demos.cifar10_real import create_few_shot_splits
    
    print("="*70)
    print("🔍 SPATIAL SPECTRAL PRODUCTION")
    print("="*70)
    print("Analyzing local patches for fine-grained details")
    
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
    classifier = SpatialSpectralProduction()
    
    # Analyze cat vs dog patches
    classifier.analyze_cat_dog_patches(train_data, train_labels)
    
    # Create splits
    np.random.seed(42)
    X_support, y_support, X_query, y_query = create_few_shot_splits(
        train_data, train_labels, test_data, test_labels,
        k_shot=10, n_test_per_class=30
    )
    
    # Fit
    classifier.fit(X_support, y_support, k_shot=10)
    
    # Test
    print("\n" + "="*60)
    print("TESTING SPATIAL APPROACH")
    print("="*60)
    
    for k in [3, 5, 7]:
        predictions = classifier.predict(X_query, k_nearest=k)
        accuracy = np.mean(predictions == y_query)
        print(f"k={k}: {accuracy:.1%}")
        
        if k == 5:
            print("\nPer-class accuracy (k=5):")
            for c in range(10):
                mask = y_query == c
                if np.any(mask):
                    class_acc = np.mean(predictions[mask] == c)
                    print(f"  {classifier.classes[c]:8s}: {class_acc:.1%}")
                    
            # Special focus on cats and dogs
            cat_mask = y_query == 3
            dog_mask = y_query == 5
            
            if np.any(cat_mask) and np.any(dog_mask):
                cat_acc = np.mean(predictions[cat_mask] == 3)
                dog_acc = np.mean(predictions[dog_mask] == 5)
                
                print(f"\n🎯 Cat-Dog Analysis:")
                print(f"  Cat accuracy: {cat_acc:.1%}")
                print(f"  Dog accuracy: {dog_acc:.1%}")
                print(f"  Difference: {abs(dog_acc - cat_acc):.1%}")
                
                # Check what cats are misclassified as
                cat_preds = predictions[cat_mask]
                unique, counts = np.unique(cat_preds, return_counts=True)
                print(f"\n  Cats classified as:")
                for cls, count in zip(unique, counts):
                    if count > 0:
                        print(f"    {classifier.classes[cls]}: {count}/{len(cat_preds)} ({count/len(cat_preds)*100:.0f}%)")
    
    print("\n💡 KEY INSIGHTS:")
    print("  • Local patches capture details lost in global analysis")
    print("  • Top center patch (ears/head) may distinguish cats vs dogs")
    print("  • Overlapping patches provide multiple views")
    print("  • Spatial hierarchy emerging naturally")
    
    return classifier


if __name__ == "__main__":
    classifier = test_spatial_production()