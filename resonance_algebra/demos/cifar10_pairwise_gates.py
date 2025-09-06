"""
PAIRWISE GATE LEARNING - Learn discriminators from actual confusions
Gates are trained on specific pairs that need disambiguation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


class PairwiseGate:
    """
    A gate that learns to distinguish between two specific groups
    Trained on actual examples from both sides
    """
    
    def __init__(self, group_a, group_b, name_a="A", name_b="B"):
        self.group_a = group_a
        self.group_b = group_b
        self.name_a = name_a
        self.name_b = name_b
        
        # Will learn discriminative features
        self.discriminative_features = None
        self.threshold = 0
        
    def learn_discrimination(self, examples_a, examples_b):
        """
        Learn what features best separate these groups
        Find the actual differences, not averages
        """
        # Extract multiple feature types
        all_features_a = []
        all_features_b = []
        
        for img in examples_a:
            features = self.extract_multi_scale_features(img)
            all_features_a.append(features)
        
        for img in examples_b:
            features = self.extract_multi_scale_features(img)
            all_features_b.append(features)
        
        all_features_a = np.array(all_features_a)
        all_features_b = np.array(all_features_b)
        
        # Find most discriminative features (maximum margin)
        n_features = all_features_a.shape[1]
        discriminability = np.zeros(n_features)
        
        for i in range(n_features):
            # Feature values for both groups
            vals_a = all_features_a[:, i]
            vals_b = all_features_b[:, i]
            
            # Separation = difference in means / sum of stds
            mean_diff = abs(np.mean(vals_a) - np.mean(vals_b))
            std_sum = np.std(vals_a) + np.std(vals_b) + 1e-8
            
            discriminability[i] = mean_diff / std_sum
        
        # Keep top discriminative features
        top_indices = np.argsort(discriminability)[-10:]  # Top 10 features
        
        self.discriminative_features = top_indices
        
        # Learn threshold for these features
        reduced_a = all_features_a[:, top_indices]
        reduced_b = all_features_b[:, top_indices]
        
        # Simple linear discriminant
        mean_a = np.mean(reduced_a, axis=0)
        mean_b = np.mean(reduced_b, axis=0)
        
        self.decision_vector = mean_a - mean_b
        self.threshold = np.dot(self.decision_vector, (mean_a + mean_b) / 2)
        
        # Report discrimination quality
        separation = discriminability[top_indices].mean()
        return separation
    
    def extract_multi_scale_features(self, image):
        """Extract features at multiple scales and types"""
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        features = []
        gray = np.mean(image, axis=2)
        
        # 1. Global features
        features.append(np.mean(gray))
        features.append(np.std(gray))
        
        # 2. Frequency features (different scales)
        for size in [8, 16]:
            fft = np.fft.fft2(gray, s=(size, size))
            fft_shifted = np.fft.fftshift(fft)
            
            # Low, mid, high frequency energy
            center = size // 2
            low_freq = np.abs(fft_shifted[center-2:center+2, center-2:center+2])
            features.append(np.mean(low_freq))
            
            mid_freq = np.abs(fft_shifted[center-4:center+4, center-4:center+4])
            features.append(np.mean(mid_freq) - np.mean(low_freq))
            
        # 3. Edge features
        edges_h = ndimage.sobel(gray, axis=0)
        edges_v = ndimage.sobel(gray, axis=1)
        
        features.append(np.mean(np.abs(edges_h)))
        features.append(np.mean(np.abs(edges_v)))
        features.append(np.std(edges_h))
        features.append(np.std(edges_v))
        
        # 4. Texture features (different regions)
        h, w = gray.shape
        regions = [
            gray[:h//2, :],      # top
            gray[h//2:, :],      # bottom
            gray[:, :w//2],      # left
            gray[:, w//2:],      # right
            gray[h//4:3*h//4, w//4:3*w//4]  # center
        ]
        
        for region in regions:
            features.append(np.var(region))
        
        # 5. Color features
        if image.ndim == 3:
            features.append(np.mean(image[:, :, 0]))  # R
            features.append(np.mean(image[:, :, 1]))  # G
            features.append(np.mean(image[:, :, 2]))  # B
            features.append(np.mean(image[:, :, 0]) - np.mean(image[:, :, 1]))  # R-G
            features.append(np.mean(image[:, :, 2]) - np.mean(image[:, :, 1]))  # B-G
        
        return np.array(features)
    
    def decide(self, image):
        """Make binary decision for this gate"""
        features = self.extract_multi_scale_features(image)
        
        if self.discriminative_features is not None:
            reduced = features[self.discriminative_features]
            score = np.dot(reduced, self.decision_vector)
            
            if score > self.threshold:
                return self.name_a, abs(score - self.threshold)
            else:
                return self.name_b, abs(score - self.threshold)
        
        return self.name_a, 0  # Default


class AdaptiveSemanticBVH:
    """
    Semantic hierarchy with pairwise-learned gates
    Gates are trained on actual confusion pairs
    """
    
    def __init__(self):
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Hierarchy (can be adapted based on confusions)
        self.hierarchy = {
            'root': {
                'geometric': [0, 1, 8, 9],  # plane, car, ship, truck
                'organic': [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
            },
            'geometric': {
                'flying': [0],  # plane
                'ground': [1, 9],  # car, truck
                'water': [8]  # ship
            },
            'organic': {
                'mammals': [3, 4, 5, 7],  # cat, deer, dog, horse
                'non_mammals': [2, 6]  # bird, frog
            },
            'mammals': {
                'pets': [3, 5],  # cat, dog
                'wild': [4, 7]  # deer, horse
            }
        }
        
        self.gates = {}
        self.productions = {}
        
        print("🌲 Adaptive Semantic BVH initialized")
        print("  Gates learn from pairwise discrimination")
    
    def identify_confusion_pairs(self, X, y):
        """
        Test current system to find which pairs are confused
        This tells us which gates need to be strongest
        """
        print("\n🔍 Identifying confusion pairs...")
        
        confusion_matrix = np.zeros((10, 10))
        
        # Simple test to find confusions
        for class_id in range(10):
            class_mask = y == class_id
            class_examples = X[class_mask][10:15]  # Use different examples
            
            for img in class_examples:
                # Quick classification using basic features
                gray = np.mean(img, axis=2) if img.ndim == 3 else img
                fft = np.fft.fft2(gray, s=(8, 8))
                features = np.abs(fft).flatten()[:20]
                
                # Find nearest class
                best_match = -1
                best_score = -np.inf
                
                for other_class in range(10):
                    if other_class == class_id:
                        continue
                    
                    other_mask = y == other_class
                    other_example = X[other_mask][0]  # Just one example
                    
                    other_gray = np.mean(other_example, axis=2) if other_example.ndim == 3 else other_example
                    other_fft = np.fft.fft2(other_gray, s=(8, 8))
                    other_features = np.abs(other_fft).flatten()[:20]
                    
                    if np.std(features) > 0 and np.std(other_features) > 0:
                        score = np.corrcoef(features, other_features)[0, 1]
                        if score > best_score:
                            best_score = score
                            best_match = other_class
                
                if best_match >= 0:
                    confusion_matrix[class_id, best_match] += 1
        
        # Find top confusion pairs
        confusion_pairs = []
        for i in range(10):
            for j in range(i+1, 10):
                total_confusion = confusion_matrix[i, j] + confusion_matrix[j, i]
                if total_confusion > 0:
                    confusion_pairs.append((i, j, total_confusion))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"  Top confusion pairs:")
        for i, j, count in confusion_pairs[:5]:
            print(f"    {self.classes[i]} ↔ {self.classes[j]}: {count:.0f} confusions")
        
        return confusion_pairs
    
    def train_pairwise_gates(self, X, y):
        """
        Train gates using pairwise discrimination
        Focus on actual confusion pairs
        """
        print("\n🎯 Training pairwise gates...")
        
        # First identify what needs discrimination
        confusion_pairs = self.identify_confusion_pairs(X, y)
        
        # Train hierarchical gates
        for level_name, groups in [('root', self.hierarchy['root']),
                                   ('geometric', self.hierarchy.get('geometric', {})),
                                   ('organic', self.hierarchy.get('organic', {})),
                                   ('mammals', self.hierarchy.get('mammals', {}))]:
            
            if not groups:
                continue
            
            group_names = list(groups.keys())
            if len(group_names) >= 2:
                print(f"\n  Level: {level_name}")
                
                # For each pair of groups at this level
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        group_a_name = group_names[i]
                        group_b_name = group_names[j]
                        
                        group_a_classes = groups[group_a_name]
                        group_b_classes = groups[group_b_name]
                        
                        # Get examples from each group
                        examples_a = []
                        examples_b = []
                        
                        for class_id in group_a_classes:
                            mask = y == class_id
                            examples_a.extend(X[mask][:5])
                        
                        for class_id in group_b_classes:
                            mask = y == class_id
                            examples_b.extend(X[mask][:5])
                        
                        if examples_a and examples_b:
                            # Train pairwise gate
                            gate = PairwiseGate(
                                group_a_classes, group_b_classes,
                                group_a_name, group_b_name
                            )
                            
                            separation = gate.learn_discrimination(examples_a, examples_b)
                            
                            gate_name = f"{level_name}_{group_a_name}_vs_{group_b_name}"
                            self.gates[gate_name] = gate
                            
                            print(f"    {group_a_name} vs {group_b_name}: separation={separation:.3f}")
        
        # Train specific discriminators for top confusion pairs
        print("\n  Specific discriminators for confused pairs:")
        
        for class_i, class_j, confusion_count in confusion_pairs[:3]:  # Top 3 confused pairs
            mask_i = y == class_i
            mask_j = y == class_j
            
            examples_i = X[mask_i][:10]
            examples_j = X[mask_j][:10]
            
            gate = PairwiseGate(
                [class_i], [class_j],
                self.classes[class_i], self.classes[class_j]
            )
            
            separation = gate.learn_discrimination(examples_i, examples_j)
            
            gate_name = f"specific_{class_i}_vs_{class_j}"
            self.gates[gate_name] = gate
            
            print(f"    {self.classes[class_i]} vs {self.classes[class_j]}: separation={separation:.3f}")
    
    def fit(self, X, y, k_shot=10):
        """Train the full system with pairwise gates"""
        print("\n🌲 Building Adaptive Semantic BVH...")
        
        # Train pairwise gates
        self.train_pairwise_gates(X, y)
        
        # Store productions for final classification
        print("\n📚 Storing production rules...")
        for class_id in range(10):
            class_mask = y == class_id
            class_examples = X[class_mask][:k_shot]
            
            self.productions[class_id] = []
            for img in class_examples:
                gray = np.mean(img, axis=2) if img.ndim == 3 else img
                fft = np.fft.fft2(gray, s=(16, 16))
                features = np.concatenate([
                    np.abs(fft).flatten()[:30],
                    np.angle(fft).flatten()[:20]
                ])
                self.productions[class_id].append(features)
        
        print("✓ Adaptive BVH trained with pairwise gates")
    
    def predict(self, X):
        """Classify using learned gates"""
        predictions = []
        
        for img in X:
            # Navigate hierarchy using gates
            
            # Level 1: Geometric vs Organic
            if 'root_geometric_vs_organic' in self.gates:
                decision, conf = self.gates['root_geometric_vs_organic'].decide(img)
                
                if decision == 'geometric':
                    candidates = [0, 1, 8, 9]
                else:
                    candidates = [2, 3, 4, 5, 6, 7]
            else:
                candidates = list(range(10))
            
            # Further refinement with specific discriminators
            # Check if we have discriminators for candidate pairs
            if len(candidates) == 2:
                i, j = candidates
                gate_name = f"specific_{min(i,j)}_vs_{max(i,j)}"
                if gate_name in self.gates:
                    decision, conf = self.gates[gate_name].decide(img)
                    # Map decision back to class
                    if decision == self.classes[i]:
                        predictions.append(i)
                    else:
                        predictions.append(j)
                    continue
            
            # Final classification among candidates
            gray = np.mean(img, axis=2) if img.ndim == 3 else img
            fft = np.fft.fft2(gray, s=(16, 16))
            features = np.concatenate([
                np.abs(fft).flatten()[:30],
                np.angle(fft).flatten()[:20]
            ])
            
            best_class = candidates[0] if candidates else 0
            best_score = -np.inf
            
            for class_id in candidates:
                if class_id in self.productions:
                    for prod_features in self.productions[class_id]:
                        if len(features) == len(prod_features):
                            if np.std(features) > 0 and np.std(prod_features) > 0:
                                score = np.corrcoef(features, prod_features)[0, 1]
                                if score > best_score:
                                    best_score = score
                                    best_class = class_id
            
            predictions.append(best_class)
        
        return np.array(predictions)


def test_pairwise_gates():
    """Test pairwise gate learning"""
    import pickle
    import os
    from resonance_algebra.demos.cifar10_real import create_few_shot_splits
    
    print("="*70)
    print("🎯 PAIRWISE GATE LEARNING")
    print("="*70)
    print("Gates learn from actual confusion pairs, not group averages")
    
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
    classifier = AdaptiveSemanticBVH()
    
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
    print("TESTING WITH PAIRWISE GATES")
    print("="*60)
    
    predictions = classifier.predict(X_query)
    accuracy = np.mean(predictions == y_query)
    print(f"Overall accuracy: {accuracy:.1%}")
    
    print("\nPer-class accuracy:")
    for c in range(10):
        mask = y_query == c
        if np.any(mask):
            class_acc = np.mean(predictions[mask] == c)
            print(f"  {classifier.classes[c]:8s}: {class_acc:.1%}")
    
    # Focus on cats and dogs
    cat_mask = y_query == 3
    dog_mask = y_query == 5
    
    if np.any(cat_mask) and np.any(dog_mask):
        cat_acc = np.mean(predictions[cat_mask] == 3)
        dog_acc = np.mean(predictions[dog_mask] == 5)
        
        print(f"\n🎯 Cat-Dog Analysis:")
        print(f"  Cat accuracy: {cat_acc:.1%}")
        print(f"  Dog accuracy: {dog_acc:.1%}")
        
        # Check specific gate performance
        if 'specific_3_vs_5' in classifier.gates:
            print("  ✓ Cat-Dog specific discriminator is active!")
    
    print("\n💡 KEY INSIGHTS:")
    print("  • Gates trained on actual confusion pairs")
    print("  • Maximum margin discrimination")
    print("  • Adaptive to data-specific confusions")
    print("  • Hierarchical + specific discriminators")
    
    return classifier


if __name__ == "__main__":
    classifier = test_pairwise_gates()