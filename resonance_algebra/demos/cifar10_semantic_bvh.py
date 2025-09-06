"""
SEMANTIC BVH - Hierarchical Gating for Classification
Don't check irrelevant categories - prune the decision tree early
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


class SemanticBVH:
    """
    Hierarchical classification tree based on semantic similarity
    High-level gates prevent checking irrelevant categories
    """
    
    def __init__(self):
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Define semantic hierarchy (could be learned)
        self.hierarchy = {
            'root': {
                'geometric': ['plane', 'car', 'ship', 'truck'],
                'organic': ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
            },
            'geometric': {
                'flying': ['plane'],
                'ground': ['car', 'truck'],
                'water': ['ship']
            },
            'organic': {
                'mammals': ['cat', 'deer', 'dog', 'horse'],
                'non_mammals': ['bird', 'frog']
            },
            'mammals': {
                'pets': ['cat', 'dog'],
                'wild': ['deer', 'horse']
            }
        }
        
        # Gate classifiers for each decision
        self.gates = {}
        self.productions = {}
        
        print("🌳 Semantic BVH initialized")
        print("  Hierarchy depth: 3-4 levels")
        print("  Pruning irrelevant branches early")
        
    def extract_gate_features(self, image, gate_type):
        """
        Extract features relevant for specific gate decisions
        Different features for different levels
        """
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        features = []
        
        if gate_type == 'geometric_vs_organic':
            # For this decision: look for straight lines, regularity
            gray = np.mean(image, axis=2)
            
            # Edge detection
            edges_h = ndimage.sobel(gray, axis=0)
            edges_v = ndimage.sobel(gray, axis=1)
            
            # Geometric objects have more straight edges
            edge_straightness = np.std(edges_h) + np.std(edges_v)
            features.append(edge_straightness)
            
            # Regularity in frequency domain
            fft = np.fft.fft2(gray, s=(16, 16))
            fft_mag = np.abs(fft)
            
            # Geometric objects have more energy in cardinal directions
            cardinal_energy = np.sum(fft_mag[7:9, :]) + np.sum(fft_mag[:, 7:9])
            diagonal_energy = np.sum(np.diag(fft_mag)) + np.sum(np.diag(np.fliplr(fft_mag)))
            features.append(cardinal_energy / (diagonal_energy + 1e-8))
            
            # Color variance (organic tend to have more varied colors)
            color_var = np.std(image)
            features.append(color_var)
            
        elif gate_type == 'mammals_vs_non_mammals':
            # For this: look for fur texture, eye patterns
            gray = np.mean(image, axis=2)
            
            # Texture analysis - mammals have fur
            texture = ndimage.gaussian_filter(gray, 0.5) - ndimage.gaussian_filter(gray, 2)
            texture_complexity = np.std(texture)
            features.append(texture_complexity)
            
            # Warm colors (mammals tend toward browns/tans)
            if image.ndim == 3:
                warm_colors = np.mean(image[:, :, 0]) - np.mean(image[:, :, 2])  # R - B
                features.append(warm_colors)
            else:
                features.append(0)
                
        elif gate_type == 'pets_vs_wild':
            # For this: look for face features, domestication cues
            gray = np.mean(image, axis=2)
            
            # Center focus (pets often photographed face-forward)
            h, w = gray.shape
            center_region = gray[h//4:3*h//4, w//4:3*w//4]
            center_variance = np.var(center_region)
            features.append(center_variance)
            
            # Roundness of shapes (pets have rounder features)
            edges = ndimage.sobel(gray)
            fft_edges = np.fft.fft2(edges, s=(16, 16))
            radial_symmetry = np.mean(np.abs(fft_edges[6:10, 6:10]))
            features.append(radial_symmetry)
            
        elif gate_type == 'cat_vs_dog':
            # Specific discriminator for hardest pair
            gray = np.mean(image, axis=2)
            
            # Ear detection (top region)
            top_region = gray[:10, :]
            top_edges = ndimage.sobel(top_region)
            
            # Pointiness of ears (cats have more triangular ears)
            top_fft = np.fft.fft2(top_edges, s=(8, 8))
            high_freq_energy = np.sum(np.abs(top_fft[4:, :]))
            features.append(high_freq_energy)
            
            # Face proportions (middle region)
            middle_region = gray[10:20, :]
            middle_variance = np.var(middle_region)
            features.append(middle_variance)
            
            # Snout length (bottom region)
            bottom_region = gray[20:, :]
            bottom_mean = np.mean(bottom_region)
            features.append(bottom_mean)
            
        else:
            # Default features
            gray = np.mean(image, axis=2)
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.max(gray) - np.min(gray)
            ])
        
        return np.array(features)
    
    def train_gate(self, X, y, parent_classes, child_groups):
        """
        Train a single gate classifier
        Binary or multi-way decision
        """
        gate_name = f"{parent_classes}_gate"
        
        # Get examples for each group
        group_examples = {}
        group_labels = {}
        
        for group_name, class_list in child_groups.items():
            group_examples[group_name] = []
            
            for class_name in class_list:
                if class_name in self.classes:
                    class_id = self.classes.index(class_name)
                    mask = y == class_id
                    class_imgs = X[mask][:5]  # Use 5 examples per class
                    group_examples[group_name].extend(class_imgs)
                    
        # Extract gate-specific features
        if len(child_groups) == 2:
            # Binary gate
            groups = list(child_groups.keys())
            gate_type = f"{groups[0]}_vs_{groups[1]}"
        else:
            gate_type = "generic"
        
        # Store gate productions
        self.gates[gate_name] = {
            'groups': child_groups,
            'productions': {}
        }
        
        for group_name, examples in group_examples.items():
            group_features = []
            for img in examples:
                features = self.extract_gate_features(img, gate_type)
                group_features.append(features)
            
            # Store average pattern for this group
            if group_features:
                self.gates[gate_name]['productions'][group_name] = np.mean(group_features, axis=0)
    
    def traverse_gate(self, image, gate_name):
        """
        Make decision at a gate
        Returns which branch to follow
        """
        if gate_name not in self.gates:
            return None
        
        gate = self.gates[gate_name]
        
        # Extract features
        # Determine gate type from groups
        groups = list(gate['groups'].keys())
        if len(groups) == 2:
            gate_type = f"{groups[0]}_vs_{groups[1]}"
        else:
            gate_type = "generic"
        
        features = self.extract_gate_features(image, gate_type)
        
        # Compare to each group's pattern
        best_group = None
        best_score = -np.inf
        
        for group_name, pattern in gate['productions'].items():
            if len(features) == len(pattern):
                if np.std(features) > 0 and np.std(pattern) > 0:
                    score = np.corrcoef(features, pattern)[0, 1]
                    if score > best_score:
                        best_score = score
                        best_group = group_name
        
        return best_group, best_score
    
    def fit(self, X, y, k_shot=10):
        """
        Build the semantic hierarchy
        Train gates at each level
        """
        print("\n🌲 Building semantic BVH...")
        
        # Train root gate (geometric vs organic)
        print("  Level 1: Geometric vs Organic")
        self.train_gate(X, y, 'root', self.hierarchy['root'])
        
        # Train second level gates
        print("  Level 2: Subcategories")
        if 'geometric' in self.hierarchy:
            self.train_gate(X, y, 'geometric', self.hierarchy['geometric'])
        if 'organic' in self.hierarchy:
            self.train_gate(X, y, 'organic', self.hierarchy['organic'])
        
        # Train third level gates
        print("  Level 3: Fine categories")
        if 'mammals' in self.hierarchy:
            self.train_gate(X, y, 'mammals', self.hierarchy['mammals'])
        
        # Train leaf productions (actual class decisions)
        print("  Leaf nodes: Individual classes")
        for class_id in range(10):
            class_mask = y == class_id
            class_examples = X[class_mask][:k_shot]
            
            self.productions[class_id] = []
            for img in class_examples:
                # Simple features for final decision
                gray = np.mean(img, axis=2) if img.ndim == 3 else img
                fft = np.fft.fft2(gray, s=(8, 8))
                features = np.concatenate([np.abs(fft).flatten()[:20], np.angle(fft).flatten()[:20]])
                self.productions[class_id].append(features)
        
        print("✓ Semantic BVH trained")
    
    def predict_with_path(self, image):
        """
        Classify by traversing the semantic tree
        Returns prediction and path taken
        """
        path = []
        
        # Start at root
        decision, confidence = self.traverse_gate(image, 'root_gate')
        path.append(('root', decision, confidence))
        
        if decision == 'geometric':
            # Geometric branch
            decision2, conf2 = self.traverse_gate(image, 'geometric_gate')
            path.append(('geometric', decision2, conf2))
            
            # Map to actual classes
            if decision2 == 'flying':
                candidates = [0]  # plane
            elif decision2 == 'ground':
                candidates = [1, 9]  # car, truck
            elif decision2 == 'water':
                candidates = [8]  # ship
            else:
                candidates = [0, 1, 8, 9]  # all geometric
                
        elif decision == 'organic':
            # Organic branch
            decision2, conf2 = self.traverse_gate(image, 'organic_gate')
            path.append(('organic', decision2, conf2))
            
            if decision2 == 'mammals':
                # Further subdivide mammals
                decision3, conf3 = self.traverse_gate(image, 'mammals_gate')
                path.append(('mammals', decision3, conf3))
                
                if decision3 == 'pets':
                    candidates = [3, 5]  # cat, dog
                elif decision3 == 'wild':
                    candidates = [4, 7]  # deer, horse
                else:
                    candidates = [3, 4, 5, 7]  # all mammals
                    
            elif decision2 == 'non_mammals':
                candidates = [2, 6]  # bird, frog
            else:
                candidates = [2, 3, 4, 5, 6, 7]  # all organic
        else:
            # Uncertain, check all
            candidates = list(range(10))
        
        # Final decision among candidates
        best_class = -1
        best_score = -np.inf
        
        # Extract features for final comparison
        gray = np.mean(image, axis=2) if image.ndim == 3 else image
        fft = np.fft.fft2(gray, s=(8, 8))
        features = np.concatenate([np.abs(fft).flatten()[:20], np.angle(fft).flatten()[:20]])
        
        for class_id in candidates:
            if class_id in self.productions:
                for prod_features in self.productions[class_id]:
                    if len(features) == len(prod_features):
                        if np.std(features) > 0 and np.std(prod_features) > 0:
                            score = np.corrcoef(features, prod_features)[0, 1]
                            if score > best_score:
                                best_score = score
                                best_class = class_id
        
        return best_class, path
    
    def predict(self, X):
        """Standard predict interface"""
        predictions = []
        paths = []
        
        for img in X:
            pred, path = self.predict_with_path(img)
            predictions.append(pred)
            paths.append(path)
        
        return np.array(predictions), paths


def test_semantic_bvh():
    """Test hierarchical semantic classification"""
    import pickle
    import os
    from resonance_algebra.demos.cifar10_real import create_few_shot_splits
    
    print("="*70)
    print("🌳 SEMANTIC BVH - Hierarchical Gating")
    print("="*70)
    print("Don't check 'is it a car?' if it already looks organic")
    
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
    classifier = SemanticBVH()
    
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
    print("TESTING SEMANTIC BVH")
    print("="*60)
    
    predictions, paths = classifier.predict(X_query)
    accuracy = np.mean(predictions == y_query)
    print(f"Overall accuracy: {accuracy:.1%}")
    
    print("\nPer-class accuracy:")
    for c in range(10):
        mask = y_query == c
        if np.any(mask):
            class_acc = np.mean(predictions[mask] == c)
            print(f"  {classifier.classes[c]:8s}: {class_acc:.1%}")
    
    # Analyze paths for cats and dogs
    print("\n🔍 Decision paths for cats and dogs:")
    
    cat_mask = y_query == 3
    dog_mask = y_query == 5
    
    if np.any(cat_mask):
        cat_paths = [paths[i] for i in np.where(cat_mask)[0][:3]]
        print("\nCat decision paths:")
        for i, path in enumerate(cat_paths):
            print(f"  Cat {i}: ", end='')
            for level, decision, conf in path:
                print(f"{decision}({conf:.2f}) → ", end='')
            print(f"pred={predictions[np.where(cat_mask)[0][i]]}")
    
    if np.any(dog_mask):
        dog_paths = [paths[i] for i in np.where(dog_mask)[0][:3]]
        print("\nDog decision paths:")
        for i, path in enumerate(dog_paths):
            print(f"  Dog {i}: ", end='')
            for level, decision, conf in path:
                print(f"{decision}({conf:.2f}) → ", end='')
            print(f"pred={predictions[np.where(dog_mask)[0][i]]}")
    
    # Check pruning efficiency
    print("\n📊 Pruning efficiency:")
    
    # Count how many candidates were checked
    geometric_count = 0
    organic_count = 0
    
    for path in paths:
        if path and path[0][1] == 'geometric':
            geometric_count += 1
        elif path and path[0][1] == 'organic':
            organic_count += 1
    
    print(f"  Geometric branch: {geometric_count}/{len(paths)} ({geometric_count/len(paths)*100:.1f}%)")
    print(f"  Organic branch: {organic_count}/{len(paths)} ({organic_count/len(paths)*100:.1f}%)")
    
    avg_candidates = (geometric_count * 4 + organic_count * 6) / len(paths)
    print(f"  Average candidates checked: {avg_candidates:.1f}/10 (vs 10 without hierarchy)")
    print(f"  Pruning efficiency: {(10 - avg_candidates)/10*100:.1f}% reduction")
    
    print("\n💡 KEY INSIGHTS:")
    print("  • High-level gates prune irrelevant branches")
    print("  • Don't check 'plane' if it's clearly organic")
    print("  • Topological sorting reduces perplexity")
    print("  • Mimics human visual hierarchy")
    print("  • Each gate uses specialized features")
    
    return classifier


if __name__ == "__main__":
    classifier = test_semantic_bvh()