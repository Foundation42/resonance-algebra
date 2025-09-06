"""
RETINAL PREPROCESSING FOR RESONANCE ALGEBRA
Implement biological vision preprocessing: edge detection, center-surround, saliency
This creates the "where to look" map before spectral analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


class RetinalPreprocessor:
    """
    Mimics retinal ganglion cells and early visual processing
    Creates feature maps that guide attention
    """
    
    def __init__(self):
        # Different "ganglion cell" types
        self.cell_types = {
            'on_center': self.on_center_off_surround,
            'off_center': self.off_center_on_surround,
            'edge_horizontal': self.horizontal_edge_detector,
            'edge_vertical': self.vertical_edge_detector,
            'color_opponent_rg': self.red_green_opponent,
            'color_opponent_by': self.blue_yellow_opponent,
        }
        
    def on_center_off_surround(self, image):
        """Detects bright spots on dark backgrounds"""
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Center (small gaussian)
        center = ndimage.gaussian_filter(gray, sigma=0.5)
        # Surround (large gaussian)
        surround = ndimage.gaussian_filter(gray, sigma=2.0)
        
        # Difference of Gaussians (DoG)
        response = center - surround
        return response
    
    def off_center_on_surround(self, image):
        """Detects dark spots on bright backgrounds"""
        return -self.on_center_off_surround(image)
    
    def horizontal_edge_detector(self, image):
        """Detects horizontal edges (like eyes, mouth)"""
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        kernel = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]])
        
        edges = ndimage.convolve(gray, kernel)
        return edges
    
    def vertical_edge_detector(self, image):
        """Detects vertical edges (like ears standing up)"""
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        kernel = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
        
        edges = ndimage.convolve(gray, kernel)
        return edges
    
    def red_green_opponent(self, image):
        """Red-green color opponency (important for natural scenes)"""
        if image.ndim != 3:
            return np.zeros_like(image)
        
        return image[:, :, 0] - image[:, :, 1]  # R - G
    
    def blue_yellow_opponent(self, image):
        """Blue-yellow color opponency"""
        if image.ndim != 3:
            return np.zeros_like(image)
        
        return image[:, :, 2] - (image[:, :, 0] + image[:, :, 1]) / 2  # B - (R+G)/2
    
    def create_saliency_map(self, image):
        """
        Combine all cell responses to create a saliency map
        This tells us "where to look"
        """
        responses = {}
        for name, detector in self.cell_types.items():
            responses[name] = detector(image)
        
        # Normalize each response
        for name in responses:
            r = responses[name]
            if np.std(r) > 0:
                responses[name] = (r - np.mean(r)) / np.std(r)
        
        # Combine with weights (learned or heuristic)
        saliency = np.zeros_like(responses['on_center'])
        
        # Edges are most important for shape
        saliency += np.abs(responses['edge_horizontal']) * 1.5
        saliency += np.abs(responses['edge_vertical']) * 1.5
        
        # Center-surround for texture
        saliency += np.abs(responses['on_center']) * 1.0
        saliency += np.abs(responses['off_center']) * 0.5
        
        # Color for natural categories
        if 'color_opponent_rg' in responses:
            saliency += np.abs(responses['color_opponent_rg']) * 0.8
            saliency += np.abs(responses['color_opponent_by']) * 0.5
        
        return saliency, responses


class AttentiveSpectralProduction:
    """
    Use retinal preprocessing to guide spectral analysis
    Focus on salient regions like biological vision
    """
    
    def __init__(self):
        self.retina = RetinalPreprocessor()
        self.productions = []
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        print("👁️ Attentive spectral production initialized")
        print("  Retinal preprocessing + saliency-guided attention")
    
    def find_attention_points(self, saliency_map, n_points=3):
        """
        Find the most salient points to "look at"
        Like saccades in human vision
        """
        # Smooth saliency to find regions, not just pixels
        smoothed = ndimage.gaussian_filter(saliency_map, sigma=2)
        
        # Find local maxima
        attention_points = []
        
        # Suppress borders (unreliable)
        smoothed[:3, :] = 0
        smoothed[-3:, :] = 0
        smoothed[:, :3] = 0
        smoothed[:, -3:] = 0
        
        for _ in range(n_points):
            # Find maximum
            max_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            attention_points.append(max_idx)
            
            # Suppress this region
            y, x = max_idx
            y_min, y_max = max(0, y-5), min(smoothed.shape[0], y+5)
            x_min, x_max = max(0, x-5), min(smoothed.shape[1], x+5)
            smoothed[y_min:y_max, x_min:x_max] = 0
        
        return attention_points
    
    def extract_attentive_features(self, image):
        """
        Extract features guided by saliency
        Focus on important regions
        """
        # Get retinal responses and saliency
        saliency, responses = self.retina.create_saliency_map(image)
        
        # Find where to look
        attention_points = self.find_attention_points(saliency, n_points=3)
        
        features = []
        
        # 1. Global features from edge maps (shape)
        edge_h = responses['edge_horizontal']
        edge_v = responses['edge_vertical']
        
        # FFT of edge maps (captures shape patterns)
        edge_combined = edge_h + 1j * edge_v  # Complex edge representation
        edge_fft = np.fft.fft2(edge_combined, s=(16, 16))
        edge_fft_shifted = np.fft.fftshift(edge_fft)
        
        # Low frequency edge patterns (global shape)
        features.extend(np.abs(edge_fft_shifted[6:10, 6:10].flatten())[:10])
        
        # 2. Local features at attention points
        for y, x in attention_points:
            # Extract local patch
            patch_size = 8
            y_min = max(0, y - patch_size//2)
            y_max = min(image.shape[0], y + patch_size//2)
            x_min = max(0, x - patch_size//2)
            x_max = min(image.shape[1], x + patch_size//2)
            
            # Get responses in this patch
            local_features = []
            
            # Edge orientation at this point (important for ears, eyes)
            if y_max > y_min and x_max > x_min:
                local_edge_h = responses['edge_horizontal'][y_min:y_max, x_min:x_max]
                local_edge_v = responses['edge_vertical'][y_min:y_max, x_min:x_max]
                
                # Dominant edge orientation
                edge_angle = np.arctan2(np.mean(local_edge_v), np.mean(local_edge_h))
                edge_strength = np.sqrt(np.mean(local_edge_h**2) + np.mean(local_edge_v**2))
                
                local_features.extend([edge_angle, edge_strength])
                
                # Texture (center-surround response)
                local_texture = responses['on_center'][y_min:y_max, x_min:x_max]
                local_features.append(np.std(local_texture))  # Texture complexity
                
                # Color if available
                if image.ndim == 3:
                    local_color = image[y_min:y_max, x_min:x_max]
                    local_features.append(np.mean(local_color[:, :, 0]))  # Red
                    local_features.append(np.mean(local_color[:, :, 1]))  # Green
                    local_features.append(np.mean(local_color[:, :, 2]))  # Blue
                else:
                    local_features.extend([0, 0, 0])
            else:
                local_features.extend([0, 0, 0, 0, 0, 0])
            
            features.extend(local_features)
        
        return np.array(features), saliency, attention_points
    
    def fit(self, X, y, k_shot=10):
        """Train with attention-guided features"""
        self.productions = []
        
        print("\n📚 Creating attentive production rules...")
        
        for class_id in range(10):
            class_mask = y == class_id
            class_examples = X[class_mask][:k_shot]
            
            print(f"  {self.classes[class_id]:8s}: ", end='')
            
            for img in class_examples:
                features, saliency, attention = self.extract_attentive_features(img)
                self.productions.append({
                    'features': features,
                    'class': class_id
                })
            
            print(f"{len(class_examples)} rules")
        
        print(f"\n📊 Total: {len(self.productions)} attentive productions")
    
    def predict(self, X, visualize_attention=False):
        """Predict using attention-guided features"""
        predictions = []
        
        for idx, img in enumerate(X):
            features, saliency, attention = self.extract_attentive_features(img)
            
            # Find best matching production
            best_score = -np.inf
            best_class = -1
            
            for prod in self.productions:
                if len(features) == len(prod['features']):
                    # Correlation-based similarity
                    if np.std(features) > 0 and np.std(prod['features']) > 0:
                        score = np.corrcoef(features, prod['features'])[0, 1]
                        if score > best_score:
                            best_score = score
                            best_class = prod['class']
            
            predictions.append(best_class)
            
            # Visualize first few
            if visualize_attention and idx < 3:
                self.visualize_attention(img, saliency, attention, best_class)
        
        return np.array(predictions)
    
    def visualize_attention(self, image, saliency, attention_points, predicted_class):
        """Show where the system is looking"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original (pred: {self.classes[predicted_class]})")
        axes[0].axis('off')
        
        # Saliency map
        axes[1].imshow(saliency, cmap='hot')
        axes[1].set_title("Saliency (where to look)")
        axes[1].axis('off')
        
        # Attention points
        axes[2].imshow(image)
        for y, x in attention_points:
            circle = plt.Circle((x, y), 3, color='red', fill=False, linewidth=2)
            axes[2].add_patch(circle)
        axes[2].set_title("Attention points (saccades)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_cat_dog_attention(self, X, y):
        """See where the system looks for cats vs dogs"""
        print("\n" + "="*60)
        print("CAT vs DOG ATTENTION ANALYSIS")
        print("="*60)
        
        cat_mask = y == 3
        dog_mask = y == 5
        
        cat_examples = X[cat_mask][:3]
        dog_examples = X[dog_mask][:3]
        
        print("\n🐱 Cat attention patterns:")
        for i, img in enumerate(cat_examples):
            features, saliency, attention = self.extract_attentive_features(img)
            print(f"  Cat {i}: attention at {attention}")
        
        print("\n🐕 Dog attention patterns:")
        for i, img in enumerate(dog_examples):
            features, saliency, attention = self.extract_attentive_features(img)
            print(f"  Dog {i}: attention at {attention}")


def test_retinal_attention():
    """Test biologically-inspired attention system"""
    import pickle
    import os
    from resonance_algebra.demos.cifar10_real import create_few_shot_splits
    
    print("="*70)
    print("👁️ RETINAL PREPROCESSING + ATTENTION")
    print("="*70)
    print("Biological vision: edge detection → saliency → attention → recognition")
    
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
    classifier = AttentiveSpectralProduction()
    
    # Analyze attention for cats vs dogs
    classifier.analyze_cat_dog_attention(train_data, train_labels)
    
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
    print("TESTING WITH ATTENTION")
    print("="*60)
    
    predictions = classifier.predict(X_query, visualize_attention=False)
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
        
        # What are cats misclassified as?
        cat_preds = predictions[cat_mask]
        unique, counts = np.unique(cat_preds, return_counts=True)
        print(f"\n  Cats classified as:")
        for cls, count in zip(unique, counts):
            if count > 0:
                print(f"    {classifier.classes[cls]}: {count}/{len(cat_preds)} ({count/len(cat_preds)*100:.0f}%)")
    
    print("\n💡 KEY INSIGHTS:")
    print("  • Edge detection highlights discriminative features")
    print("  • Saliency maps show 'where to look'")
    print("  • Attention points mimic human saccades")
    print("  • Focus on salient regions, not uniform sampling")
    print("  • Biological preprocessing before spectral analysis")
    
    return classifier


if __name__ == "__main__":
    classifier = test_retinal_attention()