"""
Debug version: Simplified CIFAR-10 resonance classifier
Let's get the basics working first on real data
"""

import numpy as np
from scipy import signal, ndimage
import warnings
warnings.filterwarnings('ignore')

class SimpleCIFAR10Resonance:
    """
    Simplified version to debug what's happening
    """
    
    def __init__(self, k_shot=10):
        self.k_shot = k_shot
        self.n_classes = 10
        self.class_prototypes = {}
    
    def extract_simple_features(self, image):
        """
        Simple but effective feature extraction
        """
        # Ensure image is properly normalized [0,1]
        if image.max() > 1.0:
            image = image / 255.0
        
        features = []
        
        # 1. Color histograms (simple but effective)
        for c in range(3):
            hist, _ = np.histogram(image[:,:,c], bins=8, range=(0,1))
            features.extend(hist / hist.sum())
        
        # 2. Simple edge detection
        gray = np.mean(image, axis=2)
        edges = ndimage.sobel(gray)
        edge_hist, _ = np.histogram(edges, bins=8)
        features.extend(edge_hist / edge_hist.sum())
        
        # 3. Basic texture via local variance
        variance = ndimage.generic_filter(gray, np.var, size=4)
        var_hist, _ = np.histogram(variance, bins=8)
        features.extend(var_hist / var_hist.sum())
        
        # 4. Simple HOG-like features
        dx = ndimage.sobel(gray, axis=1)
        dy = ndimage.sobel(gray, axis=0)
        angles = np.arctan2(dy, dx)
        angle_hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        features.extend(angle_hist / angle_hist.sum())
        
        return np.array(features)
    
    def fit(self, X_support, y_support):
        """
        Create prototypes by averaging features
        """
        for class_id in range(self.n_classes):
            class_mask = y_support == class_id
            class_images = X_support[class_mask][:self.k_shot]
            
            # Extract features and average
            class_features = []
            for img in class_images:
                feat = self.extract_simple_features(img)
                class_features.append(feat)
            
            # Simple average prototype
            self.class_prototypes[class_id] = np.mean(class_features, axis=0)
    
    def predict(self, X_query):
        """
        Nearest prototype classification
        """
        predictions = []
        
        for img in X_query:
            feat = self.extract_simple_features(img)
            
            # Find nearest prototype
            min_dist = float('inf')
            pred = -1
            
            for class_id, prototype in self.class_prototypes.items():
                # Euclidean distance
                dist = np.linalg.norm(feat - prototype)
                if dist < min_dist:
                    min_dist = dist
                    pred = class_id
            
            predictions.append(pred)
        
        return np.array(predictions)


def test_simple_classifier():
    """Test the simplified classifier"""
    import pickle
    import os
    
    print("Testing simplified CIFAR-10 classifier...")
    
    # Load one batch of CIFAR-10
    data_path = './data/cifar-10-batches-py/data_batch_1'
    
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        
        data = batch[b'data']
        labels = batch[b'labels']
        
        # Reshape and normalize
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        data = data.astype(np.float32) / 255.0
        labels = np.array(labels)
        
        # Quick test with k=10
        clf = SimpleCIFAR10Resonance(k_shot=10)
        
        # Use first 100 for support, next 100 for test
        X_support = data[:100]
        y_support = labels[:100] % 10  # Ensure 10 classes
        X_test = data[100:200]
        y_test = labels[100:200]
        
        clf.fit(X_support, y_support)
        preds = clf.predict(X_test)
        
        accuracy = np.mean(preds == y_test)
        print(f"Simple classifier accuracy: {accuracy:.1%}")
        
        # Check if features are working
        feat1 = clf.extract_simple_features(data[0])
        feat2 = clf.extract_simple_features(data[1])
        print(f"Feature vector size: {len(feat1)}")
        print(f"Feature variance: {np.var(feat1):.4f}")
        print(f"Feature difference: {np.linalg.norm(feat1 - feat2):.4f}")
        
        return accuracy
    else:
        print("CIFAR-10 data not found!")
        return 0


if __name__ == "__main__":
    accuracy = test_simple_classifier()
    
    if accuracy > 0.15:
        print("✓ Basic features working! Now we can enhance...")
    else:
        print("⚠ Need to debug feature extraction further...")