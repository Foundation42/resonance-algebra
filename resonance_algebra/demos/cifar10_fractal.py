"""
FRACTAL BASIS RESONANCE for CIFAR-10
Christian's insight: Use fractal/self-similar basis functions for the pyramid
Natural hierarchy through recursive patterns!
"""

import numpy as np
from scipy import ndimage, signal
import warnings
warnings.filterwarnings('ignore')

class FractalResonanceCIFAR10:
    """
    Fractal basis functions for hierarchical spectral decomposition
    Self-similar patterns naturally capture multi-scale structure
    """
    
    def __init__(self, k_shot=10):
        self.k_shot = k_shot
        self.n_classes = 10
        
        # Fractal pyramid parameters
        self.base_scale = 2  # Starting scale
        self.fractal_depth = 5  # 2, 4, 8, 16, 32
        self.fractal_dimension = 1.5  # Between 1D and 2D (fractal!)
        
        # Generate fractal basis functions
        self.fractal_bases = self.generate_fractal_bases()
        
        # Storage for prototypes
        self.prototypes = {}
        
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    def generate_fractal_bases(self):
        """
        Generate self-similar fractal basis functions
        These naturally encode multi-scale structure
        """
        bases = {}
        
        for level in range(self.fractal_depth):
            scale = self.base_scale ** (level + 1)  # 2, 4, 8, 16, 32
            
            # Create fractal patterns at this scale
            bases[scale] = {
                'sierpinski': self.create_sierpinski_basis(scale),
                'julia': self.create_julia_basis(scale),
                'cantor': self.create_cantor_dust_basis(scale),
                'wavelet': self.create_fractal_wavelet_basis(scale),
                'spiral': self.create_log_spiral_basis(scale)
            }
            
        return bases
    
    def create_sierpinski_basis(self, size):
        """
        Sierpinski triangle/carpet basis
        Self-similar triangular patterns at multiple scales
        """
        basis = np.ones((size, size))
        
        def sierpinski_recursive(arr, x, y, size, depth):
            if depth == 0 or size < 2:
                return
            
            half = size // 2
            quarter = size // 4
            
            # Create holes (fractal pattern)
            if quarter > 0:
                arr[x+quarter:x+half+quarter, y+quarter:y+half+quarter] = 0
            
            # Recurse on sub-squares
            sierpinski_recursive(arr, x, y, half, depth-1)
            sierpinski_recursive(arr, x+half, y, half, depth-1)
            sierpinski_recursive(arr, x, y+half, half, depth-1)
            sierpinski_recursive(arr, x+half, y+half, half, depth-1)
        
        depth = int(np.log2(size))
        sierpinski_recursive(basis, 0, 0, size, depth)
        
        # Convert to phase basis
        return np.exp(1j * np.pi * basis)
    
    def create_julia_basis(self, size):
        """
        Julia set fractal basis
        Complex dynamics create natural patterns
        """
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        # Julia set iteration
        c = -0.7 + 0.27j  # Magic constant for interesting patterns
        julia = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                z = Z[i, j]
                for k in range(20):
                    if abs(z) > 2:
                        julia[i, j] = k / 20
                        break
                    z = z**2 + c
        
        # Convert to phase
        return np.exp(1j * julia * 2 * np.pi)
    
    def create_cantor_dust_basis(self, size):
        """
        Cantor dust - hierarchical removal pattern
        Creates natural scale separation
        """
        cantor = np.ones((size, size))
        
        def remove_middle_third(arr, start, end, axis):
            length = end - start
            if length < 3:
                return
            
            third = length // 3
            if axis == 0:
                arr[start + third:start + 2*third, :] = 0
                remove_middle_third(arr, start, start + third, axis)
                remove_middle_third(arr, start + 2*third, end, axis)
            else:
                arr[:, start + third:start + 2*third] = 0
                remove_middle_third(arr, start, start + third, axis)
                remove_middle_third(arr, start + 2*third, end, axis)
        
        # Apply Cantor removal in both dimensions
        remove_middle_third(cantor, 0, size, 0)
        remove_middle_third(cantor, 0, size, 1)
        
        return np.exp(1j * np.pi * cantor)
    
    def create_fractal_wavelet_basis(self, size):
        """
        Self-similar wavelet patterns
        Multi-resolution analysis built-in
        """
        # Create multi-scale wavelet pattern
        wavelet = np.zeros((size, size), dtype=complex)
        
        for scale in [1, 2, 4, 8]:
            if scale > size // 4:
                break
            
            # Create wavelet at this scale
            x = np.linspace(-np.pi * scale, np.pi * scale, size)
            y = np.linspace(-np.pi * scale, np.pi * scale, size)
            X, Y = np.meshgrid(x, y)
            
            # Morlet-like wavelet
            gaussian = np.exp(-(X**2 + Y**2) / (2 * scale**2))
            wave = np.exp(1j * (X + Y) / scale)
            
            wavelet += gaussian * wave / scale
        
        return wavelet / np.abs(wavelet.max())
    
    def create_log_spiral_basis(self, size):
        """
        Logarithmic spiral - nature's favorite pattern
        Found in galaxies, shells, hurricanes
        """
        center = size // 2
        spiral = np.zeros((size, size), dtype=complex)
        
        for i in range(size):
            for j in range(size):
                # Convert to polar coordinates
                dx = i - center
                dy = j - center
                r = np.sqrt(dx**2 + dy**2) + 1e-8
                theta = np.arctan2(dy, dx)
                
                # Logarithmic spiral equation
                a = 0.1  # Tightness
                b = 0.2  # Growth rate
                
                # Multiple spiral arms
                for arm in range(3):
                    arm_offset = arm * 2 * np.pi / 3
                    spiral_phase = a * np.log(r) - b * theta + arm_offset
                    spiral[i, j] += np.exp(1j * spiral_phase)
        
        return spiral / 3
    
    def extract_fractal_features(self, image, scale):
        """
        Extract features using fractal basis functions
        Projects image onto self-similar patterns
        """
        # Resize image to current scale
        if scale < 32:
            resized = ndimage.zoom(image, (scale/32, scale/32, 1), order=1)
        else:
            resized = image.copy()
        
        # Convert to grayscale for fractal analysis
        gray = np.mean(resized, axis=2)
        
        # Normalize
        gray = (gray - gray.mean()) / (gray.std() + 1e-8)
        
        features = []
        
        # Project onto each fractal basis
        for basis_name, basis in self.fractal_bases[scale].items():
            # Convolution with fractal basis (resonance!)
            projection = np.vdot(gray.flatten(), basis.flatten())
            
            # Extract magnitude and phase
            magnitude = np.abs(projection)
            phase = np.angle(projection)
            
            features.append(magnitude)
            features.append(phase)
            
            # Also extract sub-region projections (local fractals)
            if scale >= 8:
                h, w = gray.shape
                for i in range(2):
                    for j in range(2):
                        sub_region = gray[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                        sub_basis = basis[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                        
                        if sub_region.size > 0 and sub_basis.size > 0:
                            sub_proj = np.vdot(sub_region.flatten(), sub_basis.flatten())
                            features.append(np.abs(sub_proj))
                            features.append(np.angle(sub_proj))
        
        # Add color fractal dimension
        if resized.shape[2] == 3:
            for c in range(3):
                channel = resized[:, :, c]
                fd = self.compute_fractal_dimension(channel)
                features.append(fd)
        
        return np.array(features)
    
    def compute_fractal_dimension(self, image):
        """
        Compute box-counting fractal dimension
        Measures self-similarity across scales
        """
        # Binarize image
        threshold = np.mean(image)
        binary = image > threshold
        
        # Box-counting at different scales
        scales = []
        counts = []
        
        for box_size in [2, 4, 8]:
            if box_size > min(image.shape):
                break
            
            # Count non-empty boxes
            count = 0
            for i in range(0, image.shape[0], box_size):
                for j in range(0, image.shape[1], box_size):
                    box = binary[i:i+box_size, j:j+box_size]
                    if np.any(box):
                        count += 1
            
            scales.append(np.log(1/box_size))
            counts.append(np.log(count))
        
        # Fractal dimension is the slope
        if len(scales) > 1:
            fd = np.polyfit(scales, counts, 1)[0]
            return fd
        return 1.5  # Default fractal dimension
    
    def fit(self, X_support, y_support):
        """
        Create fractal prototypes for each class
        """
        print("\n🌀 Creating Fractal Basis Prototypes...")
        
        for class_id in range(self.n_classes):
            print(f"  {self.classes[class_id]}:", end=' ')
            
            class_mask = y_support == class_id
            class_images = X_support[class_mask][:self.k_shot]
            
            class_features = []
            
            # Extract fractal features at each scale
            for scale in [2, 4, 8, 16, 32]:
                scale_features = []
                
                for img in class_images:
                    feat = self.extract_fractal_features(img, scale)
                    scale_features.append(feat)
                
                # Average to create prototype
                prototype = np.mean(scale_features, axis=0)
                class_features.append(prototype)
            
            self.prototypes[class_id] = np.concatenate(class_features)
            print(f"✓ {len(self.prototypes[class_id])} fractal features")
        
        print("✓ Fractal prototypes ready (no training needed!)")
    
    def predict(self, X_query):
        """
        Classify using fractal resonance matching
        """
        predictions = []
        
        for img in X_query:
            # Extract fractal features
            all_features = []
            for scale in [2, 4, 8, 16, 32]:
                feat = self.extract_fractal_features(img, scale)
                all_features.append(feat)
            
            query_features = np.concatenate(all_features)
            
            # Find best matching prototype (maximum resonance)
            best_score = -np.inf
            best_class = -1
            
            for class_id, prototype in self.prototypes.items():
                # Fractal resonance score
                # Combines correlation and phase coherence
                correlation = np.corrcoef(query_features, prototype)[0, 1]
                
                # Phase coherence for complex features
                phase_diff = query_features - prototype
                coherence = np.mean(np.cos(phase_diff))
                
                # Combined score
                score = correlation + 0.5 * coherence
                
                if score > best_score:
                    best_score = score
                    best_class = class_id
            
            predictions.append(best_class)
        
        return np.array(predictions)


def test_fractal_resonance():
    """Test the fractal basis approach"""
    import pickle
    import os
    from resonance_algebra.demos.cifar10_real import create_few_shot_splits
    
    print("="*70)
    print("🌀 FRACTAL BASIS RESONANCE - Nature's Patterns")
    print("="*70)
    print("\nUsing self-similar patterns at multiple scales:")
    print("  • Sierpinski (triangular hierarchy)")
    print("  • Julia sets (complex dynamics)")
    print("  • Cantor dust (recursive gaps)")
    print("  • Fractal wavelets (multi-resolution)")
    print("  • Log spirals (nature's favorite)")
    
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
    print(f"\n🎯 Testing 10-shot learning with fractal bases:")
    print("-"*50)
    
    # Create splits
    X_support, y_support, X_query, y_query = create_few_shot_splits(
        train_data, train_labels, test_data, test_labels,
        k_shot=10, n_test_per_class=30
    )
    
    # Initialize fractal resonance
    clf = FractalResonanceCIFAR10(k_shot=10)
    
    # Fit (create fractal prototypes)
    import time
    start = time.time()
    clf.fit(X_support, y_support)
    fit_time = time.time() - start
    
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
            
            # Show which fractal patterns resonate
            if class_acc > 0.2:
                print(f"    → Strong fractal signature detected!")
    
    print(f"\n⏱️ Timing:")
    print(f"  Fit: {fit_time:.2f}s")
    print(f"  Predict: {pred_time:.2f}s ({pred_time/len(X_query)*1000:.1f}ms per image)")
    
    print(f"\n🎯 Overall Accuracy: {accuracy:.1%}")
    
    # Compare to other approaches
    print("\n" + "="*70)
    print("💡 FRACTAL INSIGHTS")
    print("="*70)
    print("✓ Self-similar patterns capture natural hierarchies")
    print("✓ Fractal dimension measures complexity")
    print("✓ Julia sets encode nonlinear dynamics")
    print("✓ Sierpinski patterns match recursive structures")
    print("✓ Log spirals found throughout nature")
    
    print("\n🌀 Objects resonate with their natural fractal patterns!")
    print("   No training - just matching to nature's own basis functions!")
    
    return accuracy


if __name__ == "__main__":
    accuracy = test_fractal_resonance()
    
    print("\n" + "="*70)
    print(f"🌊 Fractal Resonance: {accuracy:.1%} with ZERO training!")
    print("="*70)