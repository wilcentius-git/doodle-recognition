import numpy as np

def extract_sketch_features(img_array, img_size):
    """Extract geometric & statistical features dari sketch"""
    img = img_array.reshape(img_size, img_size)
    
    # 1. Basic Statistics
    mean_val = np.mean(img)
    std_val = np.std(img)
    max_val = np.max(img)
    min_val = np.min(img)
    
    # 2. Quadrant Density (bagi 4 region)
    h, w = img_size // 2, img_size // 2
    q1_density = np.sum(img[:h, :w] > 50) / (h * w)
    q2_density = np.sum(img[:h, w:] > 50) / (h * w)
    q3_density = np.sum(img[h:, :w] > 50) / (h * w)
    q4_density = np.sum(img[h:, w:] > 50) / (h * w)
    
    # 3. Bounding Box Analysis
    non_zero = np.argwhere(img > 50)
    if len(non_zero) > 0:
        y_coords, x_coords = non_zero[:, 0], non_zero[:, 1]
        bbox_h = y_coords.max() - y_coords.min() + 1
        bbox_w = x_coords.max() - x_coords.min() + 1
        aspect_ratio = bbox_w / max(bbox_h, 1)
        fill_ratio = len(non_zero) / (img_size * img_size)
        
        # Center of mass
        center_y = np.mean(y_coords) / img_size
        center_x = np.mean(x_coords) / img_size
    else:
        aspect_ratio = 1.0
        fill_ratio = 0.0
        center_y = 0.5
        center_x = 0.5
    
    # 4. Edge Pixels Ratio (tepi vs tengah)
    edge_mask = np.zeros_like(img, dtype=bool)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
    edge_density = np.sum((img > 50) & edge_mask) / np.sum(edge_mask)
    
    # Combine semua features
    features = np.concatenate([
        img_array,  # Raw pixels (1024)
        [mean_val, std_val, max_val, min_val,  # Statistics (4)
         q1_density, q2_density, q3_density, q4_density,  # Quadrants (4)
         aspect_ratio, fill_ratio, center_y, center_x,  # Geometry (4)
         edge_density]  # Edge (1)
    ])  # Total: 1024 + 13 = 1037 features
    
    return features