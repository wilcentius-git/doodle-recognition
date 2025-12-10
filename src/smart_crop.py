import numpy as np
from PIL import Image, ImageOps
from .src.config import IMG_SIZE

def preprocess_sketch_smart_crop(img_pil):
    """Preprocessing dengan smart crop"""
    try:
        # 1. Invert dulu (Jadi BG Hitam, Garis Putih)
        img_inv = ImageOps.invert(img_pil)
        
        # 2. Cari Bounding Box (Area mana yang ada gambarnya)
        img_np = np.array(img_inv)
        non_zero_coords = np.argwhere(img_np > 0) # Cari pixel putih
        
        if len(non_zero_coords) > 0:
            y_min, x_min = non_zero_coords.min(axis=0)
            y_max, x_max = non_zero_coords.max(axis=0)
            
            # Crop area gambar
            cropped = img_inv.crop((x_min, y_min, x_max, y_max))
        else:
            # Kalau kosong, balikin aja full black
            print("Warning: Canvas kosong!")
            return img_inv.resize((IMG_SIZE, IMG_SIZE))

        # 3. Tempel ke Canvas Persegi (Biar aspek rasio ga gepeng)
        w, h = cropped.size
        new_size = max(w, h) + 20 # Tambah padding dikit
        new_img = Image.new("L", (new_size, new_size), 0) # BG Hitam
        
        # Paste di tengah
        paste_x = (new_size - w) // 2
        paste_y = (new_size - h) // 2
        new_img.paste(cropped, (paste_x, paste_y))
        
        return new_img
    except Exception as e:
        print(f"Error preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return img_pil