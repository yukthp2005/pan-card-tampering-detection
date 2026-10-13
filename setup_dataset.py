import os
import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories matching your YOLO/Roboflow raw_dataset structure
TRAIN_IMG_DIR = 'raw_dataset/train/images/'
VALID_IMG_DIR = 'raw_dataset/valid/images/'
REAL_DIR = 'dataset/real/'
TAMPERED_DIR = 'dataset/tampered/'
TOTAL_IMAGES_PER_CLASS = 1000

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(TAMPERED_DIR, exist_ok=True)

def generate_tampered_image(img):
    """
    Applies diverse, randomized forgery artifacts to simulate real-world document tampering:
    - Spliced solid boxes (white-outs/patch edits)
    - Mismatched synthetic text overlays
    - Localized blurring or pixel noise
    - JPEG compression degradation
    """
    h, w, _ = img.shape
    tampered = img.copy()
    
    forgery_type = random.choice(['patch_box', 'text_splice', 'blur_region', 'noise_splice'])
    
    if forgery_type == 'patch_box' or forgery_type == 'text_splice':
        box_w = random.randint(int(w * 0.2), int(w * 0.5))
        box_h = random.randint(int(h * 0.05), int(h * 0.12))
        x1 = random.randint(int(w * 0.2), int(w * 0.5))
        y1 = random.randint(int(h * 0.3), int(h * 0.75))
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        if forgery_type == 'patch_box':
            color_choice = random.choice([
                (255, 255, 255), # Pure white patch
                (240, 245, 250), # Off-white patch
                (200, 220, 240)  # Discolored patch
            ])
            cv2.rectangle(tampered, (x1, y1), (x2, y2), color_choice, -1)
        else:
            cv2.rectangle(tampered, (x1, y1), (x2, y2), (255, 255, 255), -1)
            fake_texts = ["ABCDE1234F", "XYZW5678G", "SRKP9876K", "TMPER0000X", "FAKEID1122A"]
            text_str = random.choice(fake_texts)
            font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX])
            font_scale = random.uniform(0.5, 0.8)
            cv2.putText(tampered, text_str, (x1 + 5, y2 - 8), font, font_scale, (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)), 1, cv2.LINE_AA)
            
    elif forgery_type == 'blur_region':
        bx1, by1 = random.randint(int(w * 0.2), int(w * 0.4)), random.randint(int(h * 0.4), int(h * 0.7))
        bw, bh = random.randint(100, 200), random.randint(30, 60)
        roi = tampered[by1:by1+bh, bx1:bx1+bw]
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
            tampered[by1:by1+bh, bx1:bx1+bw] = blurred_roi
            
    elif forgery_type == 'noise_splice':
        nx1, ny1 = random.randint(int(w * 0.2), int(w * 0.4)), random.randint(int(h * 0.3), int(h * 0.7))
        nw, nh = random.randint(120, 220), random.randint(35, 70)
        roi = tampered[ny1:ny1+nh, nx1:nx1+nw]
        if roi.size > 0:
            noise = np.random.normal(0, 25, roi.shape).astype(np.uint8)
            tampered[ny1:ny1+nh, nx1:nx1+nw] = cv2.add(roi, noise)

    quality = random.randint(40, 90)
    _, encoded_img = cv2.imencode('.jpg', tampered, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    tampered = cv2.imdecode(encoded_img, 1)
    
    return tampered

def process_dataset():
    loaded_images = []
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    # Gather clean images from both train and valid folders
    for dir_path in [TRAIN_IMG_DIR, VALID_IMG_DIR]:
        if os.path.exists(dir_path):
            for fname in os.listdir(dir_path):
                if fname.lower().endswith(valid_extensions):
                    img_path = os.path.join(dir_path, fname)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, (224, 224))
                        loaded_images.append(img_resized)

    if len(loaded_images) == 0:
        print(f"Error: No valid images found in '{TRAIN_IMG_DIR}' or '{VALID_IMG_DIR}'.")
        return

    print(f"Loaded {len(loaded_images)} base templates from raw_dataset. Generating full dataset...")

    # 1. Generate 1000 Real Images with Augmentation
    real_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )

    real_count = 0
    while real_count < TOTAL_IMAGES_PER_CLASS:
        for base_img in loaded_images:
            if real_count >= TOTAL_IMAGES_PER_CLASS:
                break
            x = np.expand_dims(base_img, axis=0)
            for batch in real_datagen.flow(x, batch_size=1):
                aug_img = batch[0].astype(np.uint8)
                cv2.imwrite(os.path.join(REAL_DIR, f"real_{real_count:04d}.jpg"), aug_img)
                real_count += 1
                break

    # 2. Generate 1000 Tampered Images
    tampered_count = 0
    while tampered_count < TOTAL_IMAGES_PER_CLASS:
        base_img = random.choice(loaded_images)
        tampered_img = generate_tampered_image(base_img)
        cv2.imwrite(os.path.join(TAMPERED_DIR, f"tampered_{tampered_count:04d}.jpg"), tampered_img)
        tampered_count += 1

    print("Dataset setup complete!")
    print(f"-> {REAL_DIR}: {TOTAL_IMAGES_PER_CLASS} images ready.")
    print(f"-> {TAMPERED_DIR}: {TOTAL_IMAGES_PER_CLASS} images ready.")

if __name__ == '__main__':
    process_dataset()