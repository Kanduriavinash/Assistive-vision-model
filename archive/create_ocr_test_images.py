"""
Create test images with clear, readable text to demonstrate OCR capability.
Also downloads high-resolution images with prominent text.
These simulate what a visually impaired user's camera would capture up close.
"""
import os
import ssl
import urllib.request
from PIL import Image, ImageDraw, ImageFont

ssl._create_default_https_context = ssl._create_unverified_context
os.makedirs("test_images", exist_ok=True)


def create_sign_image(filename, text_lines, bg_color, text_color, size=(800, 400)):
    """Create a realistic-looking sign image with clear text."""
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a decent font, fall back to default
    font_size = 60
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        small_font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Draw border
    draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=text_color, width=3)
    
    # Draw text centered
    y_offset = 50
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        draw.text((x, y_offset), line, fill=text_color, font=font)
        y_offset += 80
    
    filepath = os.path.join("test_images", filename)
    img.save(filepath, quality=95)
    print(f"Created: {filename}")
    return filepath


# Create various sign types a visually impaired person would encounter
print("Creating realistic sign images...\n")

# 1. Store entrance sign
create_sign_image(
    "ocr_test_store.jpg",
    ["WELCOME TO", "CITY MART", "OPEN 9AM - 9PM"],
    bg_color=(255, 255, 255), text_color=(0, 0, 128),
    size=(800, 400)
)

# 2. Warning/caution sign
create_sign_image(
    "ocr_test_warning.jpg",
    ["CAUTION", "WET FLOOR", "WATCH YOUR STEP"],
    bg_color=(255, 255, 0), text_color=(0, 0, 0),
    size=(800, 400)
)

# 3. Room number / office sign
create_sign_image(
    "ocr_test_room.jpg",
    ["ROOM 204", "COMPUTER LAB", "Dr. Smith"],
    bg_color=(50, 50, 50), text_color=(255, 255, 255),
    size=(800, 400)
)

# 4. Street direction sign
create_sign_image(
    "ocr_test_direction.jpg",
    ["MAIN STREET", "EXIT 5B", "HOSPITAL 2KM"],
    bg_color=(0, 100, 0), text_color=(255, 255, 255),
    size=(800, 400)
)

# 5. Menu / restaurant sign
create_sign_image(
    "ocr_test_menu.jpg",
    ["TODAY SPECIAL", "COFFEE $3.50", "SANDWICH $5.99"],
    bg_color=(139, 69, 19), text_color=(255, 255, 200),
    size=(800, 400)
)

# Also download some high-res real images
print("\nDownloading high-res real-world text images...\n")
hires_images = [
    ("ocr_real_sign_01.jpg", "https://images.unsplash.com/photo-1566932769119-7a1fb6d7ce23?w=1280&q=90"),
    ("ocr_real_sign_02.jpg", "https://images.unsplash.com/photo-1563906267088-b029e7101114?w=1280&q=90"),
]

for fname, url in hires_images:
    fpath = os.path.join("test_images", fname)
    try:
        urllib.request.urlretrieve(url, fpath)
        size = os.path.getsize(fpath) / 1024
        print(f"Downloaded: {fname} ({size:.0f} KB)")
    except Exception as e:
        print(f"Failed: {fname} - {e}")

count = len([f for f in os.listdir("test_images") if f.endswith(".jpg")])
print(f"\nTotal images now: {count}")
