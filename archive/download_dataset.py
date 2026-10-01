"""
B2: Dataset Collection Script
Downloads diverse test images for the Assistive Vision Pipeline.
Categories: street scenes, signs/text, indoor spaces.
"""
import urllib.request
import os
import ssl

# Bypass SSL verification for download reliability
ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = "test_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Curated list of free-to-use images from Unsplash (direct download links)
# Each tuple: (filename, url, category)
IMAGES = [
    # === STREET SCENES (YOLOv8 testing - people, cars, obstacles) ===
    ("street_crosswalk_01.jpg",
     "https://images.unsplash.com/photo-1517420879524-86d64ac2f339?w=640&q=80",
     "street"),
    ("street_city_02.jpg",
     "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=640&q=80",
     "street"),
    ("street_sidewalk_03.jpg",
     "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=640&q=80",
     "street"),
    ("street_traffic_04.jpg",
     "https://images.unsplash.com/photo-1494783367193-149034c05e8f?w=640&q=80",
     "street"),
    ("street_pedestrians_05.jpg",
     "https://images.unsplash.com/photo-1476231682828-37e571bc172f?w=640&q=80",
     "street"),
    ("street_cars_06.jpg",
     "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640&q=80",
     "street"),
    ("street_bench_07.jpg",
     "https://images.unsplash.com/photo-1572116469696-31de0f17cc34?w=640&q=80",
     "street"),
    ("street_bicycle_08.jpg",
     "https://images.unsplash.com/photo-1558618666-fcd25c85f82e?w=640&q=80",
     "street"),

    # === SIGNS & TEXT (EasyOCR testing - readable text on signs/storefronts) ===
    ("sign_store_01.jpg",
     "https://images.unsplash.com/photo-1528698827591-e625c96bfc19?w=640&q=80",
     "sign"),
    ("sign_neon_02.jpg",
     "https://images.unsplash.com/photo-1563906267088-b029e7101114?w=640&q=80",
     "sign"),
    ("sign_street_03.jpg",
     "https://images.unsplash.com/photo-1567449303078-57ad995bd329?w=640&q=80",
     "sign"),
    ("sign_cafe_04.jpg",
     "https://images.unsplash.com/photo-1559925393-8be0ec4767c8?w=640&q=80",
     "sign"),
    ("sign_open_05.jpg",
     "https://images.unsplash.com/photo-1526958097901-5e6d742d3371?w=640&q=80",
     "sign"),
    ("sign_road_06.jpg",
     "https://images.unsplash.com/photo-1566933293069-b55c7f326dd4?w=640&q=80",
     "sign"),
    ("sign_menu_07.jpg",
     "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=640&q=80",
     "sign"),

    # === INDOOR SCENES (assistive use case - furniture, doors, stairs) ===
    ("indoor_room_01.jpg",
     "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=640&q=80",
     "indoor"),
    ("indoor_stairs_02.jpg",
     "https://images.unsplash.com/photo-1572883454114-1cf0031ede2a?w=640&q=80",
     "indoor"),
    ("indoor_hallway_03.jpg",
     "https://images.unsplash.com/photo-1497366216548-37526070297c?w=640&q=80",
     "indoor"),
    ("indoor_kitchen_04.jpg",
     "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&q=80",
     "indoor"),
    ("indoor_door_05.jpg",
     "https://images.unsplash.com/photo-1558618666-fcd25c85f82e?w=640&q=80",
     "indoor"),

    # === MIXED / CHALLENGING (real-world complexity) ===
    ("mixed_market_01.jpg",
     "https://images.unsplash.com/photo-1533900298318-6b8da08a523e?w=640&q=80",
     "mixed"),
    ("mixed_park_02.jpg",
     "https://images.unsplash.com/photo-1519331379826-f10be5486c6f?w=640&q=80",
     "mixed"),
    ("mixed_bus_stop_03.jpg",
     "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=640&q=80",
     "mixed"),
    ("mixed_night_04.jpg",
     "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=640&q=80",
     "mixed"),
    ("mixed_rain_05.jpg",
     "https://images.unsplash.com/photo-1534274988757-a28bf1a57c17?w=640&q=80",
     "mixed"),
]


def download_images():
    """Download all images with progress tracking."""
    total = len(IMAGES)
    success = 0
    failed = []

    print(f"Downloading {total} test images to '{OUTPUT_DIR}/'...\n")

    for i, (filename, url, category) in enumerate(IMAGES, 1):
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Skip if already downloaded
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            print(f"  [{i}/{total}] SKIP (exists): {filename}")
            success += 1
            continue

        try:
            print(f"  [{i}/{total}] Downloading: {filename} [{category}]...", end=" ")
            urllib.request.urlretrieve(url, filepath)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"OK ({size_kb:.0f} KB)")
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(filename)

    # Summary
    print(f"\n{'='*50}")
    print(f"DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"  Total attempted: {total}")
    print(f"  Successfully downloaded: {success}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print(f"\n  Failed files: {', '.join(failed)}")

    # Count by category
    categories = {}
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(('.jpg', '.png', '.jpeg')):
            cat = f.split('_')[0]
            categories[cat] = categories.get(cat, 0) + 1

    print(f"\n  Images by category:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count} images")

    print(f"\n  Location: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    download_images()
