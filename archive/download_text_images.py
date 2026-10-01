"""Download images with clear, prominent text for OCR testing."""
import urllib.request, os, ssl
ssl._create_default_https_context = ssl._create_unverified_context

IMAGES = [
    # Images with large, clear, readable text
    ("test_images/text_stop_sign.jpg",
     "https://images.unsplash.com/photo-1566932769119-7a1fb6d7ce23?w=640&q=80"),
    ("test_images/text_exit_sign.jpg",
     "https://images.unsplash.com/photo-1614064641938-3bbee52942c7?w=640&q=80"),
    ("test_images/text_coffee_shop.jpg",
     "https://images.unsplash.com/photo-1511920170033-f8396924c348?w=640&q=80"),
    ("test_images/text_books.jpg",
     "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?w=640&q=80"),
    ("test_images/text_street_name.jpg",
     "https://images.unsplash.com/photo-1555899434-94d1368aa7af?w=640&q=80"),
]

for fname, url in IMAGES:
    try:
        urllib.request.urlretrieve(url, fname)
        size = os.path.getsize(fname) / 1024
        print(f"OK: {os.path.basename(fname)} ({size:.0f} KB)")
    except Exception as e:
        print(f"FAILED: {os.path.basename(fname)} - {e}")

count = len([f for f in os.listdir("test_images") if f.endswith(".jpg")])
print(f"\nTotal images now: {count}")
