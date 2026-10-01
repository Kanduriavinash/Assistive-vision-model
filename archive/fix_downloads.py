import urllib.request, os, ssl
ssl._create_default_https_context = ssl._create_unverified_context

replacements = [
    ("test_images/street_bicycle_08.jpg", "https://images.unsplash.com/photo-1507035895480-2b3156c31fc8?w=640&q=80"),
    ("test_images/sign_store_01.jpg", "https://images.unsplash.com/photo-1506368249639-73a05d6f6488?w=640&q=80"),
    ("test_images/sign_street_03.jpg", "https://images.unsplash.com/photo-1524338198850-8a2ff63aaceb?w=640&q=80"),
    ("test_images/indoor_door_05.jpg", "https://images.unsplash.com/photo-1517502884422-41eaead166d4?w=640&q=80"),
]

for fname, url in replacements:
    try:
        urllib.request.urlretrieve(url, fname)
        size = os.path.getsize(fname) / 1024
        print(f"OK: {fname} ({size:.0f} KB)")
    except Exception as e:
        print(f"FAILED: {fname} - {e}")

count = len([f for f in os.listdir("test_images") if f.endswith(".jpg")])
print(f"\nTotal images now: {count}")
