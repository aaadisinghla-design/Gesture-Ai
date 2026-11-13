import os
import cv2
from ddgs import DDGS
import easyocr
import numpy as np
from ultralytics import YOLOWorld


def measure_blur(img):
    """Calculates sharpness score using Laplacian variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def enhance_blurry_image(img):
    """Sharpens edges and boosts contrast for moving rover cameras."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    contrast_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(contrast_img, (0, 0), 3.0)
    return cv2.addWeighted(contrast_img, 1.8, gaussian, -0.8, 0)


# 1. Load YOLO-World open-vocabulary model & EasyOCR text reader
model = YOLOWorld("yolov8s-world.pt")
reader = easyocr.Reader(["en"])

# 2. Define Scrub Robotics target & safety categories
TRASH_CLASSES = [
    "plastic bottle",
    "aluminum can",
    "soda can",
    "plastic bag",
    "paper cup",
    "wrapper",
    "cardboard",
    "trash",
]
WEED_CLASSES = ["weed", "dandelion", "thistle"]
SAFETY_AVOID_CLASSES = [
    "person",
    "hand",
    "dog",
    "cat",
    "bird",
    "squirrel",
    "flower",
    "tree",
    "bush",
]

all_classes = TRASH_CLASSES + WEED_CLASSES + SAFETY_AVOID_CLASSES
model.set_classes(all_classes)

# BGR Colors
GREEN = (0, 220, 0)      # Safe to collect (Trash)
YELLOW = (0, 220, 255)   # Terrestrial weed target
RED = (0, 0, 255)        # STRICT SAFETY OVERRIDE (Humans, wildlife, native flora)
BLUE = (255, 100, 0)     # OCR Label / Text detection
WHITE = (255, 255, 255)

# 3. Load test image using your exact path
image_path = r"C:\Users\aadit\OneDrive\Documents\Python Projects\pythonw\ai.py\NLP.py\gettyimages-531954524-640x640 (1).jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# --- STEP 1: Blur Check & Enhancement ---
blur_score = measure_blur(image)
if blur_score < 100.0:
    print(f"Blurry image detected (Score: {blur_score:.1f}). Auto-sharpening...")
    process_image = enhance_blurry_image(image)
    conf_threshold = 0.20
else:
    print(f"Clear image detected (Score: {blur_score:.1f}).")
    process_image = image.copy()
    conf_threshold = 0.25

output_image = process_image.copy()
safety_override_triggered = False
detected_objects = set()

# --- STEP 2: Detect, Classify & Apply Safety Logic ---
print("Scanning image for trash, weeds, and safety zones...")
results = model.predict(process_image, conf=conf_threshold)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        obj_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        detected_objects.add(obj_name)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if obj_name in SAFETY_AVOID_CLASSES:
            color = RED
            action = "AVOID - SAFETY STOP"
            safety_override_triggered = True
        elif obj_name in TRASH_CLASSES:
            color = GREEN
            action = f"COLLECT ({cx},{cy})"
            cv2.circle(output_image, (cx, cy), 5, GREEN, -1)
        else:
            color = YELLOW
            action = f"WEED ({cx},{cy})"
            cv2.circle(output_image, (cx, cy), 5, YELLOW, -1)

        label = f"{obj_name} {conf:.2f} | {action}"

        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output_image, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            output_image, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2
        )

# --- STEP 3: Read Text / Labels on Trash or Signs (OCR) ---
print("Scanning for readable text on wrappers, bottles, or signs...")
text_results = reader.readtext(process_image)
detected_texts = []

for bbox, text, confidence in text_results:
    if confidence > 0.40 and len(text.strip()) > 1:
        cleaned_text = text.strip()
        detected_texts.append(cleaned_text)

        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        cv2.rectangle(output_image, top_left, bottom_right, BLUE, 2)
        cv2.putText(
            output_image,
            f"Text: {cleaned_text}",
            (top_left[0], top_left[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BLUE,
            2,
        )

# --- STEP 4: Live Web Search API (Disposal & Item Identification) ---
print("\n--- LIVE WEB SEARCH RESULTS ---")
if detected_texts:
    search_query = f"{' '.join(detected_texts)} recycling disposal"
elif detected_objects:
    # Search live disposal/identification info for the detected items
    items_str = ", ".join(list(detected_objects)[:3])
    search_query = f"how to recycle or dispose of {items_str}"
else:
    search_query = None

if search_query:
    print(f"Querying live web API for: '{search_query}'...\n")
    try:
        web_results = DDGS().text(search_query, max_results=3)
        if web_results:
            for i, r in enumerate(web_results, 1):
                print(f"{i}. {r['title']}")
                print(f"   Summary: {r['body']}")
                print(f"   Link: {r['href']}\n")
        else:
            print("No live web results returned.")
    except Exception as e:
        print(f"Live web search failed: {e}")
else:
    print("Nothing detected in image to search online.")

# --- STEP 5: Master Safety Status Banner ---
if safety_override_triggered:
    banner = "MASTER OVERRIDE: HUMAN / WILDLIFE / FLORA DETECTED - HALT INTAKE"
    banner_color = RED
else:
    banner = "STATUS: CLEAR TO OPERATE"
    banner_color = GREEN

cv2.rectangle(output_image, (10, 10), (630, 45), (0, 0, 0), -1)
cv2.putText(output_image, banner, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2)

# Save and open result in Windows Photo Viewer
output_path = "scrub_robotics_live_test.jpg"
cv2.imwrite(output_path, output_image)
print(f"Saved live vision test to '{output_path}'. Opening...")
os.startfile(output_path)
