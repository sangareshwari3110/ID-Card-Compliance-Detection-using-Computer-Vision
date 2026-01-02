# ===============================
# ID CARD DETECTION
# ===============================


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load person detection model (YOLOv8 pretrained)
person_model = YOLO("yolov8n.pt")  # Or yolov8s.pt for better accuracy

# Load your custom ID tag detection model
id_tag_model = YOLO("best.pt")  # Replace with your model path

# Load image
image_path = 'no_id.png'
image = cv2.imread(image_path)

# Detect persons
person_results = person_model(image_path, conf=0.3)[0]

# Detect ID tags
id_tag_results = id_tag_model(image_path, conf=0.8)[0]

# Get all person boxes
person_boxes = []
for box in person_results.boxes:
    cls_id = int(box.cls[0])
    if person_model.names[cls_id] == 'person':
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_boxes.append(((x1, y1, x2, y2), float(box.conf[0])))

# Get all ID tag boxes
id_tag_boxes = []
for box in id_tag_results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    id_tag_boxes.append((x1, y1, x2, y2))

# Function to check if tag is inside person box
def is_inside(person_box, tag_box):
    px1, py1, px2, py2 = person_box
    tx1, ty1, tx2, ty2 = tag_box
    return px1 <= tx1 <= px2 and py1 <= ty1 <= py2 and px1 <= tx2 <= px2 and py1 <= ty2 <= py2

# Draw results
for person_box, conf in person_boxes:
    has_id_tag = any(is_inside(person_box, tag_box) for tag_box in id_tag_boxes)

    color = (0, 255, 0) if has_id_tag else (0, 0, 255)
    label = "ID Tag" if has_id_tag else "No ID Tag"

    x1, y1, x2, y2 = person_box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    print(label)

# Show output
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()