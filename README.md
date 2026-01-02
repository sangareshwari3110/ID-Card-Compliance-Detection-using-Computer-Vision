# 🎫 ID Card Wearing Detection using YOLOv8

This project detects whether a person is **wearing an ID card (ID tag)** or **not wearing one** using computer vision and deep learning.
It uses a **YOLOv8 pretrained model** for person detection and a **custom-trained YOLO model** for ID card detection.

The system verifies ID compliance by checking whether the detected ID card bounding box lies **inside the person bounding box**.

---

## 🚀 Features

* Detects **persons** using YOLOv8 pretrained model
* Detects **ID cards / ID tags** using a custom-trained YOLO model
* Determines **ID compliance** using bounding box containment logic
* Labels output as:

  * ✅ *ID Tag*
  * ❌ *No ID Tag*
* Works on **images**
* Easily extendable to **video or CCTV streams**

---

## 📁 Project Structure

```
ID-Card-Detection/
│
├── idtags.py               # Main detection script
├── best.pt                 # Custom trained ID card detection model
├── yolov8n.pt              # YOLOv8 pretrained person detection model
│
├── idcard.jpg              # Sample image with ID card
├── no_id.png               # Sample image without ID card
│
├── outputs/
│   ├── ID Tag.png          # Output image (ID detected)
│   └── No ID.png           # Output image (No ID detected)
│
├── README.md
```

---

## 🧠 How It Works

1. **Person Detection**

   * Uses `yolov8n.pt` to detect all persons in the image

2. **ID Card Detection**

   * Uses `best.pt` (custom YOLO model) to detect ID cards

3. **ID Verification Logic**

   * If an ID card bounding box lies completely **inside** a person bounding box → **ID Tag**
   * Otherwise → **No ID Tag**

4. **Visualization**

   * 🟩 Green box → ID Tag detected
   * 🟥 Red box → No ID Tag detected

---

## 🛠️ Requirements

Install the required dependencies using pip:

```bash
pip install ultralytics opencv-python matplotlib
```

---

## ▶️ How to Run

1. Clone the repository or download the project files
2. Place your input image in the project directory
3. Update the image path in `idtags.py` if needed:

```python
image_path = 'no_id.png'
```

4. Run the script:

```bash
python idtags.py
```

5. The output image will be displayed and saved inside the `outputs` folder

---

## 🖼️ Output Explanation

* **Green Bounding Box** → Person wearing ID card
* **Red Bounding Box** → Person not wearing ID card

Each detected person is labeled with:

```
ID Tag / No ID Tag + Confidence Score
```

---

## 🧪 Use Cases

* Office and corporate security
* College and campus monitoring
* Industrial and factory compliance
* Restricted area access control
* Automated ID verification systems

---

## 🔮 Future Enhancements

* Real-time webcam and CCTV support
* Multi-person ID compliance tracking
* Alert system for ID violations
* Face recognition integration
* Web deployment using Flask / FastAPI

---

## 👨‍💻 Author

**Sangareshwari A**
AI Engineer | Computer Vision & Deep Learning
