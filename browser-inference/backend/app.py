import os
import ssl
import json
from collections import Counter

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.lite.python.interpreter import Interpreter
import paho.mqtt.client as mqtt

# Paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,"yolov8n_saved_model", "yolov8n_int8.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

CONFIDENCE_THRESH = 0.4
IOU_THRESHOLD = 0.5
INPUT_SIZE = 640
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}

# MQTT (disabled by default)
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "vehicles/counts")
CA_CERT = os.getenv("CA_CERT", "")
CLIENT_CERT = os.getenv("CLIENT_CERT", "")
CLIENT_KEY = os.getenv("CLIENT_KEY", "")
ENABLE_BROKER = os.getenv("ENABLE_BROKER", "0") not in {"0", "false", "False", "no", "NO"}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yolov8-tflite-detection.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]
VEHICLE_CLASS_IDS = {i for i, name in enumerate(CLASS_NAMES) if name in VEHICLE_CLASSES}

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

mqtt_client = None
if ENABLE_BROKER:
    mqtt_client = mqtt.Client()
    mqtt_client.tls_set(
        ca_certs=CA_CERT or None,
        certfile=CLIENT_CERT or None,
        keyfile=CLIENT_KEY or None,
        tls_version=ssl.PROTOCOL_TLSv1_2,
    )
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)


def iou(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    a1 = max(x2 - x1, 0) * max(y2 - y1, 0)
    a2 = max(X2 - X1, 0) * max(Y2 - Y1, 0)
    return inter / (a1 + a2 - inter + 1e-6)


def nms(boxes, thr):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        b = boxes.pop(0)
        keep.append(b)
        boxes = [x for x in boxes if x[5] != b[5] or iou(x[:4], b[:4]) < thr]
    return keep


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    img_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image"}

    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    inp = img.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)

    interpreter.set_tensor(inp_det["index"], inp)
    interpreter.invoke()

    raw = interpreter.get_tensor(out_det["index"]).copy()[0]
    preds = raw.T

    boxes = []
    for det in preds:
        class_confs = det[4:]
        cls_id = int(np.argmax(class_confs))
        conf = float(class_confs[cls_id])
        if conf < CONFIDENCE_THRESH:
            continue
        if cls_id not in VEHICLE_CLASS_IDS:
            continue

        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2), conf, cls_id])

    keep = nms(boxes, IOU_THRESHOLD)
    counts = Counter()
    detections = []
    for x1, y1, x2, y2, conf, cls in keep:
        label = CLASS_NAMES[cls]
        counts[label] += 1
        detections.append(
            {
                "label": label,
                "score": round(conf, 4),
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            }
        )

    if mqtt_client is not None:
        try:
            mqtt_client.publish(MQTT_TOPIC, json.dumps(counts))
        except Exception:
            pass

    return {"detections": detections, "counts": counts}
