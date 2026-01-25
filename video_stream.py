import cv2
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter, load_delegate
from collections import Counter
import time
import json
import ssl
import paho.mqtt.client as mqtt
import tempfile
import os
import atexit

try:
    import streamlit as st
except Exception:
    st = None

# ─── Config ────────────────────────────────────
MODEL_PATH        = "./yolov8n_saved_model/yolov8n_int8.tflite"
LABELS_PATH       = "labels.txt"
CONFIDENCE_THRESH = 0.4
IOU_THRESHOLD     = 0.5
WEBCAM_INDEX      = 0
INPUT_SIZE        = 640
VEHICLE_CLASSES   = {"car", "bus", "truck", "motorcycle", "bicycle"}
FALLBACK_IMAGE    = "vehicle-and-person.jpg"
DEFAULT_VIDEO     = "test-video.mp4"

# ─── MQTT Config ──────────────────────────────
MQTT_BROKER     = "localhost"
MQTT_PORT       = 8883  # TLS port
MQTT_TOPIC      = "vehicles/counts"
CA_CERT         = "mqtt-certs/ca.crt"
CLIENT_CERT     = "mqtt-certs/client.crt"
CLIENT_KEY      = "mqtt-certs/client.key"
ENABLE_MQTT     = os.getenv("ENABLE_MQTT", "1") not in {"0", "false", "False", "no", "NO"}

_TEMP_CERT_FILES = []

def _write_secret_to_tempfile(secret_value):
    # Accept multiline PEM or "\n"-escaped strings from secrets/env.
    if secret_value is None:
        return None
    content = secret_value.replace("\\n", "\n")
    tfh = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tfh.write(content)
    tfh.flush()
    tfh.close()
    _TEMP_CERT_FILES.append(tfh.name)
    return tfh.name

def _cleanup_tempfiles():
    for path in _TEMP_CERT_FILES:
        try:
            os.remove(path)
        except OSError:
            pass

atexit.register(_cleanup_tempfiles)

def _get_secret(name):
    if st is not None and hasattr(st, "secrets") and name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)

_ca_secret = _get_secret("CA_CERT")
_client_cert_secret = _get_secret("CLIENT_CERT")
_client_key_secret = _get_secret("CLIENT_KEY")
_enable_mqtt_secret = _get_secret("ENABLE_MQTT")
if _enable_mqtt_secret is not None:
    ENABLE_MQTT = str(_enable_mqtt_secret) not in {"0", "false", "False", "no", "NO"}

if _ca_secret and _client_cert_secret and _client_key_secret:
    CA_CERT = _write_secret_to_tempfile(_ca_secret)
    CLIENT_CERT = _write_secret_to_tempfile(_client_cert_secret)
    CLIENT_KEY = _write_secret_to_tempfile(_client_key_secret)

if ENABLE_MQTT:
    # ─── MQTT Setup with TLS ──────────────────────
    mqtt_client = mqtt.Client()
    mqtt_client.tls_set(
        ca_certs=CA_CERT,
        certfile=CLIENT_CERT,
        keyfile=CLIENT_KEY,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
else:
    mqtt_client = None

# ─── Load Class Labels ───────────────────────
with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]
VEHICLE_CLASS_IDS = {i for i, name in enumerate(CLASS_NAMES) if name in VEHICLE_CLASSES}

# ─── Initialize Interpreter ──────────────────
try:
    xnn = load_delegate('libxnnpack_delegate.so')
    interpreter = Interpreter(model_path=MODEL_PATH,
                              experimental_delegates=[xnn])
    print("✅ XNNPACK delegate loaded")
except Exception:
    interpreter = Interpreter(model_path=MODEL_PATH)
    print("⚠️ XNNPACK delegate unavailable, using CPU")

interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

# ─── IOU and NMS ─────────────────────────────
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

# ─── Frame Generator ─────────────────────────
def get_video_stream(video_path=None):
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    using_video = False
    if not cap.isOpened():
        cap.release()
        source = video_path or DEFAULT_VIDEO
        cap = cv2.VideoCapture(source)
        using_video = cap.isOpened()
    use_fallback = not cap.isOpened()
    if use_fallback:
        cap.release()
        fallback_frame = cv2.imread(FALLBACK_IMAGE)
        if fallback_frame is None:
            # Last-resort: return a blank frame so the app doesn't crash.
            fallback_frame = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

    prev = time.time()
    while True:
        if use_fallback:
            frame = fallback_frame.copy()
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                if using_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
        h, w = frame.shape[:2]

        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        inp = img.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, 0)

        interpreter.set_tensor(inp_det['index'], inp)
        interpreter.invoke()

        raw = interpreter.get_tensor(out_det['index']).copy()[0]  # (84,8400)
        preds = raw.T  # (8400,84)

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
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h

            boxes.append([max(0,x1), max(0,y1), min(w,x2), min(h,y2), conf, cls_id])

        keep = nms(boxes, IOU_THRESHOLD)

        counts = Counter()
        for x1, y1, x2, y2, conf, cls in keep:
            label = CLASS_NAMES[cls]
            if label in VEHICLE_CLASSES:
                counts[label] += 1
                # Draw box for vehicles only
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"{label}:{conf:.2f}", (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Publish counts over TLS
        if mqtt_client is not None:
            try:
                mqtt_client.publish(MQTT_TOPIC, json.dumps(counts))
            except Exception as e:
                print(f"MQTT publish failed: {e}")

        # FPS overlay
        now = time.time()
        fps = 1/(now-prev); prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Yield RGB frame + counts
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield rgb_frame, counts

    if not use_fallback:
        cap.release()
