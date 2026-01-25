# Browser Inference (Webcam, Client-Side)

This is a **separate** browser-based demo that runs inference via a **local FastAPI backend** (no server webcam required).

## Run

Start the backend:

```bash
cd browser-inference/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open `index.html` in a browser (or serve this folder with any static server).

If you need a simple local server:

```bash
python3 -m http.server 8000
```

Then visit: `http://localhost:8000`

## Notes

- This demo is intentionally standalone and does **not** modify the existing Streamlit/Python pipeline.
- Backend uses the existing `yolov8n_int8.tflite` model and filters to vehicles only.
- MQTT is disabled by default; set `ENABLE_BROKER=1` and provide broker env vars to enable.
