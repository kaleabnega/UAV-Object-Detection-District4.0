const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const scoreInput = document.getElementById("scoreInput");
const scoreValue = document.getElementById("scoreValue");
const fpsEl = document.getElementById("fps");
const countEl = document.getElementById("count");
const backendUrlInput = "http://localhost:8000";
const chartCanvas = document.getElementById("chart");
const chartCtx = chartCanvas.getContext("2d");
const yAxis = document.getElementById("yAxis");

let stream = null;
let running = false;
let lastFrameTime = performance.now();
let inFlight = false;
const SAMPLE_VIDEO = "test-video.mp4";

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

const VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle", "bicycle"];
const HISTORY_LEN = 30;
const history = VEHICLE_CLASSES.reduce((acc, key) => {
  acc[key] = [];
  return acc;
}, {});

const chartColors = {
  car: "#FFD400",
  bus: "#7CFF6B",
  truck: "#5CC8FF",
  motorcycle: "#FF7CE0",
  bicycle: "#FF9F5C",
};

scoreInput.addEventListener("input", () => {
  scoreValue.textContent = Number(scoreInput.value).toFixed(2);
});

async function startVideo() {
  video.srcObject = null;
  video.src = SAMPLE_VIDEO;
  video.loop = true;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

function resizeCanvas() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  chartCanvas.width = chartCanvas.clientWidth * window.devicePixelRatio;
  chartCanvas.height = 220 * window.devicePixelRatio;
}

function drawDetections(detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 4;
  ctx.font = "16px Fira Sans, sans-serif";
  ctx.strokeStyle = "#FFD400";
  ctx.fillStyle = "#FFD400";
  ctx.shadowColor = "rgba(0,0,0,0.7)";
  ctx.shadowBlur = 6;

  let count = 0;
  detections.forEach((d) => {
    count += 1;
    const [x, y, w, h] = d.bbox;
    ctx.strokeRect(x, y, w, h);
    const label = `${d.label} ${(d.score * 100).toFixed(0)}%`;
    const textY = Math.max(y - 6, 18);
    ctx.fillText(label, x + 4, textY);
  });
  ctx.shadowBlur = 0;
  countEl.textContent = String(count);
}

function updateHistory(counts) {
  VEHICLE_CLASSES.forEach((key) => {
    const arr = history[key];
    arr.push(counts[key] || 0);
    if (arr.length > HISTORY_LEN) arr.shift();
  });
}

function drawChart() {
  const w = chartCanvas.width;
  const h = chartCanvas.height;
  chartCtx.clearRect(0, 0, w, h);

  chartCtx.strokeStyle = "#1f2636";
  chartCtx.lineWidth = 1;
  for (let i = 1; i <= 4; i += 1) {
    const y = (h * i) / 5;
    chartCtx.beginPath();
    chartCtx.moveTo(0, y);
    chartCtx.lineTo(w, y);
    chartCtx.stroke();
  }

  const maxVal = Math.max(1, ...VEHICLE_CLASSES.flatMap((k) => history[k]));
  const stepX = w / (HISTORY_LEN - 1);

  if (yAxis) {
    const ticks = 4;
    const labels = [];
    for (let i = ticks; i >= 0; i -= 1) {
      const v = Math.round((maxVal * i) / ticks);
      labels.push(v);
    }
    yAxis.innerHTML = labels.map((v) => `<span>${v}</span>`).join("");
  }

  VEHICLE_CLASSES.forEach((key) => {
    const arr = history[key];
    if (arr.length === 0) return;
    chartCtx.strokeStyle = chartColors[key] || "#FFFFFF";
    chartCtx.lineWidth = 2;
    chartCtx.beginPath();
    arr.forEach((val, idx) => {
      const x = idx * stepX;
      const y = h - (val / maxVal) * (h - 10) - 5;
      if (idx === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();
  });
}

async function detectFrame() {
  if (!running) return;

  const t0 = performance.now();
  if (!inFlight) {
    inFlight = true;
    const minScore = Number(scoreInput.value);
    captureCtx.drawImage(
      video,
      0,
      0,
      captureCanvas.width,
      captureCanvas.height,
    );
    const blob = await new Promise((resolve) =>
      captureCanvas.toBlob(resolve, "image/jpeg", 0.8),
    );
    const form = new FormData();
    form.append("file", blob, "frame.jpg");

    try {
      const baseUrl = backendUrlInput;
      const res = await fetch(`${baseUrl}/detect`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      const detections = (data.detections || []).filter(
        (d) => d.score >= minScore,
      );
      drawDetections(detections);
      updateHistory(data.counts || {});
      drawChart();
    } catch (err) {
      // ignore network errors; keep UI responsive
    } finally {
      inFlight = false;
    }
  }

  const t1 = performance.now();
  const fps = 1000 / (t1 - lastFrameTime);
  lastFrameTime = t1;
  fpsEl.textContent = fps.toFixed(1);

  requestAnimationFrame(detectFrame);
}

async function start() {
  startBtn.disabled = true;
  await startVideo();
  resizeCanvas();
  running = true;
  stopBtn.disabled = false;
  requestAnimationFrame(detectFrame);
}

function stop() {
  running = false;
  stopBtn.disabled = true;
  startBtn.disabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  fpsEl.textContent = "0.0";
  countEl.textContent = "0";
  VEHICLE_CLASSES.forEach((key) => {
    history[key] = [];
  });
  drawChart();
  stopCamera();
  video.pause();
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
window.addEventListener("resize", () => {
  if (running) {
    resizeCanvas();
    drawChart();
  }
});
