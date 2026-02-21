const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const strokeSlider = document.getElementById("strokeSize");
const strokeValue = document.getElementById("strokeValue");
const predictBtn = document.getElementById("predictBtn");

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// API endpoint
const API_URL = "http://localhost:5000/api/predict";

// Initialize canvas
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "black";
ctx.lineWidth = 12;
ctx.lineCap = "round";
ctx.lineJoin = "round";

// Stroke size control
strokeSlider.addEventListener("input", (e) => {
  ctx.lineWidth = e.target.value;
  strokeValue.textContent = e.target.value;
});

// Mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

// Touch events
canvas.addEventListener("touchstart", handleTouchStart);
canvas.addEventListener("touchmove", handleTouchMove);
canvas.addEventListener("touchend", stopDrawing);

function startDrawing(e) {
  isDrawing = true;
  [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
  if (!isDrawing) return;
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDrawing() {
  isDrawing = false;
}

function handleTouchStart(e) {
  e.preventDefault();
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  isDrawing = true;
  [lastX, lastY] = [x, y];
}

function handleTouchMove(e) {
  if (!isDrawing) return;
  e.preventDefault();
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
}

function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").innerHTML =
    '<div class="loading">Draw a character and click Predict</div>';
}

async function predict() {
  try {
    // Disable button
    predictBtn.disabled = true;
    document.getElementById("result").innerHTML =
      '<div class="loading">Predicting...</div>';

    // Get canvas as base64
    const imageData = canvas.toDataURL("image/png");

    // Send to API
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: imageData }),
    });

    const data = await response.json();

    if (data.success) {
      // Display results
      const resultHTML = `
                <div class="prediction-title">Prediction</div>
                <div class="main-prediction">${data.prediction}</div>
                <div class="confidence">${data.confidence.toFixed(1)}% confidence</div>
                <div class="alternatives">
                    ${data.top5
                      .slice(1)
                      .map(
                        (item) =>
                          `<div class="alt-item">${item.character}: ${item.confidence.toFixed(1)}%</div>`,
                      )
                      .join("")}
                </div>
            `;
      document.getElementById("result").innerHTML = resultHTML;
    } else {
      document.getElementById("result").innerHTML =
        `<div class="error">Error: ${data.error}</div>`;
    }
  } catch (error) {
    console.error("Error:", error);
    document.getElementById("result").innerHTML =
      '<div class="error">Failed to connect to server. Make sure the Flask server is running on port 5000.</div>';
  } finally {
    predictBtn.disabled = false;
  }
}
