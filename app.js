const data = window.MODEL_DATA;
const modelKeys = ["logistic", "nn3", "dropout"];
const classColors = data.classes.map((item) => item.color);

const state = {
  modelKey: "logistic",
  point: {
    x: (data.domain.xMin + data.domain.xMax) / 2,
    y: (data.domain.yMin + data.domain.yMax) / 2,
  },
};

const els = {
  metricGrid: document.querySelector("#metricGrid"),
  modelTabs: document.querySelector("#modelTabs"),
  xInput: document.querySelector("#xInput"),
  yInput: document.querySelector("#yInput"),
  xValue: document.querySelector("#xValue"),
  yValue: document.querySelector("#yValue"),
  predictionDot: document.querySelector("#predictionDot"),
  predictionLabel: document.querySelector("#predictionLabel"),
  predictionScore: document.querySelector("#predictionScore"),
  probabilityList: document.querySelector("#probabilityList"),
  legend: document.querySelector("#legend"),
  canvas: document.querySelector("#surfaceCanvas"),
  activeModelBadge: document.querySelector("#activeModelBadge"),
  activeAccuracy: document.querySelector("#activeAccuracy"),
  correctCount: document.querySelector("#correctCount"),
  confusionMatrix: document.querySelector("#confusionMatrix"),
};

const ctx = els.canvas.getContext("2d");

function softmax(logits) {
  const maxValue = Math.max(...logits);
  const exp = logits.map((value) => Math.exp(value - maxValue));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map((value) => value / total);
}

function matVec2(point, weights, bias) {
  return weights.map((row, classIndex) => row[0] * point[0] + row[1] * point[1] + bias[classIndex]);
}

function unpackNetwork(model) {
  const units = model.hiddenUnits;
  const params = model.params;
  let offset = 0;
  const W1 = [[], []];
  for (let input = 0; input < 2; input += 1) {
    for (let unit = 0; unit < units; unit += 1) {
      W1[input][unit] = params[offset];
      offset += 1;
    }
  }
  const W2 = Array.from({ length: units }, () => []);
  for (let unit = 0; unit < units; unit += 1) {
    for (let output = 0; output < 3; output += 1) {
      W2[unit][output] = params[offset];
      offset += 1;
    }
  }
  const b1 = params.slice(offset, offset + units);
  offset += units;
  const b2 = params.slice(offset, offset + 3);
  return { W1, W2, b1, b2, units };
}

const networks = Object.fromEntries(
  modelKeys
    .filter((key) => data.models[key].type === "relu-network")
    .map((key) => [key, unpackNetwork(data.models[key])])
);

function predict(modelKey, point) {
  const model = data.models[modelKey];
  const input = Array.isArray(point) ? point : [point.x, point.y];
  if (model.type === "linear-softmax") {
    return softmax(matVec2(input, model.weights, model.bias));
  }

  const net = networks[modelKey];
  const hidden = Array.from({ length: net.units }, (_, unit) => {
    const value = input[0] * net.W1[0][unit] + input[1] * net.W1[1][unit] + net.b1[unit];
    return Math.max(0, value);
  });
  const logits = [0, 1, 2].map((output) => {
    let value = net.b2[output];
    for (let unit = 0; unit < net.units; unit += 1) {
      value += hidden[unit] * net.W2[unit][output];
    }
    return value;
  });
  return softmax(logits);
}

function classIndex(probs) {
  return probs.reduce((best, value, index) => (value > probs[best] ? index : best), 0);
}

function formatAccuracy(value) {
  return value.toFixed(6);
}

function setupControls() {
  els.xInput.min = data.domain.xMin;
  els.xInput.max = data.domain.xMax;
  els.yInput.min = data.domain.yMin;
  els.yInput.max = data.domain.yMax;
  els.xInput.value = state.point.x;
  els.yInput.value = state.point.y;

  els.xInput.addEventListener("input", () => {
    state.point.x = Number(els.xInput.value);
    render();
  });
  els.yInput.addEventListener("input", () => {
    state.point.y = Number(els.yInput.value);
    render();
  });

  window.addEventListener("resize", () => renderCanvas());
}

function renderStaticShell() {
  els.metricGrid.innerHTML = modelKeys.map((key) => {
    const model = data.models[key];
    return `
      <button class="metric-card" data-model="${key}" type="button">
        <span>
          <p>${model.name}</p>
          <strong>${formatAccuracy(model.accuracy)}</strong>
        </span>
      </button>
    `;
  }).join("");

  els.modelTabs.innerHTML = modelKeys.map((key) => {
    const model = data.models[key];
    return `<button class="model-tab" type="button" role="tab" data-model="${key}">${model.name}</button>`;
  }).join("");

  els.legend.innerHTML = data.classes.map((klass) => `
    <span class="legend-item">
      <span class="legend-swatch" style="background:${klass.color}"></span>
      ${klass.name}
    </span>
  `).join("");

  document.querySelectorAll("[data-model]").forEach((button) => {
    button.addEventListener("click", () => {
      state.modelKey = button.dataset.model;
      render();
    });
  });
}

function chartBounds() {
  const rect = els.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width * dpr));
  const height = Math.max(220, Math.floor(rect.height * dpr));
  if (els.canvas.width !== width || els.canvas.height !== height) {
    els.canvas.width = width;
    els.canvas.height = height;
  }
  return { width, height, dpr };
}

function toCanvas(point, width, height) {
  const pad = 42;
  const x = pad + ((point.x - data.domain.xMin) / (data.domain.xMax - data.domain.xMin)) * (width - pad * 2);
  const y = height - pad - ((point.y - data.domain.yMin) / (data.domain.yMax - data.domain.yMin)) * (height - pad * 2);
  return { x, y };
}

function toDomain(x, y, width, height) {
  const pad = 42;
  const dx = data.domain.xMin + ((x - pad) / (width - pad * 2)) * (data.domain.xMax - data.domain.xMin);
  const dy = data.domain.yMin + ((height - pad - y) / (height - pad * 2)) * (data.domain.yMax - data.domain.yMin);
  return [dx, dy];
}

function drawDecisionSurface(width, height) {
  const pad = 42;
  const step = Math.max(7, Math.round(width / 105));
  for (let y = pad; y < height - pad; y += step) {
    for (let x = pad; x < width - pad; x += step) {
      const probs = predict(state.modelKey, toDomain(x, y, width, height));
      ctx.fillStyle = `${classColors[classIndex(probs)]}24`;
      ctx.fillRect(x, y, step + 1, step + 1);
    }
  }
}

function drawAxes(width, height) {
  const pad = 42;
  ctx.strokeStyle = "#cfd8cc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();

  ctx.fillStyle = "#667064";
  ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("Feature 2", pad, 24);
  ctx.fillText("Feature 1", width - 98, height - 14);
}

function drawSamples(width, height) {
  data.samples.forEach((sample) => {
    const point = toCanvas(sample, width, height);
    const isValidation = sample.split === "validation";
    ctx.beginPath();
    ctx.arc(point.x, point.y, isValidation ? 4.2 : 2.6, 0, Math.PI * 2);
    ctx.fillStyle = classColors[sample.label];
    ctx.globalAlpha = isValidation ? 0.95 : 0.34;
    ctx.fill();
    ctx.globalAlpha = 1;
    if (isValidation) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  });
}

function drawSelectedPoint(width, height) {
  const point = toCanvas(state.point, width, height);
  const probs = predict(state.modelKey, state.point);
  const label = classIndex(probs);

  ctx.strokeStyle = "#111317";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(point.x - 12, point.y);
  ctx.lineTo(point.x + 12, point.y);
  ctx.moveTo(point.x, point.y - 12);
  ctx.lineTo(point.x, point.y + 12);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
  ctx.fillStyle = classColors[label];
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function renderCanvas() {
  const { width, height } = chartBounds();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfa";
  ctx.fillRect(0, 0, width, height);
  drawDecisionSurface(width, height);
  drawAxes(width, height);
  drawSamples(width, height);
  drawSelectedPoint(width, height);
}

function validationStats() {
  const matrix = Array.from({ length: 3 }, () => Array(3).fill(0));
  let correct = 0;
  const validation = data.samples.filter((sample) => sample.split === "validation");
  validation.forEach((sample) => {
    const predicted = classIndex(predict(state.modelKey, [sample.x, sample.y]));
    matrix[sample.label][predicted] += 1;
    if (predicted === sample.label) correct += 1;
  });
  return { matrix, correct, total: validation.length };
}

function renderPrediction() {
  const probs = predict(state.modelKey, state.point);
  const label = classIndex(probs);
  const klass = data.classes[label];

  els.xValue.value = state.point.x.toFixed(2);
  els.yValue.value = state.point.y.toFixed(2);
  els.predictionDot.style.background = klass.color;
  els.predictionDot.style.boxShadow = `0 0 0 5px ${klass.color}22`;
  els.predictionLabel.textContent = klass.name;
  els.predictionScore.textContent = `${probs[label].toFixed(3)} confidence`;
  els.probabilityList.innerHTML = data.classes.map((item, index) => `
    <div class="prob-row">
      <span>${item.name}</span>
      <span class="bar"><span class="bar-fill" style="width:${probs[index] * 100}%; background:${item.color}"></span></span>
      <span class="prob-value">${probs[index].toFixed(3)}</span>
    </div>
  `).join("");
}

function renderDiagnostics() {
  const model = data.models[state.modelKey];
  const stats = validationStats();
  els.activeModelBadge.textContent = model.shortName;
  els.activeAccuracy.textContent = formatAccuracy(model.accuracy);
  els.correctCount.textContent = `${stats.correct} / ${stats.total}`;
  els.confusionMatrix.innerHTML = stats.matrix.flatMap((row, actual) =>
    row.map((value, predicted) => `
      <div class="matrix-cell">
        <strong>${value}</strong>
        <span>A${actual} / P${predicted}</span>
      </div>
    `)
  ).join("");
}

function renderActiveStates() {
  document.querySelectorAll("[data-model]").forEach((element) => {
    const active = element.dataset.model === state.modelKey;
    element.classList.toggle("active", active);
    if (element.classList.contains("model-tab")) {
      element.setAttribute("aria-selected", String(active));
    }
  });
}

function render() {
  renderActiveStates();
  renderPrediction();
  renderDiagnostics();
  renderCanvas();
}

setupControls();
renderStaticShell();
render();
