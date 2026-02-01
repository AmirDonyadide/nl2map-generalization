// webapp/frontend/script.js

// ------------------------------
// Small DOM helpers
// ------------------------------
const $ = (id) => document.getElementById(id);

function show(el) {
  el.classList.remove("hidden");
}
function hide(el) {
  el.classList.add("hidden");
}
function setDisabled(el, disabled) {
  el.disabled = !!disabled;
}
function setText(el, text) {
  el.textContent = text;
}

function setLoading(btnLabelEl, spinnerEl, isLoading, labelText) {
  if (labelText) setText(btnLabelEl, labelText);
  if (isLoading) {
    hide(btnLabelEl);
    show(spinnerEl);
  } else {
    show(btnLabelEl);
    hide(spinnerEl);
  }
}

// ------------------------------
// State
// ------------------------------
let inputGeoJSON = null;
let outputGeoJSON = null;
let inputFilename = "input.geojson";
let outputFilename = "output-generalized.geojson";

let mapInput = null;
let mapOutput = null;
let inputLayer = null;
let outputLayer = null;

// ------------------------------
// Leaflet helpers
// ------------------------------
function createBaseMap(containerId) {
  const map = L.map(containerId, {
    zoomControl: true,
    preferCanvas: true,
  });

  // OSM tiles (free). If you don't want tiles, remove these 2 lines and keep a blank background.
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 22,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  // Default view (will be fit to data on load)
  map.setView([0, 0], 2);
  return map;
}

function layerStyle(feature) {
  // Keep it simple, don't set explicit colors in code if you prefer.
  // But Leaflet needs some style for polygons/lines.
  return {
    weight: 2,
    opacity: 0.9,
    fillOpacity: 0.15,
  };
}

function addGeoJSONToMap(map, geojson, existingLayer) {
  if (existingLayer) {
    map.removeLayer(existingLayer);
  }

  const layer = L.geoJSON(geojson, {
    style: layerStyle,
    pointToLayer: (feature, latlng) => L.circleMarker(latlng, { radius: 4 }),
  }).addTo(map);

  try {
    const b = layer.getBounds();
    if (b.isValid()) map.fitBounds(b.pad(0.1));
  } catch (e) {
    // ignore fitBounds issues
  }

  return layer;
}

// ------------------------------
// GeoJSON helpers
// ------------------------------
async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result || ""));
    r.onerror = () => reject(r.error || new Error("File read failed"));
    r.readAsText(file);
  });
}

function safeJsonParse(text) {
  try {
    return { ok: true, value: JSON.parse(text) };
  } catch (e) {
    return { ok: false, error: e };
  }
}

function isFeatureCollection(obj) {
  return obj && typeof obj === "object" && obj.type === "FeatureCollection" && Array.isArray(obj.features);
}

function downloadJson(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/geo+json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ------------------------------
// API helpers
// ------------------------------
function getApiBase() {
  // allows "/api" when served by FastAPI, or "http://127.0.0.1:8000/api" if needed
  const v = $("apiBase").value.trim();
  return v || "/api";
}

async function apiGetModels() {
  const base = getApiBase();
  const res = await fetch(`${base}/models`, { method: "GET" });
  if (!res.ok) throw new Error(`GET /models failed (${res.status})`);
  return await res.json();
}

async function apiGeneralize(payload) {
  const base = getApiBase();
  const res = await fetch(`${base}/generalize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`POST /generalize failed (${res.status}) ${txt}`);
  }
  return await res.json();
}

// ------------------------------
// UI wiring
// ------------------------------
function resetAll() {
  inputGeoJSON = null;
  outputGeoJSON = null;
  inputFilename = "input.geojson";
  outputFilename = "output-generalized.geojson";

  $("geojsonFile").value = "";
  $("promptText").value = "";
  // modelSelect stays loaded

  setDisabled($("clearBtn"), true);
  setDisabled($("downloadInputBtn"), true);
  setDisabled($("downloadOutputBtn"), true);

  hide($("workspace"));
  hide($("error"));
  show($("upload"));

  hide($("predictionCard"));
  hide($("warnings"));

  setText($("predOperator"), "—");
  setText($("predParam"), "—");
  setText($("predValue"), "—");
  setText($("predConf"), "—");

  if (mapInput && inputLayer) {
    mapInput.removeLayer(inputLayer);
    inputLayer = null;
  }
  if (mapOutput && outputLayer) {
    mapOutput.removeLayer(outputLayer);
    outputLayer = null;
  }
  if (mapOutput) {
    $("mapOutput").classList.add("map-muted");
  }
}

function showError(msg) {
  setText($("errorMsg"), msg);
  hide($("upload"));
  hide($("workspace"));
  show($("error"));
}

function fillModelDropdown(models) {
  const sel = $("modelSelect");
  sel.innerHTML = "";

  if (!Array.isArray(models) || models.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.disabled = true;
    opt.selected = true;
    opt.textContent = "No models found (check /models folder)";
    sel.appendChild(opt);
    return;
  }

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.disabled = true;
  placeholder.selected = true;
  placeholder.textContent = "Select a model…";
  sel.appendChild(placeholder);

  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name || m.id;
    sel.appendChild(opt);
  }
}

async function initModels() {
  try {
    const models = await apiGetModels();
    fillModelDropdown(models);
  } catch (e) {
    // If models fail, still let user proceed; dropdown will show error message.
    fillModelDropdown([]);
    console.error(e);
  }
}

// ------------------------------
// Event handlers
// ------------------------------
async function onUploadClicked() {
  const file = $("geojsonFile").files?.[0];
  if (!file) {
    showError("Please choose a GeoJSON file first.");
    return;
  }

  setLoading($("uploadLabel"), $("uploadSpin"), true);

  try {
    inputFilename = file.name || "input.geojson";
    const text = await readFileAsText(file);
    const parsed = safeJsonParse(text);

    if (!parsed.ok) throw new Error("Could not parse the file as JSON.");

    const geo = parsed.value;
    if (!isFeatureCollection(geo)) {
      throw new Error("GeoJSON must be a FeatureCollection with a 'features' array.");
    }

    inputGeoJSON = geo;
    setDisabled($("clearBtn"), false);
    setDisabled($("downloadInputBtn"), false);

    // create maps once
    if (!mapInput) mapInput = createBaseMap("mapInput");
    if (!mapOutput) mapOutput = createBaseMap("mapOutput");

    // render input preview
    inputLayer = addGeoJSONToMap(mapInput, inputGeoJSON, inputLayer);

    // clear output preview
    if (mapOutput && outputLayer) {
      mapOutput.removeLayer(outputLayer);
      outputLayer = null;
    }
    $("mapOutput").classList.add("map-muted");

    hide($("upload"));
    show($("workspace"));
    hide($("predictionCard"));
  } catch (e) {
    console.error(e);
    showError(e.message || "Upload failed.");
  } finally {
    setLoading($("uploadLabel"), $("uploadSpin"), false);
  }
}

function onClearClicked() {
  resetAll();
}

function onDownloadInput() {
  if (!inputGeoJSON) return;
  downloadJson(inputGeoJSON, inputFilename || "input.geojson");
}

function onDownloadOutput() {
  if (!outputGeoJSON) return;
  downloadJson(outputGeoJSON, outputFilename || "output-generalized.geojson");
}

async function onSendClicked() {
  if (!inputGeoJSON) {
    showError("No input GeoJSON found. Please upload a file first.");
    return;
  }

  const prompt = $("promptText").value.trim();
  const modelId = $("modelSelect").value;

  if (!prompt) {
    showError("Please write a prompt before sending.");
    return;
  }
  if (!modelId) {
    showError("Please select a model.");
    return;
  }

  setLoading($("sendLabel"), $("sendSpin"), true);

  try {
    const payload = {
      model_id: modelId,
      prompt: prompt,
      geojson: inputGeoJSON,
    };

    const res = await apiGeneralize(payload);

    // Fill prediction
    const pred = res.prediction || {};
    setText($("predOperator"), pred.operator ?? "—");
    setText($("predParam"), pred.param_name ?? "—");
    setText($("predValue"), pred.param_value ?? "—");
    setText($("predConf"), pred.confidence ?? "—");

    // Warnings
    const warnings = Array.isArray(res.warnings) ? res.warnings : [];
    if (warnings.length > 0) {
      const wEl = $("warnings");
      wEl.innerHTML = warnings.map((w) => `<div class="warning-item">${escapeHtml(w)}</div>`).join("");
      show(wEl);
    } else {
      hide($("warnings"));
    }

    // Output GeoJSON
    outputGeoJSON = res.output_geojson || null;
    if (!outputGeoJSON) throw new Error("Server response missing output_geojson.");

    // Render output preview next to input
    if (!mapOutput) mapOutput = createBaseMap("mapOutput");
    outputLayer = addGeoJSONToMap(mapOutput, outputGeoJSON, outputLayer);
    $("mapOutput").classList.remove("map-muted");

    // Enable download output
    setDisabled($("downloadOutputBtn"), false);
    outputFilename = `generalized-${inputFilename || "output.geojson"}`;

    show($("predictionCard"));
  } catch (e) {
    console.error(e);
    showError(e.message || "Request failed.");
  } finally {
    setLoading($("sendLabel"), $("sendSpin"), false);
  }
}

function onErrorBack() {
  hide($("error"));
  // go back to upload if no input, otherwise workspace
  if (inputGeoJSON) {
    hide($("upload"));
    show($("workspace"));
  } else {
    show($("upload"));
    hide($("workspace"));
  }
}

// basic HTML escape for warnings
function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ------------------------------
// Init
// ------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  // Wire events
  $("uploadBtn").addEventListener("click", onUploadClicked);
  $("clearBtn").addEventListener("click", onClearClicked);

  $("downloadInputBtn").addEventListener("click", onDownloadInput);
  $("downloadOutputBtn").addEventListener("click", onDownloadOutput);

  $("sendBtn").addEventListener("click", onSendClicked);

  $("errorBackBtn").addEventListener("click", onErrorBack);

  // Initialize models
  await initModels();

  // Start clean
  resetAll();
});
