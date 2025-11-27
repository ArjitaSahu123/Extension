/* popup.js
   Behavior:
   - When "Analyze" is clicked:
     1) Ask the active tab content script for a candidate image URL (message: {action:"getSelectedImage"})
     2) If content script returns a URL, use it. Otherwise use the URL in the input box.
     3) Normalize wrapper URLs (google imgres etc.)
     4) Show preview + loader, call backend, display result.
*/

const BACKEND_URL = "http://127.0.0.1:5000/predict-url";
const qs = id => document.getElementById(id);

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }
function setStatus(text) { const s = qs("status"); if (s) s.textContent = text; }

function clearResultUI() {
  const lbl = qs("result-label");
  const p = qs("result-prob");
  if (lbl) lbl.textContent = "";
  if (p) p.textContent = "";
  qs("prob-bar").style.width = "0%";
  const badge = qs("result-badge");
  badge.textContent = "—";
  badge.className = "badge";
  hide(qs("result-container"));
}

function showPreview(url) {
  const img = qs("preview-img");
  const card = qs("preview-card");
  if (!url) { hide(card); return; }
  img.src = url;
  img.onload = () => show(card);
  img.onerror = () => {
    // If preview failed, still show minimal card
    hide(card);
  };
}

/* ---- Normalization helpers ---- */
function decodePossiblyDoubleEncoded(s) {
  try {
    let d = decodeURIComponent(s);
    if (/%[0-9A-Fa-f]{2}/.test(d)) {
      try { d = decodeURIComponent(d); } catch (e) {}
    }
    return d;
  } catch (e) { return s; }
}

function isLikelyImageUrl(s) {
  if (!s) return false;
  const lower = s.split("?")[0].toLowerCase();
  return /\.(jpg|jpeg|png|gif|webp|bmp|tiff|svg)$/.test(lower);
}

function normalizeImageUrl(url) {
  if (!url || typeof url !== "string") return url;
  // Quick pass: if already looks like image, return
  if (isLikelyImageUrl(url)) return url;

  try {
    const u = new URL(url);
    const host = u.hostname || "";

    // Google Images /imgres
    if (host.endsWith("google.com") && u.pathname.startsWith("/imgres")) {
      const img = u.searchParams.get("imgurl") || u.searchParams.get("img_url") || u.searchParams.get("u");
      if (img) return decodePossiblyDoubleEncoded(img);
    }

    // Generic query params that may contain the actual URL
    const candidateParams = ["imgurl","img_url","u","url","source","media"];
    for (const p of candidateParams) {
      if (u.searchParams.has(p)) {
        const v = u.searchParams.get(p);
        if (v && (v.startsWith("http://") || v.startsWith("https://"))) {
          return decodePossiblyDoubleEncoded(v);
        }
      }
    }

    // Fall back: if any query param value looks like an http(s) URL, return it
    for (const [k, v] of u.searchParams) {
      if (v && (v.startsWith("http://") || v.startsWith("https://"))) {
        return decodePossiblyDoubleEncoded(v);
      }
    }

    // Try small heuristics for hosts like imgur pages to convert to direct image
    if (host.includes("imgur.com") && !isLikelyImageUrl(url)) {
      const parts = u.pathname.split("/").filter(Boolean);
      const id = parts.pop();
      if (id) return `https://i.imgur.com/${id}.jpg`;
    }

    // Last resort: return original
    return url;
  } catch (e) {
    return url;
  }
}

/* ---- Backend call ---- */
async function fetchPredictionForImageUrl(rawUrl) {
  clearError();
  clearResultUI();

  // Normalize and check
  const imageUrl = normalizeImageUrl(rawUrl);
  if (!imageUrl) {
    setError("Could not find a valid image URL to analyze.");
    return;
  }

  // UI: preview + loader
  showPreview(imageUrl);
  show(qs("loader"));
  setStatus("Analyzing image...");

  try {
    const resp = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageUrl })
    });

    hide(qs("loader"));

    if (!resp.ok) {
      const txt = await resp.text().catch(()=>"");
      let msg = resp.statusText || "Server error";
      try {
        const parsed = JSON.parse(txt);
        msg = parsed.message || parsed.error || msg;
      } catch (e) {
        if (txt) msg = txt;
      }
      throw new Error(msg);
    }

    const data = await resp.json();
    if (!data || !data.prediction) throw new Error("Invalid response from server");

    const p = data.prediction;
    const label = (p.label || (p.probability >= 0.5 ? "real" : "fake")).toString().toLowerCase();
    const prob = parseFloat(p.probability) || 0;
    displayResult(label, prob);
    setStatus("");
  } catch (err) {
    hide(qs("loader"));
    console.error("Prediction error", err);
    setStatus("Error");
    setError(err.message || "Request failed");
  }
}

function displayResult(label, prob) {
  label = (label || (prob >= 0.5 ? "real" : "fake")).toString().toLowerCase();
  const percentage = Math.round(prob * 100);

  qs("result-label").textContent = label === "real" ? "Real" : "Fake";
  qs("result-prob").textContent = percentage + "%";
  qs("prob-bar").style.width = Math.min(100, Math.max(0, percentage)) + "%";

  const badge = qs("result-badge");
  badge.textContent = percentage + "%";
  badge.className = "badge " + (label === "real" ? "real" : "fake");

  const bar = qs("prob-bar");
  bar.className = "prob-bar-fill " + (label === "real" ? "real" : "fake");

  show(qs("result-container"));
}

/* ---- Error helpers ---- */
function setError(msg) {
  const box = qs("error-box");
  box.textContent = msg;
  show(box);
}
function clearError() {
  const box = qs("error-box");
  box.textContent = "";
  hide(box);
}

/* ---- Main: wire UI and Analyze button to content script ---- */
document.addEventListener("DOMContentLoaded", function () {
  const input = qs("image-url-input");
  const analyzeBtn = qs("analyze-btn");

  // If user presses Enter in input, trigger analyze (falls back to input)
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") analyzeBtn.click();
  });

  analyzeBtn.addEventListener("click", async () => {
    // 1. Ask active tab content script for candidate
    try {
      const tabs = await new Promise((res) => chrome.tabs.query({ active: true, currentWindow: true }, res));
      if (!tabs || tabs.length === 0) {
        // fallback to input
        const url = input.value.trim();
        if (!url) { setError("Please paste an image URL to analyze."); return; }
        fetchPredictionForImageUrl(url);
        return;
      }
      const tab = tabs[0];
      // Send message to content script; it should respond synchronously
      chrome.tabs.sendMessage(tab.id, { action: "getSelectedImage" }, function (response) {
        // chrome.runtime.lastError indicates no content script
        if (chrome.runtime.lastError) {
          // No content script; fallback to input
          const url = input.value.trim();
          if (!url) { setError("Please paste an image URL to analyze."); return; }
          fetchPredictionForImageUrl(url);
          return;
        }
        const candidate = response && response.imageUrl ? response.imageUrl : null;
        if (candidate) {
          // Use candidate automatically
          input.value = normalizeImageUrl(candidate);
          fetchPredictionForImageUrl(candidate);
        } else {
          // No candidate -> fallback to input value
          const url = input.value.trim();
          if (!url) { setError("Please paste an image URL to analyze."); return; }
          fetchPredictionForImageUrl(url);
        }
      });
    } catch (e) {
      // Unexpected error querying tabs -> fallback to input
      const url = input.value.trim();
      if (!url) { setError("Please paste an image URL to analyze."); return; }
      fetchPredictionForImageUrl(url);
    }
  });

  // Auto-fill the input when popup opens, but do NOT auto-run analysis (user clicks Analyze)
  // This gives user chance to edit URL if desired.
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (!tabs || tabs.length === 0) return;
    const tab = tabs[0];
    chrome.tabs.sendMessage(tab.id, { action: "getSelectedImage" }, function (response) {
      if (chrome.runtime.lastError) {
        // content script not present; keep manual mode
        return;
      }
      if (response && response.imageUrl) {
        const norm = normalizeImageUrl(response.imageUrl);
        input.value = norm;
        showPreview(norm);
      }
    });
  });

  // initialize UI
  clearResultUI();
  clearError();
  hide(qs("loader"));
});
