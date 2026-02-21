// background.js (service worker for manifest v3)
// Context menu + Google Images / wrapper URL normalization

const BACKEND_URL = "https://extension-production-7890.up.railway.app/predict-url";

// Create right-click menu on images
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyze_image_ai_vs_real",
    title: "Analyze image (AI vs Real)",
    contexts: ["image"]
  });
});

// When user chooses "Analyze image"
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "analyze_image_ai_vs_real") return;

  let imageUrl = info.srcUrl || info.pageUrl || null;
  if (!imageUrl) {
    notify("AI vs Fake — error", "No image URL available.");
    return;
  }

  // 🔧 IMPORTANT: fix google imgres / wrapper URLs
  imageUrl = normalizeImageUrl(imageUrl);

  try {
    const resp = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageUrl })
    });

    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      console.error("Backend error:", resp.status, txt);
      notify("AI vs Fake — error", "Failed to analyze image. Is the backend running?");
      return;
    }

    const j = await resp.json().catch(() => null);
    if (!j || !j.prediction) {
      notify("AI vs Fake — error", "Invalid response from backend");
      return;
    }

    const p = j.prediction;
    const label = (p.label || (p.probability >= 0.5 ? "real" : "fake"))
      .toString()
      .toLowerCase();
    const prob = Math.round((p.probability || 0) * 100);

    notify(`Result: ${label.toUpperCase()}`, `${prob}% confidence`);
  } catch (err) {
    console.error("Context menu analyze error:", err);
    notify("AI vs Fake — error", "Could not reach backend (is it running?).");
  }
});

// Small notification helper
function notify(title, message) {
  try {
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icons/icon128.png",
      title,
      message
    });
  } catch (e) {
    console.log(title, message);
  }
}

/**
 * Normalize image preview/redirect URLs into a direct image URL.
 * Handles:
 *  - Google Images: https://www.google.com/imgres?imgurl=...
 *  - Other wrappers using imgurl / img_url / u / url parameters
 *  - Double-encoded parameters
 */
function normalizeImageUrl(url) {
  if (!url || typeof url !== "string") return url;

  try {
    const u = new URL(url);
    const host = u.hostname || "";

    // 1) Google Images redirect: /imgres?imgurl=...
    if (host.endsWith("google.com") && u.pathname.startsWith("/imgres")) {
      const img = u.searchParams.get("imgurl") ||
                  u.searchParams.get("img_url") ||
                  u.searchParams.get("u");
      if (img) return decodePossiblyDoubleEncoded(img);
    }

    // 2) Generic: imgurl / u / url params on any host
    if (u.searchParams.has("imgurl")) {
      return decodePossiblyDoubleEncoded(u.searchParams.get("imgurl"));
    }
    if (u.searchParams.has("img_url")) {
      return decodePossiblyDoubleEncoded(u.searchParams.get("img_url"));
    }
    if (u.searchParams.has("u")) {
      return decodePossiblyDoubleEncoded(u.searchParams.get("u"));
    }
    if (u.searchParams.has("url")) {
      const v = u.searchParams.get("url");
      if (v && (v.startsWith("http://") || v.startsWith("https://"))) {
        return decodePossiblyDoubleEncoded(v);
      }
    }

    // 3) Fallback: if any query param looks like a full http(s) URL, use it
    for (const [k, v] of u.searchParams) {
      if (v && (v.startsWith("http://") || v.startsWith("https://"))) {
        return decodePossiblyDoubleEncoded(v);
      }
    }

    // 4) Otherwise just return original
    return url;
  } catch (e) {
    // If URL() fails, just return original
    return url;
  }
}

function decodePossiblyDoubleEncoded(s) {
  try {
    let d = decodeURIComponent(s);
    if (/%[0-9A-Fa-f]{2}/.test(d)) {
      try {
        d = decodeURIComponent(d);
      } catch (_) {}
    }
    return d;
  } catch (e) {
    return s;
  }
}
