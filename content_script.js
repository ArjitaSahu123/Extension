// content_script.js
// Heuristics-based image finder + mutation observer.
// Responds to { action: "getSelectedImage" } with { imageUrl: string|null }
// Tracks last clicked image and tries OG meta, largest <img>, data-src/srcset, background-image.

(function () {
  let lastClickedImage = null;
  let lastHoveredImage = null;

  // Track clicks to allow user to click an image on the page to select it
  document.addEventListener("click", (e) => {
    try {
      const el = e.target;
      if (!el) return;
      // If it's an <img>
      if (el.tagName && el.tagName.toLowerCase() === "img" && el.src) {
        lastClickedImage = pickBestFromImg(el);
        flash(el);
        return;
      }
      // If element or ancestor has background-image
      const bg = findBackgroundImageInAncestors(el);
      if (bg) {
        lastClickedImage = bg;
        flash(el);
        return;
      }
      const ds = findDataSrc(el);
      if (ds) {
        lastClickedImage = ds;
        flash(el);
      }
    } catch (_) {}
  }, true);

  // Track hover to help heuristics (optional)
  document.addEventListener("mouseover", (e) => {
    try {
      const el = e.target;
      if (el && el.tagName && el.tagName.toLowerCase() === "img") {
        lastHoveredImage = pickBestFromImg(el);
      }
    } catch (_) {}
  }, true);

  // small visual feedback
  function flash(el) {
    try {
      const old = el.style.outline;
      el.style.outline = "3px solid rgba(124,58,237,0.6)";
      setTimeout(() => (el.style.outline = old), 350);
    } catch (_) {}
  }

  function pickFromSrcset(srcset) {
    try {
      const parts = srcset.split(",").map(s => s.trim()).filter(Boolean);
      if (parts.length === 0) return null;
      const last = parts[parts.length - 1].split(/\s+/)[0];
      return last || parts[0].split(/\s+/)[0];
    } catch (_) { return null; }
  }

  function pickBestFromImg(img) {
    try {
      if (!img) return null;
      // use currentSrc if available (browser picks correct src from srcset)
      if (img.currentSrc) return img.currentSrc;
      if (img.src) return img.src;
      if (img.srcset) {
        const v = pickFromSrcset(img.srcset);
        if (v) return v;
      }
      // data attributes often used for lazy loading
      const ds = findDataSrc(img);
      if (ds) return ds;
    } catch (_) {}
    return null;
  }

  function findDataSrc(el) {
    try {
      if (!el) return null;
      const attrs = ["data-src", "data-srcset", "data-original", "data-lazy", "data-lazy-src", "data-hi-res"];
      for (const a of attrs) {
        if (el.hasAttribute && el.hasAttribute(a)) {
          const v = el.getAttribute(a);
          if (v) return pickFromSrcset(v) || v;
        }
      }
      if (el.tagName && el.tagName.toLowerCase() === "img") {
        if (el.currentSrc) return el.currentSrc;
        if (el.src) return el.src;
        if (el.srcset) return pickFromSrcset(el.srcset);
      }
    } catch (_) {}
    return null;
  }

  function findBackgroundImageInAncestors(el) {
    try {
      let node = el;
      while (node && node !== document.documentElement) {
        const style = window.getComputedStyle(node);
        if (style && style.backgroundImage && style.backgroundImage !== "none") {
          const m = /url\(["']?(.*?)["']?\)/.exec(style.backgroundImage);
          if (m && m[1]) return m[1];
        }
        node = node.parentElement;
      }
    } catch (_) {}
    return null;
  }

  function findOgImage() {
    try {
      const og = document.querySelector('meta[property="og:image"], meta[name="og:image"]');
      if (og && og.content) return og.content;
      const tw = document.querySelector('meta[name="twitter:image"]');
      if (tw && tw.content) return tw.content;
    } catch (_) {}
    return null;
  }

  function findLargestImg() {
    try {
      const imgs = Array.from(document.images || []);
      let best = null, bestArea = 0;
      for (const img of imgs) {
        try {
          const w = img.naturalWidth || img.width || 0;
          const h = img.naturalHeight || img.height || 0;
          const area = (w || 0) * (h || 0);
          if (area > bestArea && img.src) { bestArea = area; best = pickBestFromImg(img); }
        } catch (_) {}
      }
      return best;
    } catch (_) { return null; }
  }

  // Observe DOM changes to capture lazy-loaded images as they appear
  const mo = new MutationObserver(muts => {
    try {
      for (const m of muts) {
        if (m.addedNodes && m.addedNodes.length) {
          for (const n of m.addedNodes) {
            if (n && n.querySelectorAll) {
              const imgs = n.querySelectorAll("img");
              for (const i of imgs) {
                // pick up the first available meaningful image
                if (i && (i.src || i.currentSrc || i.srcset)) {
                  lastHoveredImage = pickBestFromImg(i);
                  return;
                }
              }
            }
          }
        }
      }
    } catch (_) {}
  });
  try { mo.observe(document.documentElement || document.body, { childList: true, subtree: true }); } catch (_) {}

  function findCandidateFromPage() {
    try {
      // Priority: lastClicked > OG meta > hovered > largest image > first image
      if (lastClickedImage) return lastClickedImage;
      const og = findOgImage();
      if (og) return og;
      if (lastHoveredImage) return lastHoveredImage;
      const large = findLargestImg();
      if (large) return large;
      const imgs = Array.from(document.images || []).map(i => pickBestFromImg(i)).filter(Boolean);
      if (imgs.length) return imgs[0];
      return null;
    } catch (_) { return null; }
  }

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg && msg.action === "getSelectedImage") {
      const candidate = findCandidateFromPage();
      sendResponse({ imageUrl: candidate || null });
      return; // synchronous
    }
  });

  // debug statement
  try { console.debug("[content_script] image extractor ready"); } catch (_) {}
})();
