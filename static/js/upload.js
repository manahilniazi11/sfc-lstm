/*
 * Upload page: for every chosen file,
 *   1. GTM classifies it here in the browser (gtm.js),
 *   2. the file and GTM's finished scores are sent to the server,
 *   3. the server validates it, runs the Python model and makes the decision.
 * Files are sent one at a time because each carries its own GTM result.
 */
(function () {
  "use strict";
  const form = document.getElementById("upload-form");
  if (!form) return;
  const input = document.getElementById("audio-input");
  const drop = document.getElementById("drop-zone");
  const list = document.getElementById("upload-list");
  const button = document.getElementById("upload-button");
  const preview = document.getElementById("preview-player");
  const cfg = JSON.parse(document.getElementById("upload-config").textContent);
  const gtm = cfg.gtm_available ? new GtmClassifier(cfg.gtm_url, cfg.gtm_version) : null;
  const csrf = form.querySelector("input[name=csrfmiddlewaretoken]").value;
  const allowed = [".wav", ".mp3", ".flac", ".ogg", ".m4a"];

  function extension(name) { const i = name.lastIndexOf("."); return i < 0 ? "" : name.slice(i).toLowerCase(); }

  function row(file) {
    const li = document.createElement("li");
        li.innerHTML = '<i class="bi bi-file-earmark-music"></i><span class="file-name font-semibold break-words"></span>' +
      '<span class="status ms-auto text-sm text-ss-muted">Waiting</span>';
    li.querySelector(".file-name").textContent = file.name;
    list.appendChild(li);
    return li;
  }

  function setStatus(li, html, cls) {
    const s = li.querySelector(".status");
    s.className = "status ms-auto text-sm " + (cls || "text-ss-muted");
    s.innerHTML = html;
  }

  function escapeHtml(text) {
    const d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
  }

  function showPreview() {
    const file = input.files[0];
    if (!file || !preview) return;
    preview.src = URL.createObjectURL(file);
    preview.closest(".preview-box").classList.remove("hidden");
  }

  async function analyse(file, li, batch) {
    if (!allowed.includes(extension(file.name))) {
      setStatus(li, "Unsupported format: use WAV, MP3, FLAC, OGG or M4A", "text-red-600");
      return;
    }
    if (file.size > cfg.max_mb * 1048576) {
      setStatus(li, "Larger than " + cfg.max_mb + " MB", "text-red-600");
      return;
    }
    let gtmResult = { version: cfg.gtm_version, error: "GTM model not installed" };
    if (gtm) {
      setStatus(li, '<span class="spinner" aria-hidden="true"></span> GTM is listening in your browser');
      gtmResult = await gtm.classifyFile(file);
    }
    setStatus(li, '<span class="spinner" aria-hidden="true"></span> Server: validating and running the Python model');
    const body = new FormData();
    body.append("audio", file);
    body.append("gtm", JSON.stringify(gtmResult));
    if (batch) body.append("batch", "1");
    try {
      const response = await fetch(cfg.analyse_url, {
        method: "POST", body: body, credentials: "same-origin",
        headers: { "X-CSRFToken": csrf, "X-Requested-With": "fetch" },
      });
      const result = await response.json();
      if (result.ok) {
        const sev = result.severity.toLowerCase();
        setStatus(li, '<a href="' + result.url + '">' + escapeHtml(result.code) + "</a> · " +
          escapeHtml(result.final_class) + ' <span class="badge sev sev-' + sev + '">' + escapeHtml(result.severity) + "</span> · " +
          escapeHtml(result.status) + (result.duplicate ? ' · <span class="text-amber-600">duplicate</span>' : ""), "text-ss-ink");
        return result;
      }
      setStatus(li, escapeHtml(result.error || "Rejected"), "text-red-600");
    } catch (e) {
      setStatus(li, "The server could not be reached or failed to answer.", "text-red-600");
    }
  }

  async function run(files) {
    if (!files.length) return;
    button.disabled = true;
    list.innerHTML = "";
    const batch = files.length > 1;
    const rows = files.map(row);
    let last = null;
    for (let i = 0; i < files.length; i++) {
      const result = await analyse(files[i], rows[i], batch);
      if (result) last = result;
    }
    button.disabled = false;
    if (files.length === 1 && last) window.location.href = last.url; // single file: open its result
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    run(Array.from(input.files));
  });
  input.addEventListener("change", showPreview);
  ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (e) => {
    const files = Array.from(e.dataTransfer.files);
    if (!cfg.batch && files.length > 1) {
      list.innerHTML = '<li class="text-red-600">Your role can upload one file at a time.</li>';
      return;
    }
    const dt = new DataTransfer();
    files.forEach(f => dt.items.add(f));
    input.files = dt.files;
    showPreview();
  });
  if (gtm) gtm.load().catch(() => { /* reported per file when used */ });
})();
