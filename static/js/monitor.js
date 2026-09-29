/*
 * Live microphone monitoring (SRS Step 12, FR vi-vii, lxiv, lxxix).
 *
 * After the user presses Start (never before), the page:
 *   1. captures the chosen input continuously (browser echo cancellation,
 *      noise suppression and auto-gain are switched off: they are made for
 *      voice calls and would muffle sirens, glass or gunshots);
 *   2. cuts it into fixed windows (2 s by default, set by the administrator);
 *   3. runs GTM on each window here in the browser;
 *   4. sends the window's audio and GTM's scores to the server, which runs the
 *      Python model, compares the two and applies the alert rules;
 *   5. shows the result, the waveform and a scrolling spectrogram.
 * Windows are processed in order; if the server falls behind, the oldest
 * waiting window is dropped (and counted) so results stay close to real time.
 */
(function () {
  "use strict";
  const cfg = JSON.parse(document.getElementById("monitor-config").textContent);
  const csrf = document.querySelector("input[name=csrfmiddlewaretoken]").value;
  const $ = (id) => document.getElementById(id);
  const ui = {
    status: $("mic-status"), device: $("device-select"), start: $("btn-start"), pause: $("btn-pause"), stop: $("btn-stop"),
    banner: $("privacy-banner"), session: $("session-code"), elapsed: $("session-elapsed"), windows: $("session-windows"),
    dropped: $("session-dropped"), current: $("current-sound"), currentMeta: $("current-meta"), severity: $("current-severity"),
    python: $("python-result"), gtm: $("gtm-result"), agreement: $("agreement"), alertBox: $("active-alert"),
    recent: $("recent-body"), level: $("level-bar"), levelText: $("level-text"), wave: $("wave-canvas"), spec: $("spec-canvas"),
    latency: $("latency"),
  };
  const gtm = cfg.gtm_available ? new GtmClassifier(cfg.gtm_url, cfg.gtm_version) : null;
  if (gtm) gtm.load().catch(() => {});

  let stream = null, ctx = null, analyser = null, worklet = null, session = null, paused = false;
  let collected = [], collectedLength = 0, windowIndex = 0, startedAt = 0;
  const queue = [];
  let busy = false, dropped = 0, analysed = 0, silentCount = 0, animation = null, clock = null;

  const STATUS = {
    unavailable: ["No microphone found", "secondary"], available: ["Available", "success"], active: ["Active", "danger"],
    paused: ["Paused", "warning"], disconnected: ["Disconnected", "dark"], denied: ["Permission denied", "danger"],
    starting: ["Starting…", "info"],
  };
  function setStatus(key) {
    const [text, colour] = STATUS[key];
    ui.status.textContent = text;
    ui.status.className = "badge text-base text-bg-" + colour;
    const live = key === "active" || key === "paused";
    ui.banner.classList.toggle("hidden", !live);
    ui.start.disabled = live || key === "starting" || key === "unavailable";
    ui.pause.disabled = !live;
    ui.stop.disabled = !live;
    ui.device.disabled = live;
    ui.pause.innerHTML = paused ? '<i class="bi bi-play-fill"></i> Resume' : '<i class="bi bi-pause-fill"></i> Pause';
    if (key === "denied") {
      ui.currentMeta.textContent = "The browser blocks the microphone for this site. Allow it with the icon at the left of the address bar (Site settings > Microphone), then press Start.";
    } else if (key === "unavailable") {
      ui.currentMeta.textContent = "No input device found. Connect a microphone, or enable Stereo Mix in Windows' sound settings.";
    } else if (key === "disconnected") {
      ui.currentMeta.textContent = "The input device was disconnected. Reconnect it and press Start.";
    }
  }

  function escapeHtml(text) { const d = document.createElement("div"); d.textContent = text == null ? "" : String(text); return d.innerHTML; }
  function pct(v) { return v == null ? "—" : (v * 100).toFixed(0) + "%"; }

  async function listDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) { setStatus("unavailable"); return; }
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "audioinput");
    const chosen = ui.device.value;
    ui.device.innerHTML = "";
    devices.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || ("Microphone " + (i + 1));
      ui.device.appendChild(o);
    });
    if (chosen) ui.device.value = chosen;
    if (!stream) setStatus(devices.length ? "available" : "unavailable");
  }

  async function checkPermission() {
    try {
      const p = await navigator.permissions.query({ name: "microphone" });
      if (p.state === "denied") setStatus("denied");
      p.onchange = () => { if (p.state === "denied") { stopMonitoring(true); setStatus("denied"); } else listDevices(); };
    } catch (e) { /* the Permissions API is optional */ }
  }

  async function post(url, body) {
    const response = await fetch(url, { method: "POST", body, credentials: "same-origin", headers: { "X-CSRFToken": csrf } });
    return [response.status, await response.json()];
  }

  function wavBytes(samples, rate) { // 16-bit PCM mono WAV
    const buffer = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buffer);
    const text = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    text(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE"); text(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
    v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); text(36, "data");
    v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  function onSamples(block) {
    if (paused || !session) return;
    collected.push(block);
    collectedLength += block.length;
    const need = Math.round(session.window_seconds * ctx.sampleRate);
    while (collectedLength >= need) {
      const samples = new Float32Array(need);
      let filled = 0;
      while (filled < need) {
        const head = collected[0], take = Math.min(head.length, need - filled);
        samples.set(head.subarray(0, take), filled);
        filled += take;
        if (take === head.length) collected.shift(); else collected[0] = head.subarray(take);
      }
      collectedLength -= need;
      queue.push({ samples, start: windowIndex * session.window_seconds, capturedAt: performance.now() });
      windowIndex++;
      if (queue.length > 2) { queue.shift(); dropped++; }
      processQueue();
    }
  }

  async function processQueue() {
    if (busy || !queue.length || !session) return;
    busy = true;
    const item = queue.shift();
    try {
      let gtmResult = { version: cfg.gtm_version, error: "GTM model not installed" };
      if (gtm) {
        const buffer = new AudioBuffer({ length: item.samples.length, numberOfChannels: 1, sampleRate: ctx.sampleRate });
        buffer.copyToChannel(item.samples, 0);
        try { gtmResult = await gtm.classifyBuffer(buffer); } catch (e) { gtmResult = { version: cfg.gtm_version, error: "GTM failed" }; }
      }
      const body = new FormData();
      body.append("audio", wavBytes(item.samples, ctx.sampleRate), "window.wav");
      body.append("gtm", JSON.stringify(gtmResult));
      body.append("start", item.start.toFixed(2));
      const [status, result] = await post(session.window_url, body);
      analysed++;
      if (status === 200) show(result, gtmResult, performance.now() - item.capturedAt);
      else if (status === 409) { stopMonitoring(true); }
    } catch (e) {
      ui.currentMeta.textContent = "The server could not be reached; retrying with the next window.";
    } finally {
      busy = false;
      updateStats();
      if (queue.length) processQueue();
    }
  }

  function scoreRows(top3) {
    return (top3 || []).map(([name, p], i) =>
      '<div class="score-row"><span class="name">' + escapeHtml(name) + '</span><div class="progress"><div class="progress-bar ' +
      (i === 0 ? "bg-ss-link" : "bg-slate-400") + '" style="width:' + (p * 100).toFixed(1) + '%"></div></div><span class="value">' + pct(p) + "</span></div>").join("");
  }

  function show(r, gtmResult, latencyMs) {
    ui.latency.textContent = (latencyMs / 1000).toFixed(1) + " s";
    const level = r.level_dbfs;
    if (level != null) {
      ui.level.style.width = Math.max(0, Math.min(100, (level + 70) / 70 * 100)) + "%";
      ui.levelText.textContent = level.toFixed(0) + " dBFS";
    }
    const when = r.start.toFixed(0) + "–" + r.end.toFixed(0) + " s";
    if (r.silent) {
      silentCount++;
      ui.current.textContent = "Silence";
      ui.severity.className = "badge sev sev-informational";
      ui.severity.textContent = "—";
      ui.currentMeta.textContent = "Window " + when + ": no usable sound (" + (r.issues || []).join("; ") + ")";
      return;
    }
    ui.current.innerHTML = '<a href="' + r.url + '" target="_blank" rel="noopener">' + escapeHtml(r.final_class) + "</a>";
    ui.severity.className = "badge sev sev-" + r.severity.toLowerCase();
    ui.severity.textContent = r.severity;
    ui.currentMeta.textContent = "Window " + when + " · " + r.status + " · confidence " + r.confidence_level +
      " · heard " + r.repeated + "× recently" + (r.review_required ? " · needs review" : "");
    ui.python.innerHTML = '<div class="font-semibold">' + escapeHtml(r.python.class) + " " + pct(r.python.confidence) + "</div>" + scoreRows(r.python.top3);
    ui.gtm.innerHTML = r.gtm.class
      ? '<div class="font-semibold">' + escapeHtml(r.gtm.class) + " " + pct(r.gtm.confidence) + "</div>" + scoreRows(r.gtm.top3)
      : '<div class="text-amber-600 text-sm">No result: ' + escapeHtml(r.gtm.error) + "</div>";
    const colours = { "Acceptable Match": "success", "Weak Match": "warning", "Model Disagreement": "danger", "Uncertain Result": "secondary" };
    ui.agreement.innerHTML = '<span class="badge text-bg-' + (colours[r.consistency] || "secondary") + '">' + escapeHtml(r.consistency) + "</span>" +
      (r.confidence_difference != null ? ' <span class="text-sm text-ss-muted">difference ' + pct(r.confidence_difference) + "</span>" : "");
    if (r.alert) {
      ui.alertBox.className = "alert alert-danger live-alert mb-4";
      ui.alertBox.innerHTML = '<div class="flex flex-wrap items-center gap-2"><i class="bi bi-exclamation-octagon-fill text-2xl"></i>' +
        '<strong class="text-xl">' + escapeHtml(r.alert.category) + '</strong><span class="badge sev sev-' + r.alert.severity.toLowerCase() + '">' +
        escapeHtml(r.alert.severity) + "</span><span>" + escapeHtml(r.alert.code) + " · " + escapeHtml(r.alert.message) + '</span><a class="ms-auto" href="' +
        r.alert.url + '" target="_blank" rel="noopener">Handle alert</a></div><div class="mt-1 text-sm">' + escapeHtml(r.alert.action) + "</div>";
    }
    const row = document.createElement("tr");
    row.innerHTML = "<td>" + new Date().toLocaleTimeString() + '</td><td><a href="' + r.url + '" target="_blank" rel="noopener">' + escapeHtml(r.event) +
      "</a></td><td>" + escapeHtml(r.final_class) + '</td><td class="text-sm">' + escapeHtml(r.python.class) + " " + pct(r.python.confidence) +
      '</td><td class="text-sm">' + escapeHtml(r.gtm.class || "—") + " " + (r.gtm.class ? pct(r.gtm.confidence) : "") + '</td><td><span class="badge sev sev-' +
      r.severity.toLowerCase() + '">' + escapeHtml(r.severity) + "</span></td><td>" + (r.alert ? '<i class="bi bi-bell-fill text-red-600"></i>' : "") + "</td>";
    ui.recent.prepend(row);
    while (ui.recent.children.length > 20) ui.recent.lastChild.remove();
  }

  function updateStats() {
    ui.windows.textContent = analysed + (silentCount ? " (" + silentCount + " silent)" : "");
    ui.dropped.textContent = dropped;
  }

  function draw() {
    if (!analyser) return;
    const w = ui.wave, wc = w.getContext("2d");
    const time = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(time);
    wc.fillStyle = "#0f1b2d"; wc.fillRect(0, 0, w.width, w.height);
    wc.strokeStyle = "#5aa2ff"; wc.lineWidth = 1.5; wc.beginPath();
    for (let x = 0; x < w.width; x++) {
      const y = (0.5 - time[Math.floor(x / w.width * time.length)] * 0.5) * w.height;
      x ? wc.lineTo(x, y) : wc.moveTo(x, y);
    }
    wc.stroke();
    const s = ui.spec, sc = s.getContext("2d");
    const freq = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(freq);
    sc.drawImage(s, -2, 0); // scroll left
    const bins = Math.floor(freq.length / 3); // show 0-8 kHz of the 0-24 kHz range
    for (let y = 0; y < s.height; y++) {
      const v = freq[Math.floor((1 - y / s.height) * bins)];
      sc.fillStyle = "hsl(" + (260 - v) + ",90%," + (v / 255 * 60) + "%)";
      sc.fillRect(s.width - 2, y, 2, 1);
    }
    animation = requestAnimationFrame(draw);
  }

  async function startMonitoring() {
    setStatus("starting");
    paused = false;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: {
        deviceId: ui.device.value ? { exact: ui.device.value } : undefined,
        echoCancellation: false, noiseSuppression: false, autoGainControl: false,
      } });
    } catch (e) {
      setStatus(e && e.name === "NotAllowedError" ? "denied" : "unavailable");
      stream = null;
      return;
    }
    await listDevices(); // labels become visible once permission is granted
    const track = stream.getAudioTracks()[0];
    track.addEventListener("ended", () => { stopMonitoring(true); setStatus("disconnected"); });
    ctx = new AudioContext({ sampleRate: 48000 });
    const source = ctx.createMediaStreamSource(stream);
    analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    await ctx.audioWorklet.addModule(cfg.worklet_url);
    worklet = new AudioWorkletNode(ctx, "pcm-capture");
    worklet.port.onmessage = (e) => onSamples(e.data);
    const mute = ctx.createGain(); // keeps the worklet running without playing the sound back
    mute.gain.value = 0;
    source.connect(worklet).connect(mute).connect(ctx.destination);

    const body = new FormData();
    body.append("device", track.label || "");
    const [status, data] = await post(cfg.start_url, body);
    if (status !== 200) { stopMonitoring(false); setStatus("available"); return; }
    session = data;
    collected = []; collectedLength = 0; windowIndex = 0; dropped = 0; analysed = 0; silentCount = 0;
    startedAt = Date.now();
    ui.session.textContent = session.session;
    ui.alertBox.className = "hidden";
    updateStats();
    clock = setInterval(() => {
      const s = Math.floor((Date.now() - startedAt) / 1000);
      ui.elapsed.textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
    }, 1000);
    setStatus("active");
    draw();
  }

  function stopMonitoring(notifyServer) {
    if (stream) stream.getTracks().forEach(t => t.stop()); // releases the microphone: the browser's indicator goes off
    stream = null;
    if (ctx) ctx.close();
    ctx = null; analyser = null; worklet = null;
    if (animation) cancelAnimationFrame(animation);
    if (clock) clearInterval(clock);
    queue.length = 0;
    if (session && notifyServer !== false) {
      const body = new FormData();
      body.append("csrfmiddlewaretoken", csrf);
      navigator.sendBeacon(session.stop_url, body); // also works while the page is closing
    }
    session = null;
    paused = false;
    listDevices();
  }

  ui.start.addEventListener("click", startMonitoring);
  ui.stop.addEventListener("click", () => stopMonitoring(true));
  ui.pause.addEventListener("click", async () => {
    paused = !paused;
    if (paused) { collected = []; collectedLength = 0; await ctx.suspend(); setStatus("paused"); }
    else { await ctx.resume(); setStatus("active"); }
  });
  window.addEventListener("pagehide", () => stopMonitoring(true));
  if (navigator.mediaDevices) navigator.mediaDevices.addEventListener("devicechange", listDevices);
  listDevices().then(checkPermission);
})();
