/*
 * Google Teachable Machine (GTM) audio classification in the browser.
 *
 * GTM exports its audio model for TensorFlow.js and the speech-commands
 * library (the supported integration). While we recorded the training
 * samples, speech-commands turned sound into model input like this:
 *   - an AnalyserNode with a 2048-point FFT (fftSize 1024 x 2) and no smoothing,
 *     in an AudioContext running at 48 kHz (this PC's rate during training)
 *   - one frame read every 1024 / 44100 s (about 23 ms), keeping the lowest
 *     232 frequency bins in dB
 *   - 43 frames (about 1 s) form one example, normalised to mean 0 / std 1
 * For an uploaded file or a live window we build exactly that input with an
 * OfflineAudioContext and the same AnalyserNode, so GTM sees the same kind of
 * data it was trained on. GTM only ever receives audio: never the Python
 * model's prediction (SRS 1.8 items 13-14).
 *
 * Speed: the model runs on the graphics card (WebGL), which compiles its
 * programs for every new input shape. Examples are therefore always sent in
 * batches of exactly BATCH (padded), and one warm-up prediction runs when the
 * model loads, so the compiling happens once instead of on every file.
 * Results are read back synchronously, so a background tab (whose timers
 * Chrome slows to once per second) is not slowed down.
 */
(function (global) {
  "use strict";

  const SAMPLE_RATE = 48000;
  const FFT_SIZE = 1024;
  const FRAME_SECONDS = FFT_SIZE / 44100;
  const HOP_FRAMES = 21; // ~0.5 s between windows, like the Python model's 0.5 s hop
  const SILENCE_DBFS = -60; // windows quieter than this are not classified (same level as the Python side)
  const BATCH = 16; // fixed batch size: one compiled WebGL program for every file

  function mixToMono(buffer, start, end) {
    const out = new Float32Array(end - start);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const data = buffer.getChannelData(c);
      for (let i = start; i < end; i++) out[i - start] += data[i] / buffer.numberOfChannels;
    }
    return out;
  }

  function rmsDbfs(samples) {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    return 10 * Math.log10(sum / Math.max(samples.length, 1) + 1e-12);
  }

  class GtmClassifier {
    constructor(baseUrl, version) {
      // speech-commands only accepts absolute http(s) URLs, so resolve e.g. "/static/gtm_model/".
      const absolute = new URL(baseUrl, global.location.href).href;
      this.baseUrl = absolute.endsWith("/") ? absolute : absolute + "/";
      this.version = version || "gtm";
      this.recognizer = null;
      this.loading = null;
    }

    load() {
      if (!this.loading) this.loading = this._load();
      return this.loading;
    }

    async _load() {
      if (!global.speechCommands || !global.tf) throw new Error("TensorFlow.js or speech-commands did not load");
      const r = global.speechCommands.create("BROWSER_FFT", undefined,
        this.baseUrl + "model.json", this.baseUrl + "metadata.json");
      await r.ensureModelLoaded();
      const shape = r.modelInputShape(); // [null, frames, bins, 1]
      this.numFrames = shape[1];
      this.numBins = shape[2];
      this.labels = r.wordLabels();
      this.recognizer = r;
      await this.predict(new Float32Array(BATCH * this.numFrames * this.numBins), BATCH); // warm-up: compile once
      return this;
    }

    /** Scores for `count` examples packed in `input` (BATCH examples long); returns one array per example. */
    async predict(input, count) {
      const scores = global.tf.tidy(() => {
        const x = global.tf.tensor4d(input, [BATCH, this.numFrames, this.numBins, 1]);
        const moments = global.tf.moments(x, [1, 2, 3], true); // per-example normalisation, as speech-commands does
        const normalised = x.sub(moments.mean).div(moments.variance.sqrt().add(global.tf.backend().epsilon()));
        return this.recognizer.model.predict(normalised);
      });
      // Read the scores synchronously: the asynchronous read waits on a timer, and
      // Chrome slows timers to one per second in background tabs (e.g. during a batch).
      const values = scores.arraySync();
      scores.dispose();
      return values.slice(0, count);
    }

    /** Decode a File/Blob/ArrayBuffer into an AudioBuffer at 48 kHz (throws if the browser cannot decode it). */
    async decode(data) {
      const bytes = data instanceof ArrayBuffer ? data : await data.arrayBuffer();
      const ctx = new OfflineAudioContext(1, 1, SAMPLE_RATE);
      return await ctx.decodeAudioData(bytes);
    }

    /** Frequency frames of the whole buffer, read from an AnalyserNode as during GTM training. */
    async frames(buffer) {
      let source = buffer;
      const minLength = Math.ceil((this.numFrames + 2) * FRAME_SECONDS * SAMPLE_RATE);
      if (buffer.length < minLength) { // shorter than one example: pad with silence
        source = new AudioBuffer({ length: minLength, numberOfChannels: buffer.numberOfChannels, sampleRate: buffer.sampleRate });
        for (let c = 0; c < buffer.numberOfChannels; c++) source.copyToChannel(buffer.getChannelData(c), c);
      }
      const duration = source.length / source.sampleRate;
      const ctx = new OfflineAudioContext(1, Math.ceil(duration * SAMPLE_RATE), SAMPLE_RATE);
      const node = ctx.createBufferSource();
      node.buffer = source;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = FFT_SIZE * 2;
      analyser.smoothingTimeConstant = 0.0;
      node.connect(analyser);
      analyser.connect(ctx.destination);

      const frames = [];
      const scratch = new Float32Array(analyser.frequencyBinCount);
      for (let k = 1; k * FRAME_SECONDS < duration - 0.005; k++) {
        ctx.suspend(k * FRAME_SECONDS).then(() => {
          analyser.getFloatFrequencyData(scratch);
          frames.push(scratch.slice(0, this.numBins));
          ctx.resume();
        });
      }
      node.start(0);
      await ctx.startRendering();
      return frames;
    }

    /** Classify a decoded recording: per-window scores and the clip score (mean of windows). */
    async classifyBuffer(buffer) {
      await this.load();
      const started = performance.now();
      const frames = await this.frames(buffer);
      const n = this.numFrames, size = n * this.numBins;
      const starts = [];
      for (let s = 0; s + n <= frames.length; s += HOP_FRAMES) starts.push(s);
      if (starts.length && starts[starts.length - 1] + n < frames.length) starts.push(frames.length - n);

      const windows = [];
      for (const s of starts) {
        const t0 = s * FRAME_SECONDS, t1 = (s + n) * FRAME_SECONDS;
        const a = Math.min(Math.floor(t0 * buffer.sampleRate), buffer.length);
        const b = Math.min(Math.ceil(t1 * buffer.sampleRate), buffer.length);
        if (b - a < 16 || rmsDbfs(mixToMono(buffer, a, b)) < SILENCE_DBFS) continue; // silence: nothing to classify
        windows.push({ start: +t0.toFixed(3), end: +Math.min(t1, buffer.duration).toFixed(3), first: s });
      }
      if (!windows.length) {
        return { version: this.version, error: "every window is silent", elapsed_ms: Math.round(performance.now() - started) };
      }

      for (let i = 0; i < windows.length; i += BATCH) {
        const chunk = windows.slice(i, i + BATCH);
        const input = new Float32Array(BATCH * size); // unused slots stay zero and are ignored
        chunk.forEach((w, j) => {
          const example = input.subarray(j * size, (j + 1) * size);
          let floor = Infinity;
          for (let f = 0; f < n; f++) {
            const frame = frames[w.first + f];
            for (let b = 0; b < this.numBins; b++) if (isFinite(frame[b]) && frame[b] < floor) floor = frame[b];
          }
          if (!isFinite(floor)) floor = -160;
          for (let f = 0; f < n; f++) {
            const frame = frames[w.first + f];
            for (let b = 0; b < this.numBins; b++) {
              const v = frame[b];
              example[f * this.numBins + b] = isFinite(v) ? v : floor; // exact digital silence gives -Infinity
            }
          }
        });
        const values = await this.predict(input, chunk.length);
        chunk.forEach((w, j) => { w.probs = values[j]; });
      }

      const clip = this.labels.map((_, c) => windows.reduce((sum, w) => sum + w.probs[c], 0) / windows.length);
      return {
        version: this.version,
        labels: this.labels,
        clip: clip,
        windows: windows.map(w => ({ start: w.start, end: w.end, probs: w.probs })),
        elapsed_ms: Math.round(performance.now() - started),
      };
    }

    async classifyFile(file) {
      let buffer;
      try {
        buffer = await this.decode(file);
      } catch (e) {
        return { version: this.version, error: "the browser could not decode this file for GTM" };
      }
      try {
        return await this.classifyBuffer(buffer);
      } catch (e) {
        return { version: this.version, error: "GTM failed: " + (e && e.message ? e.message : e) };
      }
    }
  }

  global.GtmClassifier = GtmClassifier;
  global.GTM_CONSTANTS = { SAMPLE_RATE, FFT_SIZE, FRAME_SECONDS, HOP_FRAMES, SILENCE_DBFS, BATCH };
})(window);
