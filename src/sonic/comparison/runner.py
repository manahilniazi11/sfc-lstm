"""Steps 1 and 2: the run folder with the test clips, and the page that runs GTM on them in a browser."""

from __future__ import annotations

import csv
import json
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..dataset.config import AUDIO_DATA_DIR, DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR, REPO_ROOT
from ..gtm.model_info import GTM_MODEL_DIR, model_info

RUN_DIR = AUDIO_DATA_DIR / "comparison"
RESULTS = "gtm_results.json"
MAX_RESULTS_BYTES = 50 * 1024 * 1024


def test_clips(per_class: int | None = None) -> list[dict]:
    """The test split from dataset_metadata.csv, optionally the first `per_class` of each class (by Audio ID)."""
    rows = [r for r in csv.DictReader(open(DEFAULT_METADATA_DIR / "dataset_metadata.csv", encoding="utf-8"))
            if r["split"] == "test"]
    rows.sort(key=lambda r: (r["class_name"], r["audio_id"]))
    if not per_class:
        return rows
    chosen, taken = [], {}
    for r in rows:
        if taken.get(r["class_name"], 0) < per_class:
            taken[r["class_name"]] = taken.get(r["class_name"], 0) + 1
            chosen.append(r)
    return chosen


def prepare(per_class: int | None = None, run_dir: Path = RUN_DIR) -> list[dict]:
    """Copy the clips, the GTM model and gtm.js into `run_dir` and write the page and its manifest."""
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "clips").mkdir(parents=True)
    shutil.copytree(GTM_MODEL_DIR, run_dir / "gtm_model")
    shutil.copy(REPO_ROOT / "static" / "js" / "gtm.js", run_dir / "gtm.js")
    manifest = []
    for row in test_clips(per_class):
        name = Path(row["filename"]).name
        shutil.copy(DEFAULT_OUT_DIR / row["filename"], run_dir / "clips" / name)
        manifest.append({"audio_id": row["audio_id"], "file": name})  # no labels: the page only classifies
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "index.html").write_text(PAGE.replace("__VERSION__", model_info()["version"]), encoding="utf-8")
    return manifest


class _Handler(SimpleHTTPRequestHandler):
    """Serves the run folder and accepts one POST: the page's GTM results."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if self.path != "/" + RESULTS or not 0 < length <= MAX_RESULTS_BYTES:
            self.send_error(400, "only the GTM results can be saved here")
            return
        data = json.loads(self.rfile.read(length))
        (Path(self.directory) / RESULTS).write_text(json.dumps(data), encoding="utf-8")
        self.send_response(204)
        self.end_headers()
        print(f"saved {len(data.get('results', []))} GTM results to {Path(self.directory) / RESULTS}")

    def log_message(self, *args):  # keep the console quiet apart from the save message
        pass


def serve(port: int = 8003, run_dir: Path = RUN_DIR) -> None:
    handler = lambda *args, **kwargs: _Handler(*args, directory=str(run_dir), **kwargs)  # noqa: E731
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Open http://127.0.0.1:{port}/ in Chrome, keep the window visible and press Run. Ctrl+C to stop.")
    server.serve_forever()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GTM comparison run</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#1f2933}button{font-size:1rem;padding:.4rem 1rem}
#bar{height:.6rem;background:#e1e6ef;border-radius:.3rem;margin:1rem 0;max-width:40rem}#fill{height:100%;width:0;background:#2a78d6;border-radius:.3rem}</style>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.4.0/dist/speech-commands.min.js"></script>
<script src="gtm.js"></script></head>
<body>
<h1>GTM comparison run</h1>
<p>Classifies every test clip with the exported Teachable Machine model, the same way the web app does, and saves the scores for the comparison report. Keep this window visible while it runs.</p>
<button id="run">Run</button>
<div id="bar"><div id="fill"></div></div>
<p id="status">Ready.</p>
<script>
document.getElementById("run").addEventListener("click", async () => {
  const status = document.getElementById("status"), fill = document.getElementById("fill");
  document.getElementById("run").disabled = true;
  const manifest = await (await fetch("manifest.json")).json();
  const gtm = new GtmClassifier(new URL("gtm_model/", location.href).href, "__VERSION__");
  status.textContent = "Loading the model...";
  await gtm.load();
  const results = [], started = performance.now();
  for (const [i, clip] of manifest.entries()) {
    const blob = await (await fetch("clips/" + clip.file)).blob();
    results.push({ audio_id: clip.audio_id, file: clip.file, gtm: await gtm.classifyFile(blob) });
    fill.style.width = ((i + 1) / manifest.length * 100).toFixed(1) + "%";
    status.textContent = `${i + 1} / ${manifest.length} clips`;
  }
  const payload = { version: "__VERSION__", user_agent: navigator.userAgent, seconds: (performance.now() - started) / 1000,
                    finished: new Date().toISOString(), results };
  window.gtmResults = payload;
  const saved = await fetch("gtm_results.json", { method: "POST", body: JSON.stringify(payload) });
  status.textContent = saved.ok ? `Done: ${results.length} clips in ${payload.seconds.toFixed(0)} s, saved to gtm_results.json.`
                                : "Done, but saving failed: is `python -m sonic.comparison serve` running?";
});
</script>
</body></html>
"""
