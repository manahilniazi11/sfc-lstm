/*
 * SonicCharts: every chart in the app, drawn with Chart.js in one style.
 *
 * - Colours by role: blue for a single series; blue then orange when two
 *   series share a chart (checked for colour-blind separation). Text stays
 *   in grey ink, never in a series colour.
 * - Thin marks: bars at most 24 px thick with rounded ends, 2 px lines.
 * - Hover shows the exact values; a legend appears only for two or more
 *   series (a single series is named by the card's title).
 * - Every chart also has a "Show as table" view in the page itself.
 *
 * Data comes from <script type="application/json"> blocks written by Django's
 * json_script filter, so no numbers are built in JavaScript.
 */
(function () {
  "use strict";

  const SERIES = ["#2a78d6", "#eb6834"];
  const INK = "#52514e";
  const GRID = "#e9ecef";

  if (window.Chart) {
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.color = INK;
    Chart.defaults.animation = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 300 };
  }

  function data(id) {
    const el = document.getElementById(id);
    return el ? JSON.parse(el.textContent) : null;
  }

  function scales(horizontal, stacked, percent) {
    const ticks = percent ? { callback: (v) => `${Math.round(v * 100)}%` } : { precision: 0 }; // counts: whole numbers only
    const value = { beginAtZero: true, grid: { color: GRID }, border: { display: false }, ticks: ticks,
                    stacked: stacked, max: percent ? 1 : undefined };
    const category = { grid: { display: false }, border: { color: GRID }, stacked: stacked };
    return horizontal ? { x: value, y: category } : { x: category, y: value };
  }

  function legend(series) {
    return { display: series.length > 1, position: "top", align: "start",
             labels: { boxWidth: 10, boxHeight: 10, useBorderRadius: true, borderRadius: 2 } };
  }

  /* series: [{name, values}], options: {horizontal, stacked, percent} */
  function bar(canvasId, labels, series, options) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return null;
    options = options || {};
    return new Chart(canvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: series.map((s, i) => ({
          label: s.name, data: s.values, backgroundColor: SERIES[i % SERIES.length],
          maxBarThickness: 24, borderRadius: 4, borderSkipped: "start",
          // grouped bars keep a gap of card surface between them instead of an outline
          borderWidth: 0, categoryPercentage: 0.7, barPercentage: 0.9,
        })),
      },
      options: {
        indexAxis: options.horizontal ? "y" : "x",
        maintainAspectRatio: false,
        scales: scales(options.horizontal, options.stacked, options.percent),
        plugins: {
          legend: legend(series),
          tooltip: { callbacks: options.percent ? { label: (c) => `${c.dataset.label}: ${(c.parsed[options.horizontal ? "x" : "y"] * 100).toFixed(0)}%` } : {} },
        },
      },
    });
  }

  function line(canvasId, labels, series) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return null;
    return new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: series.map((s, i) => ({
          label: s.name, data: s.values, borderColor: SERIES[i % SERIES.length],
          backgroundColor: SERIES[i % SERIES.length], borderWidth: 2, tension: 0, // straight segments: a smoothed curve would show values that never happened
          pointRadius: 0, pointHoverRadius: 5, pointHoverBorderWidth: 2, pointHoverBorderColor: "#fff",
        })),
      },
      options: {
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false }, // hover anywhere on a day shows every series
        scales: scales(false, false),
        plugins: { legend: legend(series) },
      },
    });
  }

  window.SonicCharts = { data: data, bar: bar, line: line };
})();
