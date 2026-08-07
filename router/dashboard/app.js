/*
 * Flux router console.
 *
 * Vanilla JS on purpose — no framework, no bundler, no CDN. The page must work
 * from a wheel on a machine with no Node toolchain and no internet, because the
 * product's whole claim is that nothing leaves the operator's box.
 *
 * Everything is read-only: this page issues GETs against /v1/stats/* and never
 * mutates router state.
 */

(function () {
  "use strict";

  var TOKEN_KEY = "flux.dashboard.token";
  var THEME_KEY = "flux.dashboard.theme";

  var el = function (id) { return document.getElementById(id); };

  // ── Formatting ────────────────────────────────────────────────────────────

  function money(v) {
    if (v === null || v === undefined) return "—";
    if (v === 0) return "$0.00";
    if (v < 0.01) return "$" + v.toFixed(5);
    if (v < 1) return "$" + v.toFixed(4);
    return "$" + v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function ms(v) {
    if (v === null || v === undefined) return "—";
    if (v >= 1000) return (v / 1000).toFixed(2) + " s";
    return Math.round(v) + " ms";
  }

  function pct(v) {
    if (v === null || v === undefined) return "—";
    return v.toFixed(1) + "%";
  }

  function num(v) {
    return (v === null || v === undefined) ? "—" : v.toLocaleString();
  }

  function tokens(v) {
    if (!v) return "0";
    if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(v);
  }

  function ctx(v) {
    if (!v) return "—";
    if (v >= 1000) return Math.round(v / 1000) + "K";
    return String(v);
  }

  // Bucket labels: within a day show the clock, longer windows show the date.
  function bucketLabel(epochSeconds, bucketWidth) {
    var d = new Date(epochSeconds * 1000);
    if (bucketWidth < 86400) {
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function fullLabel(epochSeconds, bucketWidth) {
    var d = new Date(epochSeconds * 1000);
    if (bucketWidth < 86400) return d.toLocaleString();
    return d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });
  }

  // ── Transport ─────────────────────────────────────────────────────────────

  function token() {
    try { return window.localStorage.getItem(TOKEN_KEY) || ""; } catch (e) { return ""; }
  }

  function setToken(v) {
    try {
      if (v) window.localStorage.setItem(TOKEN_KEY, v);
      else window.localStorage.removeItem(TOKEN_KEY);
    } catch (e) { /* private mode — the session still works, it just won't persist */ }
  }

  function get(path) {
    var headers = {};
    var t = token();
    if (t) headers.Authorization = "Bearer " + t;
    return fetch(path, { headers: headers, cache: "no-store" }).then(function (r) {
      if (r.status === 401) {
        var err = new Error("unauthorized");
        err.unauthorized = true;
        throw err;
      }
      if (!r.ok) {
        return r.json().then(
          function (b) { throw new Error(b.detail || ("HTTP " + r.status)); },
          function () { throw new Error("HTTP " + r.status); }
        );
      }
      return r.json();
    });
  }

  // ── Banner ────────────────────────────────────────────────────────────────

  function hideBanner() {
    var b = el("banner");
    b.hidden = true;
    b.className = "banner";
    b.textContent = "";
  }

  function showError(msg) {
    var b = el("banner");
    b.className = "banner error";
    b.textContent = "";
    var s = document.createElement("strong");
    s.textContent = "Couldn't load data. ";
    b.appendChild(s);
    b.appendChild(document.createTextNode(msg));
    b.hidden = false;
  }

  function askForToken() {
    var b = el("banner");
    b.className = "banner";
    b.textContent = "";

    var s = document.createElement("strong");
    s.textContent = "This router requires a token. ";
    b.appendChild(s);
    b.appendChild(document.createTextNode(
      "Paste a bearer token from FLUX_SERVER_TOKEN or FLUX_SERVER_TOKENS. " +
      "It is stored in this browser only and sent to this local server only."
    ));
    b.appendChild(document.createElement("br"));

    var input = document.createElement("input");
    input.type = "password";
    input.placeholder = "Bearer token";
    input.autocomplete = "off";
    b.appendChild(input);

    var save = document.createElement("button");
    save.className = "btn";
    save.type = "button";
    save.textContent = "Save and load";
    b.appendChild(save);

    function submit() {
      if (!input.value) return;
      setToken(input.value.trim());
      hideBanner();
      load();
    }
    save.addEventListener("click", submit);
    input.addEventListener("keydown", function (e) { if (e.key === "Enter") submit(); });

    b.hidden = false;
    el("logout").hidden = !token();
    input.focus();
  }

  // ── Chart: single-series spend over time ─────────────────────────────────
  //
  // One series, so no legend — the panel title names it. Hand-drawn SVG rather
  // than a charting library: it is one line chart, and a vendored minified
  // dependency would be a bigger thing to audit than the 60 lines below.

  function renderChart(series, bucketWidth) {
    var host = el("chart");
    host.textContent = "";

    var W = 1000, H = 260;
    var PAD = { top: 16, right: 16, bottom: 30, left: 66 };
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("role", "img");

    function node(name, attrs, cls) {
      var n = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      if (cls) n.setAttribute("class", cls);
      return n;
    }

    if (!series.length) {
      var t = node("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, "empty");
      t.textContent = "No requests in this window yet.";
      svg.appendChild(t);
      host.appendChild(svg);
      el("chart-sub").textContent = "";
      return;
    }

    var plotW = W - PAD.left - PAD.right;
    var plotH = H - PAD.top - PAD.bottom;
    var maxCost = 0;
    series.forEach(function (d) { if (d.cost_usd > maxCost) maxCost = d.cost_usd; });
    if (maxCost <= 0) maxCost = 1e-6;
    // Headroom so the peak never touches the frame.
    var yMax = maxCost * 1.15;

    var x = function (i) {
      return PAD.left + (series.length === 1 ? plotW / 2 : (i / (series.length - 1)) * plotW);
    };
    var y = function (v) { return PAD.top + plotH - (v / yMax) * plotH; };

    // Gridlines + y ticks, recessive.
    for (var g = 0; g <= 4; g++) {
      var gv = (yMax / 4) * g;
      var gy = y(gv);
      svg.appendChild(node("line", { x1: PAD.left, y1: gy, x2: W - PAD.right, y2: gy }, "grid-line"));
      var lbl = node("text", { x: PAD.left - 9, y: gy + 4, "text-anchor": "end" }, "tick");
      lbl.textContent = money(gv);
      svg.appendChild(lbl);
    }

    var path = "", area = "";
    series.forEach(function (d, i) {
      var px = x(i), py = y(d.cost_usd);
      path += (i ? " L" : "M") + px.toFixed(2) + " " + py.toFixed(2);
    });
    area = path
      + " L" + x(series.length - 1).toFixed(2) + " " + (PAD.top + plotH)
      + " L" + x(0).toFixed(2) + " " + (PAD.top + plotH) + " Z";

    // A one-bucket series has no segment to stroke: "M x y" paints nothing and
    // the area collapses to zero width, so the panel reads as broken on any
    // instance with less than one bucket of history. Mark the single value
    // instead — a stem down to the axis plus a dot, which is what a lone
    // reading should look like.
    if (series.length === 1) {
      svg.appendChild(node("line", {
        x1: x(0), y1: y(series[0].cost_usd), x2: x(0), y2: PAD.top + plotH
      }, "series-stem"));
      svg.appendChild(node("circle", { cx: x(0), cy: y(series[0].cost_usd), r: 4.5 }, "dot"));
    } else {
      svg.appendChild(node("path", { d: area }, "series-area"));
      svg.appendChild(node("path", { d: path }, "series-line"));
    }
    svg.appendChild(node("line", {
      x1: PAD.left, y1: PAD.top + plotH, x2: W - PAD.right, y2: PAD.top + plotH
    }, "axis-line"));

    // X labels: at most six, so they never collide.
    var step = Math.max(1, Math.ceil(series.length / 6));
    series.forEach(function (d, i) {
      if (i % step && i !== series.length - 1) return;
      var tx = node("text", { x: x(i), y: H - 10, "text-anchor": "middle" }, "tick");
      tx.textContent = bucketLabel(d.bucket_start, bucketWidth);
      svg.appendChild(tx);
    });

    // Hover layer: crosshair + dot + tooltip, the default for a line chart.
    var cross = node("line", { x1: 0, y1: PAD.top, x2: 0, y2: PAD.top + plotH }, "crosshair");
    cross.style.display = "none";
    svg.appendChild(cross);
    var dot = node("circle", { r: 4.5, cx: 0, cy: 0 }, "dot");
    dot.style.display = "none";
    svg.appendChild(dot);

    var hit = node("rect", {
      x: PAD.left, y: PAD.top, width: plotW, height: plotH
    }, "hit");
    svg.appendChild(hit);

    var tip = document.createElement("div");
    tip.className = "tooltip";
    tip.style.display = "none";
    host.appendChild(tip);

    function nearest(clientX) {
      var box = svg.getBoundingClientRect();
      var rel = ((clientX - box.left) / box.width) * W;
      var best = 0, bestD = Infinity;
      for (var i = 0; i < series.length; i++) {
        var d = Math.abs(x(i) - rel);
        if (d < bestD) { bestD = d; best = i; }
      }
      return best;
    }

    hit.addEventListener("mousemove", function (ev) {
      var i = nearest(ev.clientX);
      var d = series[i];
      var box = svg.getBoundingClientRect();
      var scale = box.width / W;

      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i));
      cross.style.display = "";
      dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(d.cost_usd));
      dot.style.display = "";

      tip.textContent = "";
      var when = document.createElement("span");
      when.className = "tt-when";
      when.textContent = fullLabel(d.bucket_start, bucketWidth);
      tip.appendChild(when);
      [["Spend", money(d.cost_usd)],
       ["Requests", num(d.requests)],
       ["Saved", money(d.estimated_savings_usd)],
       ["Avg latency", ms(d.avg_latency_ms)]].forEach(function (row) {
        var r = document.createElement("div");
        r.className = "tt-row";
        var k = document.createElement("span"); k.textContent = row[0];
        var v = document.createElement("b"); v.textContent = row[1];
        r.appendChild(k); r.appendChild(v);
        tip.appendChild(r);
      });

      tip.style.display = "";
      var left = x(i) * scale + 14;
      if (left + tip.offsetWidth > box.width) left = x(i) * scale - tip.offsetWidth - 14;
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = Math.max(0, y(d.cost_usd) * scale - 10) + "px";
    });

    hit.addEventListener("mouseleave", function () {
      cross.style.display = "none";
      dot.style.display = "none";
      tip.style.display = "none";
    });

    host.appendChild(svg);
    var bucketName = bucketWidth >= 86400
      ? (bucketWidth / 86400) + " day"
      : bucketWidth >= 3600 ? (bucketWidth / 3600) + " hour" : (bucketWidth / 60) + " min";
    el("chart-sub").textContent = series.length === 1
      ? "1 bucket of " + bucketName + " — not enough history to trend yet"
      : series.length + " buckets of " + bucketName;
  }

  // ── Tables ────────────────────────────────────────────────────────────────

  function cell(row, text, cls) {
    var td = document.createElement("td");
    if (cls) td.className = cls;
    td.textContent = text;
    row.appendChild(td);
    return td;
  }

  function shareCell(row, share) {
    var td = document.createElement("td");
    td.className = "num";
    var wrap = document.createElement("div");
    wrap.className = "share";
    var n = document.createElement("span");
    n.textContent = pct(share);
    var bar = document.createElement("span");
    bar.className = "share-bar";
    var fill = document.createElement("i");
    fill.style.width = Math.max(2, Math.min(100, share)) + "%";
    bar.appendChild(fill);
    wrap.appendChild(n);
    wrap.appendChild(bar);
    td.appendChild(wrap);
    row.appendChild(td);
  }

  function emptyRow(tbody, span, text) {
    var tr = document.createElement("tr");
    var td = document.createElement("td");
    td.colSpan = span;
    td.className = "empty-row";
    td.textContent = text;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }

  function renderModels(rows) {
    var tbody = el("models").querySelector("tbody");
    tbody.textContent = "";
    if (!rows.length) return emptyRow(tbody, 5, "No traffic in this window.");
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      cell(tr, r.model_id);
      cell(tr, num(r.requests), "num");
      shareCell(tr, r.share_pct);
      cell(tr, money(r.cost_usd), "num");
      cell(tr, ms(r.avg_latency_ms), "num dim");
      tbody.appendChild(tr);
    });
  }

  function renderTasks(rows) {
    var tbody = el("tasks").querySelector("tbody");
    tbody.textContent = "";
    if (!rows.length) return emptyRow(tbody, 5, "No traffic in this window.");
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      cell(tr, r.task_type || "unknown");
      cell(tr, num(r.requests), "num");
      shareCell(tr, r.share_pct);
      cell(tr, money(r.cost_usd), "num");
      cell(tr, r.avg_complexity_score === null || r.avg_complexity_score === undefined
        ? "—" : r.avg_complexity_score.toFixed(2), "num dim");
      tbody.appendChild(tr);
    });
  }

  function renderRegistry(rows) {
    var tbody = el("registry").querySelector("tbody");
    tbody.textContent = "";
    if (!rows.length) return emptyRow(tbody, 7, "No models available.");
    rows.forEach(function (m) {
      var tr = document.createElement("tr");
      cell(tr, m.model_id);
      cell(tr, m.provider, "dim");
      var tier = document.createElement("td");
      var pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = m.tier;
      tier.appendChild(pill);
      tr.appendChild(tier);
      cell(tr, money(m.cost_per_1k_input), "num");
      cell(tr, money(m.cost_per_1k_output), "num");
      cell(tr, ctx(m.max_context_window), "num dim");
      cell(tr, m.rate_limit_rpm
        ? m.current_load_rpm + " / " + m.rate_limit_rpm
        : String(m.current_load_rpm), "num dim");
      tbody.appendChild(tr);
    });
  }

  // ── Config ────────────────────────────────────────────────────────────────

  function group(title, pairs) {
    var box = document.createElement("div");
    box.className = "config-group";
    var h = document.createElement("h3");
    h.textContent = title;
    box.appendChild(h);
    var dl = document.createElement("dl");
    dl.style.margin = "0";
    pairs.forEach(function (p) {
      var row = document.createElement("div");
      row.className = "kv";
      var dt = document.createElement("dt");
      dt.textContent = p[0];
      if (p[2]) {
        dt.appendChild(document.createTextNode(" "));
        var code = document.createElement("code");
        code.textContent = p[2];
        dt.appendChild(code);
      }
      var dd = document.createElement("dd");
      dd.textContent = p[1];
      row.appendChild(dt);
      row.appendChild(dd);
      dl.appendChild(row);
    });
    box.appendChild(dl);
    return box;
  }

  function renderConfig(c) {
    var host = el("config");
    host.textContent = "";

    var providers = Object.keys(c.providers)
      .filter(function (p) { return c.providers[p]; });

    host.appendChild(group("Server", [
      ["Bind", c.server.host + ":" + c.server.port, "FLUX_SERVER_HOST"],
      ["Auth mode", c.server.auth_mode, "FLUX_SERVER_TOKENS"],
      ["Bound tenants", num(c.server.tenant_count)],
      ["Workers", num(c.server.workers), "FLUX_SERVER_WORKERS"],
      ["Run store", c.server.run_store_backend, "FLUX_RUN_STORE"]
    ]));

    host.appendChild(group("Storage", [
      ["Usage database", c.server.usage_db_persistent ? "on disk" : "in memory (not persisted)",
        "FLUX_ATTRIBUTION_DB"],
      ["Data directory", c.server.data_dir, "FLUX_DATA_DIR"],
      ["Configured providers", providers.length ? providers.join(", ") : "none"]
    ]));

    host.appendChild(group("Run budget", [
      ["Max cost", money(c.run_limits.max_cost_usd), "FLUX_RUN_MAX_COST_USD"],
      ["Max steps", num(c.run_limits.max_steps)],
      ["Max tokens", tokens(c.run_limits.max_tokens)],
      ["Max duration", Math.round(c.run_limits.max_duration_seconds) + " s"],
      ["Degrade / warn at", pct(c.run_limits.degrade_threshold * 100) + " / "
        + pct(c.run_limits.warn_threshold * 100)]
    ]));

    var plans = Object.keys(c.budgets.plans).map(function (p) {
      return [p.replace("_plan", ""),
        money(c.budgets.plans[p].daily) + " / day"];
    });
    plans.push(["Tenant daily cap",
      c.budgets.tenant_daily_cap_usd === null ? "none" : money(c.budgets.tenant_daily_cap_usd),
      "FLUX_TENANT_DAILY_CAP_USD"]);
    plans.push(["Rate limit", num(c.rate_limit.rpm) + " rpm, burst " + num(c.rate_limit.burst),
      "FLUX_RATE_LIMIT_RPM"]);
    host.appendChild(group("Budgets & limits", plans));

    el("foot-db").textContent = c.server.usage_db_persistent
      ? "Usage database: " + c.server.usage_db
      // FLUX_DATA_DIR alone does NOT enable persistence: nothing in the library
      // reads it, and only router/cli.py translates it into FLUX_ATTRIBUTION_DB
      // (see config.py::DATA_DIR). Naming it here sent operators to a variable
      // they could set correctly and still see this same warning.
      : "Usage is not being persisted — run `flux serve`, or set FLUX_ATTRIBUTION_DB "
        + "to a file path, to keep history across restarts.";
  }

  // ── Load ──────────────────────────────────────────────────────────────────

  function renderSummary(s) {
    el("t-cost").textContent = money(s.total_cost_usd);
    el("t-cost-note").textContent = s.requests
      ? pct(s.actual_cost_pct) + " from provider-reported usage"
      : "";

    el("t-savings").textContent = money(s.estimated_savings_usd);
    el("t-savings-note").textContent = s.requests
      ? pct(s.savings_pct) + " below " + money(s.baseline_cost_usd) + " baseline"
      : "";

    el("t-requests").textContent = num(s.requests);
    el("t-requests-note").textContent = s.requests
      ? num(s.runs) + " runs · " + num(s.distinct_models) + " models · "
        + pct(s.cache_hit_rate) + " cached"
      : "No traffic yet";

    el("t-latency").textContent = ms(s.p95_latency_ms);
    el("t-latency-note").textContent = s.p50_latency_ms === null
      ? "No latency recorded"
      : "p50 " + ms(s.p50_latency_ms) + " · avg " + ms(s.avg_latency_ms);
  }

  var loading = false;

  function load() {
    if (loading) return;
    loading = true;
    var w = el("window").value;

    Promise.all([
      get("/v1/stats/summary?window=" + w),
      get("/v1/stats/timeseries?window=" + w),
      get("/v1/stats/models?window=" + w),
      get("/v1/stats/tasks?window=" + w),
      get("/v1/stats/registry"),
      get("/v1/stats/config")
    ]).then(function (r) {
      hideBanner();
      renderSummary(r[0]);
      renderChart(r[1].data, r[1].bucket_seconds);
      renderModels(r[2].data);
      renderTasks(r[3].data);
      renderRegistry(r[4].data);
      renderConfig(r[5]);
      el("logout").hidden = !token();
    }).catch(function (err) {
      if (err.unauthorized) askForToken();
      else showError(err.message);
    }).then(function () {
      loading = false;
    });
  }

  // ── Theme ─────────────────────────────────────────────────────────────────
  //
  // Follows the OS by default; the toggle stamps data-theme on <html>, which
  // wins over the media query in both directions.

  function applyTheme(v) {
    if (v) document.documentElement.setAttribute("data-theme", v);
    else document.documentElement.removeAttribute("data-theme");
  }

  function initTheme() {
    var saved;
    try { saved = window.localStorage.getItem(THEME_KEY); } catch (e) { saved = null; }
    applyTheme(saved);
    el("theme").addEventListener("click", function () {
      var dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      var current = document.documentElement.getAttribute("data-theme") || (dark ? "dark" : "light");
      var next = current === "dark" ? "light" : "dark";
      applyTheme(next);
      try { window.localStorage.setItem(THEME_KEY, next); } catch (e) { /* not fatal */ }
    });
  }

  // ── Wire up ───────────────────────────────────────────────────────────────

  initTheme();
  el("window").addEventListener("change", load);
  el("refresh").addEventListener("click", load);
  el("logout").addEventListener("click", function () {
    setToken("");
    el("logout").hidden = true;
    load();
  });
  load();
})();
