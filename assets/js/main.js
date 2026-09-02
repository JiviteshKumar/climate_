/* ==========================================================================
   main.js - Heatwave Intelligence Platform
   Dependency-free SVG rendering for the visualization and decision-support
   modules described in Phase V of Use Case KJS-CES-01.
   ========================================================================== */
(function () {
  "use strict";

  var SEV_COLOR = {
    normal:"#12805c", caution:"#c98a04", heat:"#e2620d",
    severe:"#c0341c", extreme:"#8b1c34"
  };
  var SEV_LABEL = {
    normal:"Normal", caution:"Caution", heat:"Heat Wave",
    severe:"Severe Heat Wave", extreme:"Extreme Heat Wave"
  };
  var NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) { if (attrs[k] !== null) n.setAttribute(k, attrs[k]); }
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function byId(id) { return document.getElementById(id); }
  function region(id) {
    for (var i = 0; i < REGIONS.length; i++) if (REGIONS[i].id === id) return REGIONS[i];
    return null;
  }
  function severityFor(dep, tmax) {
    if (dep > 6.4 || tmax >= 47) return "severe";
    if (dep >= 4.5 || tmax >= 45) return "heat";
    if (dep >= 2.5 || tmax >= 40) return "caution";
    return "normal";
  }

  /* ---------------- Navigation: mark the active page ---------------- */
  function markActiveNav() {
    var here = location.pathname.split("/").pop() || "index.html";
    var links = document.querySelectorAll(".nav a");
    for (var i = 0; i < links.length; i++) {
      var href = links[i].getAttribute("href");
      if (href === here) links[i].setAttribute("aria-current", "page");
    }
  }

  /* ---------------- Horizontal bar chart ---------------- */
  function barChart(mount, rows, opts) {
    opts = opts || {};
    var W = 720, rowH = 38, padL = 168, padR = 66, padT = 26, padB = 34;
    var H = padT + rows.length * rowH + padB;
    var max = opts.max || Math.ceil(Math.max.apply(null, rows.map(function (r) { return r.value; })) / 5) * 5 + 3;
    var min = opts.min || 0;
    var plotW = W - padL - padR;
    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": opts.aria || "Bar chart", width: "100%"
    });
    var x = function (v) { return padL + (v - min) / (max - min) * plotW; };

    // gridlines
    var step = opts.step || 10;
    for (var g = Math.ceil(min / step) * step; g <= max; g += step) {
      svg.appendChild(el("line", { x1: x(g), y1: padT - 8, x2: x(g), y2: H - padB + 4,
        stroke: "#e4e7ec", "stroke-width": 1 }));
      svg.appendChild(el("text", { x: x(g), y: H - padB + 20, "text-anchor": "middle",
        "font-size": 11, fill: "#667085" }, g + (opts.unit || "")));
    }

    rows.forEach(function (r, i) {
      var y = padT + i * rowH;
      var bw = Math.max(2, x(r.value) - padL);
      svg.appendChild(el("text", { x: padL - 12, y: y + 20, "text-anchor": "end",
        "font-size": 13, "font-weight": 600, fill: "#344054" }, r.label));
      svg.appendChild(el("rect", { x: padL, y: y + 6, width: plotW, height: 22, rx: 5, fill: "#f1f5f9" }));
      var bar = el("rect", { x: padL, y: y + 6, width: bw, height: 22, rx: 5, fill: r.color || "#0b5fa5" });
      bar.appendChild(el("title", {}, r.label + ": " + r.value + (opts.unit || "")));
      svg.appendChild(bar);
      svg.appendChild(el("text", { x: padL + bw + 9, y: y + 22, "font-size": 12.5,
        "font-weight": 700, fill: "#101828" }, r.value.toFixed(1) + (opts.unit || "")));
    });
    mount.innerHTML = "";
    mount.appendChild(svg);
  }

  /* ---------------- Multi-series line chart ---------------- */
  function lineChart(mount, labels, series, opts) {
    opts = opts || {};
    var W = 760, H = 340, padL = 54, padR = 24, padT = 20, padB = 44;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var all = [];
    series.forEach(function (s) { all = all.concat(s.values); });
    var lo = Math.floor((opts.min !== undefined ? opts.min : Math.min.apply(null, all) - 2) / 2) * 2;
    var hi = Math.ceil((opts.max !== undefined ? opts.max : Math.max.apply(null, all) + 2) / 2) * 2;
    var x = function (i) { return padL + (labels.length === 1 ? plotW / 2 : i / (labels.length - 1) * plotW); };
    var y = function (v) { return padT + plotH - (v - lo) / (hi - lo) * plotH; };

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": opts.aria || "Line chart", width: "100%" });

    // y gridlines
    var ticks = 5;
    for (var t = 0; t <= ticks; t++) {
      var v = lo + (hi - lo) * t / ticks;
      svg.appendChild(el("line", { x1: padL, y1: y(v), x2: W - padR, y2: y(v),
        stroke: "#eef1f4", "stroke-width": 1 }));
      svg.appendChild(el("text", { x: padL - 10, y: y(v) + 4, "text-anchor": "end",
        "font-size": 11, fill: "#667085" }, v.toFixed(0) + (opts.unit || "")));
    }
    // x labels
    labels.forEach(function (lb, i) {
      svg.appendChild(el("text", { x: x(i), y: H - padB + 20, "text-anchor": "middle",
        "font-size": 11.5, fill: "#667085" }, lb));
    });
    svg.appendChild(el("line", { x1: padL, y1: padT + plotH, x2: W - padR, y2: padT + plotH,
      stroke: "#d0d5dd", "stroke-width": 1 }));

    series.forEach(function (s) {
      var d = s.values.map(function (v, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); }).join(" ");
      if (s.area) {
        svg.appendChild(el("path", {
          d: d + " L" + x(s.values.length - 1) + " " + (padT + plotH) + " L" + x(0) + " " + (padT + plotH) + " Z",
          fill: s.color, "fill-opacity": .09, stroke: "none"
        }));
      }
      svg.appendChild(el("path", { d: d, fill: "none", stroke: s.color,
        "stroke-width": s.dashed ? 2 : 2.6, "stroke-dasharray": s.dashed ? "6 5" : null,
        "stroke-linejoin": "round", "stroke-linecap": "round" }));
      if (!s.dashed) {
        s.values.forEach(function (v, i) {
          var c = el("circle", { cx: x(i), cy: y(v), r: 4.2, fill: "#fff", stroke: s.color, "stroke-width": 2.4 });
          c.appendChild(el("title", {}, s.name + " " + labels[i] + ": " + v + (opts.unit || "")));
          svg.appendChild(c);
        });
      }
    });
    mount.innerHTML = "";
    mount.appendChild(svg);
  }

  /* ---------------- Season x region heatmap ---------------- */
  function heatmap(mount) {
    var cw = 118, ch = 46, padL = 176, padT = 46;
    var W = padL + SEASONS.length * cw + 20, H = padT + REGIONS.length * ch + 22;
    var vals = [];
    REGIONS.forEach(function (r) { SEASONS.forEach(function (s) { vals.push(SEASONAL_TMAX[r.id][s.id]); }); });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", role: "img",
      "aria-label": "Heatmap of mean seasonal maximum temperature for seven IMD regions across four seasons" });

    SEASONS.forEach(function (s, j) {
      svg.appendChild(el("text", { x: padL + j * cw + cw / 2, y: padT - 22, "text-anchor": "middle",
        "font-size": 12.5, "font-weight": 700, fill: "#344054" }, s.name));
      svg.appendChild(el("text", { x: padL + j * cw + cw / 2, y: padT - 8, "text-anchor": "middle",
        "font-size": 10.5, fill: "#98a2b3" }, s.months));
    });
    REGIONS.forEach(function (r, i) {
      svg.appendChild(el("text", { x: padL - 14, y: padT + i * ch + ch / 2 + 4, "text-anchor": "end",
        "font-size": 12.5, "font-weight": 600, fill: "#344054" }, r.name));
      SEASONS.forEach(function (s, j) {
        var v = SEASONAL_TMAX[r.id][s.id];
        var t = (v - lo) / (hi - lo);
        var fill = "hsl(" + (212 - t * 208) + ", " + (52 + t * 30) + "%, " + (94 - t * 44) + "%)";
        var g = el("g", {});
        var rect = el("rect", { x: padL + j * cw + 3, y: padT + i * ch + 3, width: cw - 6, height: ch - 6,
          rx: 7, fill: fill, stroke: "#ffffff", "stroke-width": 1.5 });
        rect.appendChild(el("title", {}, r.name + " / " + s.name + ": " + v + " °C"));
        g.appendChild(rect);
        g.appendChild(el("text", { x: padL + j * cw + cw / 2, y: padT + i * ch + ch / 2 + 4.5,
          "text-anchor": "middle", "font-size": 13, "font-weight": 700,
          fill: t > .58 ? "#ffffff" : "#101828" }, v.toFixed(1)));
        svg.appendChild(g);
      });
    });
    mount.innerHTML = "";
    mount.appendChild(svg);
  }

  /* ---------------- Schematic regional hotspot map ---------------- */
  var INDIA = [[74.0,34.5],[76.5,35.5],[78.9,34.3],[79.5,32.5],[81.0,30.3],[83.5,29.2],
    [86.0,27.0],[88.2,27.9],[89.5,26.7],[92.0,27.8],[95.5,28.0],[97.4,28.2],[96.5,27.0],
    [97.0,24.5],[94.5,23.0],[93.3,24.0],[92.5,22.0],[91.0,23.5],[89.0,22.0],[87.0,21.5],
    [85.0,19.5],[82.0,16.8],[80.3,13.1],[79.8,10.3],[78.2,8.9],[77.5,8.1],[76.0,9.5],
    [74.9,13.0],[73.5,15.5],[72.8,18.9],[72.6,21.2],[69.0,22.3],[68.2,23.8],[70.0,25.5],
    [71.0,27.5],[73.0,29.5],[74.5,31.5]];

  function hotspotMap(mount) {
    var W = 430, H = 480, pad = 16;
    var lon0 = 67.5, lon1 = 98.0, lat0 = 6.5, lat1 = 36.5;
    var px = function (lon) { return pad + (lon - lon0) / (lon1 - lon0) * (W - pad * 2); };
    var py = function (lat) { return pad + (lat1 - lat) / (lat1 - lat0) * (H - pad * 2); };

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", role: "img",
      "aria-label": "Schematic map of India showing current heatwave severity for seven IMD homogeneous regions" });

    var d = INDIA.map(function (p, i) { return (i ? "L" : "M") + px(p[0]).toFixed(1) + " " + py(p[1]).toFixed(1); }).join(" ") + " Z";
    svg.appendChild(el("path", { d: d, fill: "#f4f7fa", stroke: "#c3cdd8", "stroke-width": 1.4, "stroke-linejoin": "round" }));

    WATCH.forEach(function (w) {
      var r = region(w.region);
      var c = SEV_COLOR[w.severity];
      var cx = px(r.lon), cy = py(r.lat);
      var g = el("g", {});
      if (w.severity !== "normal") {
        g.appendChild(el("circle", { cx: cx, cy: cy, r: 26, fill: c, "fill-opacity": .13 }));
      }
      var dot = el("circle", { cx: cx, cy: cy, r: 11, fill: c, stroke: "#fff", "stroke-width": 2.5 });
      dot.appendChild(el("title", {}, r.name + ": " + w.tmax + " °C, " + SEV_LABEL[w.severity] +
        (w.hotspots ? ", " + w.hotspots + " hotspots" : "")));
      g.appendChild(dot);
      g.appendChild(el("text", { x: cx, y: cy + 3.8, "text-anchor": "middle", "font-size": 9,
        "font-weight": 700, fill: "#fff" }, w.hotspots || ""));
      g.appendChild(el("text", { x: cx, y: cy + 27, "text-anchor": "middle", "font-size": 10.5,
        "font-weight": 700, fill: "#344054" }, r.short));
      g.appendChild(el("text", { x: cx, y: cy + 39, "text-anchor": "middle", "font-size": 10.5,
        fill: "#667085" }, w.tmax.toFixed(1) + " °C"));
      svg.appendChild(g);
    });
    mount.innerHTML = "";
    mount.appendChild(svg);
  }

  /* ---------------- Dashboard page ---------------- */
  function initDashboard() {
    var mapMount = byId("hotspot-map");
    if (mapMount) hotspotMap(mapMount);

    var barMount = byId("watch-bars");
    if (barMount) {
      barChart(barMount, WATCH.map(function (w) {
        return { label: region(w.region).name, value: w.tmax, color: SEV_COLOR[w.severity] };
      }), { min: 20, max: 50, step: 5, unit: " °C",
        aria: "Region-wise forecast maximum temperature coloured by heatwave severity" });
    }

    var tbody = byId("watch-body");
    if (tbody) {
      var html = "";
      WATCH.forEach(function (w) {
        var r = region(w.region);
        html += "<tr>" +
          "<td class='region'>" + r.name + "<br><span style='font-weight:400;font-size:12px;color:var(--ink-3)'>" + r.states + "</span></td>" +
          "<td class='num'>" + w.tmax.toFixed(1) + " °C</td>" +
          "<td class='num'>" + (w.dep >= 0 ? "+" : "") + w.dep.toFixed(1) + " °C</td>" +
          "<td><span class='chip " + w.severity + "'>" + SEV_LABEL[w.severity] + "</span></td>" +
          "<td class='num'>" + w.hotspots + "</td>" +
          "<td class='num'>" + w.pop + "</td>" +
          "</tr>";
      });
      tbody.innerHTML = html;
    }

    var scale = byId("severity-scale");
    if (scale) {
      var s = "";
      SEVERITY_SCALE.forEach(function (x) {
        s += "<tr><td><span class='chip " + x.key + "'>" + x.label + "</span></td><td>" + x.rule + "</td></tr>";
      });
      scale.innerHTML = s;
    }
  }

  /* ---------------- Forecast page ---------------- */
  function initForecast() {
    var sel = byId("region-select");
    var mount = byId("forecast-chart");
    if (mount && sel) {
      REGIONS.forEach(function (r) {
        var o = document.createElement("option");
        o.value = r.id; o.textContent = r.name;
        sel.appendChild(o);
      });
      sel.value = "ip";
      var draw = function () {
        var f = FORECAST[sel.value], r = region(sel.value);
        lineChart(mount, FORECAST_DAYS, [
          { name: "Forecast Tmax", color: "#e2620d", values: f.values, area: true },
          { name: "Long-period normal", color: "#0b5fa5",
            values: FORECAST_DAYS.map(function () { return f.normal; }), dashed: true }
        ], { unit: " °C", aria: "Seven day maximum temperature forecast for " + r.name });

        var peak = Math.max.apply(null, f.values);
        var dep = peak - f.normal;
        var sev = severityFor(dep, peak);
        byId("f-peak").textContent = peak.toFixed(1);
        byId("f-normal").textContent = f.normal.toFixed(1);
        byId("f-dep").textContent = (dep >= 0 ? "+" : "") + dep.toFixed(1);
        var chip = byId("f-sev");
        chip.className = "chip " + sev;
        chip.textContent = SEV_LABEL[sev];
        byId("f-region-name").textContent = r.name;
        byId("f-region-states").textContent = r.states;
      };
      sel.addEventListener("change", draw);
      draw();
    }

    var hm = byId("season-heatmap");
    if (hm) heatmap(hm);

    var tr = byId("trend-chart");
    if (tr) {
      barChart(tr, REGIONS.map(function (r) {
        return { label: r.name, value: REGION_TREND[r.id].trend, color: "#c0341c" };
      }), { min: 0, max: 0.36, step: 0.1, unit: " °C",
        aria: "Decadal warming trend in pre-monsoon maximum temperature by region" });
    }

    var hw = byId("hwdays-chart");
    if (hw) {
      lineChart(hw, TREND_DECADES,
        REGIONS.filter(function (r) { return ["nwi", "nci", "ip", "ec"].indexOf(r.id) > -1; })
          .map(function (r, i) {
            return { name: r.name, color: ["#c0341c", "#e2620d", "#c98a04", "#0b5fa5"][i],
              values: REGION_TREND[r.id].hwDays };
          }),
        { min: 0, max: 16, unit: " d", aria: "Mean heatwave days per year by decade for four regions" });
    }

    var ms = byId("model-body");
    if (ms) {
      var h = "";
      MODEL_SCORES.forEach(function (m) {
        h += "<tr><td class='region'>" + m.model + "</td><td>" + m.task + "</td><td>" + m.metric +
          "</td><td class='num'>" + m.value + "</td><td style='color:var(--ink-3);font-size:13px'>" + m.note + "</td></tr>";
      });
      ms.innerHTML = h;
    }
  }

  /* ---------------- Monitoring page ---------------- */
  function initMonitoring() {
    var body = byId("station-body");
    if (body) {
      var render = function () {
        var h = "";
        STATIONS.forEach(function (s) {
          var r = region(s.region);
          var st = s.status === "online"
            ? "<span class='chip online'>Online</span>"
            : "<span class='chip offline'>Maintenance</span>";
          h += "<tr>" +
            "<td class='region'><code>" + s.id + "</code></td>" +
            "<td>" + s.site + "<br><span style='font-size:12px;color:var(--ink-3)'>" + r.name + "</span></td>" +
            "<td class='num'>" + s.temp.toFixed(1) + " °C</td>" +
            "<td class='num'>" + s.rh + " %</td>" +
            "<td class='num'>" + s.wind.toFixed(1) + "</td>" +
            "<td class='num'>" + s.battery + " %</td>" +
            "<td>" + st + "</td>" +
            "<td style='font-size:13px;color:var(--ink-3)'>" + s.seen + "</td>" +
            "</tr>";
        });
        body.innerHTML = h;
      };
      render();
      // Simulated telemetry refresh from the AWS ingest endpoint.
      setInterval(function () {
        STATIONS.forEach(function (s) {
          if (s.status !== "online") return;
          s.temp = Math.round((s.temp + (Math.random() - 0.5) * 0.3) * 10) / 10;
          s.rh = Math.max(10, Math.min(95, s.rh + Math.round((Math.random() - 0.5) * 2)));
          s.wind = Math.round((Math.max(0, s.wind + (Math.random() - 0.5) * 0.8)) * 10) / 10;
        });
        render();
        var stamp = byId("last-sync");
        if (stamp) stamp.textContent = new Date().toLocaleTimeString("en-IN", { hour12: false });
      }, 5000);
    }

    var vm = byId("validation-chart");
    if (vm) {
      var W = 760, H = 330, pad = 52;
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", role: "img",
        "aria-label": "Comparison of AI model forecast against co-located automated weather station observations" });
      var lo = 34, hi = 46;
      var n = VALIDATION.length, gw = (W - pad * 2) / n;
      var y = function (v) { return H - pad - (v - lo) / (hi - lo) * (H - pad - 26); };
      for (var t = lo; t <= hi; t += 2) {
        svg.appendChild(el("line", { x1: pad, y1: y(t), x2: W - 14, y2: y(t), stroke: "#eef1f4" }));
        svg.appendChild(el("text", { x: pad - 10, y: y(t) + 4, "text-anchor": "end",
          "font-size": 11, fill: "#667085" }, t + " °C"));
      }
      VALIDATION.forEach(function (v, i) {
        var cx = pad + gw * i + gw / 2;
        var bw = Math.min(20, gw / 2.6);
        svg.appendChild(el("rect", { x: cx - bw - 2, y: y(v.forecast), width: bw,
          height: H - pad - y(v.forecast), rx: 3, fill: "#0b5fa5" }))
          .appendChild(el("title", {}, v.station + " forecast: " + v.forecast + " °C"));
        svg.appendChild(el("rect", { x: cx + 2, y: y(v.observed), width: bw,
          height: H - pad - y(v.observed), rx: 3, fill: "#e2620d" }))
          .appendChild(el("title", {}, v.station + " observed: " + v.observed + " °C"));
        svg.appendChild(el("text", { x: cx, y: H - pad + 17, "text-anchor": "middle",
          "font-size": 10, fill: "#667085" }, v.station.replace("AWS-", "")));
      });
      svg.appendChild(el("line", { x1: pad, y1: H - pad, x2: W - 14, y2: H - pad, stroke: "#d0d5dd" }));
      vm.innerHTML = "";
      vm.appendChild(svg);

      var errs = VALIDATION.map(function (v) { return Math.abs(v.forecast - v.observed); });
      var mae = errs.reduce(function (a, b) { return a + b; }, 0) / errs.length;
      var rmse = Math.sqrt(errs.reduce(function (a, b) { return a + b * b; }, 0) / errs.length);
      if (byId("v-mae")) byId("v-mae").textContent = mae.toFixed(2);
      if (byId("v-rmse")) byId("v-rmse").textContent = rmse.toFixed(2);
      if (byId("v-max")) byId("v-max").textContent = Math.max.apply(null, errs).toFixed(2);
    }
  }

  /* ---------------- Advisories page ---------------- */
  function initAdvisories() {
    var mount = byId("advisory-list");
    if (!mount) return;
    var h = "";
    ADVISORIES.forEach(function (a) {
      h += "<article class='advisory'>" +
        "<div class='meta'><span class='chip " + a.severity + "'>" + SEV_LABEL[a.severity] + "</span>" +
        "<span>" + a.audience + "</span><span>&middot;</span><span>" + a.region + "</span>" +
        "<span>&middot;</span><span>" + a.issued + "</span></div>" +
        "<h3>" + a.headline + "</h3><ul>";
      a.points.forEach(function (p) { h += "<li>" + p + "</li>"; });
      h += "</ul><p class='src'>Generated by the LLM advisory module from the region-wise forecast " +
        "and severity classification, then reviewed by a duty meteorologist before release " +
        "(human-in-the-loop checkpoint).</p></article>";
    });
    mount.innerHTML = h;

    var filter = byId("audience-filter");
    if (filter) {
      var opts = ["All audiences"].concat(ADVISORIES.map(function (a) { return a.audience; }));
      opts.forEach(function (o) {
        var n = document.createElement("option");
        n.value = o; n.textContent = o; filter.appendChild(n);
      });
      filter.addEventListener("change", function () {
        var v = filter.value;
        var cards = mount.querySelectorAll(".advisory");
        for (var i = 0; i < cards.length; i++) {
          cards[i].style.display = (v === "All audiences" || ADVISORIES[i].audience === v) ? "" : "none";
        }
      });
    }
  }

  /* ---------------- Home page summary tiles ---------------- */
  function initHome() {
    var mount = byId("home-bars");
    if (mount) {
      barChart(mount, WATCH.map(function (w) {
        return { label: region(w.region).short + " — " + region(w.region).name, value: w.tmax,
          color: SEV_COLOR[w.severity] };
      }), { min: 20, max: 50, step: 5, unit: " °C",
        aria: "Current region-wise maximum temperature outlook for the seven IMD regions" });
    }
    var alertCount = byId("alert-count");
    if (alertCount) {
      alertCount.textContent = WATCH.filter(function (w) {
        return w.severity === "heat" || w.severity === "severe" || w.severity === "extreme";
      }).length;
    }
    var hs = byId("hotspot-count");
    if (hs) {
      hs.textContent = WATCH.reduce(function (a, w) { return a + w.hotspots; }, 0);
    }
  }

  /* ---------------- Year stamp ---------------- */
  function initYear() {
    var y = document.querySelectorAll("[data-year]");
    for (var i = 0; i < y.length; i++) y[i].textContent = new Date().getFullYear();
  }

  document.addEventListener("DOMContentLoaded", function () {
    markActiveNav();
    initYear();
    initHome();
    initDashboard();
    initForecast();
    initMonitoring();
    initAdvisories();
  });
})();
