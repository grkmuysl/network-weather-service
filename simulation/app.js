// Tarayıcı simülasyonunun arayüzü. Hesaplar engine.js içindedir.
(function () {
  "use strict";
  const { SLOT, DAY, RUN_DAYS, MARGIN_DAYS, THRESHOLD } = NWL.constants;
  const $ = (id) => document.getElementById(id);
  const COLORS = { a: "#93a9b3", b: "#ff6f61", c: "#f2b84b", d: "#4fd1b8" };
  const TEXT = {
    en: {
      lead: "A live simulation of one 5G cell. The base station broadcasts its 24-hour load forecast, and AI agents move deferrable work to quiet times.",
      cellTitle: "Cell view", kSending: "Agent sending", kUrgent: "Urgent task", kWaiting: "Waiting for a quiet time",
      kDeaf: "Cannot hear the forecast", kHuman: "Human traffic", scenarioTitle: "Scenario", stripTitle: "Forecast for the next 24 hours",
      chartTitle: "Cell load across the day", tryTitle: "Try a situation", trace: "Traffic", forecaster: "Forecast model",
      share: "Agent traffic", urgent: "Urgent tasks", maxSlack: "Longest deadline", noise: "Forecast error", copies: "Broadcast delivery",
      seedLabel: "Random tasks", newTasks: "New tasks", adaptive: "Widen the random window when the forecast is uncertain",
      tableTitle: "Scenario results", packetTitle: "Latest broadcast",
      footer: "Runs in your browser with the same rules as the project's Python simulator. Milan data: BigDataChallenge, Barlacchi et al. 2015 (ODbL), via CESNET TS-Zoo. Capacity and message loss: NVIDIA Sionna. All agent settings are assumptions.",
      play: "Play", pause: "Pause", clear: "Clear", busy: "Busy", jam: "Congested", load: "load",
      sA: "No agents", sB: "Ignore forecast", sC: "No randomness", sD: "Full NWL",
      peak: "Daily peak", congested: "Congested min/day", delay: "Agent delay (min)", violations: "Missed deadlines",
      par: "Peak/average", unserved: "Lost human demand", vsB: "against ignoring the forecast", same: "same as ignoring the forecast",
      tEvening: "Milan, evening area", tCentre: "Milan, city centre", tAverage: "Milan, average area", tSynthetic: "Synthetic day",
      fProfile: "Weekly shape + AR(1)", fYesterday: "Same time yesterday", fWeek: "Same time last week", fAverage: "Moving average", fPerfect: "Perfect (human load)",
      cPerfect: "Always delivered", c1: "1 copy (35% lost)", c2: "2 copies (19% lost)", c4: "4 copies (9% lost)",
      day: "Day", date: (i) => `${13 + i} Dec 2013`,
      pDefault: "Normal day", pHerd: "Herd effect", pBad: "Bad forecast", pFix: "Bad forecast, fixed", pEdge: "Weak coverage", pPeak: "Too many agents",
      storyDefault: "Agents move deferrable work into the night. The evening peak stays lower, and no deadline is missed.",
      storyHerd: "Every agent picks the same quiet slot, so the quiet slot becomes the worst moment of the day. Watch the amber line.",
      storyBad: "A persistent forecast error creates false quiet dips. Agents gather in them, so a wrong forecast is worse than none.",
      storyFix: "Same error, but agents treat uncertain slots as equal and spread out across more of the day.",
      storyEdge: "With one copy, agents at the cell edge rarely hear the forecast and send at once. Coverage matters more than average delivery.",
      storyPeak: "With 75% agent traffic the cell has no quiet time left. Moving work cannot fix a lack of capacity.",
      noForecast: "This scenario ignores the broadcast. The strip shows what Full NWL would hear.",
      stripNote: (q, c) => `Quiet window covers ${q} of the next 96 slots. Mean confidence ${c}%.`,
      packet: (b, r) => `Header in amber, then slot, load and confidence for each entry. ${b} bytes in total, about ${r} bit/s of overhead.`,
      tableNote: (d) => `Averages over ${d} measured days. Cell capacity 132 Mbps from Sionna, congestion above 80%.`,
    },
    tr: {
      lead: "Tek bir 5G hücresinin canlı simülasyonu. Baz istasyonu 24 saatlik yük tahminini yayınlar, yapay zekâ ajanları ertelenebilir işleri sakin saatlere taşır.",
      cellTitle: "Hücre görünümü", kSending: "Gönderen ajan", kUrgent: "Acil görev", kWaiting: "Sakin saati bekliyor",
      kDeaf: "Tahmini alamıyor", kHuman: "İnsan trafiği", scenarioTitle: "Senaryo", stripTitle: "Önümüzdeki 24 saatin tahmini",
      chartTitle: "Gün boyunca hücre yükü", tryTitle: "Bir durum dene", trace: "Trafik", forecaster: "Tahmin modeli",
      share: "Ajan trafiği", urgent: "Acil görevler", maxSlack: "En uzun teslim süresi", noise: "Tahmin hatası", copies: "Yayın iletimi",
      seedLabel: "Rastgele görevler", newTasks: "Yeni görevler", adaptive: "Tahmin belirsizse rastgele pencereyi genişlet",
      tableTitle: "Senaryo sonuçları", packetTitle: "Son yayın",
      footer: "Tarayıcınızda, projenin Python simülatörüyle aynı kurallarla çalışır. Milano verisi: BigDataChallenge, Barlacchi vd. 2015 (ODbL), CESNET TS-Zoo üzerinden. Kapasite ve mesaj kaybı: NVIDIA Sionna. Ajan ayarlarının tümü varsayımdır.",
      play: "Oynat", pause: "Duraklat", clear: "Açık", busy: "Yoğun", jam: "Tıkanık", load: "yük",
      sA: "Ajan yok", sB: "Tahmini yok say", sC: "Rastgelelik yok", sD: "Tam NWL",
      peak: "Günlük tepe", congested: "Tıkanık dk/gün", delay: "Ajan gecikmesi (dk)", violations: "Kaçan teslim",
      par: "Tepe/ortalama", unserved: "Kayıp insan talebi", vsB: "tahmini yok saymaya göre", same: "tahmini yok saymakla aynı",
      tEvening: "Milano, akşam yoğun bölge", tCentre: "Milano, şehir merkezi", tAverage: "Milano, ortalama bölge", tSynthetic: "Sentetik gün",
      fProfile: "Haftalık şekil + AR(1)", fYesterday: "Dün aynı saat", fWeek: "Geçen hafta aynı saat", fAverage: "Hareketli ortalama", fPerfect: "Kusursuz (insan yükü)",
      cPerfect: "Her zaman ulaşır", c1: "1 kopya (%35 kayıp)", c2: "2 kopya (%19 kayıp)", c4: "4 kopya (%9 kayıp)",
      day: "Gün", date: (i) => `${13 + i} Aralık 2013`,
      pDefault: "Normal gün", pHerd: "Sürü etkisi", pBad: "Kötü tahmin", pFix: "Kötü tahmin, çözüldü", pEdge: "Zayıf kapsama", pPeak: "Çok fazla ajan",
      storyDefault: "Ajanlar ertelenebilir işleri geceye taşır. Akşam tepesi düşük kalır, hiçbir teslim süresi kaçmaz.",
      storyHerd: "Her ajan aynı sakin slotu seçer ve o slot günün en kötü anına dönüşür. Sarı çizgiyi izleyin.",
      storyBad: "Kalıcı tahmin hatası sahte sakin boşluklar yaratır. Ajanlar oraya yığılır, yanlış tahmin hiç tahminden kötü olur.",
      storyFix: "Hata aynı, ama ajanlar belirsiz slotları eşit sayar ve günün daha geniş bir kısmına dağılır.",
      storyEdge: "Tek kopyada hücre kenarındaki ajanlar tahmini nadiren alır ve hemen gönderir. Kapsama, ortalama iletimden önemlidir.",
      storyPeak: "%75 ajan trafiğinde hücrede sakin zaman kalmaz. İşi kaydırmak kapasite eksiğini çözemez.",
      noForecast: "Bu senaryo yayını yok sayar. Şerit, Tam NWL'nin duyacağı tahmini gösterir.",
      stripNote: (q, c) => `Sakin pencere önümüzdeki 96 slotun ${q} tanesini kapsıyor. Ortalama güven %${c}.`,
      packet: (b, r) => `Sarı kısım başlık, sonra her kayıt için slot, yük ve güven gelir. Toplam ${b} bayt, yaklaşık ${r} bit/s ek yük.`,
      tableNote: (d) => `${d} ölçüm gününün ortalaması. Hücre kapasitesi Sionna'dan 132 Mbps, %80 üstü tıkanıklıktır.`,
    },
  };

  const PRESETS = {
    pDefault: { set: {}, scenario: "d", story: "storyDefault", time: 19 * 60 },
    pHerd: { set: {}, scenario: "c", story: "storyHerd", time: 2 * 60 },
    pBad: { set: { forecaster: "perfect", noise: 20, adaptive: false }, scenario: "d", story: "storyBad", time: 3 * 60 },
    pFix: { set: { forecaster: "perfect", noise: 20, adaptive: true }, scenario: "d", story: "storyFix", time: 3 * 60 },
    pEdge: { set: { copies: 1 }, scenario: "d", story: "storyEdge", time: 20 * 60 },
    pPeak: { set: { share: 75 }, scenario: "d", story: "storyPeak", time: 20 * 60 },
  };
  const BASE = { trace: "evening", forecaster: "profile", share: 25, urgent: 30, maxSlack: 12, noise: 0, copies: 0, adaptive: true };

  let lang = "en";
  try { lang = localStorage.getItem("nwl-lang") || "en"; } catch (e) { /* tarayıcı depolamayı kapatmış olabilir */ }
  const state = {
    scenario: "d", day: 4, minute: 19 * 60, playing: false, seed: 1, preset: "pDefault", story: "storyDefault",
    visible: { a: true, b: true, c: false, d: true }, result: null, humans: [],
  };
  const t = (key) => TEXT[lang][key];

  function setupHumans() {
    let seed = 12345;
    const next = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    state.humans = Array.from({ length: 160 }, () => ({ r: Math.sqrt(next()), a: next() * 6.283, z: next() }));
  }

  function fitCanvas(canvas) {
    const scale = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(box.width * scale)), height = Math.max(1, Math.round(box.height * scale));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    return { ctx, w: box.width, h: box.height };
  }

  const absolute = () => state.day * DAY + Math.floor(state.minute);
  const offeredLoad = (key, at) => {
    const r = state.result;
    return (r.human[at] + r[key].agent[at]) / r.capacity;
  };
  const loadColor = (load) => (load > THRESHOLD ? COLORS.b : load > 0.6 ? COLORS.c : COLORS.d);
  const clock = (m) => `${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(Math.floor(m) % 60).padStart(2, "0")}`;

  function drawRadar() {
    const { ctx, w, h } = fitCanvas($("radar"));
    const r = state.result, at = absolute(), load = offeredLoad(state.scenario, at);
    const cx = w / 2, cy = h / 2 + 4, R = Math.min(w, h) * 0.46, run = r[state.scenario];
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "#1b3440"; ctx.lineWidth = 1;
    for (let ring = 1; ring <= 4; ring++) { ctx.beginPath(); ctx.arc(cx, cy, (R * ring) / 4, 0, 6.2832); ctx.stroke(); }
    ctx.beginPath();
    for (let i = 0; i <= 6; i++) {
      const a = (i * Math.PI) / 3 + Math.PI / 6, x = cx + Math.cos(a) * R * 1.03, y = cy + Math.sin(a) * R * 1.03;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.strokeStyle = loadColor(load); ctx.globalAlpha = 0.75; ctx.lineWidth = 1.5; ctx.stroke(); ctx.globalAlpha = 1;

    const humans = Math.round(state.humans.length * Math.min(1, (r.human[at] / r.capacity) * 1.25));
    ctx.fillStyle = "#6d8791";
    for (let i = 0; i < humans; i++) {
      const p = state.humans[i], a = p.a + Math.sin(state.minute / 90 + p.z * 6) * 0.03;
      ctx.globalAlpha = 0.35;
      ctx.beginPath(); ctx.arc(cx + Math.cos(a) * p.r * R, cy + Math.sin(a) * p.r * R, 1.6, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;

    if (r.params.copies !== undefined && state.scenario !== "a") {
      const phase = (at % SLOT) / SLOT;
      if (state.scenario === "c" || state.scenario === "d") {
        ctx.strokeStyle = COLORS.d; ctx.lineWidth = 1.5; ctx.globalAlpha = (1 - phase) * 0.6;
        ctx.beginPath(); ctx.arc(cx, cy, 10 + phase * R, 0, 6.2832); ctx.stroke(); ctx.globalAlpha = 1;
      }
    }

    if (state.scenario !== "a") {
      const tasks = r.tasks, starts = run.starts, finish = run.finish, copies = r.params.copies;
      for (let i = 0; i < tasks.length; i++) {
        const task = tasks[i];
        if (task.arrival > at) break;
        const end = Number.isNaN(finish[i]) ? Infinity : finish[i];
        if (end <= at) continue;
        const radius = Math.sqrt(1 - task.position * 0.97) * R;
        const x = cx + Math.cos(task.angle) * radius, y = cy + Math.sin(task.angle) * radius;
        const sending = starts[i] <= at;
        const deaf = copies && !task.urgent && NWL.lossAt(task.position, copies) >= 0.5;
        if (sending) {
          ctx.strokeStyle = task.urgent ? COLORS.b : COLORS.d; ctx.globalAlpha = 0.22; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(cx, cy); ctx.stroke(); ctx.globalAlpha = 1;
          ctx.fillStyle = task.urgent ? COLORS.b : COLORS.d;
          ctx.beginPath(); ctx.arc(x, y, 3, 0, 6.2832); ctx.fill();
        } else {
          ctx.strokeStyle = COLORS.c; ctx.lineWidth = 1; ctx.globalAlpha = 0.55;
          ctx.beginPath(); ctx.arc(x, y, 2.4, 0, 6.2832); ctx.stroke(); ctx.globalAlpha = 1;
        }
        if (deaf && sending) {
          ctx.strokeStyle = "#e8f0f1"; ctx.setLineDash([2, 2]); ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(x, y, 5.5, 0, 6.2832); ctx.stroke(); ctx.setLineDash([]);
        }
      }
    }

    ctx.fillStyle = "#e8f0f1";
    ctx.beginPath(); ctx.moveTo(cx, cy - 13); ctx.lineTo(cx + 7, cy + 9); ctx.lineTo(cx - 7, cy + 9); ctx.closePath(); ctx.fill();

    $("clock").textContent = `${dayLabel(state.day)}  ${clock(state.minute)}`;
    $("load").textContent = `${Math.round(load * 100)}%`;
    const pill = $("status"), kind = load > THRESHOLD ? "jam" : load > 0.6 ? "busy" : "clear";
    pill.textContent = t(kind); pill.className = `pill ${kind === "clear" ? "" : kind}`;
  }

  function dayLabel(i) {
    return state.result && state.result.params.trace === "synthetic" ? `${t("day")} ${i}` : t("date")(i);
  }

  function currentBroadcast() {
    const key = state.scenario === "c" ? "c" : "d";
    return state.result[key].broadcasts[Math.floor(absolute() / SLOT)];
  }

  function drawStrip() {
    const { ctx, w, h } = fitCanvas($("strip"));
    const b = currentBroadcast(), pad = 18, top = 8, base = h - pad, n = 96, gap = 1;
    ctx.clearRect(0, 0, w, h);
    if (!b) return;
    const adaptive = state.result.params.adaptive;
    let best = Infinity, bestError = 0, sum = 0;
    for (let i = 0; i < b.loads.length; i++) {
      const error = (1 - b.confidence[i]) * 0.5, cautious = b.loads[i] + error;
      sum += b.confidence[i];
      if (cautious < best) { best = cautious; bestError = error; }
    }
    const limit = best + 0.05 + (adaptive ? 2 * bestError : 0), max = Math.max(1, ...b.loads) * 1.08, bar = (w - gap * n) / n;
    let quiet = 0;
    for (let i = 0; i < b.loads.length; i++) {
      const load = b.loads[i], x = i * (bar + gap), error = (1 - b.confidence[i]) * 0.5;
      const height = (load / max) * (base - top), band = (error / max) * (base - top);
      const inWindow = load + error <= limit;
      if (inWindow) quiet++;
      ctx.fillStyle = loadColor(load);
      ctx.globalAlpha = inWindow ? 1 : 0.5;
      ctx.fillRect(x, base - height, bar, height);
      ctx.globalAlpha = 0.25;
      ctx.fillRect(x, base - height - band, bar, band);
    }
    ctx.globalAlpha = 1;
    const y = base - (THRESHOLD / max) * (base - top);
    ctx.strokeStyle = "#9fb4bc"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#9fb4bc"; ctx.font = "11px JetBrains Mono, monospace"; ctx.textAlign = "left";
    for (let i = 0; i <= 3; i++) {
      const slot = b.slot + i * 24, x = Math.min((i * 24) * (bar + gap), w - 38);
      ctx.fillText(clock((slot * SLOT) % DAY), x, h - 4);
    }
    $("stripNote").textContent = state.scenario === "a" || state.scenario === "b"
      ? t("noForecast") : t("stripNote")(quiet, Math.round((100 * sum) / b.loads.length));
  }

  function drawChart() {
    const { ctx, w, h } = fitCanvas($("chart"));
    const left = 44, right = 12, top = 10, bottom = 26, r = state.result, first = state.day * DAY;
    const keys = "abcd".split("").filter((k) => state.visible[k]);
    let max = 1;
    for (const k of keys) for (let m = 0; m < DAY; m += 3) max = Math.max(max, offeredLoad(k, first + m));
    max *= 1.06;
    const X = (m) => left + (m / DAY) * (w - left - right), Y = (v) => h - bottom - (v / max) * (h - top - bottom);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.textAlign = "right"; ctx.fillStyle = "#9fb4bc";
    const step = max > 2 ? 0.5 : 0.25;
    for (let v = 0; v <= max; v += step) {
      ctx.strokeStyle = "#1f3945"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(w - right, Y(v)); ctx.stroke();
      ctx.fillText(`${Math.round(v * 100)}%`, left - 6, Y(v) + 4);
    }
    ctx.textAlign = "center";
    for (let hour = 0; hour <= 24; hour += w < 620 ? 6 : 3) ctx.fillText(`${String(hour).padStart(2, "0")}:00`, Math.min(X(hour * 60), w - 22), h - 8);
    ctx.strokeStyle = "#e8f0f1"; ctx.setLineDash([5, 4]); ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(left, Y(THRESHOLD)); ctx.lineTo(w - right, Y(THRESHOLD)); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    for (const k of keys) {
      ctx.strokeStyle = COLORS[k]; ctx.lineWidth = k === state.scenario ? 2.4 : 1.4;
      ctx.beginPath();
      for (let m = 0; m < DAY; m += 2) {
        const y = Y(offeredLoad(k, first + m));
        if (m) ctx.lineTo(X(m), y); else ctx.moveTo(X(m), y);
      }
      ctx.stroke();
    }
    ctx.strokeStyle = "#e8f0f1"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(state.minute), top); ctx.lineTo(X(state.minute), h - bottom); ctx.stroke();
    ctx.fillStyle = COLORS[state.scenario];
    ctx.beginPath(); ctx.arc(X(state.minute), Y(offeredLoad(state.scenario, absolute())), 4, 0, 6.2832); ctx.fill();
    state.chartX = (x) => Math.max(0, Math.min(DAY - 1, Math.round(((x - left) / (w - left - right)) * DAY)));
  }

  function drawHex() {
    const b = currentBroadcast();
    if (!b) return;
    const bytes = NWL.encodeBroadcast(b), hex = (v) => v.toString(16).padStart(2, "0");
    const head = bytes.slice(0, 6).map(hex).join(" ");
    const entries = [0, 1, 2].map((i) => {
      const e = bytes.slice(6 + i * 5, 11 + i * 5);
      return `<span class="s">${hex(e[0])} ${hex(e[1])}</span> <span class="l">${hex(e[2])} ${hex(e[3])}</span> <span class="c">${hex(e[4])}</span>`;
    });
    $("hex").innerHTML = `<span class="h">${head}</span>\n${entries.join("\n")}\n...`;
    $("packetNote").textContent = t("packet")(6 + b.loads.length * 5, 4.3);
  }

  function renderFrame() {
    drawRadar(); drawStrip(); drawChart(); drawHex();
    $("time").value = Math.floor(state.minute);
    $("play").textContent = state.playing ? t("pause") : t("play");
  }

  function fmt(key, value) {
    if (key === "peak") return `${Math.round(value * 100)}%`;
    if (key === "unserved" || key === "violations") return `${value.toFixed(2)}%`;
    if (key === "peakToAverage") return value.toFixed(2);
    return value.toFixed(key === "congested" ? 1 : 0);
  }

  function renderSummary() {
    const r = state.result, m = r[state.scenario].metrics, b = r.b.metrics;
    const change = m.congested - b.congested, percent = b.congested ? Math.round((100 * change) / b.congested) : 0;
    const delta = state.scenario === "b" ? "" : `<small class="delta ${change > 0 ? "worse" : ""}">${percent > 0 ? "+" : ""}${percent}% ${t("vsB")}</small>`;
    const stat = (value, label, extra = "") => `<div class="stat"><strong>${value}</strong><small>${label}</small>${extra}</div>`;
    $("stats").innerHTML = stat(fmt("peak", m.peak), t("peak"))
      + stat(fmt("congested", m.congested), t("congested"), delta)
      + stat(fmt("delay", m.delay), t("delay"))
      + stat(fmt("violations", m.violations), t("violations"));

    const rows = [["peak", "peak"], ["par", "peakToAverage"], ["congested", "congested"], ["unserved", "unserved"], ["delay", "delay"], ["violations", "violations"]];
    $("results").innerHTML = `<thead><tr><th></th>${rows.map(([label]) => `<th>${t(label)}</th>`).join("")}</tr></thead><tbody>`
      + "abcd".split("").map((k) => `<tr class="${k === state.scenario ? "on" : ""}"><td>(${k}) ${t(`s${k.toUpperCase()}`)}</td>${rows.map(([, field]) => `<td>${fmt(field, r[k].metrics[field])}</td>`).join("")}</tr>`).join("")
      + "</tbody>";
    $("tableNote").textContent = t("tableNote")(RUN_DAYS - 2 * MARGIN_DAYS);
  }

  function renderChrome() {
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-t]").forEach((el) => { el.textContent = t(el.dataset.t); });
    $("lang").textContent = lang === "en" ? "Türkçe" : "English";
    $("segments").innerHTML = "abcd".split("").map((k) => `<button type="button" data-s="${k}" aria-pressed="${k === state.scenario}"><b>(${k})</b>${t(`s${k.toUpperCase()}`)}</button>`).join("");
    $("series").innerHTML = "abcd".split("").map((k) => `<button class="chip" type="button" data-v="${k}" aria-pressed="${state.visible[k]}"><i style="background:${COLORS[k]}"></i>(${k}) ${t(`s${k.toUpperCase()}`)}</button>`).join("");
    $("presets").innerHTML = Object.keys(PRESETS).map((k) => `<button type="button" data-p="${k}" aria-pressed="${k === state.preset}">${t(k)}</button>`).join("");
    $("story").textContent = t(state.story);
    const fill = (id, list) => {
      const el = $(id), value = el.value;
      el.innerHTML = list.map(([v, label]) => `<option value="${v}">${label}</option>`).join("");
      if (value) el.value = value;
    };
    fill("trace", [["evening", t("tEvening")], ["centre", t("tCentre")], ["average", t("tAverage")], ["synthetic", t("tSynthetic")]]);
    fill("forecaster", [["profile", t("fProfile")], ["yesterday", t("fYesterday")], ["week", t("fWeek")], ["average", t("fAverage")], ["perfect", t("fPerfect")]]);
    fill("copies", [["0", t("cPerfect")], ["1", t("c1")], ["2", t("c2")], ["4", t("c4")]]);
    const dayValue = String(state.day);
    $("day").innerHTML = Array.from({ length: RUN_DAYS - 2 * MARGIN_DAYS }, (_, i) => i + MARGIN_DAYS).map((i) => `<option value="${i}">${dayLabel(i)}</option>`).join("");
    $("day").value = dayValue;
    $("shareOut").textContent = `${$("share").value}%`;
    $("urgentOut").textContent = `${$("urgent").value}%`;
    $("maxSlackOut").textContent = `${$("maxSlack").value} h`;
    $("noiseOut").textContent = `${$("noise").value}%`;
  }

  function recompute() {
    state.result = NWL.runAll({
      trace: $("trace").value, forecaster: $("forecaster").value,
      share: +$("share").value / 100, urgent: +$("urgent").value / 100, maxSlack: +$("maxSlack").value * 60,
      noise: +$("noise").value / 100, copies: +$("copies").value, adaptive: $("adaptive").checked, seed: state.seed,
    });
    renderChrome(); renderSummary(); renderFrame();
  }

  function applyForm(values) {
    for (const [key, value] of Object.entries(values)) {
      if (key === "adaptive") $(key).checked = value; else $(key).value = value;
    }
  }

  let last = 0;
  function tick(now) {
    if (state.playing) {
      const elapsed = Math.min(0.1, (now - last) / 1000);
      state.minute += elapsed * +$("speed").value;
      if (state.minute >= DAY) {
        state.minute -= DAY;
        state.day = state.day + 1 >= RUN_DAYS - MARGIN_DAYS ? MARGIN_DAYS : state.day + 1;
        $("day").value = state.day;
      }
      renderFrame();
    }
    last = now;
    requestAnimationFrame(tick);
  }

  $("lang").addEventListener("click", () => {
    lang = lang === "en" ? "tr" : "en";
    try { localStorage.setItem("nwl-lang", lang); } catch (e) { /* depolama kapalıysa dil oturumla sınırlı kalır */ }
    renderChrome(); renderSummary(); renderFrame();
  });
  $("play").addEventListener("click", () => { state.playing = !state.playing; renderFrame(); });
  $("day").addEventListener("change", (e) => { state.day = +e.target.value; renderFrame(); });
  $("time").addEventListener("input", (e) => { state.minute = +e.target.value; renderFrame(); });
  $("seed").addEventListener("click", () => { state.seed += 1; recompute(); });
  document.querySelector(".split").addEventListener("input", (e) => {
    if (e.target.closest(".controls")) { state.preset = ""; clearTimeout(state.timer); state.timer = setTimeout(recompute, 60); }
  });
  $("segments").addEventListener("click", (e) => {
    const b = e.target.closest("[data-s]");
    if (!b) return;
    state.scenario = b.dataset.s; state.visible[state.scenario] = true;
    renderChrome(); renderSummary(); renderFrame();
  });
  $("series").addEventListener("click", (e) => {
    const b = e.target.closest("[data-v]");
    if (!b) return;
    state.visible[b.dataset.v] = !state.visible[b.dataset.v];
    renderChrome(); renderFrame();
  });
  $("presets").addEventListener("click", (e) => {
    const b = e.target.closest("[data-p]");
    if (!b) return;
    const preset = PRESETS[b.dataset.p];
    state.preset = b.dataset.p; state.scenario = preset.scenario; state.story = preset.story; state.minute = preset.time;
    state.visible = { a: true, b: true, c: preset.scenario === "c", d: true };
    applyForm({ ...BASE, ...preset.set });
    recompute();
  });
  const scrub = (e) => {
    const box = $("chart").getBoundingClientRect();
    state.minute = state.chartX(e.clientX - box.left);
    renderFrame();
  };
  $("chart").addEventListener("pointerdown", (e) => { scrub(e); $("chart").setPointerCapture(e.pointerId); });
  $("chart").addEventListener("pointermove", (e) => { if (e.buttons) scrub(e); });
  window.addEventListener("resize", renderFrame);

  setupHumans();
  renderChrome();
  applyForm(BASE);
  recompute();
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) state.playing = true;
  renderFrame();
  requestAnimationFrame((now) => { last = now; tick(now); });
})();
