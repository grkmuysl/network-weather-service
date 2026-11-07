// Python simülatörünün tarayıcıdaki karşılığı. Hücre, görev, zamanlayıcı ve ölçütler
// nwl/ paketindeki kurallarla aynıdır. LSTM ve statsmodels burada çalışmadığı için
// ARIMA yerine "haftalık şekil + AR(1)" modeli kullanılır.
(function (root) {
  "use strict";
  const SLOT = 15, HORIZON = 96, DAY = 1440, WEEK_SLOTS = 672;
  const SCALE = 0.5, QUIET_MARGIN = 0.05, ERROR_WEIGHT = 2, SAFETY = 30;
  const TASK_MBPS = 2, MEAN_MINUTES = 10, SPREAD = 0.8, URGENT_SLACK = 10;
  const PAST_DAYS = 42, RUN_DAYS = 10, MARGIN_DAYS = 1, THRESHOLD = 0.8;
  const ERROR_WINDOW = 2 * HORIZON, WARMUP_STEP = 4;
  const DEFAULTS = {
    trace: "evening", forecaster: "profile", share: 0.25, urgent: 0.3, maxSlack: 720,
    noise: 0, memory: 0.95, copies: 0, adaptive: true, seed: 1,
  };

  function makeRandom(seed) {
    let a = seed >>> 0;
    const next = () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normal = () => Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next());
    const poisson = (mean) => {
      const limit = Math.exp(-mean);
      let count = 0, product = next();
      while (product > limit) { count++; product *= next(); }
      return count;
    };
    return { next, normal, poisson };
  }

  function syntheticLoad() {
    const bump = (h, c, w) => {
      const d = Math.min(Math.abs(h - c), 24 - Math.abs(h - c));
      return Math.exp(-0.5 * (d / w) ** 2);
    };
    const days = PAST_DAYS + RUN_DAYS, random = makeRandom(7);
    const out = new Float64Array(days * HORIZON);
    let noise = 0;
    for (let d = 0; d < days; d++) {
      const factor = 1 + 0.05 * random.normal();
      for (let s = 0; s < HORIZON; s++) {
        noise = 0.74 * noise + 0.034 * random.normal();
        const h = (s + 0.5) / 4;
        const shape = 0.15 + 0.3 * bump(h, 13, 3.5) + 0.58 * bump(h, 21.5, 1.8);
        out[d * HORIZON + s] = shape * factor * (1 + noise);
      }
    }
    return out;
  }

  function buildWorld(trace) {
    const slots = trace === "synthetic" ? syntheticLoad() : Float64Array.from(root.NWL_DATA.traces[trace]);
    const pastSlots = PAST_DAYS * HORIZON, runSlots = RUN_DAYS * HORIZON;
    const run = new Float64Array(runSlots * SLOT);
    for (let i = 0; i < run.length; i++) run[i] = slots[pastSlots + Math.floor(i / SLOT)];
    const profile = new Float64Array(DAY);
    for (let i = 0; i < pastSlots; i++) {
      for (let m = 0; m < SLOT; m++) profile[(i % HORIZON) * SLOT + m] += slots[i] / PAST_DAYS;
    }
    return { past: slots.subarray(0, pastSlots), run, profile };
  }

  function generateTasks(world, capacity, p, seed) {
    const random = makeRandom(seed), meanSize = TASK_MBPS * 60 * MEAN_MINUTES;
    const logMean = Math.log(MEAN_MINUTES) - SPREAD ** 2 / 2, tasks = [];
    for (let t = 0; t < RUN_DAYS * DAY; t++) {
      const rate = p.share * capacity * world.profile[t % DAY] * 60 / meanSize;
      const count = random.poisson(rate);
      for (let i = 0; i < count; i++) {
        const duration = Math.exp(logMean + SPREAD * random.normal());
        const urgent = random.next() < p.urgent;
        const slack = urgent ? URGENT_SLACK : 120 + random.next() * (p.maxSlack - 120);
        tasks.push({
          arrival: t, duration, urgent, deadline: t + duration + slack,
          position: random.next(), angle: random.next() * 2 * Math.PI,
        });
      }
    }
    return tasks;
  }

  // Sionna: SINR yüzdelik dağılımı ve kopya sayısına göre mesaj hata eğrisi.
  function lossAt(position, copies) {
    const d = root.NWL_DATA, sinr = d.sinrDbPercentiles[Math.min(98, Math.floor(position * 99))];
    const curve = d.messageErrorRate[String(copies)], snr = d.snrDb;
    if (sinr <= snr[0]) return curve[0];
    for (let i = 1; i < snr.length; i++) {
      if (sinr <= snr[i]) {
        const f = (sinr - snr[i - 1]) / (snr[i] - snr[i - 1]);
        return curve[i - 1] + f * (curve[i] - curve[i - 1]);
      }
    }
    return curve[curve.length - 1];
  }

  function missedBroadcasts(position, copies, random) {
    if (!copies) return 0;
    const loss = lossAt(position, copies);
    if (loss >= 1) return HORIZON;
    if (loss <= 0) return 0;
    return Math.min(Math.floor(Math.log(1 - random.next()) / Math.log(loss)), HORIZON);
  }

  function errorCurve(p, slots, seed) {
    const out = new Float64Array(slots);
    if (!p.noise) return out;
    const random = makeRandom(seed), m = p.memory;
    out[0] = p.noise * random.normal();
    for (let i = 1; i < slots; i++) {
      out[i] = m * out[i - 1] + p.noise * Math.sqrt(1 - m * m) * random.normal();
    }
    return out;
  }

  function measuredPast(world, p) {
    return Float64Array.from(world.past, (load) => load * (1 + p.share));
  }

  function makeModel(kind, past) {
    const profile = new Float64Array(WEEK_SLOTS), counts = new Float64Array(WEEK_SLOTS);
    for (let i = 0; i < past.length; i++) { profile[i % WEEK_SLOTS] += past[i]; counts[i % WEEK_SLOTS]++; }
    for (let i = 0; i < WEEK_SLOTS; i++) profile[i] /= counts[i];
    let top = 0, bottom = 0;
    for (let i = 1; i < past.length; i++) {
      const a = past[i - 1] - profile[(i - 1) % WEEK_SLOTS], b = past[i] - profile[i % WEEK_SLOTS];
      top += a * b; bottom += a * a;
    }
    const phi = bottom ? top / bottom : 0;
    return (series, n, out) => {
      if (kind === "average") {
        let sum = 0;
        for (let i = 1; i <= 8; i++) sum += series[n - i];
        out.fill(sum / 8);
      } else if (kind === "yesterday") {
        for (let h = 0; h < HORIZON; h++) out[h] = series[n - HORIZON + h];
      } else if (kind === "week") {
        for (let h = 0; h < HORIZON; h++) out[h] = series[n - WEEK_SLOTS + h];
      } else {
        let residual = series[n - 1] - profile[(n - 1) % WEEK_SLOTS];
        for (let h = 0; h < HORIZON; h++) {
          residual *= phi;
          out[h] = Math.max(profile[(n + h) % WEEK_SLOTS] + residual, 0);
        }
      }
    };
  }

  // Görevler sırayla işlenir, böylece taban istasyonu yalnızca bitmiş slotları görür.
  function runPolicy(tasks, world, capacity, p, randomised, seed) {
    const run = world.run, runSlots = run.length / SLOT, past = measuredPast(world, p);
    const random = makeRandom(seed), meter = Float64Array.from(run), taskLoad = TASK_MBPS / capacity;
    const addTask = (start, duration) => {
      const end = Math.min(start + Math.ceil(duration), meter.length);
      for (let m = start; m < end; m++) meter[m] += taskLoad * Math.min(start + duration - m, 1);
    };
    for (const t of tasks) if (t.urgent) addTask(t.arrival, t.duration);

    const n0 = past.length, series = new Float64Array(n0 + runSlots);
    series.set(past);
    const curve = errorCurve(p, runSlots + HORIZON, seed + 11), perfect = p.forecaster === "perfect";
    const model = perfect ? null : makeModel(p.forecaster, past), rows = new Map();
    const issue = (n) => { const out = new Float64Array(HORIZON); model(series, n, out); return out; };
    if (!perfect) for (let t = n0 - ERROR_WINDOW - HORIZON; t < n0; t += WARMUP_STEP) rows.set(t, issue(t));

    const errorNow = new Float64Array(HORIZON);
    const refreshError = (now) => {
      if (perfect) { errorNow.fill(p.noise * Math.sqrt(2 / Math.PI)); return; }
      if (now % WARMUP_STEP) return;
      const sum = new Float64Array(HORIZON), count = new Float64Array(HORIZON);
      for (const [origin, values] of rows) {
        if (origin < now - ERROR_WINDOW - HORIZON) { rows.delete(origin); continue; }
        for (let h = 0; h < HORIZON && origin + h < now; h++) {
          sum[h] += Math.abs(values[h] - series[origin + h]); count[h]++;
        }
      }
      for (let h = 0; h < HORIZON; h++) if (count[h]) errorNow[h] = sum[h] / count[h];
    };

    let measured = 0;
    const broadcasts = new Array(runSlots);
    const hear = (slot) => {
      if (broadcasts[slot]) return broadcasts[slot];
      for (; measured < slot; measured++) {
        let sum = 0;
        for (let m = 0; m < SLOT; m++) sum += meter[measured * SLOT + m];
        series[n0 + measured] = sum / SLOT;
      }
      const length = Math.min(HORIZON, runSlots - slot), now = n0 + slot;
      const values = perfect ? null : issue(now);
      if (!perfect) rows.set(now, values);
      refreshError(now);
      const loads = new Float32Array(length), confidence = new Float32Array(length);
      for (let h = 0; h < length; h++) {
        let load = perfect ? 0 : values[h];
        if (perfect) for (let m = 0; m < SLOT; m++) load += run[(slot + h) * SLOT + m] / SLOT;
        loads[h] = Math.max(load + curve[slot + h], 0);
        confidence[h] = Math.round(100 * Math.min(1, Math.max(0, 1 - errorNow[h] / SCALE))) / 100;
      }
      return (broadcasts[slot] = { slot, loads, confidence });
    };

    const starts = new Int32Array(tasks.length);
    tasks.forEach((task, i) => {
      starts[i] = task.arrival;
      if (task.urgent) return;
      const slot = Math.floor(task.arrival / SLOT);
      hear(slot);
      const heard = broadcasts[slot - missedBroadcasts(task.position, p.copies, random)];
      if (heard) {
        const latest = Math.floor(task.deadline - task.duration - SAFETY);
        let best = Infinity, bestError = 0, bestFirst = -1;
        const options = [];
        for (let h = 0; h < heard.loads.length; h++) {
          const begin = (heard.slot + h) * SLOT;
          const first = Math.max(begin, task.arrival), last = Math.min(begin + SLOT - 1, latest);
          if (first > last) continue;
          const error = (1 - heard.confidence[h]) * SCALE, cautious = heard.loads[h] + error;
          options.push([first, last, cautious]);
          if (cautious < best) { best = cautious; bestError = error; bestFirst = first; }
        }
        if (options.length) {
          if (!randomised) starts[i] = bestFirst;
          else {
            const limit = best + QUIET_MARGIN + (p.adaptive ? ERROR_WEIGHT * bestError : 0);
            let total = 0;
            for (const o of options) if (o[2] <= limit) total += o[1] - o[0] + 1;
            let pick = Math.floor(random.next() * total);
            for (const o of options) {
              if (o[2] > limit) continue;
              const size = o[1] - o[0] + 1;
              if (pick < size) { starts[i] = o[0] + pick; break; }
              pick -= size;
            }
          }
        }
      }
      addTask(starts[i], task.duration);
    });
    for (let s = 0; s < runSlots; s++) hear(s);
    return { starts, broadcasts };
  }

  function simulate(tasks, starts, human, capacity) {
    const steps = human.length, n = tasks.length;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => starts[a] - starts[b]);
    const remaining = Float64Array.from(tasks, (t) => t.duration * TASK_MBPS * 60);
    const finish = new Float64Array(n).fill(NaN), agent = new Float64Array(steps);
    let active = [], next = 0;
    for (let t = 0; t < steps; t++) {
      while (next < n && starts[order[next]] <= t) active.push(order[next++]);
      if (!active.length) continue;
      let wanted = 0;
      for (const id of active) wanted += Math.min(TASK_MBPS, remaining[id] / 60);
      agent[t] = wanted;
      const share = Math.min(1, capacity / (human[t] + wanted)), budget = TASK_MBPS * share * 60;
      active = active.filter((id) => {
        remaining[id] -= Math.min(remaining[id], budget);
        if (remaining[id] > 1e-9) return true;
        finish[id] = t + 1;
        return false;
      });
    }
    return { agent, finish };
  }

  function metricsOf(human, agent, tasks, finish, capacity) {
    const first = MARGIN_DAYS * DAY, last = human.length - first, days = (last - first) / DAY;
    let peakSum = 0, loadSum = 0, congested = 0, unserved = 0, humanSum = 0;
    for (let d = 0; d < days; d++) {
      let peak = 0;
      for (let t = first + d * DAY; t < first + (d + 1) * DAY; t++) {
        const offered = human[t] + agent[t], load = offered / capacity;
        peak = Math.max(peak, load); loadSum += load;
        if (load > THRESHOLD) congested++;
        unserved += human[t] * (1 - capacity / Math.max(offered, capacity)); humanSum += human[t];
      }
      peakSum += peak;
    }
    const peakLoad = peakSum / days;
    let count = 0, delay = 0, late = 0;
    tasks.forEach((task, i) => {
      if (task.arrival < first || task.arrival >= last || task.deadline > last + first) return;
      const done = Number.isNaN(finish[i]) ? last + first : finish[i];
      count++; delay += done - task.arrival;
      if (done > task.deadline) late++;
    });
    return {
      peak: peakLoad, peakToAverage: peakLoad / (loadSum / (days * DAY)),
      congested: congested / days, unserved: 100 * unserved / humanSum,
      delay: count ? delay / count : 0, violations: count ? 100 * late / count : 0,
    };
  }

  function runAll(options) {
    const p = { ...DEFAULTS, ...options }, capacity = root.NWL_DATA.capacityMbps;
    const world = buildWorld(p.trace), human = world.run.map((load) => load * capacity);
    const tasks = generateTasks(world, capacity, p, p.seed * 7919 + 1);
    const out = { params: p, capacity, tasks, human, world };
    const finishRun = (name, list, starts, broadcasts) => {
      const { agent, finish } = simulate(list, starts, human, capacity);
      out[name] = { agent, finish, starts, broadcasts, metrics: metricsOf(human, agent, list, finish, capacity) };
    };
    finishRun("a", [], new Int32Array(0), null);
    finishRun("b", tasks, Int32Array.from(tasks, (t) => t.arrival), null);
    for (const [name, randomised] of [["c", false], ["d", true]]) {
      const { starts, broadcasts } = runPolicy(tasks, world, capacity, p, randomised, p.seed * 104729 + 3);
      finishRun(name, tasks, starts, broadcasts);
    }
    return out;
  }

  function encodeBroadcast(message) {
    const bytes = [];
    const u16 = (v) => bytes.push((v >> 8) & 255, v & 255);
    const minute = message.slot * SLOT;
    bytes.push((minute >>> 24) & 255, (minute >>> 16) & 255, (minute >>> 8) & 255, minute & 255);
    u16(message.loads.length);
    message.loads.forEach((load, h) => {
      u16(message.slot + h); u16(Math.min(Math.round(load * 1000), 65535));
      bytes.push(Math.round(message.confidence[h] * 100));
    });
    return bytes;
  }

  root.NWL = { runAll, encodeBroadcast, lossAt, DEFAULTS, constants: { SLOT, HORIZON, DAY, RUN_DAYS, MARGIN_DAYS, THRESHOLD, TASK_MBPS } };
})(globalThis);
