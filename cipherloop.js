// CipherLoop inference in plain JavaScript (no dependencies, no server).
// A line-by-line port of cipherloop/model.py + the Hungarian decoding of cipherloop/evaluate.py.
// Checked against PyTorch with `node web/test/check.mjs`.

export const VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?-;:'\"";
const NL = 26; // letters
const SPACE = 26;
const APOS = VOCAB.indexOf("'");
const NF = 13; // per-symbol statistics features
export const MAX_MODEL_LEN = 1024; // RoPE table size used in training

// ------------------------------------------------------------------ tensor helpers (row-major Float32Array)

function linear(x, n, din, W, dout, b) {
  const y = new Float32Array(n * dout);
  for (let i = 0; i < n; i++) {
    const xo = i * din, yo = i * dout;
    for (let o = 0; o < dout; o++) {
      const wo = o * din;
      let s = 0;
      for (let k = 0; k < din; k++) s += x[xo + k] * W[wo + k];
      y[yo + o] = b ? s + b[o] : s;
    }
  }
  return y;
}

function addInto(a, b) {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
  return a;
}

function rmsnorm(x, n, d, w, eps = 1e-5) {
  for (let i = 0; i < n; i++) {
    const o = i * d;
    let ss = 0;
    for (let k = 0; k < d; k++) ss += x[o + k] * x[o + k];
    const r = 1 / Math.sqrt(ss / d + eps);
    for (let k = 0; k < d; k++) x[o + k] = x[o + k] * r * w[k];
  }
  return x;
}

const silu = (v) => v / (1 + Math.exp(-v));

function swiglu(x, n, d, p, hidden) {
  const a = linear(x, n, d, p.w1, hidden);
  const g = linear(x, n, d, p.w3, hidden);
  for (let i = 0; i < a.length; i++) a[i] = silu(a[i]) * g[i];
  return linear(a, n, hidden, p.w2, d);
}

function softmaxRows(x, rows, cols) {
  const y = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    const o = r * cols;
    let m = -Infinity;
    for (let c = 0; c < cols; c++) m = Math.max(m, x[o + c]);
    let z = 0;
    for (let c = 0; c < cols; c++) z += (y[o + c] = Math.exp(x[o + c] - m));
    for (let c = 0; c < cols; c++) y[o + c] /= z;
  }
  return y;
}

function ropeTable(n, hd) {
  // precompute_freqs_cis: float32 angles, as in PyTorch
  const half = hd / 2, cos = new Float32Array(n * half), sin = new Float32Array(n * half);
  for (let p = 0; p < half; p++) {
    const freq = Math.fround(1 / Math.fround(Math.pow(10000, Math.fround((2 * p) / hd))));
    for (let t = 0; t < n; t++) {
      const ang = Math.fround(t * freq);
      cos[t * half + p] = Math.cos(ang);
      sin[t * half + p] = Math.sin(ang);
    }
  }
  return { cos, sin };
}

// Multi-head attention (F.scaled_dot_product_attention semantics).
// bias(h, i, j) -> additive score bias, or -Infinity to mask.
function attention(x, n, d, H, p, rope, bias) {
  const hd = d / H, half = hd / 2, scale = 1 / Math.sqrt(hd), d3 = 3 * d;
  const qkv = linear(x, n, d, p.wqkv, d3);
  if (rope) {
    for (let i = 0; i < n; i++) {
      for (const base of [0, d]) { // q then k
        for (let h = 0; h < H; h++) {
          for (let q = 0; q < half; q++) {
            const idx = i * d3 + base + h * hd + 2 * q;
            const c = rope.cos[i * half + q], s = rope.sin[i * half + q];
            const x0 = qkv[idx], x1 = qkv[idx + 1];
            qkv[idx] = x0 * c - x1 * s;
            qkv[idx + 1] = x1 * c + x0 * s;
          }
        }
      }
    }
  }
  const out = new Float32Array(n * d), sc = new Float32Array(n);
  for (let h = 0; h < H; h++) {
    const ho = h * hd;
    for (let i = 0; i < n; i++) {
      const qo = i * d3 + ho;
      let m = -Infinity;
      for (let j = 0; j < n; j++) {
        const ko = j * d3 + d + ho;
        let s = 0;
        for (let e = 0; e < hd; e++) s += qkv[qo + e] * qkv[ko + e];
        s = s * scale + bias(h, i, j);
        sc[j] = s;
        if (s > m) m = s;
      }
      let z = 0;
      for (let j = 0; j < n; j++) z += (sc[j] = sc[j] === -Infinity ? 0 : Math.exp(sc[j] - m));
      const oo = i * d + ho;
      for (let j = 0; j < n; j++) {
        const pj = sc[j] / z;
        if (pj === 0) continue;
        const vo = j * d3 + 2 * d + ho;
        for (let e = 0; e < hd; e++) out[oo + e] += pj * qkv[vo + e];
      }
    }
  }
  return linear(out, n, d, p.wo, d);
}

// ------------------------------------------------------------------ text <-> ids

const NORMALIZE = { "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "." };

/** Map raw text to model ids: letters/punctuation in VOCAB keep their id, anything else becomes a space. */
export function toIds(text) {
  const ids = [];
  for (const ch of text.slice(0, MAX_MODEL_LEN)) {
    const u = (NORMALIZE[ch] || ch).toUpperCase();
    const k = u.length === 1 ? VOCAB.indexOf(u) : -1;
    ids.push(k >= 0 ? k : SPACE);
  }
  return ids;
}

/** Apply a cipher->plain letter key (array of 26 letter indices) to text, keeping case and non-letters. */
export function applyKey(text, key) {
  let out = "";
  for (const ch of text) {
    const up = ch.toUpperCase(), k = up.length === 1 ? VOCAB.indexOf(up) : -1;
    if (k >= 0 && k < NL) {
      const p = VOCAB[key[k]];
      out += ch === up ? p : p.toLowerCase();
    } else out += ch;
  }
  return out;
}

// ------------------------------------------------------------------ random IDs (seeded)

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(rand(), 1e-12), v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

// ------------------------------------------------------------------ Hungarian (rows <= cols, minimise cost)

export function hungarian(cost, n, m) {
  // e-maxx potentials algorithm; returns assignment row -> column
  const INF = Infinity, u = new Float64Array(n + 1), v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1), way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(INF), used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const ans = new Int32Array(n);
  for (let j = 1; j <= m; j++) if (p[j]) ans[p[j] - 1] = j - 1;
  return ans;
}

/** Bijective key from (26x26) logits: Hungarian on count-weighted log-probs over present symbols. */
export function decodeKey(logits, counts) {
  const lp = new Float32Array(NL * NL);
  for (let r = 0; r < NL; r++) {
    let m = -Infinity, z = 0;
    for (let c = 0; c < NL; c++) m = Math.max(m, logits[r * NL + c]);
    for (let c = 0; c < NL; c++) z += Math.exp(logits[r * NL + c] - m);
    for (let c = 0; c < NL; c++) lp[r * NL + c] = logits[r * NL + c] - m - Math.log(z);
  }
  const key = new Int32Array(NL);
  for (let r = 0; r < NL; r++) { // argmax for every row (absent symbols keep this)
    let best = 0;
    for (let c = 1; c < NL; c++) if (lp[r * NL + c] > lp[r * NL + best]) best = c;
    key[r] = best;
  }
  const present = [];
  for (let r = 0; r < NL; r++) if (counts[r] > 0) present.push(r);
  if (present.length) {
    const cost = new Float64Array(present.length * NL);
    present.forEach((r, i) => { for (let c = 0; c < NL; c++) cost[i * NL + c] = -lp[r * NL + c] * counts[r]; });
    const a = hungarian(cost, present.length, NL);
    present.forEach((r, i) => { key[r] = a[i]; });
  }
  return { key, logprobs: lp };
}

// ------------------------------------------------------------------ the model

export class CipherLoop {
  constructor(manifest, buffer) {
    if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error("big-endian platforms are not supported");
    this.manifest = manifest;
    this.cfg = manifest.config;
    const all = new Float32Array(buffer);
    const W = {};
    for (const t of manifest.tensors) W[t.name] = all.subarray(t.offset, t.offset + t.length);
    this.W = W;
    const g = (pfx) => ({ wqkv: W[pfx + ".wqkv.weight"], wo: W[pfx + ".wo.weight"] });
    const f = (pfx) => ({ w1: W[pfx + ".w1.weight"], w2: W[pfx + ".w2.weight"], w3: W[pfx + ".w3.weight"] });
    this.tokAttn = g("cell.tok_attn");
    this.symAttn = g("cell.sym_attn");
    this.tokFfn = f("cell.tok_ffn");
    this.symFfn = f("cell.sym_ffn");
    this.norms = [0, 1, 2, 3, 4, 5].map((i) => W[`cell.norms.${i}.weight`]);
    this.rope = ropeTable(MAX_MODEL_LEN, this.cfg.dim / this.cfg.n_head);
  }

  /** symbol_features(): cipher-invariant statistics per symbol, (26 x 13), plus counts. */
  static features(ids) {
    const n = ids.length, isL = ids.map((c) => c < NL);
    const wlen = new Int32Array(n);
    for (let i = 0; i < n; ) { // length of each letter run
      if (!isL[i]) { i++; continue; }
      let j = i;
      while (j < n && isL[j]) j++;
      for (let k = i; k < j; k++) wlen[k] = j - i;
      i = j;
    }
    const sums = new Float32Array(NL * 7), counts = new Float32Array(NL);
    for (let i = 0; i < n; i++) {
      if (!isL[i]) continue;
      const c = ids[i], prevL = i > 0 && isL[i - 1], nextL = i < n - 1 && isL[i + 1];
      const prev = i > 0 ? ids[i - 1] : -1, next = i < n - 1 ? ids[i + 1] : -1;
      const fl = [!prevL, !nextL, wlen[i] === 1, wlen[i] === 2, wlen[i] === 3,
        prev === c || next === c, prev === APOS || next === APOS];
      for (let k = 0; k < 7; k++) sums[c * 7 + k] += fl[k] ? 1 : 0;
      counts[c] += 1;
    }
    let nLetters = 0, nDistinct = 0;
    for (let c = 0; c < NL; c++) { nLetters += counts[c]; nDistinct += counts[c] > 0 ? 1 : 0; }
    const nL = Math.max(nLetters, 1), log300 = Math.log(301);
    const feats = new Float32Array(NL * NF);
    for (let c = 0; c < NL; c++) {
      const o = c * NF, cnt = Math.max(counts[c], 1);
      for (let k = 0; k < 7; k++) feats[o + k] = sums[c * 7 + k] / cnt;
      let rank = 0;
      for (let c2 = 0; c2 < NL; c2++) if (counts[c2] > counts[c]) rank++;
      feats[o + 7] = Math.log1p(counts[c]) / log300;
      feats[o + 8] = counts[c] / nL;
      feats[o + 9] = rank / NL;
      feats[o + 10] = Math.log1p(nL) / log300;
      feats[o + 11] = nDistinct / NL;
      feats[o + 12] = counts[c] > 0 ? 1 : 0;
    }
    return { feats, counts };
  }

  /** Cell.forward: token block -> pool -> symbol block (post-norm). */
  cell(h, s, xt, xs, ctx) {
    const { n, slot, isL, present, ids } = ctx, d = this.cfg.dim, H = this.cfg.n_head, W = this.W, nm = this.norms;
    const s2t = linear(s, NL, d, W["cell.s2t.weight"], d);
    const h1 = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < d; k++) {
        h1[i * d + k] = h[i * d + k] + xt[i * d + k] + (isL[i] ? s2t[slot[i] * d + k] : 0);
      }
    }
    rmsnorm(h1, n, d, nm[0]);
    const eq = W["cell.eq_w"];
    const tokBias = (hh, i, j) => (isL[i] && isL[j] && ids[i] === ids[j] ? eq[hh] : 0);
    rmsnorm(addInto(h1, attention(h1, n, d, H, this.tokAttn, this.rope, tokBias)), n, d, nm[1]);
    rmsnorm(addInto(h1, swiglu(h1, n, d, this.tokFfn, this.cfg.ffn_tok)), n, d, nm[2]);

    const pooled = new Float32Array(NL * d), cnt = new Float32Array(NL);
    for (let i = 0; i < n; i++) {
      if (!isL[i]) continue;
      cnt[slot[i]] += 1;
      for (let k = 0; k < d; k++) pooled[slot[i] * d + k] += h1[i * d + k];
    }
    for (let c = 0; c < NL; c++) { const z = Math.max(cnt[c], 1); for (let k = 0; k < d; k++) pooled[c * d + k] /= z; }
    const s1 = addInto(addInto(Float32Array.from(s), xs), linear(pooled, NL, d, W["cell.t2s.weight"], d));
    rmsnorm(s1, NL, d, nm[3]);
    const symMask = (hh, i, j) => (present[j] ? 0 : -Infinity);
    rmsnorm(addInto(s1, attention(s1, NL, d, H, this.symAttn, null, symMask)), NL, d, nm[4]);
    rmsnorm(addInto(s1, swiglu(s1, NL, d, this.symFfn, this.cfg.ffn_sym)), NL, d, nm[5]);
    return [h1, s1];
  }

  /**
   * Solve a ciphertext. Returns { text, key, counts, steps: [{key, text}] }.
   * opts: nOuter (default from config), seed (random-ID seed), randIds (Float32Array 26*id_dim, overrides seed),
   *       onStep(stepIndex, {key, text, logits}) awaited after every outer step.
   */
  async solve(text, opts = {}) {
    const cfg = this.cfg, d = cfg.dim, W = this.W;
    const ids = toIds(text), n = ids.length;
    const isL = ids.map((c) => c < NL), slot = ids.map((c) => Math.min(c, NL - 1));
    const { feats, counts } = CipherLoop.features(ids);
    const present = Array.from(counts, (c) => c > 0);
    if (!present.some(Boolean)) throw new Error("no letters to decipher");
    const ctx = { n, ids, isL, slot, present };

    // random symbol IDs, unit-normalised (fresh per pass, seeded for reproducibility)
    let rid = opts.randIds ? Float32Array.from(opts.randIds) : gaussian(mulberry32(opts.seed ?? 0), NL * cfg.id_dim);
    for (let c = 0; c < NL; c++) {
      let ss = 0;
      for (let k = 0; k < cfg.id_dim; k++) ss += rid[c * cfg.id_dim + k] ** 2;
      const z = Math.max(Math.sqrt(ss), 1e-6);
      for (let k = 0; k < cfg.id_dim; k++) rid[c * cfg.id_dim + k] /= z;
    }

    // _embed(): x_sym (26 x d), x_tok (n x d)
    const xSym = addInto(linear(rid, NL, cfg.id_dim, W["id_proj.weight"], d),
      linear(linear(feats, NL, NF, W["stat_mlp.0.weight"], d, W["stat_mlp.0.bias"]).map(silu), NL, d, W["stat_mlp.2.weight"], d, W["stat_mlp.2.bias"]));
    for (let c = 0; c < NL; c++) for (let k = 0; k < d; k++) xSym[c * d + k] = present[c] ? xSym[c * d + k] + W.sym_base[k] : 0;
    const xTok = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      const cls = isL[i] ? 0 : ids[i] - (NL - 1);
      for (let k = 0; k < d; k++) xTok[i * d + k] = W["tok_embed.weight"][cls * d + k] + (isL[i] ? xSym[slot[i] * d + k] : 0);
    }

    let h = Float32Array.from(xTok), s = Float32Array.from(xSym), probs = null, last = null;
    const steps = [];
    for (let t = 0; t < (opts.nOuter || cfg.n_outer); t++) {
      let xt = xTok, xs = xSym;
      if (probs) { // hypothesis feedback: softmax(logits) @ plain_embed
        const y = new Float32Array(NL * d), E = W["plain_embed.weight"];
        for (let c = 0; c < NL; c++) for (let l = 0; l < NL; l++) {
          const pr = probs[c * NL + l];
          for (let k = 0; k < d; k++) y[c * d + k] += pr * E[l * d + k];
        }
        xs = addInto(Float32Array.from(xSym), y);
        xt = Float32Array.from(xTok);
        for (let i = 0; i < n; i++) if (isL[i]) for (let k = 0; k < d; k++) xt[i * d + k] += y[slot[i] * d + k];
      }
      for (let r = 0; r < cfg.n_inner; r++) [h, s] = this.cell(h, s, xt, xs, ctx);
      const logits = linear(s, NL, d, W["head.weight"], NL, W["head.bias"]);
      probs = softmaxRows(logits, NL, NL);
      const { key } = decodeKey(logits, counts);
      last = { key, text: applyKey(text, key), logits };
      steps.push({ key, text: last.text });
      if (opts.onStep) await opts.onStep(t, last);
    }
    return { text: last.text, key: last.key, counts, steps, logits: last.logits };
  }
}

// ------------------------------------------------------------------ loading (Cache Storage = the model lives in the browser cache)

const CACHE = "cipherloop-model";

async function sha256Hex(buf) {
  if (!globalThis.crypto?.subtle) return null; // insecure context: skip the check
  const h = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(h), (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Load the model, preferring the browser cache. The manifest is revalidated online (so a new model
 * version is picked up) and served from cache offline; weights are cached per version.
 * Returns { model, manifest, source: "cache" | "network" }.
 */
export async function loadModel(base = "model/") {
  const cache = globalThis.caches ? await caches.open(CACHE).catch(() => null) : null;
  const manifestUrl = new URL(base + "manifest.json", location.href).href;
  let manifest;
  try {
    const r = await fetch(manifestUrl, { cache: "no-cache" });
    if (!r.ok) throw new Error(`manifest: HTTP ${r.status}`);
    if (cache) await cache.put(manifestUrl, r.clone());
    manifest = await r.json();
  } catch (e) {
    const hit = cache && (await cache.match(manifestUrl));
    if (!hit) throw e;
    manifest = await hit.json(); // offline
  }
  const weightsUrl = new URL(`${base}weights.bin?v=${manifest.version}`, location.href).href;
  let buf = null, source = "cache";
  const hit = cache && (await cache.match(weightsUrl));
  if (hit) {
    buf = await hit.arrayBuffer();
    const sha = await sha256Hex(buf);
    if (sha && sha !== manifest.sha256) { buf = null; await cache.delete(weightsUrl); }
  }
  if (!buf) {
    source = "network";
    const r = await fetch(weightsUrl);
    if (!r.ok) throw new Error(`weights: HTTP ${r.status}`);
    buf = await r.clone().arrayBuffer();
    const sha = await sha256Hex(buf);
    if (sha && sha !== manifest.sha256) throw new Error("weights checksum mismatch");
    if (cache) {
      await cache.put(weightsUrl, r);
      for (const req of await cache.keys()) { // drop older model versions
        if (req.url.includes("weights.bin") && req.url !== weightsUrl) await cache.delete(req);
      }
    }
  }
  return { model: new CipherLoop(manifest, buf), manifest, source };
}

export async function clearCachedModel() {
  if (globalThis.caches) await caches.delete(CACHE);
}
