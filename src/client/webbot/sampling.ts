/**
 * Categorical + Beta sampling matching rl/policy.py's Policy.act(). No
 * seeded PRNG - live play doesn't need reproducibility, just Math.random().
 */

/** Softmax + sample (or argmax if greedy) over raw logits. Returns the
 * chosen index. -1e9-masked entries (baked into the ONNX head outputs,
 * mirroring MASKED_NEG in rl/policy.py) end up with ~0 probability. */
export function sampleCategorical(logits: Float32Array, greedy: boolean): number {
  const n = logits.length;
  let maxLogit = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > maxLogit) maxLogit = logits[i];
  if (greedy) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const exps = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    exps[i] = e;
    sum += e;
  }
  let r = Math.random() * sum;
  for (let i = 0; i < n; i++) {
    r -= exps[i];
    if (r <= 0) return i;
  }
  return n - 1;
}

/** Standard gamma(shape) sample via Marsaglia-Tsang (shape >= 1 required;
 * alpha/beta from quantityDist are always >= 1 thanks to 1+softplus). */
function sampleGamma(shape: number): number {
  const d = shape - 1.0 / 3.0;
  const c = 1.0 / Math.sqrt(9.0 * d);
  for (;;) {
    let x: number, v: number;
    do {
      // Box-Muller for a standard normal sample.
      const u1 = Math.random() || 1e-12;
      const u2 = Math.random();
      x = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      v = 1.0 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** (alpha, beta) -> Beta sample via two Gamma draws, matching
 * torch.distributions.Beta used for the quantity head. */
export function sampleBeta(alpha: number, beta: number, greedy: boolean): number {
  if (greedy) return alpha / (alpha + beta);
  const x = sampleGamma(alpha);
  const y = sampleGamma(beta);
  return x / (x + y);
}

function softplus(x: number): number {
  // Numerically stable: log(1+e^x), avoiding overflow for large x.
  return x > 20 ? x : Math.log1p(Math.exp(x));
}

/** Raw (alpha_logit, beta_logit) head output -> (alpha, beta) params,
 * matching Policy.quantity_dist's `1 + softplus(...)` parameterization. */
export function betaParams(raw: Float32Array): [number, number] {
  return [1.0 + softplus(raw[0]), 1.0 + softplus(raw[1])];
}
