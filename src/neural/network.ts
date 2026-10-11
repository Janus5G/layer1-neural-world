// Small, hand-written neural network with an Adam optimizer.
// No external ML dependencies. All arrays use Float32Array.

export function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function xavier(fanIn: number, fanOut: number): number {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return (Math.random() * 2 - 1) * limit;
}

export class Linear {
  W: Float32Array; b: Float32Array;
  mW: Float32Array; vW: Float32Array;
  mb: Float32Array; vb: Float32Array;
  gW: Float32Array; gb: Float32Array;
  x: Float32Array | null = null;
  out: Float32Array;

  constructor(public inDim: number, public outDim: number) {
    this.W = new Float32Array(outDim * inDim);
    this.b = new Float32Array(outDim);
    this.mW = new Float32Array(outDim * inDim);
    this.vW = new Float32Array(outDim * inDim);
    this.mb = new Float32Array(outDim);
    this.vb = new Float32Array(outDim);
    this.gW = new Float32Array(outDim * inDim);
    this.gb = new Float32Array(outDim);
    this.out = new Float32Array(outDim);
    for (let i = 0; i < this.W.length; i++) this.W[i] = xavier(inDim, outDim);
  }

  forward(x: Float32Array): Float32Array {
    this.x = x;
    for (let o = 0; o < this.outDim; o++) {
      let s = this.b[o];
      const base = o * this.inDim;
      for (let i = 0; i < this.inDim; i++) s += this.W[base + i] * x[i];
      this.out[o] = s;
    }
    return this.out;
  }

  backward(dOut: Float32Array): Float32Array {
    const dX = new Float32Array(this.inDim);
    this.gW.fill(0);
    this.gb.fill(0);
    for (let o = 0; o < this.outDim; o++) {
      const go = dOut[o];
      this.gb[o] += go;
      const base = o * this.inDim;
      for (let i = 0; i < this.inDim; i++) {
        this.gW[base + i] += go * this.x![i];
        dX[i] += go * this.W[base + i];
      }
    }
    return dX;
  }

  adam(lr: number, t: number, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    for (let i = 0; i < this.W.length; i++) {
      const g = this.gW[i];
      this.mW[i] = b1 * this.mW[i] + (1 - b1) * g;
      this.vW[i] = b2 * this.vW[i] + (1 - b2) * g * g;
      this.W[i] -= lr * (this.mW[i] / bc1) / (Math.sqrt(this.vW[i] / bc2) + eps);
    }
    for (let i = 0; i < this.b.length; i++) {
      const g = this.gb[i];
      this.mb[i] = b1 * this.mb[i] + (1 - b1) * g;
      this.vb[i] = b2 * this.vb[i] + (1 - b2) * g * g;
      this.b[i] -= lr * (this.mb[i] / bc1) / (Math.sqrt(this.vb[i] / bc2) + eps);
    }
  }

  clone(): Linear {
    const l = new Linear(this.inDim, this.outDim);
    l.W.set(this.W); l.b.set(this.b);
    return l;
  }

  inheritFrom(a: Linear, b: Linear, mutationRate: number, mutationScale = 0.05) {
    for (let i = 0; i < this.W.length; i++) {
      this.W[i] = Math.random() < 0.5 ? a.W[i] : b.W[i];
      if (Math.random() < mutationRate) this.W[i] += randn() * mutationScale;
    }
    for (let i = 0; i < this.b.length; i++) {
      this.b[i] = Math.random() < 0.5 ? a.b[i] : b.b[i];
      if (Math.random() < mutationRate) this.b[i] += randn() * mutationScale;
    }
  }
}

export interface Transition {
  state: Float32Array;
  action: number;
  logProb: number;
  value: number;
  reward: number;
  done: boolean;
}

export interface BrainTrace {
  input: Float32Array;
  h1: Float32Array;
  h2: Float32Array;
  logits: Float32Array;
  probs: Float32Array;
  value: number;
}

export interface BrainSnapshot {
  params: number;
  policyEntropy: number;
  lastLoss: number;
  lastValueLoss: number;
  updates: number;
}

export class Brain {
  l1: Linear;
  l2: Linear;
  policyHead: Linear;
  valueHead: Linear;

  h1: Float32Array;
  h1Relu: Float32Array;
  h2: Float32Array;
  h2Relu: Float32Array;
  logits: Float32Array;
  probs: Float32Array;
  value = 0;

  buffer: Transition[] = [];
  adamT = 0;
  updates = 0;
  lastLoss = 0;
  lastValueLoss = 0;
  lastEntropy = 0;

  constructor(public inputDim: number, public hiddenDim: number, public numActions: number) {
    this.l1 = new Linear(inputDim, hiddenDim);
    this.l2 = new Linear(hiddenDim, hiddenDim);
    this.policyHead = new Linear(hiddenDim, numActions);
    this.valueHead = new Linear(hiddenDim, 1);
    this.h1 = new Float32Array(hiddenDim);
    this.h1Relu = new Float32Array(hiddenDim);
    this.h2 = new Float32Array(hiddenDim);
    this.h2Relu = new Float32Array(hiddenDim);
    this.logits = new Float32Array(numActions);
    this.probs = new Float32Array(numActions);
  }

  forward(x: Float32Array): { probs: Float32Array; value: number } {
    this.h1.set(this.l1.forward(x));
    for (let i = 0; i < this.hiddenDim; i++) this.h1Relu[i] = Math.max(0, this.h1[i]);
    this.h2.set(this.l2.forward(this.h1Relu));
    for (let i = 0; i < this.hiddenDim; i++) this.h2Relu[i] = Math.max(0, this.h2[i]);
    const logits = this.policyHead.forward(this.h2Relu);
    this.logits.set(logits);

    let maxL = -Infinity;
    for (let i = 0; i < this.numActions; i++) if (this.logits[i] > maxL) maxL = this.logits[i];
    let sum = 0;
    for (let i = 0; i < this.numActions; i++) {
      this.probs[i] = Math.exp(this.logits[i] - maxL);
      sum += this.probs[i];
    }
    for (let i = 0; i < this.numActions; i++) this.probs[i] /= sum;

    const v = this.valueHead.forward(this.h2Relu);
    this.value = v[0];
    return { probs: this.probs, value: this.value };
  }

  forwardWithTrace(x: Float32Array): BrainTrace {
    const { probs, value } = this.forward(x);
    return {
      input: new Float32Array(x),
      h1: new Float32Array(this.h1Relu),
      h2: new Float32Array(this.h2Relu),
      logits: new Float32Array(this.logits),
      probs: new Float32Array(probs),
      value,
    };
  }

  sampleAction(): { action: number; logProb: number; value: number } {
    let r = Math.random();
    let action = 0;
    for (let i = 0; i < this.numActions; i++) {
      r -= this.probs[i];
      if (r <= 0) { action = i; break; }
      action = i;
    }
    const logProb = Math.log(this.probs[action] + 1e-9);
    return { action, logProb, value: this.value };
  }

  // A2C update: REINFORCE with value baseline + entropy bonus + GAE.
  train(gamma = 0.99, lambda = 0.95, lr = 3e-4, valueCoef = 0.5, entropyCoef = 0.01) {
    if (this.buffer.length < 2) return;

    const n = this.buffer.length;
    const returns = new Float32Array(n);
    const advantages = new Float32Array(n);
    let lastAdv = 0;
    let nextValue = 0;
    for (let t = n - 1; t >= 0; t--) {
      const tr = this.buffer[t];
      const mask = tr.done ? 0 : 1;
      const delta = tr.reward + gamma * nextValue * mask - tr.value;
      lastAdv = delta + gamma * lambda * mask * lastAdv;
      advantages[t] = lastAdv;
      returns[t] = tr.value + lastAdv;
      nextValue = tr.value;
    }

    let mean = 0;
    for (let i = 0; i < n; i++) mean += advantages[i];
    mean /= n;
    let varSum = 0;
    for (let i = 0; i < n; i++) varSum += (advantages[i] - mean) ** 2;
    const std = Math.sqrt(varSum / n) + 1e-6;
    for (let i = 0; i < n; i++) advantages[i] = (advantages[i] - mean) / std;

    this.adamT++;
    let totalPolicyLoss = 0;
    let totalValueLoss = 0;
    let totalEntropy = 0;

    for (let t = 0; t < n; t++) {
      const tr = this.buffer[t];
      this.forward(tr.state);

      const dLogits = new Float32Array(this.numActions);
      for (let i = 0; i < this.numActions; i++) dLogits[i] = this.probs[i];
      dLogits[tr.action] -= 1;
      const adv = advantages[t];
      for (let i = 0; i < this.numActions; i++) dLogits[i] *= adv;

      let H = 0;
      for (let i = 0; i < this.numActions; i++) H -= this.probs[i] * Math.log(this.probs[i] + 1e-9);
      for (let i = 0; i < this.numActions; i++) {
        const dH = -this.probs[i] * (Math.log(this.probs[i] + 1e-9) + H);
        dLogits[i] -= entropyCoef * dH;
      }

      const dValue = new Float32Array([(this.value - returns[t]) * valueCoef]);

      const dH2FromPolicy = this.policyHead.backward(dLogits);
      const dH2FromValue = this.valueHead.backward(dValue);
      const dH2 = new Float32Array(this.hiddenDim);
      for (let i = 0; i < this.hiddenDim; i++) dH2[i] = dH2FromPolicy[i] + dH2FromValue[i];
      for (let i = 0; i < this.hiddenDim; i++) if (this.h2[i] <= 0) dH2[i] = 0;

      const dH1 = this.l2.backward(dH2);
      for (let i = 0; i < this.hiddenDim; i++) if (this.h1[i] <= 0) dH1[i] = 0;
      this.l1.backward(dH1);

      this.l1.adam(lr, this.adamT);
      this.l2.adam(lr, this.adamT);
      this.policyHead.adam(lr, this.adamT);
      this.valueHead.adam(lr, this.adamT);

      totalPolicyLoss += -Math.log(this.probs[tr.action] + 1e-9) * advantages[t];
      totalValueLoss += (returns[t] - this.value) ** 2;
      totalEntropy += H;
    }

    this.lastLoss = totalPolicyLoss / n;
    this.lastValueLoss = totalValueLoss / n;
    this.lastEntropy = totalEntropy / n;
    this.updates++;
    this.buffer.length = 0;
  }

  snapshot(): BrainSnapshot {
    const p =
      this.l1.W.length + this.l1.b.length +
      this.l2.W.length + this.l2.b.length +
      this.policyHead.W.length + this.policyHead.b.length +
      this.valueHead.W.length + this.valueHead.b.length;
    return {
      params: p,
      policyEntropy: this.lastEntropy,
      lastLoss: this.lastLoss,
      lastValueLoss: this.lastValueLoss,
      updates: this.updates,
    };
  }
}