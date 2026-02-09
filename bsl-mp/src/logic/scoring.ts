import type { AngleMap } from "./types";

// Weighted mean absolute error → 0..100 score
export function poseScore(
  current: AngleMap,
  target: AngleMap,
  weights: Record<string, number> = { INDEX: 1.2, THUMB: 1.0, OTHERS: 0.8 },
  toleranceDeg = 12
): { score: number; perJoint: Record<string, number> } {
  let totW = 0, acc = 0;
  const perJoint: Record<string, number> = {};
  for (const j of Object.keys(target)) {
    const group = groupFromJoint(j); // e.g., "INDEX", "THUMB"
    const w = weights[group] ?? weights["OTHERS"] ?? 1;
    const err = Math.abs((current[j] ?? 0) - target[j]);
    const js = Math.max(0, 1 - err / toleranceDeg); // 0..1
    perJoint[j] = js;
    totW += w; acc += w * js;
  }
  const s = totW ? Math.round((acc / totW) * 100) : 0;
  return { score: s, perJoint };
}

function groupFromJoint(j: string) {
  if (j.includes("INDEX")) return "INDEX";
  if (j.includes("THUMB")) return "THUMB";
  if (j.includes("MIDDLE")) return "MIDDLE";
  if (j.includes("RING")) return "RING";
  if (j.includes("PINKY")) return "PINKY";
  return "OTHERS";
}

function zNormalizeSeq(seq: number[][]): number[][] {
  const d = seq[0]?.length ?? 0;
  if (!d) return seq;

  const mean = Array(d).fill(0);
  const std = Array(d).fill(0);

  for (const row of seq) for (let i = 0; i < d; i++) mean[i] += row[i];
  for (let i = 0; i < d; i++) mean[i] /= seq.length;

  for (const row of seq) for (let i = 0; i < d; i++) std[i] += (row[i] - mean[i]) ** 2;
  for (let i = 0; i < d; i++) std[i] = Math.sqrt(std[i] / seq.length) || 1;

  return seq.map((row) => row.map((v, i) => (v - mean[i]) / std[i]));
}

// === Simple DTW for dynamic signs over feature vectors ===
export function dtwCost(seqA: number[][], seqB: number[][]): number {
  const A = zNormalizeSeq(seqA);
  const B = zNormalizeSeq(seqB);

  const n = A.length, m = B.length;
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(Infinity));

  const dist = (a: number[], b: number[]) =>
    Math.sqrt(a.reduce((s, ai, i) => s + (ai - b[i]) ** 2, 0));

  dp[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = dist(A[i - 1], B[j - 1]);
      dp[i][j] = cost + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }

  return dp[n][m] / (n + m);
}

export function dtwScore(cost: number, good = 0.15, bad = 0.45) {
  // Map cost to 0..100 (lower cost better)
  if (cost <= good) return 100;
  if (cost >= bad) return 0;
  return Math.round(100 * (1 - (cost - good) / (bad - good)));
}
