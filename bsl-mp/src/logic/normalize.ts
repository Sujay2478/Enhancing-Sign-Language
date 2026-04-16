import type { Vec3, Landmarks } from "./types";
import normStats from "../data/bsl_norm.json";

export function toPixels(lms: Landmarks, video: HTMLVideoElement): Landmarks {
  const w = video.videoWidth;
  const h = video.videoHeight;
  return lms.map(([x, y, z]) => [x * w, y * h, z * Math.max(w, h)]);
}

export function normalize(lms: Landmarks, opts: { mirror?: boolean } = {}): Landmarks {
  const w0 = lms[0];
  const midMCP = lms[9];
  const refLen =
    Math.hypot(midMCP[0] - w0[0], midMCP[1] - w0[1], midMCP[2] - w0[2]) || 1;

  const out = lms.map(([x, y, z]) => [
    (x - w0[0]) / refLen,
    (y - w0[1]) / refLen,
    (z - w0[2]) / refLen,
  ]) as Landmarks;

  if (opts.mirror) {
    for (const p of out) p[0] = -p[0];
  }
  return out;
}

export function zerosHand(): number[] {
  return new Array(63).fill(0);
}

export function flattenLandmarks(norm: Landmarks): number[] {
  return norm.flatMap((p) => [p[0], p[1], p[2]]);
}

export function standardize(features: number[]): number[] {
  const mean = (normStats as any).mean as number[];
  const std = (normStats as any).std as number[];

  if (!mean || !std || mean.length !== features.length || std.length !== features.length) {
    console.warn(
      `⚠️ standardize(): stats length mismatch. features=${features.length}, mean=${mean?.length}, std=${std?.length}`
    );
    return features;
  }

  return features.map((x, i) => {
    const s = std[i] || 1;
    return (x - mean[i]) / s;
  });
}

/**
 * Normalize each hand independently (wrist origin + wrist->middle_mcp scale),
 * then concatenate in stable order [Left, Right].
 *
 * If a hand is missing, it is zero-padded so output is always 126 floats.
 */
export function normalizeTwoHands(
  leftPx: Landmarks | null,
  rightPx: Landmarks | null,
  opts: { mirror?: boolean } = {}
): number[] {
  const leftNorm = leftPx ? flattenLandmarks(normalize(leftPx, opts)) : zerosHand();
  const rightNorm = rightPx ? flattenLandmarks(normalize(rightPx, opts)) : zerosHand();
  const features = [...leftNorm, ...rightNorm];
  return standardize(features);
}

export type { Vec3, Landmarks } from "./types";
