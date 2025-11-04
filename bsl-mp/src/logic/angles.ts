import type { Landmarks, Vec3, AngleMap } from "./types";

// Compute angle at joint B between AB and CB
function angle(a: Vec3, b: Vec3, c: Vec3): number {
  const ab: Vec3 = [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const cb: Vec3 = [c[0]-b[0], c[1]-b[1], c[2]-b[2]];
  const dot = ab[0]*cb[0] + ab[1]*cb[1] + ab[2]*cb[2];
  const nab = Math.hypot(...ab);
  const ncb = Math.hypot(...cb);
  const cos = Math.min(1, Math.max(-1, dot / (nab * ncb + 1e-8)));
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * Define joints. You can refine these triplets per finger if you prefer different
 * anatomical references. These work well for coarse coaching.
 */
const J = {
  // index finger
  R_INDEX_MCP: [0, 5, 6],
  R_INDEX_PIP: [5, 6, 7],
  R_INDEX_DIP: [6, 7, 8],
  // middle
  R_MIDDLE_MCP: [0, 9, 10],
  R_MIDDLE_PIP: [9, 10, 11],
  R_MIDDLE_DIP: [10, 11, 12],
  // ring
  R_RING_MCP: [0, 13, 14],
  R_RING_PIP: [13, 14, 15],
  R_RING_DIP: [14, 15, 16],
  // pinky
  R_PINKY_MCP: [0, 17, 18],
  R_PINKY_PIP: [17, 18, 19],
  R_PINKY_DIP: [18, 19, 20],
  // thumb
  R_THUMB_MCP: [0, 2, 3],
  R_THUMB_IP: [2, 3, 4],
} as const;

export function computeAngles(lms: Landmarks): AngleMap {
  const out: AngleMap = {};
  const entries = Object.entries(J) as [keyof typeof J, number[]][];
  for (const [name, [ia, ib, ic]] of entries) {
    out[name] = angle(lms[ia], lms[ib], lms[ic]);
  }
  return out;
}
