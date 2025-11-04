import type { Vec3, Landmarks } from "./types";

export function toPixels(
  lms: Landmarks,
  video: HTMLVideoElement
): Landmarks {
  const w = video.videoWidth;
  const h = video.videoHeight;
  return lms.map(([x, y, z]) => [x * w, y * h, z * Math.max(w, h)]);
}

export function normalize(
  lms: Landmarks,
  opts: { mirror?: boolean } = {}
): Landmarks {
  // translate so wrist (0) at origin; scale by wrist->middle_mcp length
  const w0 = lms[0];
  const midMCP = lms[9]; // middle finger MCP index=9
  const refLen =
    Math.hypot(midMCP[0] - w0[0], midMCP[1] - w0[1], midMCP[2] - w0[2]) || 1;
  const out = lms.map(([x, y, z]) => [(x - w0[0]) / refLen, (y - w0[1]) / refLen, (z - w0[2]) / refLen]) as Landmarks;
  if (opts.mirror) {
    for (const p of out) p[0] = -p[0];
  }
  return out;
}

export type { Vec3, Landmarks } from "./types";
