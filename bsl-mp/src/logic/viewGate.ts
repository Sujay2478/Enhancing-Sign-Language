import type { Landmarks } from "./types";

export type ViewGate = { ok: boolean; advice?: string };

export function viewGate(
  pxLms: Landmarks,
  video: HTMLVideoElement
): ViewGate {
  if (!pxLms?.length) return { ok: false, advice: "Show your hand to the camera" };
  // Distance proxy: bbox height
  const ys = pxLms.map((p) => p[1]);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const bboxH = maxY - minY;
  if (bboxH < video.videoHeight * 0.15) {
    return { ok: false, advice: "Move closer to the camera" };
  }
  // Angle proxy: palm plane (thumb CMC ~1, pinky MCP ~17, wrist 0)
  const n = palmNormal(pxLms[1], pxLms[17], pxLms[0]);
  const facing = Math.abs(n[2]); // Z component ~ facing camera
  if (facing < 0.5) return { ok: false, advice: "Rotate your palm towards the camera" };

  // Lighting proxy: quick luma estimate via bbox area (rough)
  // (Optional) Keep simple for MVP — can be added later

  return { ok: true };
}

function palmNormal(a: number[], b: number[], c: number[]) {
  const u = [a[0] - c[0], a[1] - c[1], a[2] - c[2]];
  const v = [b[0] - c[0], b[1] - c[1], b[2] - c[2]];
  const x = u[1] * v[2] - u[2] * v[1];
  const y = u[2] * v[0] - u[0] * v[2];
  const z = u[0] * v[1] - u[1] * v[0];
  const n = Math.hypot(x, y, z) || 1;
  return [x / n, y / n, z / n];
}
