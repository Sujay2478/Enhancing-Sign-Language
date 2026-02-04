import type { Landmarks } from "./types";

const KEY_IDXS = [0, 5, 9, 13, 17]; // wrist + MCPs (stable & informative)

export function motionEnergy(prev: Landmarks, curr: Landmarks): number {
  if (!prev?.length || !curr?.length) return 0;

  let acc = 0;
  for (const idx of KEY_IDXS) {
    const [x0, y0, z0] = prev[idx];
    const [x1, y1, z1] = curr[idx];
    const dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
    acc += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return acc / KEY_IDXS.length; // average movement per frame
}