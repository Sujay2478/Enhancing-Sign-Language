import { useEffect, useRef } from "react";
import type { Landmarks } from "../logic/types";

function drawSkeleton(ctx: CanvasRenderingContext2D, px: Landmarks) {
  const lines: [number, number][][] = [
    [[0,1],[1,2],[2,3],[3,4]],       // thumb-ish
    [[0,5],[5,6],[6,7],[7,8]],       // index
    [[0,9],[9,10],[10,11],[11,12]],  // middle
    [[0,13],[13,14],[14,15],[15,16]],
    [[0,17],[17,18],[18,19],[19,20]],
  ];
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#00FFFF";
  for (const chain of lines) {
    ctx.beginPath();
    ctx.moveTo(px[chain[0][0]][0], px[chain[0][0]][1]);
    for (const [, j] of chain) ctx.lineTo(px[j][0], px[j][1]);
    ctx.stroke();
  }
  ctx.fillStyle = "#00FFFF";
  for (const [x,y] of px) {
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI*2); ctx.fill();
  }
}

export default function CanvasOverlay({
  width, height, handsPx, ghostPx
}: { width: number; height: number; handsPx: Landmarks[]; ghostPx?: Landmarks }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current!;
    c.width = width; c.height = height;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0,0,width,height);
    if (ghostPx) {
      ctx.globalAlpha = 0.25; ctx.strokeStyle = "#00FF00"; ctx.fillStyle = "#00FF00";
      drawSkeleton(ctx, ghostPx); ctx.globalAlpha = 1;
    }
    for (const h of handsPx) drawSkeleton(ctx, h);
  }, [handsPx, ghostPx, width, height]);
  return (
    <canvas
      ref={ref}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", transform: "scaleX(-1)" }}
    />
  );
}
