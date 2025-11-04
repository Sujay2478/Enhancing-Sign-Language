// Live hand landmarks via MediaPipe Tasks Vision
import {
  FilesetResolver,
  HandLandmarker
} from "@mediapipe/tasks-vision";


export type HandResult = {
  // Each landmark is [x,y,z] normalized to video size by MediaPipe (0..1)
  landmarks: [number, number, number][];
  handedness: "Left" | "Right";
  score: number; // confidence
};
export type FrameResult = { hands: HandResult[]; videoEl: HTMLVideoElement };

type Props = {
  videoEl: HTMLVideoElement;
  onResults: (r: FrameResult) => void;
  maxHands?: number;
};

export class HandTracker {
  private landmarker?: HandLandmarker;
  private rafId = 0;
  private running = false;
  private video!: HTMLVideoElement;
  private onResults!: (r: FrameResult) => void;
  private lastTs = -1;

  async init(p: Props) {
    this.video = p.videoEl;
    this.onResults = p.onResults;
    const vision = await FilesetResolver.forVisionTasks("/wasm");
    this.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "/models/hand_landmarker.task" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
  }

  start() {
    if (this.running) return;
    this.running = true;
    const loop = () => {
      if (!this.running || !this.landmarker) return;
      const now = performance.now();
      if (this.lastTs === now) {
        this.rafId = requestAnimationFrame(loop);
        return;
      }
      this.lastTs = now;
      const res = this.landmarker.detectForVideo(this.video, now);
      const hands: HandResult[] = [];
      if (res.landmarks) {
        const hd = res.handednesses ?? [];
        const conf = res.handednesses?.map((h) => h[0].score ?? 0) ?? [];
        for (let i = 0; i < res.landmarks.length; i++) {
          const lm = res.landmarks[i];
          hands.push({
            landmarks: lm.map((p) => [p.x, p.y, p.z]) as any,
            handedness: (hd[i]?.[0]?.categoryName ?? "Right") as "Left" | "Right",
            score: conf[i] ?? 0,
          });
        }
      }
      this.onResults({ hands, videoEl: this.video });
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }
}
