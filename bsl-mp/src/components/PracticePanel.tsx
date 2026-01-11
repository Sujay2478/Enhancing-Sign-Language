import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import CameraFeed from "./CameraFeed";
import CanvasOverlay from "./CanvasOverlay";
import { HandTracker } from "../mediapipe/handTracker";
import { toPixels, normalize } from "../logic/normalize";
import { computeAngles } from "../logic/angles";
import { poseScore, dtwCost, dtwScore } from "../logic/scoring";
import { viewGate } from "../logic/viewGate";
import signs from "../data/bsl_signs.json";
import labels from "../data/bsl_labels.json";
import normStats from "../data/bsl_norm.json"; // mean/std from training
import type { Landmarks } from "../logic/types";

interface TargetPose {
  angles: Record<string, number>;
  toleranceDefault?: number;
}
interface SignType {
  id: string;
  name: string;
  type: "static" | "dynamic";
  hands?: "one" | "two";
  dominant?: "left" | "right";
  weights?: Record<string, number>;
  targetPose?: TargetPose;
  template?: { sequence: number[][]; length?: number };
  tolerance?: { dtw?: number };
}

type NormStats = { mean: number[]; std: number[] };

function safeNumberDim(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = Number.parseInt(v, 10);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function zeros(n: number): number[] {
  return new Array(n).fill(0);
}

// Flatten Landmarks -> 63 floats
function flatten63(lms: Landmarks): number[] {
  const out = new Array(63);
  let k = 0;
  for (let i = 0; i < lms.length; i++) {
    const [x, y, z] = lms[i];
    out[k++] = x;
    out[k++] = y;
    out[k++] = z;
  }
  return out;
}

// Mirror raw MediaPipe normalized landmarks (x in [0..1]) if user toggles mirror view
function maybeMirrorRaw(lms: Landmarks, mirror: boolean): Landmarks {
  if (!mirror) return lms;
  return lms.map(([x, y, z]) => [1 - x, y, z]) as Landmarks;
}

function zscore(features: number[], stats: NormStats): number[] {
  const mean = stats.mean;
  const std = stats.std;
  const eps = 1e-8;

  if (mean.length !== features.length || std.length !== features.length) {
    console.warn(
      `⚠️ Norm stats length mismatch: features=${features.length}, mean=${mean.length}, std=${std.length}. Using raw features.`
    );
    return features;
  }

  const out = new Array(features.length);
  for (let i = 0; i < features.length; i++) {
    const s = std[i] ?? 1;
    out[i] = (features[i] - (mean[i] ?? 0)) / (Math.abs(s) < eps ? 1 : s);
  }
  return out;
}

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((v) => v / sum);
}

/**
 * Canonical key for labels:
 * "C - c" -> "c"
 * "Ten - 10" -> "10"
 * fallback -> lowercase trimmed
 */
function labelKey(lbl: string): string {
  const parts = lbl.split("-");
  if (parts.length >= 2) return parts[1].trim().toLowerCase();
  return lbl.trim().toLowerCase();
}

/**
 * Expected key from sign name:
 * "BSL - A" -> "a"
 * "BSL - 10" -> "10"
 * otherwise null
 */
function expectedKeyFromSign(current: SignType): string | null {
  const name = current.name ?? "";

  const letter = name.match(/([A-Za-z])\s*$/);
  if (letter?.[1]) return letter[1].toLowerCase();

  const num = name.match(/(\d+)\s*$/);
  if (num?.[1]) return num[1];

  return null;
}

export default function PracticePanel() {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const trackerRef = useRef<HandTracker>();

  // overlay can draw 0/1/2 hands
  const [handsPx, setHandsPx] = useState<Landmarks[]>([]);

  const [score, setScore] = useState<number>(0);
  const [advice, setAdvice] = useState<string>("");
  const [selectedId, setSelectedId] = useState<string>((signs as any)[0].id);
  const [mirror, setMirror] = useState(true);
  const [prediction, setPrediction] = useState<string>("");

  const current: SignType = useMemo(
    () => (signs as any).find((s: SignType) => s.id === selectedId)!,
    [selectedId]
  );

  const windowRef = useRef<number[][]>([]);
  const [seqScore, setSeqScore] = useState<number | null>(null);

  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const labelMap = useRef<string[]>(labels as string[]);

  // we expect 126 for two-hand model, but read from ONNX to be safe
  const expectedFeatureDim = useRef<number>(126);

  useEffect(() => {
    async function loadModel(): Promise<void> {
      try {
        console.log("🔄 Loading ONNX model...");

        ort.env.wasm.wasmPaths = window.location.origin + "/ort/";
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
        ort.env.wasm.simd = true;
        ort.env.wasm.useDynamicImport = false;

        const modelURL = import.meta.env.BASE_URL + "models/bsl_sign_model.onnx";
        const sess = await ort.InferenceSession.create(modelURL, {
          executionProviders: ["wasm"],
        });

        const inName = sess.inputNames[0];
        const meta = sess.inputMetadata?.[inName];
        const dims = meta?.dimensions ?? [];
        const last = dims[dims.length - 1];
        const parsed = safeNumberDim(last);
        expectedFeatureDim.current = parsed ?? expectedFeatureDim.current;

        console.log("✅ ONNX model loaded.", {
          inputName: inName,
          dims,
          expectedFeatureDim: expectedFeatureDim.current,
        });

        setSession(sess);
      } catch (err) {
        console.error("❌ Failed to load ONNX model:", err);
      }
    }

    void loadModel();
  }, []);

  const runInference = useCallback(
    async (rawFeatures: number[]): Promise<void> => {
      if (!session) return;

      if (rawFeatures.length !== expectedFeatureDim.current) {
        console.warn(
          `⚠️ Feature length mismatch: got=${rawFeatures.length}, expected=${expectedFeatureDim.current}`
        );
        return;
      }

      try {
        // apply same z-score normalization used in training
        const features = zscore(rawFeatures, normStats as NormStats);

        const inputName = session.inputNames[0];
        const inputTensor = new ort.Tensor(
          "float32",
          new Float32Array(features),
          [1, features.length]
        );

        const results = await session.run({ [inputName]: inputTensor });
        const outputTensor = Object.values(results)[0] as ort.Tensor;
        const logits = Array.from(outputTensor.data as Float32Array);
        if (!logits.length) return;

        const probs = softmax(logits);
        const maxProb = Math.max(...probs);
        const maxIdx = probs.indexOf(maxProb);

        const label = labelMap.current[maxIdx] ?? "Unknown";
        setPrediction(label);

        // keep as integer percent, but don’t force 100 unless it truly is ~1.0
        setScore(Math.round(maxProb * 100));
      } catch (err) {
        console.error("❌ Inference error:", err);
      }
    },
    [session]
  );

  useEffect(() => {
    if (!video) return;

    const ht = new HandTracker();
    trackerRef.current = ht;

    (async () => {
      await ht.init({
        videoEl: video,
        onResults: ({ hands, videoEl }) => {
          if (!hands.length) {
            setHandsPx([]);
            setAdvice("Show your hand(s) to the camera");
            setPrediction("");
            setScore(0);
            return;
          }

          // stable left/right selection from MediaPipe handedness
          const left = hands.find((h) => h.handedness === "Left") ?? null;
          const right = hands.find((h) => h.handedness === "Right") ?? null;

          // raw normalized landmarks (0..1-ish) for the MODEL
          const leftRaw = left ? (left.landmarks as Landmarks) : null;
          const rightRaw = right ? (right.landmarks as Landmarks) : null;

          // pixels for overlay + gating ONLY
          const leftPx = leftRaw ? toPixels(leftRaw, videoEl) : null;
          const rightPx = rightRaw ? toPixels(rightRaw, videoEl) : null;

          const overlayHands: Landmarks[] = [];
          if (leftPx) overlayHands.push(leftPx);
          if (rightPx) overlayHands.push(rightPx);
          setHandsPx(overlayHands);

          // require at least one good hand in view
          const gate =
            (rightPx && viewGate(rightPx, videoEl)) ||
            (leftPx && viewGate(leftPx, videoEl));

          if (!gate || !gate.ok) {
            setAdvice(gate?.advice ?? "Adjust hands into view");
            setPrediction("");
            setScore(0);
            return;
          }
          setAdvice("");

          // ✅ Build model features to match CSV format:
          // two-hand => 126 floats [Left(63), Right(63)] with zero-padding
          // one-hand => 63 floats from dominant/best hand
          let rawFeatures: number[] = [];

          if (expectedFeatureDim.current === 126) {
            const L = leftRaw ? flatten63(maybeMirrorRaw(leftRaw, mirror)) : zeros(63);
            const R = rightRaw ? flatten63(maybeMirrorRaw(rightRaw, mirror)) : zeros(63);
            rawFeatures = [...L, ...R];
          } else {
            // single-hand model fallback
            const dominant = (current.dominant ?? "right").toLowerCase();
            const bestRaw =
              dominant === "left"
                ? leftRaw ?? rightRaw
                : rightRaw ?? leftRaw;

            if (!bestRaw) {
              setPrediction("");
              setScore(0);
              return;
            }

            rawFeatures = flatten63(maybeMirrorRaw(bestRaw, mirror));
          }

          void runInference(rawFeatures);

          // keep your existing pose / DTW scoring (still single-hand)
          const bestPx = (current.dominant ?? "right") === "left"
            ? leftPx ?? rightPx
            : rightPx ?? leftPx;

          if (!bestPx) return;

          const bestNorm = normalize(bestPx, { mirror });

          if (current.type === "static") {
            const ang = computeAngles(bestNorm);
            const target = current.targetPose?.angles ?? {};
            poseScore(
              ang,
              target,
              current.weights ?? {},
              current.targetPose?.toleranceDefault ?? 12
            );
          } else {
            const ang = computeAngles(bestNorm);
            const row = [bestNorm[0][1], ang.R_INDEX_MCP ?? 0, ang.R_MIDDLE_MCP ?? 0];
            const buf = windowRef.current;
            buf.push(row);
            if (buf.length > 60) buf.shift();
          }
        },
      });

      ht.start();
    })();

    return () => trackerRef.current?.stop();
  }, [video, mirror, current, runInference]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (current.type !== "dynamic") return;
      if (e.code === "Space") {
        const buf = windowRef.current.slice();
        if (buf.length < 20) return;
        const tmpl = current.template;
        if (!tmpl) return;

        const resampled = resample(buf, tmpl.length ?? buf.length);
        const cost = dtwCost(resampled, tmpl.sequence);
        setSeqScore(dtwScore(cost, 0.15, current.tolerance?.dtw ?? 0.45));
        windowRef.current = [];
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current]);

  const expectedKey = expectedKeyFromSign(current);
  const predKey = prediction ? labelKey(prediction) : null;
  const isCorrect =
    current.type === "static" && expectedKey && predKey
      ? predKey === expectedKey
      : null;

  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <label>
          Sign:&nbsp;
          <select value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
            {(signs as any).map((s: SignType) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </label>

        <label>
          <input
            type="checkbox"
            checked={mirror}
            onChange={(e) => setMirror(e.target.checked)}
          />{" "}
          Mirror tutor view
        </label>

        {current.type === "static" ? (
          <b>Confidence: {score}%</b>
        ) : (
          <b>Seq Score: {seqScore ?? "-"}</b>
        )}

        <span style={{ color: advice ? "#d33" : "#0a0" }}>{advice || "View OK"}</span>

        <b style={{ color: "#0077cc" }}>{prediction ? `Pred: ${prediction}` : ""}</b>

        {current.type === "static" && prediction && expectedKey && (
          <b style={{ color: isCorrect ? "#18a34a" : "#dc2626" }}>
          </b>
        )}
      </div>

      <div style={{ position: "relative", width: "100%", maxWidth: 960 }}>
        <CameraFeed onReady={setVideo} />
        {video && (
          <CanvasOverlay
            width={video.videoWidth || 1280}
            height={video.videoHeight || 720}
            handsPx={handsPx}
            ghostPx={undefined}
          />
        )}
      </div>
    </div>
  );
}

function resample(seq: number[][], len: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < len; i++) {
    const t = (i * (seq.length - 1)) / (len - 1);
    const i0 = Math.floor(t);
    const i1 = Math.min(seq.length - 1, i0 + 1);
    const a = t - i0;
    const row = seq[i0].map((v, k) => v * (1 - a) + seq[i1][k] * a);
    out.push(row);
  }
  return out;
}
