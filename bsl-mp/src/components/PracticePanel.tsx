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
import type { Landmarks } from "../logic/types";

interface TargetPose {
  angles: Record<string, number>;
  toleranceDefault?: number;
}

interface SignType {
  id: string;
  name: string;
  type: "static" | "dynamic";
  weights?: Record<string, number>;
  targetPose?: TargetPose;
  template?: { sequence: number[][]; length?: number };
  tolerance?: { dtw?: number };
}

export default function PracticePanel() {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const trackerRef = useRef<HandTracker>();
  const [handsPx, setHandsPx] = useState<Landmarks[]>([]);
  const [score, setScore] = useState<number>(0); // shown as "Confidence" for static
  const [advice, setAdvice] = useState<string>("");
  const [selectedId, setSelectedId] = useState<string>(signs[0].id);
  const [mirror, setMirror] = useState(true);
  const [prediction, setPrediction] = useState<string>("");

  const current: SignType = useMemo(
    () => signs.find((s: SignType) => s.id === selectedId)!,
    [selectedId]
  );

  const windowRef = useRef<number[][]>([]);
  const [seqScore, setSeqScore] = useState<number | null>(null);

  // ---------------- ONNX runtime setup ----------------
  const [session, setSession] = useState<ort.InferenceSession | null>(null);

  // Label map is now synced to training via JSON
  const labelMap = useRef<string[]>(labels as string[]);

  useEffect(() => {
    async function loadModel(): Promise<void> {
      try {
        console.log("🔄 Loading ONNX model...");

        // WASM runtime configuration (paths must match your public/ort folder)
        ort.env.wasm.wasmPaths = window.location.origin + "/ort/";
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
        ort.env.wasm.simd = true;
        ort.env.wasm.useDynamicImport = false;

        // Model served from public/models/
        const modelURL = import.meta.env.BASE_URL + "models/bsl_sign_model.onnx";

        const sess = await ort.InferenceSession.create(modelURL, {
          executionProviders: ["wasm"],
        });

        setSession(sess);
        console.log("✅ ONNX model loaded successfully!");
      } catch (err) {
        console.error("Failed to load ONNX model:", err);
      }
    }

    void loadModel();
  }, []);

  // ---------------- Inference (with softmax) ----------------
  const runInference = useCallback(
    async (landmarks: number[]): Promise<void> => {
      if (!session || landmarks.length !== 63) return;

      try {
        const inputTensor = new ort.Tensor("float32", new Float32Array(landmarks), [1, 63]);
        const results = await session.run({ input: inputTensor });

        // Assume single output
        const outputTensor = Object.values(results)[0] as ort.Tensor;
        const logits = Array.from(outputTensor.data as Float32Array);

        if (!logits.length) return;

        // Softmax for stable probabilities
        const maxLogit = Math.max(...logits);
        const exp = logits.map((v) => Math.exp(v - maxLogit));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        const probs = exp.map((v) => v / sumExp);

        const maxProb = Math.max(...probs);
        const maxIdx = probs.indexOf(maxProb);

        const label = labelMap.current[maxIdx] ?? "Unknown";

        setPrediction(label);
        setScore(Math.round(maxProb * 100)); // 0–100%
      } catch (err) {
        console.error("Inference error:", err);
      }
    },
    [session]
  );

  // ---------------- MediaPipe + scoring ----------------
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
            setAdvice("Show your hand to the camera");
            setPrediction("");
            setScore(0);
            return;
          }

          const px = toPixels(hands[0].landmarks, videoEl);
          setHandsPx([px]);

          const gate = viewGate(px, videoEl);
          if (!gate.ok) {
            setAdvice(gate.advice!);
            setScore(0);
            setPrediction("");
            return;
          }
          setAdvice("");

          const norm = normalize(px, { mirror });
          const landmarks = norm.flatMap((p) => [p[0], p[1], p[2]]);

          // Run ONNX classifier
          void runInference(landmarks);

          // Existing pose / DTW scoring logic
          if (current.type === "static") {
            const ang = computeAngles(norm);
            const target = current.targetPose?.angles ?? {};
            const s = poseScore(
              ang,
              target,
              current.weights ?? {},
              current.targetPose?.toleranceDefault ?? 12
            );
            // If you want the ONNX score only, comment the next line:
            // setScore(s.score);
          } else {
            const ang = computeAngles(norm);
            const row = [norm[0][1], ang.R_INDEX_MCP ?? 0, ang.R_MIDDLE_MCP ?? 0];
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

  // Dynamic sequence scoring (unchanged)
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

  // ---------------- UI ----------------
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <label>
          Sign:&nbsp;
          <select value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
            {signs.map((s: SignType) => (
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
        <b style={{ color: "#0077cc" }}>{prediction}</b>
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
