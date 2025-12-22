import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import CameraFeed from "./CameraFeed";
import CanvasOverlay from "./CanvasOverlay";
import { HandTracker } from "../mediapipe/handTracker";
import { toPixels, normalizeTwoHands, normalize, flattenLandmarks } from "../logic/normalize";
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

  // ✅ now can hold 0, 1, or 2 hands for overlay
  const [handsPx, setHandsPx] = useState<Landmarks[]>([]);

  const [score, setScore] = useState<number>(0);
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

  const [session, setSession] = useState<ort.InferenceSession | null>(null);

  // synced labels
  const labelMap = useRef<string[]>(labels as string[]);

  // ✅ detect expected feature size from ONNX model (63 or 126)
  const expectedFeatureDim = useRef<number>(63);

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

        // infer input dim
        const inName = sess.inputNames[0];
        const meta = sess.inputMetadata[inName];
        const dims = meta?.dimensions ?? [];
        const last = dims[dims.length - 1];
        if (typeof last === "number") expectedFeatureDim.current = last;

        setSession(sess);
        console.log("✅ ONNX model loaded! Expected features:", expectedFeatureDim.current);
      } catch (err) {
        console.error("Failed to load ONNX model:", err);
      }
    }

    void loadModel();
  }, []);

  // ---------------- Inference (softmax) ----------------
  const runInference = useCallback(
    async (features: number[]): Promise<void> => {
      if (!session) return;
      if (features.length !== expectedFeatureDim.current) return;

      try {
        const inputName = session.inputNames[0];
        const inputTensor = new ort.Tensor("float32", new Float32Array(features), [1, features.length]);
        const results = await session.run({ [inputName]: inputTensor });

        const outputTensor = Object.values(results)[0] as ort.Tensor;
        const logits = Array.from(outputTensor.data as Float32Array);
        if (!logits.length) return;

        // softmax
        const maxLogit = Math.max(...logits);
        const exp = logits.map((v) => Math.exp(v - maxLogit));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        const probs = exp.map((v) => v / sumExp);

        const maxProb = Math.max(...probs);
        const maxIdx = probs.indexOf(maxProb);

        const label = labelMap.current[maxIdx] ?? "Unknown";
        setPrediction(label);
        setScore(Math.round(maxProb * 100));
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
            setAdvice("Show your hand(s) to the camera");
            setPrediction("");
            setScore(0);
            return;
          }

          // ✅ split into left/right (stable)
          const left = hands.find((h) => h.handedness === "Left") ?? null;
          const right = hands.find((h) => h.handedness === "Right") ?? null;

          // ✅ convert to pixels (for drawing + gating)
          const leftPx = left ? toPixels(left.landmarks as any, videoEl) : null;
          const rightPx = right ? toPixels(right.landmarks as any, videoEl) : null;

          // ✅ draw both (whichever exist)
          const overlayHands: Landmarks[] = [];
          if (leftPx) overlayHands.push(leftPx);
          if (rightPx) overlayHands.push(rightPx);
          setHandsPx(overlayHands);

          // gating policy: require at least one “good” hand
          const gateHand = (leftPx && viewGate(leftPx, videoEl)) || (rightPx && viewGate(rightPx, videoEl));
          if (!gateHand || (gateHand && !gateHand.ok)) {
            setAdvice(gateHand?.advice ?? "Adjust hands into view");
            setPrediction("");
            setScore(0);
            return;
          }
          setAdvice("");

          // ✅ build features depending on model input size
          let features: number[] = [];

          if (expectedFeatureDim.current === 126) {
            // true 2-hand model
            features = normalizeTwoHands(leftPx, rightPx, { mirror });
          } else {
            // fallback: single-hand model -> use the “best” hand
            // prefer Right if present, else Left
            const bestPx = rightPx ?? leftPx!;
            const norm = normalize(bestPx, { mirror });
            features = flattenLandmarks(norm);
          }

          void runInference(features);

          // keep your existing scoring (still uses a single hand here)
          // If you want two-hand pose scoring later, we can extend computeAngles for both.
          const bestPx = rightPx ?? leftPx!;
          const bestNorm = normalize(bestPx, { mirror });

          if (current.type === "static") {
            const ang = computeAngles(bestNorm);
            const target = current.targetPose?.angles ?? {};
            const s = poseScore(
              ang,
              target,
              current.weights ?? {},
              current.targetPose?.toleranceDefault ?? 12
            );
            // optional: if you want poseScore to override ONNX confidence, uncomment:
            // setScore(s.score);
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

  // dynamic scoring unchanged
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
          <input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} />{" "}
          Mirror tutor view
        </label>

        {current.type === "static" ? <b>Confidence: {score}%</b> : <b>Seq Score: {seqScore ?? "-"}</b>}

        <span style={{ color: advice ? "#d33" : "#0a0" }}>{advice || "View OK"}</span>
        <b style={{ color: "#0077cc" }}>{prediction}</b>
      </div>

      <div style={{ position: "relative", width: "100%", maxWidth: 960 }}>
        <CameraFeed onReady={setVideo} />
        {video && (
          <CanvasOverlay
            width={video.videoWidth || 1280}
            height={video.videoHeight || 720}
            handsPx={handsPx} // ✅ now can draw both
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
