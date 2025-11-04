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
import type { Landmarks } from "../logic/types";

/* -------------------------------------------------------------------------- */
/*                               Custom Types                                 */
/* -------------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------- */
/*                               Main Component                               */
/* -------------------------------------------------------------------------- */
export default function PracticePanel() {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const trackerRef = useRef<HandTracker>();
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

  /* ---------------------------- ONNX Model State --------------------------- */
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const labelMap = useRef<string[]>([
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "0","1","2","3","4","5","6","7","8","9"
  ]);

  /* ---------------------------- Load ONNX Model ---------------------------- */
  useEffect(() => {
    async function loadModel(): Promise<void> {
      try {
        console.log("🔄 Loading ONNX model...");

        // ✅ Ensure correct path for ONNX wasm files
        ort.env.wasm.wasmPaths = window.location.origin + "/ort/";
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
        ort.env.wasm.simd = true;
        ort.env.wasm.useDynamicImport = false;

        const modelURL = import.meta.env.BASE_URL + "models/bsl_sign_model.ort";

        const session = await ort.InferenceSession.create(modelURL, {
          executionProviders: ["wasm"],
        });

        setSession(session);
        console.log("✅ ONNX model loaded successfully!");
      } catch (err: any) {
        console.error("❌ Failed to load ONNX model:", err);
      }
    }

    void loadModel();
  }, []);

  /* ----------------------------- Run Inference ----------------------------- */
  const runInference = useCallback(
    async (landmarks: number[]): Promise<void> => {
      if (!session || landmarks.length !== 63) return;

      try {
        const inputTensor = new ort.Tensor<Float32Array>("float32", new Float32Array(landmarks), [1, 63]);
        const results: Record<string, ort.Tensor> = await session.run({ input: inputTensor });
        const outputTensor = Object.values(results)[0] as ort.Tensor;
        const scores = Array.from(outputTensor.data as Float32Array);

        const maxScore = Math.max(...scores);
        const maxIdx = scores.indexOf(maxScore);
        const label = labelMap.current[maxIdx] || "Unknown";

        // ✅ Update UI label and score directly from model confidence
        setPrediction(`${label}`);
        setScore(Math.round(maxScore * 100)); // model confidence as % score
      } catch (err) {
        console.error("❌ Inference error:", err);
      }
    },
    [session]
  );


  /* ------------------------ MediaPipe Hand Tracking ------------------------ */
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
            return;
          }

          const px = toPixels(hands[0].landmarks, videoEl);
          setHandsPx([px]);

          const gate = viewGate(px, videoEl);
          if (!gate.ok) {
            setAdvice(gate.advice!);
            setScore(0);
            return;
          }
          setAdvice("");

          const norm = normalize(px, { mirror });
          const landmarks = norm.flatMap((p) => [p[0], p[1], p[2]]);
          void runInference(landmarks);

          if (current.type === "static") {
            const ang = computeAngles(norm);
            const target = current.targetPose?.angles ?? {};
            const s = poseScore(
              ang,
              target,
              current.weights ?? {},
              current.targetPose?.toleranceDefault ?? 12
            );
            setScore(s.score);
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

  /* -------------------------- Dynamic Key Scoring -------------------------- */
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

  /* ------------------------------- UI Render ------------------------------- */
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

/* -------------------------------------------------------------------------- */
/*                                Helper Utils                                */
/* -------------------------------------------------------------------------- */
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
