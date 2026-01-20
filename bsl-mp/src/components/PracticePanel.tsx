import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import CameraFeed from "./CameraFeed";
import CanvasOverlay from "./CanvasOverlay";
import SentenceBuilder from "./SentenceBuilder";

import { HandTracker } from "../mediapipe/handTracker";
import { toPixels, normalize } from "../logic/normalize";
import { computeAngles } from "../logic/angles";
import { poseScore, dtwCost, dtwScore } from "../logic/scoring";
import { viewGate } from "../logic/viewGate";

import signs from "../data/bsl_signs.json";
import labels from "../data/bsl_labels.json";
import normStats from "../data/bsl_norm.json";
import vocab from "../data/bsl_vocab.json";

import {
  createDecoderState,
  commitChar,
  commitSpace,
  backspace as decoderBackspace,
  clearAll as decoderClearAll,
  commitSuggestion,
  suggestionsFor,
  type DecoderState,
} from "../logic/sentenceDecoder";

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
      `⚠️ Norm stats mismatch: features=${features.length}, mean=${mean.length}, std=${std.length}. Using raw.`
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

function softmax(x: number[]): number[] {
  const m = Math.max(...x);
  const exps = x.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((v) => v / sum);
}

/**
 * If the output already looks like probabilities (all >=0 and sum ~ 1),
 * don’t softmax again.
 */
function toProbabilities(output: number[]): number[] {
  if (!output.length) return output;

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const v of output) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }

  const looksLikeProbs = min >= 0 && max <= 1.0 + 1e-3 && Math.abs(sum - 1) < 1e-2;
  return looksLikeProbs ? output : softmax(output);
}

/**
 * "C - c" -> "c"
 * "Ten - 10" -> "10"
 */
function labelToToken(lbl: string): string | null {
  const parts = lbl.split("-").map((s) => s.trim());
  if (parts.length >= 2) {
    const rhs = parts[1];
    if (/^[a-z]$/i.test(rhs)) return rhs.toLowerCase();
    if (/^\d+$/.test(rhs)) return rhs;
  }
  return null;
}

function majorityVote(items: string[]): { label: string; frac: number } | null {
  if (!items.length) return null;
  const counts = new Map<string, number>();
  for (const s of items) counts.set(s, (counts.get(s) ?? 0) + 1);

  let best = items[0];
  let bestC = 0;
  for (const [k, v] of counts.entries()) {
    if (v > bestC) {
      best = k;
      bestC = v;
    }
  }
  return { label: best, frac: bestC / items.length };
}

export default function PracticePanel() {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const trackerRef = useRef<HandTracker>();

  const [handsPx, setHandsPx] = useState<Landmarks[]>([]);

  const [advice, setAdvice] = useState<string>("");
  const [selectedId, setSelectedId] = useState<string>((signs as any)[0].id);
  const [mirror, setMirror] = useState(true);

  const [prediction, setPrediction] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);

  const [decoder, setDecoder] = useState<DecoderState>(() => createDecoderState());

  const current: SignType = useMemo(
    () => (signs as any).find((s: SignType) => s.id === selectedId)!,
    [selectedId]
  );

  const windowRef = useRef<number[][]>([]);
  const [seqScore, setSeqScore] = useState<number | null>(null);

  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const labelMap = useRef<string[]>(labels as string[]);
  const expectedFeatureDim = useRef<number>(126);

  // Stabilisation
  const predWindowRef = useRef<string[]>([]);
  const stableLabelRef = useRef<string>("");
  const stableSinceRef = useRef<number>(0);
  const lastCommitAtRef = useRef<number>(0);

  // Tunables
  const CONF_THRESH = 55;
  const VOTE_WINDOW = 10;
  const VOTE_FRAC = 0.75;
  const HOLD_MS = 550;
  const COOLDOWN_MS = 700;

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

        console.log("✅ ONNX loaded:", {
          inputName: inName,
          dims,
          expectedFeatureDim: expectedFeatureDim.current,
          outputs: sess.outputNames,
        });

        setSession(sess);
      } catch (err) {
        console.error("❌ Failed to load ONNX model:", err);
      }
    }

    void loadModel();
  }, []);

  const runInference = useCallback(
    async (rawFeatures: number[]): Promise<{ label: string; conf: number } | null> => {
      if (!session) return null;
      if (rawFeatures.length !== expectedFeatureDim.current) return null;

      const features = zscore(rawFeatures, normStats as NormStats);

      const inputName = session.inputNames[0];
      const inputTensor = new ort.Tensor("float32", new Float32Array(features), [
        1,
        features.length,
      ]);

      const results = await session.run({ [inputName]: inputTensor });

      // Prefer first outputName explicitly
      const outName = session.outputNames[0];
      const outTensor = (results[outName] ?? Object.values(results)[0]) as ort.Tensor;

      const rawOut = Array.from(outTensor.data as Float32Array);
      if (!rawOut.length) return null;

      const probs = toProbabilities(rawOut);

      let maxProb = -1;
      let maxIdx = -1;
      for (let i = 0; i < probs.length; i++) {
        if (probs[i] > maxProb) {
          maxProb = probs[i];
          maxIdx = i;
        }
      }

      const label = labelMap.current[maxIdx] ?? "Unknown";
      const conf = Math.round(maxProb * 100);

      return { label, conf };
    },
    [session]
  );

  // Decoder actions
  const doCommitSpace = useCallback(() => setDecoder((s) => commitSpace(s)), []);
  const doBackspace = useCallback(() => setDecoder((s) => decoderBackspace(s)), []);
  const doClear = useCallback(() => setDecoder(decoderClearAll()), []);
  const doPickSuggestion = useCallback((w: string) => {
    setDecoder((s) => commitSuggestion(s, w));
  }, []);

  // Keyboard helpers (sentence building)
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.code === "Backspace") doBackspace();
      if (e.code === "Space" && current.type === "static") doCommitSpace();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [doBackspace, doCommitSpace, current.type]);

  // MediaPipe loop
  useEffect(() => {
    if (!video) return;

    const ht = new HandTracker();
    trackerRef.current = ht;

    (async () => {
      await ht.init({
        videoEl: video,
        onResults: async ({ hands, videoEl }) => {
          if (!hands.length) {
            setHandsPx([]);
            setAdvice("Show your hand(s) to the camera");
            setPrediction("");
            setConfidence(0);
            predWindowRef.current = [];
            stableLabelRef.current = "";
            return;
          }

          const left = hands.find((h) => h.handedness === "Left") ?? null;
          const right = hands.find((h) => h.handedness === "Right") ?? null;

          const leftRaw = left ? (left.landmarks as Landmarks) : null;
          const rightRaw = right ? (right.landmarks as Landmarks) : null;

          const leftPx = leftRaw ? toPixels(leftRaw, videoEl) : null;
          const rightPx = rightRaw ? toPixels(rightRaw, videoEl) : null;

          const overlayHands: Landmarks[] = [];
          if (leftPx) overlayHands.push(leftPx);
          if (rightPx) overlayHands.push(rightPx);
          setHandsPx(overlayHands);

          const gate =
            (rightPx && viewGate(rightPx, videoEl)) || (leftPx && viewGate(leftPx, videoEl));

          if (!gate || !gate.ok) {
            setAdvice(gate?.advice ?? "Adjust hands into view");
            setPrediction("");
            setConfidence(0);
            predWindowRef.current = [];
            stableLabelRef.current = "";
            return;
          }
          setAdvice("");

          // Build feature vector (matches your CSV style: raw 0..1 landmarks)
          let rawFeatures: number[] = [];
          if (expectedFeatureDim.current === 126) {
            const L = leftRaw ? flatten63(maybeMirrorRaw(leftRaw, mirror)) : zeros(63);
            const R = rightRaw ? flatten63(maybeMirrorRaw(rightRaw, mirror)) : zeros(63);
            rawFeatures = [...L, ...R];
          } else {
            const dominant = (current.dominant ?? "right").toLowerCase();
            const bestRaw = dominant === "left" ? leftRaw ?? rightRaw : rightRaw ?? leftRaw;
            if (!bestRaw) return;
            rawFeatures = flatten63(maybeMirrorRaw(bestRaw, mirror));
          }

          const res = await runInference(rawFeatures);
          if (!res) return;

          setPrediction(res.label);
          setConfidence(res.conf);

          // Auto-commit ONLY in static mode (fingerspelling)
          if (current.type === "static") {
            const win = predWindowRef.current;
            win.push(res.label);
            if (win.length > VOTE_WINDOW) win.shift();

            const mv = majorityVote(win);
            if (!mv) return;

            const candidate = mv.label;
            const now = performance.now();

            if (res.conf < CONF_THRESH || mv.frac < VOTE_FRAC) {
              stableLabelRef.current = "";
              return;
            }

            if (candidate !== stableLabelRef.current) {
              stableLabelRef.current = candidate;
              stableSinceRef.current = now;
              return;
            }

            const stableFor = now - stableSinceRef.current;
            const sinceLast = now - lastCommitAtRef.current;

            if (stableFor >= HOLD_MS && sinceLast >= COOLDOWN_MS) {
              const tok = labelToToken(candidate);
              if (tok) {
                setDecoder((s) => commitChar(s, tok));
                lastCommitAtRef.current = now;
              }
            }
          }

          // Keep your pose/DTW logic untouched (single-hand)
          const bestPx =
            (current.dominant ?? "right") === "left" ? leftPx ?? rightPx : rightPx ?? leftPx;
          if (!bestPx) return;

          const bestNorm = normalize(bestPx, { mirror });

          if (current.type === "static") {
            const ang = computeAngles(bestNorm);
            const target = current.targetPose?.angles ?? {};
            poseScore(ang, target, current.weights ?? {}, current.targetPose?.toleranceDefault ?? 12);
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

  // Dynamic DTW scoring
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

  const stableLabel = stableLabelRef.current;
  const sugg = suggestionsFor(decoder.bufferWord, vocab as string[], 6);

  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
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
          <input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} />{" "}
          Mirror tutor view
        </label>

        {current.type === "static" ? <b>Confidence: {confidence}%</b> : <b>Seq Score: {seqScore ?? "-"}</b>}

        <span style={{ color: advice ? "#d33" : "#0a0" }}>{advice || "View OK"}</span>

        <b style={{ color: "#0077cc" }}>
          {prediction ? `Pred: ${prediction}` : ""}
          {stableLabel ? ` (stable: ${stableLabel})` : ""}
        </b>
      </div>

      {/* Sentence builder works TODAY even with only alphabet labels */}
      <SentenceBuilder
        transcript={decoder.transcript}
        bufferWord={decoder.bufferWord}
        suggestions={sugg}
        onCommitSpace={doCommitSpace}
        onBackspace={doBackspace}
        onClear={doClear}
        onPickSuggestion={doPickSuggestion}
      />

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
