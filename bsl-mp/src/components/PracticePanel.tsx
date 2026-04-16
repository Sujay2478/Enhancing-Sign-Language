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

import { motionEnergy } from "../logic/motion";

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
  template?: { sequence: number[][]; length?: number; features?: string[] };
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

function featureValue(
  feat: string,
  norm: Landmarks,
  ang: Record<string, number>,
  px: Landmarks
): number {

  if (feat === "R_WRIST_X") return px[0]?.[0] ?? 0;
  if (feat === "R_WRIST_Y") return px[0]?.[1] ?? 0;
  if (feat === "R_WRIST_Z") return px[0]?.[2] ?? 0;

  if (feat in ang) return ang[feat] ?? 0;

  return 0;
}

function frameFeatures(
  norm: Landmarks,
  ang: Record<string, number>,
  feats: string[],
  px: Landmarks
): number[] {
  return feats.map(f => featureValue(f, norm, ang, px));
}

function padTo21(lms: Landmarks | null): Landmarks {
  if (!lms) return Array.from({ length: 21 }, () => [0, 0, 0]) as Landmarks;
  return lms;
}

function featureValue2H(
  feat: string,
  leftNorm: Landmarks,
  rightNorm: Landmarks,
  angL: Record<string, number>,
  angR: Record<string, number>,
  leftPx: Landmarks,
  rightPx: Landmarks
): number {
  // Pixel wrist features
  if (feat === "L_WRIST_X") return leftPx[0]?.[0] ?? 0;
  if (feat === "L_WRIST_Y") return leftPx[0]?.[1] ?? 0;
  if (feat === "L_WRIST_Z") return leftPx[0]?.[2] ?? 0;

  if (feat === "R_WRIST_X") return rightPx[0]?.[0] ?? 0;
  if (feat === "R_WRIST_Y") return rightPx[0]?.[1] ?? 0;
  if (feat === "R_WRIST_Z") return rightPx[0]?.[2] ?? 0;

  // Angle features
  if (feat.startsWith("L_")) {
    const k = feat.replace("L_", "R_");
    return angL[k] ?? 0;
  }
  if (feat.startsWith("R_")) return angR[feat] ?? 0;

  return 0;
}

function frameFeatures2H(
  feats: string[],
  leftNorm: Landmarks,
  rightNorm: Landmarks,
  angL: Record<string, number>,
  angR: Record<string, number>,
  leftPx: Landmarks,
  rightPx: Landmarks
): number[] {
  return feats.map((f) => featureValue2H(f, leftNorm, rightNorm, angL, angR, leftPx, rightPx));
}

export default function PracticePanel() {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const trackerRef = useRef<HandTracker>();

  const [handsPx, setHandsPx] = useState<Landmarks[]>([]);

  const [advice, setAdvice] = useState<string>("");
  const [selectedId, setSelectedId] = useState<string>((signs as any)[0].id);
  const [mirror, setMirror] = useState(true);

  // Static (ONNX) prediction display
  const [prediction, setPrediction] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);

  const [decoder, setDecoder] = useState<DecoderState>(() => createDecoderState());

  const current: SignType = useMemo(
    () => (signs as any).find((s: SignType) => s.id === selectedId)!,
    [selectedId]
  );

  const [seqScore, setSeqScore] = useState<number | null>(null);
  const [dynResult, setDynResult] = useState<{ label: string; score: number } | null>(null);

  const [dynDebug, setDynDebug] = useState<string>("");

  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const labelMap = useRef<string[]>(labels as string[]);
  const expectedFeatureDim = useRef<number>(126);

  const predWindowRef = useRef<string[]>([]);
  const stableLabelRef = useRef<string>("");
  const stableSinceRef = useRef<number>(0);
  const lastCommitAtRef = useRef<number>(0);

  const dynActiveRef = useRef(false);
  const dynIdleRef = useRef(0);
  const dynBufRef = useRef<number[][]>([]);
  const dynLastCommitAtRef = useRef<number>(0);

  const dynPrevPxRef = useRef<Landmarks | null>(null);

  const CONF_THRESH = 55;
  const VOTE_WINDOW = 10;
  const VOTE_FRAC = 0.75;
  const HOLD_MS = 550;
  const COOLDOWN_MS = 700;

  const DYN_START = 0.006;
  const DYN_END = 0.0035;
  const DYN_END_FRAMES = 12;
  const DYN_MIN_FRAMES = 8;
  const DYN_MAX_FRAMES = 90;
  const DYN_COMMIT_COOLDOWN = 900;

  const resetDynamic = useCallback(() => {
    dynActiveRef.current = false;
    dynIdleRef.current = 0;
    dynBufRef.current = [];
    dynPrevPxRef.current = null;
    setDynDebug("");
  }, []);

  useEffect(() => {
    async function loadModel(): Promise<void> {
      try {
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

        setSession(sess);
      } catch (err) {
        console.error("Failed to load ONNX model:", err);
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

  // Finalize dynamic buffer → DTW score
  const finalizeDynamic = useCallback(() => {
    const buf = dynBufRef.current.slice();
    resetDynamic();

    if (current.type !== "dynamic") return;
    const tmpl = current.template;
    if (!tmpl?.sequence?.length) return;

    if (buf.length < DYN_MIN_FRAMES) {
      setDynResult(null);
      setSeqScore(0);
      return;
    }

    const targetLen = tmpl.length ?? buf.length;
    const resampled = resample(buf, targetLen);

    if (current.id === "bsl_hello" || current.id === "bsl_how_are_you" || "bsl_help") {
      console.log("📦 COPY THIS TEMPLATE:");
      console.log(JSON.stringify(resampled));
    }

    const cost = dtwCost(resampled, tmpl.sequence);
    const score = dtwScore(cost, 0.4, 1.2);

    console.log("DTW ranges:", {
      rowMinMax: [
        Math.min(...resampled.map(r => r[0])), Math.max(...resampled.map(r => r[0])),
        Math.min(...resampled.map(r => r[1])), Math.max(...resampled.map(r => r[1])),
        Math.min(...resampled.map(r => r[2])), Math.max(...resampled.map(r => r[2])),
      ],
      tmplMinMax: [
        Math.min(...tmpl.sequence.map(r => r[0])), Math.max(...tmpl.sequence.map(r => r[0])),
        Math.min(...tmpl.sequence.map(r => r[1])), Math.max(...tmpl.sequence.map(r => r[1])),
        Math.min(...tmpl.sequence.map(r => r[2])), Math.max(...tmpl.sequence.map(r => r[2])),
      ],
    });

    console.log("DTW debug:", {
      bufLen: buf.length,
      targetLen,
      cost,
      tol: current.tolerance?.dtw ?? 0.45,
      score,
      sampleResampled0: resampled[0],
      sampleTemplate0: tmpl.sequence[0],
    });

    setSeqScore(score);
    setDynResult({ label: current.name, score });

    const now = performance.now();
    if (score >= 75 && now - dynLastCommitAtRef.current >= DYN_COMMIT_COOLDOWN) {
      const word = current.name.split("-").pop()?.trim().toLowerCase();
      if (word) setDecoder((s) => commitSuggestion(s, word));
      dynLastCommitAtRef.current = now;
    }
  }, [current, DYN_MIN_FRAMES, DYN_COMMIT_COOLDOWN, resetDynamic]);

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
            resetDynamic();
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
            resetDynamic();
            return;
          }
          setAdvice("");

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

          if (res && current.type === "static") {
            setPrediction(res.label);
            setConfidence(res.conf);
          }

          if (res && current.type === "static") {
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

          const bestPx =
            (current.dominant ?? "right") === "left" ? leftPx ?? rightPx : rightPx ?? leftPx;
          if (!bestPx) return;

          const bestNorm = normalize(bestPx, { mirror });

          if (current.type === "static") {
            const ang = computeAngles(bestNorm);
            const target = current.targetPose?.angles ?? {};
            poseScore(ang, target, current.weights ?? {}, current.targetPose?.toleranceDefault ?? 12);
            resetDynamic();
            setDynResult(null);
            return;
          }

          const isTwoHand = current.hands === "two";

          let row: number[] = [];
          let motionPx: Landmarks | null = null;

          if (!isTwoHand) {
            const bestPx =
              (current.dominant ?? "right") === "left" ? leftPx ?? rightPx : rightPx ?? leftPx;
            if (!bestPx) return;

            const bestNorm = normalize(bestPx, { mirror });
            const ang = computeAngles(bestNorm);

            const feats = current.template?.features ?? ["R_WRIST_Y", "R_INDEX_MCP", "R_MIDDLE_MCP"];
            row = frameFeatures(bestNorm, ang, feats, bestPx);

            motionPx = bestPx;
          } else {
            // TWO-hand dynamic
            const LPx = padTo21(leftPx);
            const RPx = padTo21(rightPx);

            const LNorm = normalize(LPx, { mirror });
            const RNorm = normalize(RPx, { mirror });

            const angL = computeAngles(LNorm);
            const angR = computeAngles(RNorm);

            const feats =
              current.template?.features ??
              ["L_INDEX_MCP", "L_MIDDLE_MCP", "R_INDEX_MCP", "R_MIDDLE_MCP"];

            row = frameFeatures2H(feats, LNorm, RNorm, angL, angR, LPx, RPx);

            motionPx = RPx;
          }

          const printedRef =
            (window as any).__printedDyn ?? ((window as any).__printedDyn = new Set());

          if (!printedRef.has(current.id)) {
            printedRef.add(current.id);

            console.log("Dynamic sign:", current.id, current.name);
            console.log("Template features:", current.template?.features);
            console.log("Template first row:", current.template?.sequence?.[0]);

            console.log("Row (computed features):", row);

            const keys = Object.keys(ang);
            console.log("Angle keys (first 40):", keys.slice(0, 40));
            console.log(
              "Angle sample values:",
              keys.slice(0, 10).map((k) => [k, ang[k]])
            );
          }

          const prevPx = dynPrevPxRef.current;
          const rawMovePx = prevPx && motionPx ? motionEnergy(prevPx, motionPx) : 0;
          dynPrevPxRef.current = motionPx;

          const denom = Math.max(videoEl.videoWidth || 1, videoEl.videoHeight || 1);
          const move = rawMovePx / denom;

          if (!dynActiveRef.current) {
            if (move > DYN_START) {
              dynActiveRef.current = true;
              dynIdleRef.current = 0;
              dynBufRef.current = [row];
              setDynResult(null);
            }

            setDynDebug(
              `move=${move.toFixed(4)} active=${dynActiveRef.current} idle=${dynIdleRef.current} len=${dynBufRef.current.length}`
            );
            return;
          }

          dynBufRef.current.push(row);

          console.log("LIVE FRAME:", dynBufRef.current.length, row);

          if (dynBufRef.current.length <= 40) {
            console.log("LIVE FRAME:", dynBufRef.current.length, row);
          }
          if (dynBufRef.current.length === 40) {
            console.log("Captured 40 frames — stop gesture and let it finalize.");
          }

          if (move < DYN_END) dynIdleRef.current += 1;
          else dynIdleRef.current = 0;

          if (dynIdleRef.current >= DYN_END_FRAMES) {
            finalizeDynamic();
            return;
          }

          if (dynBufRef.current.length >= DYN_MAX_FRAMES) {
            finalizeDynamic();
            return;
          }

          setDynDebug(
            `move=${move.toFixed(4)} active=${dynActiveRef.current} idle=${dynIdleRef.current} len=${dynBufRef.current.length}`
          );
        },
      });

      ht.start();
    })();

    return () => trackerRef.current?.stop();
  }, [
    video,
    mirror,
    current,
    runInference,
    resetDynamic,
    finalizeDynamic,
    CONF_THRESH,
    VOTE_WINDOW,
    VOTE_FRAC,
    HOLD_MS,
    COOLDOWN_MS,
    DYN_START,
    DYN_END,
    DYN_END_FRAMES,
    DYN_MAX_FRAMES,
  ]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (current.type !== "dynamic") return;
      if (e.code === "Space") {
        if (dynBufRef.current.length >= DYN_MIN_FRAMES) finalizeDynamic();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current, finalizeDynamic, DYN_MIN_FRAMES]);

  const stableLabel = stableLabelRef.current;
  const sugg = suggestionsFor(decoder.bufferWord, vocab as string[], 6);

  const headerPred =
    current.type === "dynamic"
      ? dynResult
        ? `${dynResult.label} (${dynResult.score}%)`
        : dynActiveRef.current
          ? `Recording… (${dynBufRef.current.length}f)`
          : "—"
      : prediction;

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

        {current.type === "static" ? (
          <b>Confidence: {confidence}%</b>
        ) : (
          <b>Seq Score: {seqScore ?? "-"}</b>
        )}

        {current.type === "dynamic" && (
          <span style={{ color: "#999", fontFamily: "monospace" }}>{dynDebug}</span>
        )}

        <span style={{ color: advice ? "#d33" : "#0a0" }}>{advice || "View OK"}</span>

        <b style={{ color: "#0077cc" }}>
          {headerPred ? `Pred: ${headerPred}` : ""}
          {current.type === "static" && stableLabel ? ` (stable: ${stableLabel})` : ""}
        </b>
      </div>

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
  if (len <= 1 || seq.length <= 1) return seq.slice(0, Math.max(1, len));

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