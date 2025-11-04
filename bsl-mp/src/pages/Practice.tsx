import PracticePanel from "../components/PracticePanel";

export default function Practice() {
  return (
    <div style={{ padding: 16 }}>
      <h2>BSL Practice (MediaPipe)</h2>
      <PracticePanel />
      <p style={{ opacity: 0.7, marginTop: 12 }}>
        Tip: For dynamic signs, perform the motion for ~2 seconds then press <kbd>Space</kbd> to score the attempt.
      </p>
    </div>
  );
}
