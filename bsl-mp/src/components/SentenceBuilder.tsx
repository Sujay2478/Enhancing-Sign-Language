// src/components/SentenceBuilder.tsx
import React from "react";

export default function SentenceBuilder(props: {
  transcript: string;
  bufferWord: string;
  suggestions: string[];
  onCommitSpace: () => void;
  onBackspace: () => void;
  onClear: () => void;
  onPickSuggestion: (w: string) => void;
}) {
  const {
    transcript,
    bufferWord,
    suggestions,
    onCommitSpace,
    onBackspace,
    onClear,
    onPickSuggestion,
  } = props;

  return (
    <div
      style={{
        padding: 10,
        border: "1px solid #333",
        borderRadius: 8,
        background: "#111",
        display: "grid",
        gap: 10,
        maxWidth: 960,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
        <b style={{ color: "#ddd" }}>Sentence Builder</b>
        <span style={{ color: "#888" }}>
          Hold stable sign = add letter • Space=commit word • Backspace=delete
        </span>
      </div>

      <div
        style={{
          minHeight: 44,
          padding: "10px 12px",
          borderRadius: 8,
          background: "#1a1a1a",
          color: "#fff",
          fontSize: 18,
          wordBreak: "break-word",
        }}
      >
        {transcript || <span style={{ color: "#777" }}>Start signing…</span>}
        <span style={{ color: "#60a5fa" }}>{bufferWord}</span>
      </div>

      {suggestions.length > 0 && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {suggestions.map((w) => (
            <button key={w} onClick={() => onPickSuggestion(w)} style={chipStyle}>
              {w}
            </button>
          ))}
        </div>
      )}

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button onClick={onCommitSpace} style={btnStyle}>
          Space / Commit word
        </button>
        <button onClick={onBackspace} style={btnStyle}>
          Backspace
        </button>
        <button onClick={onClear} style={{ ...btnStyle, borderColor: "#444" }}>
          Clear
        </button>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: "8px 12px",
  borderRadius: 8,
  border: "1px solid #333",
  background: "#1f1f1f",
  color: "#eee",
  cursor: "pointer",
};

const chipStyle: React.CSSProperties = {
  padding: "6px 10px",
  borderRadius: 999,
  border: "1px solid #2b2b2b",
  background: "#181818",
  color: "#e5e5e5",
  cursor: "pointer",
};
