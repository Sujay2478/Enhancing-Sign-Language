// src/logic/sentenceDecoder.ts
export type DecoderState = {
  transcript: string;   // committed words + spaces
  bufferWord: string;   // currently building word (letters/digits)
};

export function createDecoderState(): DecoderState {
  return { transcript: "", bufferWord: "" };
}

export function commitChar(state: DecoderState, ch: string): DecoderState {
  return { ...state, bufferWord: state.bufferWord + ch };
}

export function backspace(state: DecoderState): DecoderState {
  if (state.bufferWord.length > 0) {
    return { ...state, bufferWord: state.bufferWord.slice(0, -1) };
  }
  // backspace transcript (remove last char)
  if (state.transcript.length > 0) {
    return { ...state, transcript: state.transcript.slice(0, -1) };
  }
  return state;
}

export function commitSpace(state: DecoderState): DecoderState {
  // if bufferWord exists, commit it as a word first
  const word = state.bufferWord.trim();
  if (word) {
    const base = state.transcript;
    const next = base.length === 0 || base.endsWith(" ") ? base : base + " ";
    return { transcript: next + word + " ", bufferWord: "" };
  }
  // otherwise just add a space (avoid double spaces)
  if (state.transcript.endsWith(" ") || state.transcript.length === 0) return state;
  return { ...state, transcript: state.transcript + " " };
}

export function clearAll(): DecoderState {
  return createDecoderState();
}

export function commitSuggestion(state: DecoderState, word: string): DecoderState {
  const base = state.transcript;
  const next = base.length === 0 || base.endsWith(" ") ? base : base + " ";
  return { transcript: next + word + " ", bufferWord: "" };
}

export function suggestionsFor(bufferWord: string, vocab: string[], limit = 6): string[] {
  const w = bufferWord.trim().toLowerCase();
  if (!w) return [];

  // prefix matches first
  const prefix = vocab.filter((x) => x.startsWith(w));
  if (prefix.length >= limit) return prefix.slice(0, limit);

  // contains matches fallback
  const contains = vocab.filter((x) => !x.startsWith(w) && x.includes(w));
  return [...prefix, ...contains].slice(0, limit);
}
