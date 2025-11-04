import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ["onnxruntime-web"], // prevent bundling ONNXRuntime
  },
  assetsInclude: ["**/*.wasm", "**/*.mjs"],
  server: {
    watch: { usePolling: true },
    hmr: { overlay: false },
    fs: {
      strict: false, // allow reading from /public without rewriting imports
    },
    mimeTypes: {
      "application/javascript": ["mjs"],
    },
  },
  build: {
    rollupOptions: {
      external: [
        "/ort/ort-wasm-simd-threaded.jsep.mjs",
        "/ort/ort-wasm-simd-threaded.wasm",
        "/ort/ort-wasm.wasm",
        "/ort/ort-wasm-simd.wasm",
      ],
    },
  },
});
