import { useEffect, useRef, useState } from "react";

export default function CameraFeed({
  onReady,
}: { onReady: (video: HTMLVideoElement) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        const v = videoRef.current!;
        v.srcObject = stream;
        await v.play();
        onReady(v);
      } catch (e: any) {
        setErr(e?.message ?? "Camera permission denied");
      }
    })();
    return () => { stream && stream.getTracks().forEach((t) => t.stop()); };
  }, [onReady]);

  return (
    <div style={{ position: "relative" }}>
      {err && <div style={{ color: "red" }}>{err}</div>}
      <video ref={videoRef} playsInline style={{ width: "100%", transform: "scaleX(-1)" }} />
    </div>
  );
}
