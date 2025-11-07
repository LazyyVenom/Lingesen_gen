import { useEffect, useRef, useState } from "react";
import "./App.css";

const HUMAN_CONFIG = {
  modelBasePath: "https://cdn.jsdelivr.net/npm/@vladmandic/human/models",
  cacheSensitivity: 0,
  warmup: "none",
  face: {
    enabled: true,
    detector: { rotation: true },
    mesh: { enabled: true, refineLandmarks: true },
  },
  body: { enabled: false },
  hand: { enabled: false },
  gesture: { enabled: false },
  object: { enabled: false },
  segmentation: { enabled: false },
  filter: { enabled: true },
};

const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109, 10,
];

const SRC_LEFT_EYE = 33;
const SRC_RIGHT_EYE = 263;
const SRC_NOSE = 1;

const easeInOutQuad = (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t);

const loadImageFromFile = (file) =>
  new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = (err) => reject(err);
    img.src = url;
  });

const drawContain = (ctx, img, cw, ch) => {
  const ratio = Math.min(cw / img.width, ch / img.height);
  const w = img.width * ratio;
  const h = img.height * ratio;
  const x = (cw - w) / 2;
  const y = (ch - h) / 2;
  ctx.drawImage(img, x, y, w, h);
  return { x, y, w, h, ratio };
};

const affineFrom3Points = (src1, src2, src3, dst1, dst2, dst3) => {
  const A = [
    [src1.x, src1.y, 1, 0, 0, 0],
    [0, 0, 0, src1.x, src1.y, 1],
    [src2.x, src2.y, 1, 0, 0, 0],
    [0, 0, 0, src2.x, src2.y, 1],
    [src3.x, src3.y, 1, 0, 0, 0],
    [0, 0, 0, src3.x, src3.y, 1],
  ];
  const b = [dst1.x, dst1.y, dst2.x, dst2.y, dst3.x, dst3.y];

  const M = A.map((row, i) => [...row, b[i]]);
  const n = 6;
  for (let i = 0; i < n; i += 1) {
    let maxRow = i;
    for (let k = i + 1; k < n; k += 1) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];
    const pivot = M[i][i] || 1e-12;
    for (let j = i; j <= n; j += 1) M[i][j] /= pivot;
    for (let k = 0; k < n; k += 1) {
      if (k === i) continue;
      const factor = M[k][i];
      for (let j = i; j <= n; j += 1) M[k][j] -= factor * M[i][j];
    }
  }

  const a = M[0][n];
  const c = M[1][n];
  const e = M[2][n];
  const bRes = M[3][n];
  const d = M[4][n];
  const f = M[5][n];
  return [a, bRes, c, d, e, f];
};

const facePath = (ctx, pts) => {
  ctx.beginPath();
  const start = pts[0];
  ctx.moveTo(start.x, start.y);
  for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
};

function App() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const humanRef = useRef(null);
  const humanReadyRef = useRef(false);
  const audioInputRef = useRef(null);
  const rafRef = useRef(null);

  const stateRef = useRef({
    src: null,
    dst: null,
    srcDet: null,
    dstDet: null,
    faceMaskCanvas: null,
    audio: null,
    audioUrl: null,
    ready: false,
  });

  const [hasSource, setHasSource] = useState(false);
  const [hasTarget, setHasTarget] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [ready, setReady] = useState(false);
  const [downloadEnabled, setDownloadEnabled] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    ctxRef.current = canvas.getContext("2d");

    const faceMaskCanvas = document.createElement("canvas");
    faceMaskCanvas.width = canvas.width;
    faceMaskCanvas.height = canvas.height;
    stateRef.current.faceMaskCanvas = faceMaskCanvas;

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const { src, dst, audio, audioUrl } = stateRef.current;
      if (src?.url) URL.revokeObjectURL(src.url);
      if (dst?.url) URL.revokeObjectURL(dst.url);
      if (audio) {
        try {
          audio.pause();
        } catch (err) {
          console.error(err);
        }
      }
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, []);

  const getHuman = async () => {
    if (!humanRef.current) {
      if (!window?.Human?.Human) throw new Error("Human.js failed to load.");
      humanRef.current = new window.Human.Human(HUMAN_CONFIG);
    }
    const human = humanRef.current;
    if (!humanReadyRef.current) {
      await human.load();
      await human.warmup();
      humanReadyRef.current = true;
    }
    return human;
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const renderPreview = () => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    const state = stateRef.current;
    if (!canvas || !ctx) return;
    clearCanvas();
    if (!state.src && !state.dst) return;
    const half = canvas.width / 2;
    ctx.save();
    if (state.dst) drawContain(ctx, state.dst.img, half, canvas.height);
    ctx.translate(half, 0);
    if (state.src) drawContain(ctx, state.src.img, half, canvas.height);
    ctx.restore();
  };

  const handleImageChange = (key) => async (event) => {
    const file = event.target.files?.[0];
    const state = stateRef.current;

    if (state[key]?.url) URL.revokeObjectURL(state[key].url);
    state[key] = null;
    state.ready = false;
    state.srcDet = key === "src" ? null : state.srcDet;
    state.dstDet = key === "dst" ? null : state.dstDet;
    setReady(false);
    setDownloadEnabled(false);

    if (!file) {
      if (key === "src") setHasSource(false);
      if (key === "dst") setHasTarget(false);
      renderPreview();
      return;
    }

    try {
      const loaded = await loadImageFromFile(file);
      state[key] = loaded;
      if (key === "src") setHasSource(true);
      if (key === "dst") setHasTarget(true);
      renderPreview();
    } catch (err) {
      console.error(err);
      window.alert("Failed to load selected image.");
      if (key === "src") setHasSource(false);
      if (key === "dst") setHasTarget(false);
    }
  };

  const handleAudioChange = () => {
    const state = stateRef.current;
    if (state.audio) {
      try {
        state.audio.pause();
      } catch (err) {
        console.error(err);
      }
      state.audio = null;
    }
    if (state.audioUrl) {
      URL.revokeObjectURL(state.audioUrl);
      state.audioUrl = null;
    }
  };

  const playAudio = () => {
    const state = stateRef.current;
    const file = audioInputRef.current?.files?.[0];
    if (!file) return null;
    if (state.audio) {
      try {
        state.audio.pause();
      } catch (err) {
        console.error(err);
      }
      state.audio = null;
    }
    if (state.audioUrl) {
      URL.revokeObjectURL(state.audioUrl);
      state.audioUrl = null;
    }
    const url = URL.createObjectURL(file);
    const audio = new Audio(url);
    audio.play().catch(() => {
      // Autoplay may be blocked until the user interacts with the page.
    });
    audio.addEventListener("ended", () => {
      URL.revokeObjectURL(url);
      if (state.audioUrl === url) state.audioUrl = null;
    });
    state.audio = audio;
    state.audioUrl = url;
    return audio;
  };

  const detectAll = async () => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    const state = stateRef.current;
    const faceMaskCanvas = state.faceMaskCanvas;
    if (!canvas || !ctx || !faceMaskCanvas) return;

    clearCanvas();
    const half = canvas.width / 2;
    ctx.save();
    if (state.dst) drawContain(ctx, state.dst.img, half, canvas.height);
    ctx.translate(half, 0);
    if (state.src) drawContain(ctx, state.src.img, half, canvas.height);
    ctx.restore();

    const human = await getHuman();

    state.srcDet = state.src ? await human.detect(state.src.img) : null;
    state.dstDet = state.dst ? await human.detect(state.dst.img) : null;

    const srcFace = state.srcDet?.face?.[0];
    const dstFace = state.dstDet?.face?.[0];
    if (!srcFace) throw new Error("No face found in source image.");
    if (!dstFace) throw new Error("No face found in target image.");

    faceMaskCanvas.width = canvas.width;
    faceMaskCanvas.height = canvas.height;
    const maskCtx = faceMaskCanvas.getContext("2d");
    maskCtx.clearRect(0, 0, canvas.width, canvas.height);

    const tmp = document.createElement("canvas");
    tmp.width = canvas.width;
    tmp.height = canvas.height;
    const tctx = tmp.getContext("2d");

    const rightLayout = drawContain(tctx, state.src.img, half, canvas.height);
    const srcOffsetX = half + rightLayout.x;
    const srcOffsetY = rightLayout.y;
    const srcScale = rightLayout.ratio;

    const mesh = srcFace.mesh.map((p) => ({
      x: srcOffsetX + p.x * srcScale,
      y: srcOffsetY + p.y * srcScale,
    }));

    const ovalPts = FACE_OVAL.map((index) => mesh[index]);
    maskCtx.save();
    facePath(maskCtx, ovalPts);
    maskCtx.clip();
    maskCtx.drawImage(
      state.src.img,
      srcOffsetX,
      srcOffsetY,
      state.src.img.width * srcScale,
      state.src.img.height * srcScale
    );
    maskCtx.restore();

    state.ready = true;
    setReady(true);
  };

  const animatePaste = () =>
    new Promise((resolve, reject) => {
      const canvas = canvasRef.current;
      const ctx = ctxRef.current;
      const state = stateRef.current;
      if (!canvas || !ctx) {
        resolve();
        return;
      }
      if (!state.ready) {
        reject(new Error("Run Detect & Prepare first."));
        return;
      }

      if (!state.srcDet?.face?.[0] || !state.dstDet?.face?.[0]) {
        reject(
          new Error("Missing detection data. Run Detect & Prepare again.")
        );
        return;
      }

      const half = canvas.width / 2;

      const tmp = document.createElement("canvas");
      tmp.width = canvas.width;
      tmp.height = canvas.height;
      const tctx = tmp.getContext("2d");

      const dstLayout = drawContain(tctx, state.dst.img, half, canvas.height);
      const dstOffsetX = dstLayout.x;
      const dstOffsetY = dstLayout.y;
      const dstScale = dstLayout.ratio;

      const dstMesh = state.dstDet.face[0].mesh.map((p) => ({
        x: dstOffsetX + p.x * dstScale,
        y: dstOffsetY + p.y * dstScale,
      }));

      const srcLayout = drawContain(tctx, state.src.img, half, canvas.height);
      const srcOffsetX = half + srcLayout.x;
      const srcOffsetY = srcLayout.y;
      const srcScale = srcLayout.ratio;

      const srcMesh = state.srcDet.face[0].mesh.map((p) => ({
        x: srcOffsetX + p.x * srcScale,
        y: srcOffsetY + p.y * srcScale,
      }));

      const src1 = srcMesh[SRC_LEFT_EYE];
      const src2 = srcMesh[SRC_RIGHT_EYE];
      const src3 = srcMesh[SRC_NOSE];

      const dst1 = dstMesh[SRC_LEFT_EYE];
      const dst2 = dstMesh[SRC_RIGHT_EYE];
      const dst3 = dstMesh[SRC_NOSE];

      const finalMatrix = affineFrom3Points(src1, src2, src3, dst1, dst2, dst3);

      const startMatrix = [
        finalMatrix[0],
        finalMatrix[1],
        finalMatrix[2],
        finalMatrix[3],
        finalMatrix[4] - 400,
        finalMatrix[5] - 300,
      ];

      const duration = 1200;
      const t0 = performance.now();
      const faceMask = state.faceMaskCanvas;

      playAudio();

      if (rafRef.current) cancelAnimationFrame(rafRef.current);

      const frame = (now) => {
        const t = Math.min(1, (now - t0) / duration);
        const ease = easeInOutQuad(t);

        clearCanvas();
        drawContain(ctx, state.dst.img, half, canvas.height);
        ctx.save();
        ctx.translate(half, 0);
        drawContain(ctx, state.src.img, half, canvas.height);
        ctx.restore();

        const m = [
          startMatrix[0] + (finalMatrix[0] - startMatrix[0]) * ease,
          startMatrix[1] + (finalMatrix[1] - startMatrix[1]) * ease,
          startMatrix[2] + (finalMatrix[2] - startMatrix[2]) * ease,
          startMatrix[3] + (finalMatrix[3] - startMatrix[3]) * ease,
          startMatrix[4] + (finalMatrix[4] - startMatrix[4]) * ease,
          startMatrix[5] + (finalMatrix[5] - startMatrix[5]) * ease,
        ];

        ctx.save();
        ctx.transform(m[0], m[1], m[2], m[3], m[4], m[5]);
        ctx.drawImage(faceMask, 0, 0);
        ctx.restore();

        if (t < 1) {
          rafRef.current = requestAnimationFrame(frame);
        } else {
          rafRef.current = null;
          resolve();
        }
      };

      rafRef.current = requestAnimationFrame(frame);
    });

  const handleDetect = async () => {
    stateRef.current.ready = false;
    setReady(false);
    setDownloadEnabled(false);
    try {
      setIsDetecting(true);
      await detectAll();
      window.alert("Faces detected and mask prepared. Ready to animate.");
    } catch (err) {
      console.error(err);
      window.alert(err.message || "Detection failed.");
      stateRef.current.ready = false;
      setReady(false);
      setDownloadEnabled(false);
    } finally {
      setIsDetecting(false);
    }
  };

  const handleAnimate = async () => {
    try {
      setIsAnimating(true);
      await animatePaste();
      setDownloadEnabled(true);
    } catch (err) {
      console.error(err);
      window.alert(err.message || "Animation failed.");
    } finally {
      setIsAnimating(false);
    }
  };

  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = "face-paste-result.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  const canDetect = hasSource && hasTarget && !isDetecting;
  const canAnimate = ready && !isAnimating;

  return (
    <div className="app">
      <h1>Face Segmentation -&gt; Animated Paste with Audio</h1>
      <p>
        Upload a source face and a target image, then animate the segmented face
        into the target with optional audio playback.
      </p>

      <div className="row">
        <div className="card col">
          <label htmlFor="srcInput">Source image (face to extract)</label>
          <input
            id="srcInput"
            type="file"
            accept="image/*"
            onChange={handleImageChange("src")}
          />
        </div>
        <div className="card col">
          <label htmlFor="dstInput">Target image (where the face goes)</label>
          <input
            id="dstInput"
            type="file"
            accept="image/*"
            onChange={handleImageChange("dst")}
          />
        </div>
        <div className="card col">
          <label htmlFor="audioInput">Audio (mp3/wav/ogg) - optional</label>
          <input
            id="audioInput"
            ref={audioInputRef}
            type="file"
            accept="audio/*"
            onChange={handleAudioChange}
          />
          <small className="code">If empty, no audio plays.</small>
        </div>
      </div>

      <div className="row actions">
        <button
          type="button"
          className="btn"
          disabled={!canDetect}
          onClick={handleDetect}
        >
          {isDetecting ? "Detecting..." : "Detect & Prepare"}
        </button>
        <button
          type="button"
          className="btn"
          disabled={!canAnimate}
          onClick={handleAnimate}
        >
          {isAnimating ? "Animating..." : "Animate & Play"}
        </button>
        <button
          type="button"
          className="btn"
          disabled={!downloadEnabled}
          onClick={handleDownload}
        >
          Download Result (PNG)
        </button>
      </div>

      <div className="row preview">
        <div className="col">
          <label htmlFor="canvas">Preview</label>
          <canvas id="canvas" ref={canvasRef} width={1040} height={780} />
          <small className="code">
            Canvas is 1040x780 for clarity. Images are fit to canvas.
          </small>
        </div>
      </div>
    </div>
  );
}

export default App;
