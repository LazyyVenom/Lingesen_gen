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

const KEYPOINT_LEFT_EYE = 33;
const KEYPOINT_RIGHT_EYE = 263;
const KEYPOINT_NOSE = 1;

const HERO_FACE_TUNING = {
  hero1: {
    maskScale: 1.65,
    clipScale: 2.1,
    scale: 2.3,
    offsetX: 0,
    offsetY: 0,
    removeOriginal: true,
  },
  hero2: {
    removeOriginal: false,
  },
};
const DEFAULT_FACE_TUNING = {
  maskScale: 1.3,
  clipScale: 1.5,
  scale: 1.6,
  offsetX: 0,
  offsetY: 0,
  removeOriginal: true,
};

const TIMINGS = {
  DOCTOR_START: 600,
  DOCTOR_DURATION: 1400,
  MED_START: 1800,
  MED_DURATION: 800,
  HERO1_FADEOUT: 2600,
  FADEOUT_DURATION: 400,
  HERO2_START: 3000,
  TOTAL: 3500,
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const easeOutCubic = (t) => 1 - (1 - t) * (1 - t) * (1 - t);
const easeInOutQuad = (t) =>
  t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

const normalizePoint = (pt) => {
  if (!pt) return null;
  if (Number.isFinite(pt.x) && Number.isFinite(pt.y))
    return { x: pt.x, y: pt.y };
  if (Array.isArray(pt) || typeof pt.length === "number") {
    const x = pt[0];
    const y = pt[1];
    if (Number.isFinite(x) && Number.isFinite(y)) return { x, y };
  }
  return null;
};

const waitForHuman = () =>
  new Promise((resolve, reject) => {
    if (window?.Human?.Human) {
      resolve();
      return;
    }
    const timeoutAt = Date.now() + 8000;
    const check = () => {
      if (window?.Human?.Human) {
        resolve();
        return;
      }
      if (Date.now() > timeoutAt) {
        reject(new Error("Human.js failed to load."));
        return;
      }
      requestAnimationFrame(check);
    };
    check();
  });

const loadStaticImage = (src) =>
  new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load ${src}`));
    img.src = src;
  });

const loadImageFromFile = (file) =>
  new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = (err) => reject(err);
    img.src = url;
  });

const facePath = (ctx, pts) => {
  if (!pts?.length) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
};

const expandPolygon = (points, scale = 1) => {
  if (!points?.length) return [];
  const center = points.reduce(
    (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
    { x: 0, y: 0 }
  );
  center.x /= points.length;
  center.y /= points.length;
  return points.map((point) => ({
    x: center.x + (point.x - center.x) * scale,
    y: center.y + (point.y - center.y) * scale,
  }));
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

const prepareFaceData = (image, face) => {
  const w = image.naturalWidth || image.width;
  const h = image.naturalHeight || image.height;
  if (!w || !h) throw new Error("Unable to read image dimensions.");

  const mesh = face?.mesh;
  if (!Array.isArray(mesh) || mesh.length === 0) {
    throw new Error("Face landmarks missing. Try another photo.");
  }

  const normalizedMesh = mesh.map(normalizePoint);
  const validMesh = normalizedMesh.filter(Boolean);
  if (validMesh.length === 0) {
    throw new Error("Face landmarks invalid. Try another photo.");
  }

  const ovalPts = FACE_OVAL.map((index) => normalizedMesh[index]).filter(
    Boolean
  );

  const xs = validMesh.map((pt) => pt.x);
  const ys = validMesh.map((pt) => pt.y);
  const pad = Math.max(50, Math.round(Math.max(w, h) * 0.18));

  let minX = Math.max(0, Math.floor(Math.min(...xs) - pad));
  let maxX = Math.min(w, Math.ceil(Math.max(...xs) + pad));
  let minY = Math.max(0, Math.floor(Math.min(...ys) - pad));
  let maxY = Math.min(h, Math.ceil(Math.max(...ys) + pad));

  if (maxX <= minX) maxX = Math.min(w, minX + 4);
  if (maxY <= minY) maxY = Math.min(h, minY + 4);

  if (
    !Number.isFinite(minX) ||
    !Number.isFinite(maxX) ||
    !Number.isFinite(minY) ||
    !Number.isFinite(maxY)
  ) {
    throw new Error("Invalid face crop bounds. Try a clearer face photo.");
  }

  const cropWidth = Math.max(1, Math.round(maxX - minX));
  const cropHeight = Math.max(1, Math.round(maxY - minY));

  const canvas = document.createElement("canvas");
  canvas.width = cropWidth;
  canvas.height = cropHeight;
  const ctx = canvas.getContext("2d");

  ctx.save();
  ctx.translate(-minX, -minY);
  if (ovalPts.length >= 6) {
    const clipPts = expandPolygon(ovalPts, 1.5).map((pt) => ({
      x: pt.x - minX,
      y: pt.y - minY,
    }));
    facePath(ctx, clipPts);
    ctx.clip();
  }
  ctx.drawImage(image, 0, 0);
  ctx.restore();

  const fetchPoint = (index) => {
    const pt = normalizedMesh[index];
    if (!pt) throw new Error("Missing facial landmarks for alignment.");
    return { x: pt.x - minX, y: pt.y - minY };
  };

  return {
    canvas,
    srcPoints: {
      leftEye: fetchPoint(KEYPOINT_LEFT_EYE),
      rightEye: fetchPoint(KEYPOINT_RIGHT_EYE),
      nose: fetchPoint(KEYPOINT_NOSE),
    },
  };
};

const detectPrimaryFace = async (human, image) => {
  const result = await human.detect(image);
  return result.face?.[0] ?? null;
};

const drawContainIntoBox = (ctx, img, box) => {
  const imgWidth = img.naturalWidth || img.width;
  const imgHeight = img.naturalHeight || img.height;
  if (!imgWidth || !imgHeight) return;
  const ratio = Math.min(box.width / imgWidth, box.height / imgHeight);
  const dw = imgWidth * ratio;
  const dh = imgHeight * ratio;
  const dx = box.x + (box.width - dw) / 2;
  const dy = box.y + (box.height - dh) / 2;
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, dx, dy, dw, dh);
  ctx.restore();
};

const computeLayout = (width, height, villainImg, medImg) => {
  const centerHeroBox = {
    x: width * 0.33,
    y: height * 0.18,
    width: width * 0.34,
    height: height * 0.68,
  };

  const villainRatio =
    (villainImg?.naturalHeight || villainImg?.height || 1) /
    (villainImg?.naturalWidth || villainImg?.width || 1);
  const villainWidth = width * 0.18;
  const villainHeight = villainWidth * villainRatio;

  const medRatio =
    (medImg?.naturalHeight || medImg?.height || 1) /
    (medImg?.naturalWidth || medImg?.width || 1);
  const medWidth = width * 0.12;
  const medHeight = medWidth * medRatio;

  return {
    hero1Box: centerHeroBox,
    hero2Box: centerHeroBox,
    doctor: {
      width: villainWidth,
      height: villainHeight,
      startX: width + villainWidth * 0.1,
      targetX: centerHeroBox.x + centerHeroBox.width + width * 0.05,
      y: centerHeroBox.y + centerHeroBox.height - villainHeight,
    },
    med: {
      width: medWidth,
      height: medHeight,
      startX: width + medWidth * 0.1,
      targetX: centerHeroBox.x + centerHeroBox.width * 0.5,
      y: centerHeroBox.y + centerHeroBox.height * 0.3,
    },
  };
};

function App() {
  const canvasRef = useRef(null);
  const humanRef = useRef(null);
  const humanReadyRef = useRef(false);
  const heroDataRef = useRef({ hero1: null, hero2: null });
  const assetsRef = useRef({ bg: null, villain: null, med: null });
  const compositesRef = useRef({ hero1: null, hero2: null });
  const layoutRef = useRef(null);
  const animationRef = useRef({ raf: null, playing: false, start: 0 });
  const latestFaceRef = useRef(null);

  const [isBootstrapped, setIsBootstrapped] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [initError, setInitError] = useState(null);
  const [hasFace, setHasFace] = useState(false);

  const getHuman = async () => {
    if (!humanRef.current) {
      await waitForHuman();
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

  const stopAnimation = () => {
    if (animationRef.current.raf) {
      cancelAnimationFrame(animationRef.current.raf);
      animationRef.current.raf = null;
    }
    animationRef.current.playing = false;
  };

  const drawBackground = () => {
    const canvas = canvasRef.current;
    const bg = assetsRef.current.bg;
    if (!canvas || !bg) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);
  };

  const buildHeroComposite = (key, faceData) => {
    const heroEntry = heroDataRef.current[key];
    if (!heroEntry?.image) return null;
    const heroFace = heroEntry.face;
    const heroImage = heroEntry.image;
    const width = heroImage.naturalWidth || heroImage.width;
    const height = heroImage.naturalHeight || heroImage.height;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(heroImage, 0, 0, width, height);

    const mesh = heroFace?.mesh?.map(normalizePoint) || [];
    const left = mesh[KEYPOINT_LEFT_EYE];
    const right = mesh[KEYPOINT_RIGHT_EYE];
    const nose = mesh[KEYPOINT_NOSE];
    if (!left || !right || !nose) return canvas;

    const heroOval = FACE_OVAL.map((index) => mesh[index]).filter(Boolean);
    const tuning = {
      ...DEFAULT_FACE_TUNING,
      ...HERO_FACE_TUNING[key],
    };

    if (heroOval.length && tuning.removeOriginal) {
      const expanded = expandPolygon(heroOval, tuning.maskScale);
      ctx.save();
      ctx.globalCompositeOperation = "destination-out";
      facePath(ctx, expanded);
      ctx.fill();
      ctx.restore();
      ctx.globalCompositeOperation = "source-over";
    }

    const matrix = affineFrom3Points(
      faceData.srcPoints.leftEye,
      faceData.srcPoints.rightEye,
      faceData.srcPoints.nose,
      left,
      right,
      nose
    );

    ctx.save();
    if (heroOval.length && tuning.clipScale) {
      const clipShape = expandPolygon(heroOval, tuning.clipScale);
      facePath(ctx, clipShape);
      ctx.clip();
    }
    if (tuning.scale && Math.abs(tuning.scale - 1) > 0.01) {
      ctx.translate(nose.x, nose.y);
      ctx.scale(tuning.scale, tuning.scale);
      ctx.translate(-nose.x, -nose.y);
    }
    ctx.transform(
      matrix[0],
      matrix[1],
      matrix[2],
      matrix[3],
      matrix[4],
      matrix[5]
    );
    if (tuning.offsetX || tuning.offsetY) {
      ctx.translate(tuning.offsetX || 0, tuning.offsetY || 0);
    }
    ctx.drawImage(faceData.canvas, 0, 0);
    ctx.restore();

    return canvas;
  };

  const drawFrame = (elapsed) => {
    const canvas = canvasRef.current;
    const layout = layoutRef.current;
    const heroes = compositesRef.current;
    const { bg, villain, med } = assetsRef.current;
    if (!canvas || !layout || !bg) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);

    if (!heroes.hero1) return;

    // Draw hero1 with fade out effect
    if (elapsed < TIMINGS.HERO2_START) {
      ctx.save();
      if (elapsed >= TIMINGS.HERO1_FADEOUT) {
        const fadeProgress = clamp(
          (elapsed - TIMINGS.HERO1_FADEOUT) / TIMINGS.FADEOUT_DURATION,
          0,
          1
        );
        ctx.globalAlpha = 1 - fadeProgress;
      }
      drawContainIntoBox(ctx, heroes.hero1, layout.hero1Box);
      ctx.restore();
    }

    // Doctor slides in from right
    if (
      elapsed >= TIMINGS.DOCTOR_START &&
      elapsed < TIMINGS.HERO2_START &&
      villain
    ) {
      const progress = clamp(
        (elapsed - TIMINGS.DOCTOR_START) / TIMINGS.DOCTOR_DURATION,
        0,
        1
      );
      const eased = easeOutCubic(progress);
      const x =
        layout.doctor.startX +
        (layout.doctor.targetX - layout.doctor.startX) * eased;
      ctx.drawImage(
        villain,
        x,
        layout.doctor.y,
        layout.doctor.width,
        layout.doctor.height
      );
    }

    // Med moves from doctor to hero1
    if (elapsed >= TIMINGS.MED_START && elapsed < TIMINGS.HERO2_START && med) {
      const progress = clamp(
        (elapsed - TIMINGS.MED_START) / TIMINGS.MED_DURATION,
        0,
        1
      );
      const eased = easeInOutQuad(progress);
      const x =
        layout.med.startX + (layout.med.targetX - layout.med.startX) * eased;
      ctx.drawImage(
        med,
        x - layout.med.width / 2,
        layout.med.y,
        layout.med.width,
        layout.med.height
      );
    }

    // Hero2 appears in center after hero1 fades
    if (elapsed >= TIMINGS.HERO2_START && heroes.hero2) {
      drawContainIntoBox(ctx, heroes.hero2, layout.hero2Box);
    }
  };

  const startSequence = () => {
    stopAnimation();
    animationRef.current.playing = true;
    animationRef.current.start = performance.now();
    drawFrame(0);

    const tick = (time) => {
      const elapsed = time - animationRef.current.start;
      drawFrame(elapsed);
      if (elapsed < TIMINGS.TOTAL) {
        animationRef.current.raf = requestAnimationFrame(tick);
      } else {
        animationRef.current.playing = false;
        drawFrame(TIMINGS.TOTAL);
      }
    };

    animationRef.current.raf = requestAnimationFrame(tick);
  };

  useEffect(() => () => stopAnimation(), []);

  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      try {
        const human = await getHuman();
        const [bg, hero1, hero2, villain, med] = await Promise.all([
          loadStaticImage("/bg.jpg"),
          loadStaticImage("/hero1.png"),
          loadStaticImage("/hero2.png"),
          loadStaticImage("/villian.png"),
          loadStaticImage("/med.png"),
        ]);
        if (cancelled) return;

        const hero1Face = await detectPrimaryFace(human, hero1);
        const hero2Face = await detectPrimaryFace(human, hero2);

        heroDataRef.current.hero1 = { image: hero1, face: hero1Face };
        heroDataRef.current.hero2 = { image: hero2, face: hero2Face };
        assetsRef.current = { bg, villain, med };

        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = bg.naturalWidth || bg.width;
        canvas.height = bg.naturalHeight || bg.height;

        layoutRef.current = computeLayout(
          canvas.width,
          canvas.height,
          villain,
          med
        );
        drawBackground();

        if (!hero1Face || !hero2Face) {
          setInitError(
            "Could not find every face in the hero images. Results may look goofy."
          );
        }

        if (latestFaceRef.current) {
          const faceData = latestFaceRef.current;
          compositesRef.current.hero1 = buildHeroComposite("hero1", faceData);
          compositesRef.current.hero2 =
            heroDataRef.current.hero2?.image ?? null;
          drawFrame(0);
          setHasFace(true);
        }

        setIsBootstrapped(true);
      } catch (err) {
        console.error(err);
        if (!cancelled) setInitError(err.message || "Failed to load assets.");
      }
    };

    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleFaceUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(null);
    setIsProcessing(true);
    setHasFace(false);

    let loaded = null;
    try {
      if (!isBootstrapped) throw new Error("Scene is still loading.");

      const human = await getHuman();
      loaded = await loadImageFromFile(file);
      const detection = await human.detect(loaded.img);
      const face = detection.face?.[0];
      if (!face) throw new Error("No face detected in that photo.");

      const faceData = prepareFaceData(loaded.img, face);
      latestFaceRef.current = faceData;

      const hero1Composite = buildHeroComposite("hero1", faceData);
      if (!hero1Composite) throw new Error("Failed to align face with hero 1.");

      compositesRef.current.hero1 = hero1Composite;
      compositesRef.current.hero2 = heroDataRef.current.hero2?.image ?? null;
      setHasFace(true);
      startSequence();
    } catch (err) {
      console.error(err);
      setError(err.message || "Face transfer failed.");
      drawBackground();
    } finally {
      setIsProcessing(false);
      if (loaded?.url) URL.revokeObjectURL(loaded.url);
    }
  };

  return (
    <div className="app">
      <label className="upload-card" htmlFor="faceInput">
        <input
          id="faceInput"
          type="file"
          accept="image/*"
          onChange={handleFaceUpload}
          disabled={isProcessing}
        />
      </label>

      {isProcessing && <p className="status">Processing your photo…</p>}
      {error && <p className="error">{error}</p>}
      {initError && <p className="warning">{initError}</p>}

      <div className="stage">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}

export default App;
