/**
 * preprocessor.js — Tight non-square crop + convex hull overlay
 *
 * Pipeline:
 *  1. Mirror video frame
 *  2. MediaPipe Hands (selfieMode) → landmarks
 *  3. Tight bounding box (natural aspect ratio, no forced square)
 *  4. Convex hull polygon drawn on display canvas
 *  5. Letterbox crop → 224×224 (preserves proportions)
 *  6. [NEW] Wrist-relative, scale-normalized landmark flattening
 *  7. [NEW] Palm orientation detection (palm vs back of hand)
 *
 * ─────────────────────────────────────────────────────────────
 * ⚠️  IMPORTANT — NORMALIZATION & MODEL RETRAINING
 * ─────────────────────────────────────────────────────────────
 * This file now uses _normalizeLandmarks() instead of raw MediaPipe
 * coordinates. The normalization makes the model invariant to:
 *   • Hand position anywhere in the camera frame
 *   • Distance to the camera (scale)
 *
 * Because of this change, the existing model.json / group1-shard1of1.bin
 * WILL produce wrong predictions until it is RETRAINED with the same
 * normalization applied to all training samples.
 *
 * Training-side pseudocode (Python / NumPy):
 *   wrist  = landmarks[0]           # shape (3,)
 *   midMCP = landmarks[9]           # shape (3,)
 *   scale  = np.linalg.norm(midMCP - wrist)
 *   norm   = (landmarks - wrist) / scale   # shape (21, 3)
 *   flat   = norm.flatten()                # shape (63,)
 *
 * If you want to keep the old model temporarily, set
 * USE_NORMALIZED_LANDMARKS = false below (reverts to raw coords).
 * ─────────────────────────────────────────────────────────────
 */

const PreprocessorModule = (() => {

  /* ── Constants ─────────────────────────────────────────── */
  const CROP_SIZE     = 224;
  const MIN_PADDING   = 28;
  const PAD_RATIO     = 0.05;
  const MOTION_THRESH = 0.07;

  /**
   * Toggle between normalized (new) and raw (legacy) landmark encoding.
   *  true  → wrist-relative + scale-normalized  (requires retrained model)
   *  false → raw MediaPipe coords 0–1           (original behaviour)
   */
  const USE_NORMALIZED_LANDMARKS = true;

  /* ── Private state ─────────────────────────────────────── */
  let _hands      = null;
  let _lastResult = null;
  let _ready      = false;
  let _prevCenter = null; // {x, y} pixel coords, for motion detection

  /* ── Off-screen canvases ────────────────────────────────── */
  const _mirrorCanvas = document.createElement('canvas');
  const _mirrorCtx    = _mirrorCanvas.getContext('2d');
  const _cropCanvas   = document.createElement('canvas');
  const _cropCtx      = _cropCanvas.getContext('2d');
  _cropCanvas.width   = CROP_SIZE;
  _cropCanvas.height  = CROP_SIZE;

  /* ══════════════════════════════════════════════════════════
   * PRIVATE HELPERS
   * ══════════════════════════════════════════════════════════ */

  /**
   * Graham-scan convex hull of 2-D points [{x, y}].
   * Used for the overlay polygon on the display canvas.
   */
  function _convexHull(pts) {
    if (pts.length < 3) return pts;
    const s = pts.slice().sort((a, b) => a.x !== b.x ? a.x - b.x : a.y - b.y);
    const cross = (O, A, B) =>
      (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);

    const lower = [];
    for (const p of s) {
      while (lower.length >= 2 &&
             cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
        lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = s.length - 1; i >= 0; i--) {
      const p = s[i];
      while (upper.length >= 2 &&
             cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
        upper.pop();
      upper.push(p);
    }
    upper.pop();
    lower.pop();
    return lower.concat(upper);
  }

  /**
   * _normalizeLandmarks
   * ─────────────────────────────────────────────────────────
   * Converts 21 raw MediaPipe landmarks into a 63-element
   * wrist-relative, scale-normalised flat array.
   *
   * Why this matters:
   *  • Raw MediaPipe x/y are ABSOLUTE positions within the
   *    camera frame (0–1). If the hand moves to the left edge
   *    vs the centre, every x-value shifts — the gesture is
   *    the same but the numbers are completely different.
   *    This forces the MLP to memorise positions rather than
   *    shapes, which causes the confusion errors you see.
   *
   *  • By subtracting the wrist and dividing by the
   *    wrist→middle-MCP distance we get coordinates that are
   *    purely about finger configuration, regardless of where
   *    in the frame the hand sits or how close it is.
   *
   * Normalization steps:
   *  1. Translate  → subtract wrist (landmark 0) from all points
   *  2. Scale      → divide by Euclidean distance
   *                  wrist (lm 0) → middle finger MCP (lm 9)
   *  3. Mirror X   → negate x for physical left hand so the MLP
   *                  always sees a right-hand-equivalent input
   *
   * @param {Array<{x,y,z}>} landmarks  - 21-element MediaPipe array
   * @param {boolean}        isLeftHand - physical left hand flag
   * @returns {number[]} 63-element flat array [x0,y0,z0, x1,y1,z1 …]
   */
  function _normalizeLandmarks(landmarks, isLeftHand) {
    const wrist  = landmarks[0]; // origin
    const midMCP = landmarks[9]; // middle finger MCP — stable scale reference

    // Euclidean distance in MediaPipe normalised space
    const refDist = Math.sqrt(
      (midMCP.x - wrist.x) ** 2 +
      (midMCP.y - wrist.y) ** 2 +
      (midMCP.z - wrist.z) ** 2
    );

    // Guard against division by near-zero (hand barely in frame)
    const scale = refDist > 0.001 ? refDist : 1.0;

    const flat = [];
    for (let i = 0; i < landmarks.length; i++) {
      // Step 1 & 2: translate to wrist origin, then scale
      let x = (landmarks[i].x - wrist.x) / scale;
      let y = (landmarks[i].y - wrist.y) / scale;
      let z = (landmarks[i].z - wrist.z) / scale;

      // Step 3: mirror X for left hand
      if (isLeftHand) x = -x;

      flat.push(x, y, z);
    }
    return flat;
  }

  /**
   * _flattenRaw  (legacy — used when USE_NORMALIZED_LANDMARKS = false)
   * Original behaviour: raw MediaPipe coords with left-hand X mirror.
   *
   * @param {Array<{x,y,z}>} landmarks
   * @param {boolean}        isLeftHand
   * @returns {number[]} 63-element flat array
   */
  function _flattenRaw(landmarks, isLeftHand) {
    const flat = [];
    for (let i = 0; i < landmarks.length; i++) {
      const x = isLeftHand ? 1.0 - landmarks[i].x : landmarks[i].x;
      flat.push(x, landmarks[i].y, landmarks[i].z);
    }
    return flat;
  }

  /**
   * _isPalmFacing
   * ─────────────────────────────────────────────────────────
   * Determines whether the camera is seeing the PALM or the
   * BACK of the hand, using the 3D cross product of two
   * vectors that lie in the palm plane.
   *
   * How it works:
   *  • Vector A: wrist (lm 0) → index finger MCP (lm 5)
   *  • Vector B: wrist (lm 0) → pinky MCP (lm 17)
   *  • We calculate the 3D normal vector (A x B)
   *  • Since we are using MediaPipe 3D coordinates, the Z-component
   *    of this normal vector directly indicates the palm's facing direction.
   *
   * @param {Array<{x,y,z}>} landmarks
   * @param {boolean}        isLeftHand - physical left hand flag
   * @returns {boolean} true = palm is facing the camera
   */
  function _isPalmFacing(landmarks, isLeftHand) {
    const w  = landmarks[0];  // wrist
    const im = landmarks[5];  // index finger MCP
    const pm = landmarks[17]; // pinky MCP

    // Vector A: wrist to index MCP
    const ax = im.x - w.x;
    const ay = im.y - w.y;

    // Vector B: wrist to pinky MCP
    const bx = pm.x - w.x;
    const by = pm.y - w.y;

    // 3D Cross Product normal vector (A x B)
    // Only the Z component is strictly needed for facing direction relative to camera
    const crossZ = (ax * by) - (ay * bx);

    // Sign is inverted for left vs right hand in selfie mode
    return isLeftHand ? (crossZ > 0) : (crossZ < 0);
  }

  /* ══════════════════════════════════════════════════════════
   * PUBLIC API
   * ══════════════════════════════════════════════════════════ */

  /**
   * init — Initialise MediaPipe Hands.
   * Must be called once before process().
   */
  async function init() {
    _hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    _hands.setOptions({
      maxNumHands:            1,
      modelComplexity:        1,
      minDetectionConfidence: 0.75,
      minTrackingConfidence:  0.60,
      selfieMode:             true,
    });
    _hands.onResults((r) => { _lastResult = r; });
    await _hands.initialize();
    _ready = true;
  }

  /**
   * process — Run one frame through the full pipeline.
   *
   * @param {HTMLVideoElement}         videoEl    - live camera feed
   * @param {CanvasRenderingContext2D} displayCtx - overlay canvas context
   *
   * @returns {{
   *   handDetected:  boolean,
   *   croppedCanvas: HTMLCanvasElement|null,
   *   landmarks:     Array|null,
   *   flatLandmarks: number[]|null,
   *   bbox:          {x,y,w,h}|null,
   *   isMoving:      boolean,
   *   isPalmFacing:  boolean
   * }}
   */
  async function process(videoEl, displayCtx) {
    // Empty result returned when not ready or video not loaded
    const EMPTY = {
      handDetected: false,
      croppedCanvas: null,
      landmarks: null,
      flatLandmarks: null,
      bbox: null,
      isMoving: false,
      isPalmFacing: false,
    };

    if (!_ready || !videoEl || videoEl.readyState < 2) return EMPTY;

    const vw = videoEl.videoWidth  || 640;
    const vh = videoEl.videoHeight || 480;

    /* ── 1. Mirror video to offscreen canvas ────────────── */
    _mirrorCanvas.width  = vw;
    _mirrorCanvas.height = vh;
    _mirrorCtx.save();
    _mirrorCtx.translate(vw, 0);
    _mirrorCtx.scale(-1, 1);
    _mirrorCtx.drawImage(videoEl, 0, 0, vw, vh);
    _mirrorCtx.restore();

    /* ── 2. Send raw video to MediaPipe (selfieMode mirrors internally) */
    await _hands.send({ image: videoEl });

    const result = _lastResult;

    /* ── 3. Draw mirrored video on display canvas ────────── */
    if (displayCtx) {
      displayCtx.canvas.width  = vw;
      displayCtx.canvas.height = vh;
      displayCtx.drawImage(_mirrorCanvas, 0, 0);
    }

    /* ── No hand detected ────────────────────────────────── */
    if (!result?.multiHandLandmarks?.length) {
      _prevCenter = null;
      return EMPTY;
    }

    const landmarks = result.multiHandLandmarks[0];

    // Convert normalised landmarks to pixel coords for drawing/bbox
    const pts = landmarks.map(lm => ({ x: lm.x * vw, y: lm.y * vh }));

    /* ── 4. Draw skeleton overlay ────────────────────────── */
    if (displayCtx) {
      drawConnectors(displayCtx, landmarks, HAND_CONNECTIONS,
        { color: '#ffffff', lineWidth: 2 });
      drawLandmarks(displayCtx, landmarks,
        { color: '#ef4444', lineWidth: 1, radius: 4 });
    }

    /* ── 5. Tight bounding box (natural aspect ratio) ────── */
    let minX = Infinity, minY = Infinity,
        maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }

    const rawW = maxX - minX;
    const rawH = maxY - minY;
    const pad  = Math.max(
      MIN_PADDING,
      Math.round(Math.max(rawW, rawH) * PAD_RATIO)
    );

    minX = Math.max(0,  minX - pad);
    minY = Math.max(0,  minY - pad);
    maxX = Math.min(vw, maxX + pad);
    maxY = Math.min(vh, maxY + pad);

    const cropW = maxX - minX;
    const cropH = maxY - minY;

    /* ── 6. Motion detection (centre-based, normalised) ──── */
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    let isMoving = false;

    if (_prevCenter) {
      const dx = Math.abs(_prevCenter.x - cx) / vw;
      const dy = Math.abs(_prevCenter.y - cy) / vh;
      isMoving = (dx + dy) > MOTION_THRESH;
    }
    _prevCenter = { x: cx, y: cy };

    const bbox = { x: minX, y: minY, w: cropW, h: cropH };

    /* ── 7. Draw bounding rect + convex hull on display ──── */
    if (displayCtx) {
      // Bounding rectangle — orange if moving, blue if stable
      displayCtx.strokeStyle = isMoving ? '#f97316' : '#38bdf8';
      displayCtx.lineWidth   = 2;
      displayCtx.strokeRect(minX, minY, cropW, cropH);

      // Convex hull — follows the shape of the hand
      const hull = _convexHull(pts);
      if (hull.length >= 3) {
        displayCtx.beginPath();
        displayCtx.moveTo(hull[0].x, hull[0].y);
        for (let i = 1; i < hull.length; i++)
          displayCtx.lineTo(hull[i].x, hull[i].y);
        displayCtx.closePath();
        displayCtx.strokeStyle = isMoving
          ? 'rgba(249,115,22,0.7)'
          : 'rgba(255,255,255,0.65)';
        displayCtx.lineWidth = 1.5;
        displayCtx.stroke();
      }
    }

    /* ── 8. Letterbox crop → 224×224 ────────────────────── */
    _cropCtx.fillStyle = '#111111';
    _cropCtx.fillRect(0, 0, CROP_SIZE, CROP_SIZE);

    const cropScale = Math.min(CROP_SIZE / cropW, CROP_SIZE / cropH);
    const drawW = Math.round(cropW * cropScale);
    const drawH = Math.round(cropH * cropScale);
    const offX  = Math.round((CROP_SIZE - drawW) / 2);
    const offY  = Math.round((CROP_SIZE - drawH) / 2);

    _cropCtx.drawImage(
      _mirrorCanvas,
      minX, minY, cropW, cropH,
      offX, offY, drawW, drawH
    );

    /* ── 9. Determine physical handedness ───────────────── */
    // In selfie mode, MediaPipe labels are inverted relative to
    // the physical hand — a physical right hand is labelled "Left".
    let isPhysicalLeftHand = false;
    if (result.multiHandedness?.length > 0) {
      isPhysicalLeftHand = (result.multiHandedness[0].label === 'Right');
    }

    /* ── 10. Palm orientation detection ─────────────────── */
    const isPalmFacing = _isPalmFacing(landmarks, isPhysicalLeftHand);

    /* ── 11. Flatten landmarks for MLP ──────────────────── */
    const flatLandmarks = USE_NORMALIZED_LANDMARKS
      ? _normalizeLandmarks(landmarks, isPhysicalLeftHand)
      : _flattenRaw(landmarks, isPhysicalLeftHand);

    return {
      handDetected: true,
      croppedCanvas: _cropCanvas,
      landmarks,
      flatLandmarks,
      bbox,
      isMoving,
      isPalmFacing,
    };
  }

  /* ── Accessors ───────────────────────────────────────────── */
  function isReady()       { return _ready; }
  function getCropCanvas() { return _cropCanvas; }

  return { init, process, isReady, getCropCanvas };
})();