/**
 * preprocessor.js — Tight non-square crop + convex hull overlay
 * Pipeline:
 *  1. Mirror video frame
 *  2. MediaPipe Hands (selfieMode) → landmarks
 *  3. Tight bounding box (natural aspect ratio, no forced square)
 *  4. Convex hull polygon drawn on display canvas
 *  5. Letterbox crop → 224×224 (preserves proportions)
 */

const PreprocessorModule = (() => {

  const CROP_SIZE     = 224;
  const MIN_PADDING   = 28;   // enough to include wrist base
  const PAD_RATIO     = 0.05; // 5% of hand's natural size
  const MOTION_THRESH = 0.07;

  let _hands      = null;
  let _lastResult = null;
  let _ready      = false;
  let _prevCenter = null;  // {x,y} normalized, for motion detection

  const _mirrorCanvas = document.createElement('canvas');
  const _mirrorCtx    = _mirrorCanvas.getContext('2d');
  const _cropCanvas   = document.createElement('canvas');
  const _cropCtx      = _cropCanvas.getContext('2d');
  _cropCanvas.width   = CROP_SIZE;
  _cropCanvas.height  = CROP_SIZE;

  /** Graham-scan convex hull of 2-D points [{x,y}] */
  function _convexHull(pts) {
    if (pts.length < 3) return pts;
    const s = pts.slice().sort((a, b) => a.x !== b.x ? a.x - b.x : a.y - b.y);
    const cross = (O, A, B) => (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    const lower = [];
    for (const p of s) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = s.length - 1; i >= 0; i--) {
      const p = s[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    upper.pop(); lower.pop();
    return lower.concat(upper);
  }

  async function init() {
    _hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
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

  async function process(videoEl, displayCtx) {
    if (!_ready || !videoEl || videoEl.readyState < 2) {
      return { handDetected: false, croppedCanvas: null, landmarks: null, bbox: null, isMoving: false };
    }

    const vw = videoEl.videoWidth  || 640;
    const vh = videoEl.videoHeight || 480;

    // 1. Mirror to offscreen canvas
    _mirrorCanvas.width  = vw;
    _mirrorCanvas.height = vh;
    _mirrorCtx.save();
    _mirrorCtx.translate(vw, 0);
    _mirrorCtx.scale(-1, 1);
    _mirrorCtx.drawImage(videoEl, 0, 0, vw, vh);
    _mirrorCtx.restore();

    // 2. Send raw video to MediaPipe (selfieMode handles mirroring)
    await _hands.send({ image: videoEl });

    const result = _lastResult;

    // 3. Draw mirrored video on display canvas
    if (displayCtx) {
      displayCtx.canvas.width  = vw;
      displayCtx.canvas.height = vh;
      displayCtx.drawImage(_mirrorCanvas, 0, 0);
    }

    if (!result?.multiHandLandmarks?.length) {
      _prevCenter = null;
      return { handDetected: false, croppedCanvas: null, landmarks: null, bbox: null, isMoving: false };
    }

    const landmarks = result.multiHandLandmarks[0];
    const pts = landmarks.map(lm => ({ x: lm.x * vw, y: lm.y * vh }));

    // 4. Draw skeleton: white connectors + red dots (matches reference style)
    if (displayCtx) {
      drawConnectors(displayCtx, landmarks, HAND_CONNECTIONS, { color: '#ffffff', lineWidth: 2 });
      drawLandmarks(displayCtx, landmarks, { color: '#ef4444', lineWidth: 1, radius: 4 });
    }

    // 5. Tight bounding box — natural aspect ratio (NO forced square)
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    const rawW = maxX - minX;
    const rawH = maxY - minY;
    const pad  = Math.max(MIN_PADDING, Math.round(Math.max(rawW, rawH) * PAD_RATIO));

    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(vw, maxX + pad);
    maxY = Math.min(vh, maxY + pad);

    const cropW = maxX - minX;
    const cropH = maxY - minY;

    // 6. Motion detection (center-based, normalized)
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

    // 7. Draw tight rect + convex hull on display canvas
    if (displayCtx) {
      // Tight bounding rectangle
      displayCtx.strokeStyle = isMoving ? '#f97316' : '#38bdf8';
      displayCtx.lineWidth   = 2;
      displayCtx.strokeRect(minX, minY, cropW, cropH);

      // Convex hull — "follows the shape of the hand"
      const hull = _convexHull(pts);
      if (hull.length >= 3) {
        displayCtx.beginPath();
        displayCtx.moveTo(hull[0].x, hull[0].y);
        for (let i = 1; i < hull.length; i++) displayCtx.lineTo(hull[i].x, hull[i].y);
        displayCtx.closePath();
        displayCtx.strokeStyle = isMoving ? 'rgba(249,115,22,0.7)' : 'rgba(255,255,255,0.65)';
        displayCtx.lineWidth   = 1.5;
        displayCtx.stroke();
      }
    }

    // 8. Letterbox crop → 224×224 (preserves natural proportions, no distortion)
    _cropCtx.fillStyle = '#111111';
    _cropCtx.fillRect(0, 0, CROP_SIZE, CROP_SIZE);
    const scale = Math.min(CROP_SIZE / cropW, CROP_SIZE / cropH);
    const drawW = Math.round(cropW * scale);
    const drawH = Math.round(cropH * scale);
    const offX  = Math.round((CROP_SIZE - drawW) / 2);
    const offY  = Math.round((CROP_SIZE - drawH) / 2);
    _cropCtx.drawImage(_mirrorCanvas, minX, minY, cropW, cropH, offX, offY, drawW, drawH);

    // 9. Extract handedness and mirror X-axis for physical left hand
    // MediaPipe Hands returns the handedness of the hand in the image.
    // Because we use selfieMode (mirroring the video), the labels 'Left' and 'Right' are inverted.
    // We adjust the X coordinates for the left hand to normalize the data for our model,
    // which was trained to recognize both hands consistently.
    let isPhysicalLeftHand = false;
    if (result.multiHandedness && result.multiHandedness.length > 0) {
      const label = result.multiHandedness[0].label;
      isPhysicalLeftHand = (label === 'Right');
    }

    /**
     * Flatten landmarks for MLP model
     * Extracts X, Y, Z coordinates for all 21 landmarks into a 63-element array.
     * Mirrors the X-axis for left hands to ensure the model receives standardized right-hand equivalent data.
     */
    const flatLandmarks = [];
    for (let i = 0; i < landmarks.length; i++) {
      // Mirror spatial data along X-axis if it's the left hand
      const x = isPhysicalLeftHand ? 1.0 - landmarks[i].x : landmarks[i].x;
      flatLandmarks.push(x, landmarks[i].y, landmarks[i].z);
    }

    return { handDetected: true, croppedCanvas: _cropCanvas, landmarks, flatLandmarks, bbox, isMoving };
  }

  function isReady()       { return _ready; }
  function getCropCanvas() { return _cropCanvas; }

  return { init, process, isReady, getCropCanvas };
})();
