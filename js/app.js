/**
 * app.js – Main orchestrator.
 *
 * Pipeline per-frame (throttled to MAX_FPS):
 *   video → PreprocessorModule (MediaPipe crop 224×224)
 *         → [NEW] TrajectoryTracker.push(indexTip)  ← J / Z detection
 *         → [NEW] isPalmFacing check                 ← back-of-hand warning
 *         → ModelModule.predict(flatLandmarks)
 *         → VotingBuffer.push(topLabel)
 *         → if winner → update UI + appendLetter()
 *
 * Changes from original:
 *  1. Added TrajectoryTracker for J and Z (motion gestures)
 *  2. Added palm-orientation guard — warns user to show palm
 *  3. Destructures isPalmFacing from PreprocessorModule.process()
 */

/* ══════════════════════════════════════════════════════════════
 * TRAJECTORY TRACKER  — J and Z detection
 * ══════════════════════════════════════════════════════════════
 *
 * J and Z are the only two ASL letters defined by MOVEMENT, not
 * a static pose. The MLP model only sees a single frame at a
 * time, so it is structurally incapable of recognising them.
 *
 * This module records the last BUFFER_SIZE positions of the
 * index fingertip (MediaPipe landmark 8) and applies simple
 * heuristics to decide if the recent trajectory matches J or Z.
 *
 * Heuristics (simplified):
 *  J → significant downward movement followed by an inward hook
 *       (like handwriting a capital J)
 *  Z → two distinct rightward strokes separated by a diagonal
 *       (like handwriting a capital Z)
 *
 * These are intentionally conservative to avoid false positives.
 * Tune MIN_MOVE_DIST and the per-letter thresholds if needed.
 */
const TrajectoryTracker = (() => {

  /* ── Tuning constants ───────────────────────────────────── */
  const BUFFER_SIZE   = 22;   // how many frames of history to keep
  const MIN_MOVE_DIST = 0.13; // min total displacement (normalised 0–1)
                               // to even bother classifying; filters idle

  const _history = []; // [{x, y}] in MediaPipe normalised coords (0–1)

  /**
   * push — Record the current index fingertip position.
   * Call every frame when a hand is detected.
   *
   * @param {{ x: number, y: number }} pt - MediaPipe lm 8 (normalised)
   */
  function push(pt) {
    _history.push({ x: pt.x, y: pt.y });
    if (_history.length > BUFFER_SIZE) _history.shift();
  }

  /**
   * detect — Classify the accumulated trajectory.
   *
   * Returns 'J', 'Z', or null.
   * null means "not enough movement" or "pattern not recognised".
   */
  function detect() {
    if (_history.length < 12) return null;

    const pts   = _history;
    const n     = pts.length;
    const first = pts[0];
    const mid   = pts[Math.floor(n / 2)];
    const last  = pts[n - 1];

    // Total displacement vector (start → end)
    const totalDx   = last.x - first.x;
    const totalDy   = last.y - first.y;
    const totalDist = Math.sqrt(totalDx ** 2 + totalDy ** 2);

    // Ignore micro-movements (e.g. subtle hand tremor)
    if (totalDist < MIN_MOVE_DIST) return null;

    // First half and second half displacement vectors
    const fhDx = mid.x - first.x; // first-half horizontal
    const fhDy = mid.y - first.y; // first-half vertical
    const shDx = last.x - mid.x;  // second-half horizontal
    const shDy = last.y - mid.y;  // second-half vertical

    /* ── Z detection ─────────────────────────────────────────
     * Signature: rightward → diagonal down-left → rightward
     *   • First half:  moves right (fhDx > threshold)
     *   • Second half: also moves right (shDx > threshold)
     *   • Overall vertical displacement is modest (not J-like)
     *   • The mid-point dips lower than both start and end
     *     (the diagonal crossing stroke of the Z)
     */
    const midDipY = mid.y - Math.min(first.y, last.y); // positive = dip
    const isZ = (
      fhDx  >  0.06 &&    // first half goes right
      shDx  >  0.04 &&    // second half also goes right
      Math.abs(totalDy) < 0.18 && // not predominantly vertical
      midDipY > 0.02      // there is a noticeable downward dip in the middle
    );
    if (isZ) return 'Z';

    /* ── J detection ─────────────────────────────────────────
     * Signature: downward movement with an inward hook at bottom
     *   • First half: clearly downward (fhDy > threshold)
     *   • Second half: curves sideways — the hook (|shDx| > threshold)
     *   • Overall movement is mostly vertical
     */
    const isJ = (
      fhDy              >  0.08 &&   // first half goes down
      totalDy           >  0.10 &&   // overall downward arc
      Math.abs(shDx)    >  0.05 &&   // hook at the bottom
      Math.abs(totalDx) <  0.18      // not predominantly horizontal
    );
    if (isJ) return 'J';

    return null;
  }

  /** reset — Clear history (call when hand is lost). */
  function reset() { _history.length = 0; }

  return { push, detect, reset };
})();


/* ══════════════════════════════════════════════════════════════
 * DOM REFERENCES
 * ══════════════════════════════════════════════════════════════ */
const videoEl       = document.getElementById('webcam-video');
const displayCanvas = document.getElementById('display-canvas');
const cropPreview   = document.getElementById('crop-preview');
const overlayEl     = document.getElementById('camera-overlay');
const overlayMsgEl  = document.getElementById('overlay-message');
const spinnerEl     = document.getElementById('overlay-spinner');
const statusBadge   = document.getElementById('status-badge');
const statusTextEl  = document.getElementById('status-text');
const statusDotEl   = document.getElementById('status-dot');
const btnStart      = document.getElementById('btn-start');
const btnStop       = document.getElementById('btn-stop');
const topLetterEl   = document.getElementById('top-letter');
const topConfEl     = document.getElementById('top-confidence');
const topCardEl     = document.getElementById('top-prediction-card');
const handStatusEl  = document.getElementById('hand-status');
const textOutputEl  = document.getElementById('text-output');
const btnClearText  = document.getElementById('btn-clear-text');
const btnBackspace  = document.getElementById('btn-backspace');
const btnSpeak      = document.getElementById('btn-speak');


/* ══════════════════════════════════════════════════════════════
 * CONFIG
 * ══════════════════════════════════════════════════════════════ */
const MAX_FPS       = 15;
const FRAME_INTERVAL = 1000 / MAX_FPS;
const CONF_THRESHOLD = 0.75; // minimum confidence for a vote to count
const VOTER = new VotingBuffer(10, 7); // 10-frame window, need ≥7 to win


/* ══════════════════════════════════════════════════════════════
 * STATE
 * ══════════════════════════════════════════════════════════════ */
let rafId          = null;
let lastPredictAt  = 0;
let displayCtx     = null;
let stableChar     = '';
let stableStartTime = 0;
let isCharConfirmed = false;


/* ══════════════════════════════════════════════════════════════
 * HELPERS
 * ══════════════════════════════════════════════════════════════ */
function setStatus(state, msg) {
  statusBadge.className = `status-badge status-${state}`;
  statusTextEl.textContent = msg;
}

function showOverlay(msg, spin = false) {
  overlayMsgEl.textContent    = msg;
  spinnerEl.style.display     = spin ? 'block' : 'none';
  overlayEl.classList.remove('hidden');
}

function hideOverlay() {
  overlayEl.classList.add('hidden');
}


/* ══════════════════════════════════════════════════════════════
 * UI UPDATER
 * ══════════════════════════════════════════════════════════════ */
function updateUI(predictions, winner) {
  if (!predictions?.length) return;

  const top = predictions[0]; // already sorted desc by ModelModule
  const pct = (top.probability * 100).toFixed(1);

  // Top prediction card
  topLetterEl.textContent = winner || '?';
  topConfEl.textContent   = `${pct}%`;
  topCardEl.classList.toggle('detected', !!winner);

  // Crop preview
  const crop = PreprocessorModule.getCropCanvas();
  if (crop && cropPreview) {
    cropPreview.width  = crop.width;
    cropPreview.height = crop.height;
    cropPreview.getContext('2d').drawImage(crop, 0, 0);
  }
}

/**
 * appendLetter — Custom text-input handler.
 *
 * Intercepts special model class names ('space', 'del', 'nothing')
 * and maps them to formatting commands instead of literal characters.
 * Keeps the native cursor blinking at the end of the textarea.
 *
 * @param {string} letter - Predicted gesture class name
 */
function appendLetter(letter) {
  if (!textOutputEl) return;

  if (letter === 'space') {
    textOutputEl.value += ' ';
  } else if (letter === 'del') {
    textOutputEl.value = textOutputEl.value.slice(0, -1);
  } else if (letter === 'nothing') {
    return; // explicit no-op gesture — do nothing
  } else {
    textOutputEl.value += letter;
  }

  // Force native blinking cursor to the end of the text
  textOutputEl.focus();
  textOutputEl.setSelectionRange(
    textOutputEl.value.length,
    textOutputEl.value.length
  );
}


/* ══════════════════════════════════════════════════════════════
 * MAIN PREDICTION LOOP
 * ══════════════════════════════════════════════════════════════ */
async function loop(now) {
  if (!CameraModule.isRunning()) return;

  rafId = requestAnimationFrame(loop);

  // Throttle to MAX_FPS
  if (now - lastPredictAt < FRAME_INTERVAL) return;
  lastPredictAt = now;

  try {
    const video = CameraModule.getVideoElement();

    /* ── Step 1: Preprocess ─────────────────────────────── */
    // Destructure isPalmFacing — new field added by updated preprocessor
    const {
      handDetected,
      croppedCanvas,
      landmarks,
      flatLandmarks,
      isMoving,
      isPalmFacing,
    } = await PreprocessorModule.process(video, displayCtx);

    /* ── Step 2: Hand-lost reset ────────────────────────── */
    if (!handDetected) {
      VOTER.reset();
      TrajectoryTracker.reset(); // clear trajectory on hand loss

      topLetterEl.textContent = '?';
      topConfEl.textContent   = '—';
      topCardEl.classList.remove('detected');
      handStatusEl.textContent = '– No hand';
      handStatusEl.className   = 'hand-tag hand-none';

      stableChar      = '';
      stableStartTime = 0;
      isCharConfirmed = false;
      return;
    }

    /* ── Step 3: Update hand-status badge ───────────────── */
    // Priority order: moving > back-of-hand > stable
    if (isMoving) {
      handStatusEl.textContent = '↔ Moving';
      handStatusEl.className   = 'hand-tag hand-moving';
    } else if (!isPalmFacing) {
      // ── Palm orientation warning ──────────────────────────
      // The model was trained on palm-facing data only.
      // When the back of the hand is shown the Z-depth profile
      // of every landmark inverts, confusing the classifier.
      // Show a persistent warning so the user knows to flip.
      handStatusEl.textContent = '🔄 Show palm';
      handStatusEl.className   = 'hand-tag hand-moving'; // reuse orange style
    } else {
      handStatusEl.textContent = '✋ Stable';
      handStatusEl.className   = 'hand-tag hand-ok';
    }

    /* ── Step 4: Feed index fingertip to trajectory tracker  */
    // landmark 8 = index fingertip in MediaPipe hand model
    if (landmarks && landmarks[8]) {
      TrajectoryTracker.push(landmarks[8]); // {x, y} normalised 0–1
    }

    /* ── Step 5: Check for J / Z trajectory gesture ──────── */
    // Do this BEFORE the static MLP prediction so a trajectory
    // match can short-circuit the rest of the frame processing.
    // J and Z cannot be recognised by a single-frame snapshot.
    const trajectoryResult = TrajectoryTracker.detect();
    if (trajectoryResult) {
      // Confirmed dynamic gesture — bypass voting, append directly
      appendLetter(trajectoryResult);
      TrajectoryTracker.reset(); // prevent double-firing the same gesture

      // Reflect in UI briefly
      topLetterEl.textContent = trajectoryResult;
      topConfEl.textContent   = '(motion)';
      topCardEl.classList.add('detected');
      return; // skip static MLP prediction for this frame
    }

    /* ── Step 6: Inject sentinel if hand is moving ──────── */
    if (isMoving) {
      VOTER.push('_moving');
      topCardEl.classList.remove('detected');
      stableChar      = '';
      stableStartTime = 0;
      isCharConfirmed = false;
      return;
    }

    /* ── Step 7: Run static MLP model ───────────────────── */
    const predictions = await ModelModule.predict(flatLandmarks);
    let top = predictions[0];

    /* ── Step 7.5: Angle-based correction ────────────────── */
    // Koreksi prediksi berdasarkan sudut jari untuk membedakan:
    //  • I vs J (kemiringan pinky)
    //  • D vs Z (kemiringan index)
    //  • L guard (mencegah T dan Z terbaca sebagai L)
    const correctedClass = AngleDetectorModule.correctPrediction(
      top.className,
      landmarks,
      top.probability
    );
    
    // Jika ada koreksi, update top prediction
    if (correctedClass !== top.className) {
      top = {
        className: correctedClass,
        probability: top.probability // Pertahankan confidence asli
      };
    }

    /* ── Step 8: Temporal debounce (2-second hold) ──────── */
    // A gesture must be held steadily for ≥2 s before it is
    // appended as text. This prevents accidental single-frame
    // triggers while the hand settles into a pose.
    if (top.probability > 0.80) {
      if (top.className !== stableChar) {
        // New candidate — start the hold timer fresh
        stableChar      = top.className;
        stableStartTime = Date.now();
        isCharConfirmed = false;
      } else if ((Date.now() - stableStartTime) >= 2000 && !isCharConfirmed) {
        // Held for 2 seconds — confirm and append
        appendLetter(stableChar);
        isCharConfirmed = true;
      }
    } else {
      // Confidence dropped — reset hold timer
      stableChar      = '';
      stableStartTime = 0;
      isCharConfirmed = false;
    }

    /* ── Step 9: Push into voting buffer ────────────────── */
    if (top.probability >= CONF_THRESHOLD) {
      VOTER.push(top.className);
    } else {
      VOTER.push('_low'); // non-letter sentinel keeps buffer alive
    }

    const rawWinner = VOTER.getWinner();
    // Filter out internal sentinels before passing to UI
    const winner = (rawWinner === '_low' || rawWinner === '_moving')
      ? null
      : rawWinner;

    /* ── Step 10: Update UI ─────────────────────────────── */
    updateUI(predictions, winner);

  } catch (err) {
    console.error('[app] loop error:', err.message);
  }
}


/* ══════════════════════════════════════════════════════════════
 * START
 * ══════════════════════════════════════════════════════════════ */
async function handleStart() {
  btnStart.disabled = true;
  btnStop.disabled  = true;
  setStatus('loading', 'Loading…');
  showOverlay('Loading ASL MLP model…', true);

  try {
    if (!ModelModule.isLoaded()) await ModelModule.load();

    overlayMsgEl.textContent = 'Initializing MediaPipe Hands…';
    if (!PreprocessorModule.isReady()) await PreprocessorModule.init();

    overlayMsgEl.textContent = 'Requesting camera access…';
    CameraModule.init(videoEl);
    await CameraModule.start();

    displayCtx = displayCanvas.getContext('2d');
    hideOverlay();
    setStatus('live', 'Live');
    if (statusDotEl) statusDotEl.className = 'status-dot active';
    btnStop.disabled = false;

    VOTER.reset();
    TrajectoryTracker.reset();
    lastPredictAt = 0;
    rafId = requestAnimationFrame(loop);

  } catch (err) {
    console.error('[app] start error:', err);
    const msg =
      err.name === 'NotAllowedError'
        ? 'Camera permission denied. Please allow and refresh.'
        : err.name === 'NotFoundError'
          ? 'No camera found on this device.'
          : err.message || 'Unknown error. Please try again.';

    showOverlay(msg, false);
    setStatus('error', 'Error');
    btnStart.disabled = false;
  }
}


/* ══════════════════════════════════════════════════════════════
 * STOP
 * ══════════════════════════════════════════════════════════════ */
function handleStop() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  CameraModule.stop();

  VOTER.reset();
  TrajectoryTracker.reset();

  stableChar      = '';
  stableStartTime = 0;
  isCharConfirmed = false;

  // Clear display canvas
  if (displayCtx) {
    displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  }

  topLetterEl.textContent  = '—';
  topConfEl.textContent    = '0%';
  topCardEl.classList.remove('detected');
  handStatusEl.textContent = '– Camera off';
  handStatusEl.className   = 'hand-tag hand-none';

  showOverlay('Camera stopped. Press Start to begin.', false);
  setStatus('idle', 'Stopped');
  if (statusDotEl) statusDotEl.className = 'status-dot idle';
  btnStart.disabled = false;
  btnStop.disabled  = true;
}


/* ══════════════════════════════════════════════════════════════
 * TEXT BUILDER CONTROLS
 * ══════════════════════════════════════════════════════════════ */
btnClearText?.addEventListener('click', () => {
  if (textOutputEl) textOutputEl.value = '';
});

btnBackspace?.addEventListener('click', () => {
  if (textOutputEl)
    textOutputEl.value = textOutputEl.value.slice(0, -1);
});

btnSpeak?.addEventListener('click', () => {
  if (textOutputEl && textOutputEl.value.trim() !== '') {
    const utterance = new SpeechSynthesisUtterance(textOutputEl.value);
    utterance.lang  = 'en-US';
    window.speechSynthesis.speak(utterance);
  }
});

/**
 * Block physical keyboard input on the text output textarea.
 * The field is effectively read-only but retains focus so the
 * native blinking cursor is still visible.
 */
textOutputEl?.addEventListener('keydown', (event) => {
  event.preventDefault();
});


/* ══════════════════════════════════════════════════════════════
 * BUTTON WIRING
 * ══════════════════════════════════════════════════════════════ */
btnStart?.addEventListener('click', handleStart);
btnStop?.addEventListener('click', handleStop);


/* ══════════════════════════════════════════════════════════════
 * BOOT
 * ══════════════════════════════════════════════════════════════ */
setStatus('idle', 'Ready');
showOverlay('Press "Start Camera" to begin.', false);
btnStart.disabled = false;


/* ══════════════════════════════════════════════════════════════
 * VISIBILITY CHANGE — pause/resume on tab switch
 * ══════════════════════════════════════════════════════════════ */
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    // User switched tabs — pause the prediction loop
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  } else {
    // User returned — resume only if camera is still running
    if (CameraModule.isRunning() && !rafId) {
      lastPredictAt = performance.now();
      rafId = requestAnimationFrame(loop);
    }
  }
});