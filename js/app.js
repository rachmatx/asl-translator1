/**
 * app.js – Main orchestrator.
 *
 * Pipeline per-frame (throttled to MAX_FPS):
 *   video → PreprocessorModule (MediaPipe crop 224×224)
 *         → ModelModule.predict(croppedCanvas)
 *         → VotingBuffer.push(topLabel)
 *         → if winner → update UI + EvaluatorModule.record()
 */

/* ── DOM ─────────────────────────────────────────────────── */
const videoEl = document.getElementById('webcam-video');
const displayCanvas = document.getElementById('display-canvas');
const cropPreview = document.getElementById('crop-preview');
const overlayEl = document.getElementById('camera-overlay');
const overlayMsgEl = document.getElementById('overlay-message');
const spinnerEl = document.getElementById('overlay-spinner');
const statusBadge = document.getElementById('status-badge');
const statusTextEl = document.getElementById('status-text');
const statusDotEl = document.getElementById('status-dot');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const topLetterEl = document.getElementById('top-letter');
const topConfEl = document.getElementById('top-confidence');
const topCardEl = document.getElementById('top-prediction-card');
const handStatusEl = document.getElementById('hand-status');
const textOutputEl = document.getElementById('text-output');
const btnClearText = document.getElementById('btn-clear-text');
const btnBackspace = document.getElementById('btn-backspace');
const btnSpeak = document.getElementById('btn-speak');

/* ── Config ──────────────────────────────────────────────── */
const MAX_FPS = 15;
const FRAME_INTERVAL = 1000 / MAX_FPS;
const CONF_THRESHOLD = 0.75;                     // raised: only high-confidence votes count
const VOTER = new VotingBuffer(10, 7);  // 10-frame window, need ≥7 to win

/* ── State ───────────────────────────────────────────────── */
let rafId = null;
let lastPredictAt = 0;
let displayCtx = null;
let stableChar = "";
let stableStartTime = 0;
let isCharConfirmed = false;

/* ── Helpers ─────────────────────────────────────────────── */
function setStatus(state, msg) {
  statusBadge.className = `status-badge status-${state}`;
  statusTextEl.textContent = msg;
}
function showOverlay(msg, spin = false) {
  overlayMsgEl.textContent = msg;
  spinnerEl.style.display = spin ? 'block' : 'none';
  overlayEl.classList.remove('hidden');
}
function hideOverlay() { overlayEl.classList.add('hidden'); }

/* ── UI updater ──────────────────────────────────────────── */
function updateUI(predictions, winner) {
  if (!predictions?.length) return;

  const top = predictions[0]; // already sorted desc by ModelModule
  const pct = (top.probability * 100).toFixed(1);

  // Top prediction card
  topLetterEl.textContent = winner || '?';
  topConfEl.textContent = `${pct}%`;
  topCardEl.classList.toggle('detected', !!winner);

  // Crop preview
  const crop = PreprocessorModule.getCropCanvas();
  if (crop && cropPreview) {
    cropPreview.width = crop.width;
    cropPreview.height = crop.height;
    cropPreview.getContext('2d').drawImage(crop, 0, 0);
  }
}

/**
 * appendLetter - Custom text input handling
 * Intercepts specific model classes to execute formatting commands (space, backspace, etc.)
 * rather than simply appending the raw string. Also maintains native cursor focus.
 * @param {string} letter - The predicted gesture class to handle.
 */
function appendLetter(letter) {
  if (!textOutputEl) return;

  if (letter === 'space') {
    textOutputEl.value += ' ';
  } else if (letter === 'del') {
    textOutputEl.value = textOutputEl.value.slice(0, -1);
  } else if (letter === 'nothing') {
    return;
  } else {
    textOutputEl.value += letter;
  }

  // Force native blinking cursor to the end of the text
  textOutputEl.focus();
  textOutputEl.setSelectionRange(textOutputEl.value.length, textOutputEl.value.length);
}

/* ── Main prediction loop ────────────────────────────────── */
async function loop(now) {
  if (!CameraModule.isRunning()) return;

  rafId = requestAnimationFrame(loop);

  // Throttle predictions
  if (now - lastPredictAt < FRAME_INTERVAL) return;
  lastPredictAt = now;

  try {
    const video = CameraModule.getVideoElement();

    // 1. Preprocess (MediaPipe Hands → landmarks)
    const { handDetected, croppedCanvas, landmarks, flatLandmarks, isMoving } =
      await PreprocessorModule.process(video, displayCtx);

    // Show hand state in header tag
    if (!handDetected) {
      handStatusEl.textContent = '– No hand';
      handStatusEl.className = 'hand-tag hand-none';
    } else if (isMoving) {
      handStatusEl.textContent = '↔ Moving';
      handStatusEl.className = 'hand-tag hand-moving';
    } else {
      handStatusEl.textContent = '✋ Stable';
      handStatusEl.className = 'hand-tag hand-ok';
    }

    if (!handDetected) {
      VOTER.reset();
      topLetterEl.textContent = '?';
      topConfEl.textContent = '—';
      topCardEl.classList.remove('detected');
      stableChar = "";
      stableStartTime = 0;
      isCharConfirmed = false;
      return;
    }

    // 2. Run model on flatLandmarks
    const predictions = await ModelModule.predict(flatLandmarks);
    const top = predictions[0];

    if (isMoving) {
      // Hand is transitioning — inject sentinel to break any building consensus
      VOTER.push('_moving');
      topCardEl.classList.remove('detected');
      stableChar = "";
      stableStartTime = 0;
      isCharConfirmed = false;
      return;
    }

    // 2.5 Temporal Smoothing / Auto-append logic (Debounce)
    // Ensures a gesture is held steadily for 2 seconds before confirming it as a valid input.
    if (top.probability > 0.85) {
      if (top.className !== stableChar) {
        stableChar = top.className;
        stableStartTime = Date.now();
        isCharConfirmed = false;
      } else if ((Date.now() - stableStartTime) >= 2000 && !isCharConfirmed) {
        appendLetter(stableChar);
        isCharConfirmed = true;
      }
    } else {
      stableChar = "";
      stableStartTime = 0;
      isCharConfirmed = false;
    }

    // 3. Vote – only push if confidence crosses threshold
    if (top.probability >= CONF_THRESHOLD) {
      VOTER.push(top.className);
    } else {
      VOTER.push('_low');   // non-letter sentinel keeps buffer alive
    }

    const rawWinner = VOTER.getWinner();
    const winner = (rawWinner === '_low' || rawWinner === '_moving') ? null : rawWinner;

    // 4. Update UI
    updateUI(predictions, winner);
  } catch (err) {
    console.error('[app] loop error:', err.message);
  }
}

/* ── Start ───────────────────────────────────────────────── */
async function handleStart() {
  btnStart.disabled = true;
  btnStop.disabled = true;
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
    lastPredictAt = 0;
    rafId = requestAnimationFrame(loop);

  } catch (err) {
    console.error('[app] start error:', err);
    const msg =
      err.name === 'NotAllowedError' ? 'Camera permission denied. Please allow and refresh.' :
        err.name === 'NotFoundError' ? 'No camera found on this device.' :
          err.message || 'Unknown error. Please try again.';
    showOverlay(msg, false);
    setStatus('error', 'Error');
    btnStart.disabled = false;
  }
}

/* ── Stop ────────────────────────────────────────────────── */
function handleStop() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  CameraModule.stop();
  VOTER.reset();
  stableChar = "";
  stableStartTime = 0;
  isCharConfirmed = false;

  // Clear display canvas
  if (displayCtx) {
    displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  }

  topLetterEl.textContent = '—';
  topConfEl.textContent = '0%';
  topCardEl.classList.remove('detected');
  handStatusEl.textContent = '– Camera off';
  handStatusEl.className = 'hand-tag hand-none';

  showOverlay('Camera stopped. Press Start to begin.', false);
  setStatus('idle', 'Stopped');
  if (statusDotEl) statusDotEl.className = 'status-dot idle';
  btnStart.disabled = false;
  btnStop.disabled = true;
}

/* ── Text builder controls ──────────────────────────────── */
btnClearText?.addEventListener('click', () => { if (textOutputEl) textOutputEl.value = ''; });
btnBackspace?.addEventListener('click', () => {
  if (textOutputEl) textOutputEl.value = textOutputEl.value.slice(0, -1);
});
btnSpeak?.addEventListener('click', () => {
  if (textOutputEl && textOutputEl.value.trim() !== '') {
    const utterance = new SpeechSynthesisUtterance(textOutputEl.value);
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
  }
});

/**
 * Block physical keyboard input to text output
 * Retains read-only feel while allowing focus for the blinking cursor
 */
textOutputEl?.addEventListener('keydown', (event) => {
  event.preventDefault();
});

/* ── Wire buttons ────────────────────────────────────────── */
btnStart?.addEventListener('click', handleStart);
btnStop?.addEventListener('click', handleStop);

/* ── Boot ────────────────────────────────────────────────── */
setStatus('idle', 'Ready');
showOverlay('Press "Start Camera" to begin.', false);
btnStart.disabled = false;
