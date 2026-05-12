/**
 * camera.js – Webcam stream management.
 */

const CameraModule = (() => {

  let _videoEl = null;
  let _stream  = null;
  let _running = false;

  function init(videoElement) { _videoEl = videoElement; }

  async function start() {
    if (_running) return;
    _stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    _videoEl.srcObject = _stream;
    await new Promise((res, rej) => {
      _videoEl.onloadedmetadata = res;
      _videoEl.onerror = rej;
    });
    await _videoEl.play();
    _running = true;
  }

  function stop() {
    _stream?.getTracks().forEach(t => t.stop());
    _stream = null;
    if (_videoEl) _videoEl.srcObject = null;
    _running = false;
  }

  function isRunning()       { return _running; }
  function getVideoElement() { return _videoEl; }

  return { init, start, stop, isRunning, getVideoElement };
})();
