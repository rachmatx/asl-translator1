/**
 * voting.js  —  Stricter rolling prediction voting buffer.
 * Window: 10 frames, threshold: 7 votes needed to win.
 * Eliminates flicker and forces the model to be consistently confident.
 */

class VotingBuffer {
  /**
   * @param {number} size      – rolling window length
   * @param {number} threshold – votes needed for a class to "win"
   */
  constructor(size = 10, threshold = 7) {
    this._size      = size;
    this._threshold = threshold;
    this._buffer    = [];
  }

  push(label) {
    this._buffer.push(label);
    if (this._buffer.length > this._size) this._buffer.shift();
  }

  /** @returns {string|null} winning class or null */
  getWinner() {
    if (!this._buffer.length) return null;
    const counts = {};
    for (const l of this._buffer) counts[l] = (counts[l] || 0) + 1;
    let best = null, bestCount = 0;
    for (const [l, c] of Object.entries(counts)) {
      if (c > bestCount) { bestCount = c; best = l; }
    }
    return bestCount >= this._threshold ? best : null;
  }

  reset() { this._buffer = []; }
}
