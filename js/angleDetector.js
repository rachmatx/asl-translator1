/**
 * angleDetector.js — Deteksi sudut kemiringan jari untuk membedakan
 * huruf-huruf ASL yang mirip secara visual.
 *
 * Digunakan untuk:
 *  • I vs J  → kemiringan jari kelingking (pinky)
 *  • D vs Z  → kemiringan jari telunjuk (index)
 *  • L guard → mencegah T dan Z terbaca sebagai L
 *
 * MediaPipe Hand Landmarks (21 titik):
 *  0  = WRIST
 *  4  = THUMB_TIP
 *  5  = INDEX_FINGER_MCP
 *  6  = INDEX_FINGER_PIP
 *  7  = INDEX_FINGER_DIP
 *  8  = INDEX_FINGER_TIP
 *  17 = PINKY_MCP
 *  18 = PINKY_PIP
 *  19 = PINKY_DIP
 *  20 = PINKY_TIP
 */

const AngleDetectorModule = (() => {

  /* ══════════════════════════════════════════════════════════
   * HELPER FUNCTIONS
   * ══════════════════════════════════════════════════════════ */

  /**
   * Menghitung sudut kemiringan dari garis vertikal (dalam derajat).
   * 
   * @param {Object} p1 - Titik awal {x, y, z}
   * @param {Object} p2 - Titik akhir {x, y, z}
   * @returns {number} Sudut dalam derajat (0° = vertikal, 90° = horizontal)
   */
  function calculateAngleFromVertical(p1, p2) {
    // Hitung vektor dari p1 ke p2
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    
    // Hitung sudut dari sumbu Y (vertikal)
    // atan2(dx, dy) memberikan sudut dari vertikal
    // Kita gunakan nilai absolut untuk mendapatkan kemiringan tanpa arah
    const angleRad = Math.atan2(Math.abs(dx), Math.abs(dy));
    const angleDeg = angleRad * (180 / Math.PI);
    
    return angleDeg;
  }

  /**
   * Menghitung sudut kemiringan jari kelingking (pinky).
   * Menggunakan garis dari MCP (17) ke TIP (20).
   * 
   * @param {Array} landmarks - 21 MediaPipe landmarks
   * @returns {number} Sudut kemiringan dalam derajat
   */
  function getPinkyAngle(landmarks) {
    const pinkyMCP = landmarks[17]; // Base jari kelingking
    const pinkyTIP = landmarks[20]; // Ujung jari kelingking
    
    return calculateAngleFromVertical(pinkyMCP, pinkyTIP);
  }

  /**
   * Menghitung sudut kemiringan jari telunjuk (index).
   * Menggunakan garis dari MCP (5) ke TIP (8).
   * 
   * @param {Array} landmarks - 21 MediaPipe landmarks
   * @returns {number} Sudut kemiringan dalam derajat
   */
  function getIndexFingerAngle(landmarks) {
    const indexMCP = landmarks[5];  // Base jari telunjuk
    const indexTIP = landmarks[8];  // Ujung jari telunjuk
    
    return calculateAngleFromVertical(indexMCP, indexTIP);
  }



  /* ══════════════════════════════════════════════════════════
   * POST-PROCESSING LOGIC
   * ══════════════════════════════════════════════════════════ */

  /**
   * Memperbaiki prediksi model berdasarkan analisis sudut jari.
   * 
   * Aturan koreksi:
   *  1. I → J: Jika model prediksi I dan pinky angle ≥ 50°, ubah ke J
   *  2. D → Z: Jika model prediksi D dan index angle ≥ 50°, ubah ke Z
   *  3. Z → L guard: Jika model prediksi L tapi index angle ≥ 45°, kembalikan ke Z
   * 
   * CATATAN: Aturan L → T dihapus karena menyebabkan false positive.
   * Model sudah cukup baik membedakan L dan T tanpa koreksi sudut.
   * 
   * @param {string} predictedClass - Kelas yang diprediksi oleh model
   * @param {Array}  landmarks      - 21 MediaPipe landmarks
   * @param {number} confidence     - Confidence score dari model (0-1)
   * @returns {string} Kelas yang sudah dikoreksi
   */
  function correctPrediction(predictedClass, landmarks, confidence) {
    if (!landmarks || landmarks.length < 21) {
      return predictedClass; // Tidak bisa koreksi tanpa landmarks lengkap
    }

    const pinkyAngle = getPinkyAngle(landmarks);
    const indexAngle = getIndexFingerAngle(landmarks);

    // Debug log (bisa diaktifkan untuk tuning)
    // console.log(`[AngleDetector] ${predictedClass} | Pinky: ${pinkyAngle.toFixed(1)}° | Index: ${indexAngle.toFixed(1)}°`);

    /* ── Aturan 1: I → J (kemiringan pinky) ──────────────── */
    if (predictedClass === 'I' && pinkyAngle >= 50) {
      console.log(`[AngleDetector] I → J (pinky angle: ${pinkyAngle.toFixed(1)}°)`);
      return 'J';
    }

    /* ── Aturan 2: D → Z (kemiringan index) ───────────────── */
    if (predictedClass === 'D' && indexAngle >= 50) {
      console.log(`[AngleDetector] D → Z (index angle: ${indexAngle.toFixed(1)}°)`);
      return 'Z';
    }

    /* ── Aturan 3: L guard untuk Z ─────────────────────────── */
    // Huruf Z memiliki index finger yang miring
    // Jika model bilang L tapi index miring, kemungkinan ini Z
    // Threshold dinaikkan dari 40° ke 45° untuk mengurangi false positive
    if (predictedClass === 'L' && indexAngle >= 45) {
      console.log(`[AngleDetector] L → Z (index angle: ${indexAngle.toFixed(1)}° ≥ 45°)`);
      return 'Z';
    }

    // Tidak ada koreksi diperlukan
    return predictedClass;
  }



  /* ══════════════════════════════════════════════════════════
   * PUBLIC API
   * ══════════════════════════════════════════════════════════ */
  return {
    correctPrediction,
    getPinkyAngle,
    getIndexFingerAngle
  };

})();
