/**
 * model.js – TensorFlow.js Layers model loader & predictor.
 * Receives the 63-element landmarks array from PreprocessorModule.
 */

const ModelModule = (() => {

  const MODEL_URL = './tfjs_model/model.json';
  let _model  = null;
  let _loaded = false;

  const CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
  ];

  async function load() {
    if (_loaded) return;
    _model  = await tf.loadLayersModel(MODEL_URL);
    _loaded = true;
  }

  /**
   * Predict on a 63-element landmarks array.
   * Returns array sorted by probability DESCENDING.
   * @param {Array<number>} flatLandmarks 
   * @returns {Promise<Array<{className:string, probability:number}>>}
   */
  async function predict(flatLandmarks) {
    if (!_loaded || !_model) throw new Error('Model not loaded.');
    if (!flatLandmarks || flatLandmarks.length !== 63) {
      throw new Error('Invalid landmarks array length (expected 63).');
    }
    
    // Create a 2D tensor of shape [1, 63]
    const inputTensor = tf.tensor2d([flatLandmarks], [1, 63]);
    
    // Predict
    const predictionTensor = _model.predict(inputTensor);
    const predictionData = await predictionTensor.data();
    
    // Cleanup tensors
    inputTensor.dispose();
    predictionTensor.dispose();

    // Format output
    const raw = Array.from(predictionData).map((prob, i) => ({
      className: CLASSES[i] || `Class_${i}`,
      probability: prob
    }));

    return raw.sort((a, b) => b.probability - a.probability);
  }

  function isLoaded() { return _loaded; }

  return { load, predict, isLoaded };
})();
