# Angle Detection Guide

## Overview

The `AngleDetectorModule` (located in `js/angleDetector.js`) is designed to correct misclassifications from the static TensorFlow.js model by analyzing the 3D geometry of the hand.

Because the neural network is trained on single, flat snapshots of hand positions, it can easily confuse letters that look similar straight-on but have slight variations in finger angle or depth.

## Current Correction Rules

The post-processing step intercepts the model's prediction and applies rules:

1. **I → J Correction**
   - **Trigger**: The model predicts `I`.
   - **Condition**: The pinky finger angle is ≥ 50° from the vertical axis.
   - **Action**: Correct prediction to `J`.

2. **D → Z Correction**
   - **Trigger**: The model predicts `D`.
   - **Condition**: The index finger angle is ≥ 50° from the vertical axis.
   - **Action**: Correct prediction to `Z`.

3. **L Guard (L → Z)**
   - **Trigger**: The model predicts `L`.
   - **Condition**: The index finger angle is ≥ 45° from the vertical axis.
   - **Action**: Correct prediction to `Z`.

*(Note: The old rule `L → T` was removed because the model proved capable of differentiating them without geometry assistance, and it caused false positives.)*

## How to Tune the Thresholds

Thresholds are centrally located in the `AngleDetectorModule` configuration at the top of `js/angleDetector.js`.

```javascript
const CONFIG = {
  // Threshold in degrees from vertical (0 = vertical)
  PINKY_J_THRESHOLD: 50,
  INDEX_Z_THRESHOLD: 50,
  INDEX_L_GUARD_THRESHOLD: 45
};
```

If you notice:
- **False positives** (e.g., normal `I` is being corrected to `J` when it shouldn't): **Increase** the threshold (e.g., from 50 to 55 or 60).
- **False negatives** (e.g., tilted `J` is not triggering the correction and remains `I`): **Decrease** the threshold (e.g., from 50 to 40 or 45).

## Debugging

To view real-time angle calculations, uncomment the debugging console log in the `correctPrediction` function inside `js/angleDetector.js`:

```javascript
// Debug log
console.log(`[AngleDetector] ${predictedClass} | Pinky: ${pinkyAngle.toFixed(1)}° | Index: ${indexAngle.toFixed(1)}°`);
```

This will print the calculated angles for the index and pinky fingers every frame, allowing you to see exactly what angles your natural hand signs are producing and adjust the thresholds accordingly.
