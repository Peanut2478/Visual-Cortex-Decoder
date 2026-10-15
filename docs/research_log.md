# Visual Cortex Decoder — Research Log

## Research Objective

Investigate whether the direction of a visual stimulus can be reconstructed from population-level neural activity recorded in the mouse visual cortex.

**Dataset:** Allen Brain Observatory, experiment 652094901  
**Recorded neurons:** 207  
**Stimulus:** Drifting gratings  
**Neural measurement:** Calcium fluorescence (ΔF/F)

## Experiment 1 — Population Vector Decoding

**Method:** Estimate each neuron's preferred direction from its average fluorescence response to different grating directions. Decode a presented stimulus using the response-weighted population vector.

**Evaluation:** Mean circular absolute angular error across the first 20 valid stimulus presentations.

| Model                | Mean angular error |
| -------------------- | -----------------: |
| 5-neuron population  |            100.76° |
| 50-neuron population |            101.96° |

### Observations

Increasing the number of neurons from 5 to 50 did not improve decoding accuracy in this preliminary test.

The 50-neuron population had an uneven distribution of preferred directions. Twenty neurons preferred 0°, while only one preferred 135°.

Predictions frequently clustered near 0°, suggesting possible population bias.

### Limitations

- Only 20 stimulus presentations were evaluated.
- Neurons were selected by recording order rather than tuning quality.
- Preferred directions were estimated using data that included the evaluated presentations.
- Raw ΔF/F responses were used as decoding weights.
- No independent test set or uncertainty estimates were used.

## Experiment 2 — Response Normalization

**Hypothesis:** Differences in baseline activity and response magnitude may bias the population vector toward certain neurons.

**Proposed method:** Examine baseline-corrected responses and neuron-specific normalization before comparing decoding accuracy.

**Status:** In progress.

**Next steps:** Define a baseline estimation procedure, implement response correction, and evaluate the model on held-out stimulus presentations.

### Preliminary Results — Mean-Centered Population Decoder

I compared the original population vector decoder against a version that subtracts each neuron's whole-recording mean ΔF/F before decoding.

| Method | Mean angular error |
| Original, 50 neurons | 101.96° |
| Mean-centered, 50 neurons | 101.27° |

**Observation:** Mean-centering reduced the mean angular error by approximately 0.69°, but the decoder continued to make large errors for several stimulus directions.

**Interpretation:** Differences in average neural activity may contribute to decoding bias, but mean-centering alone did not substantially improve performance.

**Limitations:** Results are based on 20 stimulus presentations. Whole-recording means and preferred directions were calculated using data that included evaluation presentations.

**Next experiment:** Evaluate direction decoding on held-out data and compare against an appropriate baseline.

## Experiment 3 — Held-Out Population Decoding

**Objective:** Evaluate whether neural population activity can predict stimulus direction on presentations excluded from preferred-direction estimation.

**Dataset:** 598 valid drifting-grating presentations, split chronologically into 478 training and 120 testing presentations.

**Method:** Estimate preferred directions for 50 neurons using training presentations. Apply an uncentered population vector decoder to held-out presentations.

### Results

| Metric                                         |  Result |
| ---------------------------------------------- | ------: |
| Neural decoder mean angular error              |  82.51° |
| Constant-direction baseline mean angular error |  91.13° |
| Difference in mean error                       |   8.61° |
| Valid neural predictions                       | 119/120 |

**Interpretation:** The population vector decoder achieved a lower observed angular error than a constant-direction predictor on this test set. However, performance remains relatively poor, and the comparison excludes one undefined neural prediction.

**Limitations:** The train/test split is chronological rather than randomized or stratified. Only 50 of 207 recorded neurons were used. The baseline is simple, and the difference has not been tested for statistical significance.

**Next steps:** Compare models on identical valid trials, investigate direction-specific errors, and evaluate more robust decoding methods.

### Direction-Specific Decoding Performance

I evaluated the population vector decoder separately for each drifting-grating direction using held-out stimulus presentations.

| Stimulus direction | Mean angular error |
| ------------------ | -----------------: |
| 0°                 |              8.29° |
| 45°                |             52.24° |
| 90°                |             65.70° |
| 135°               |            146.03° |
| 180°               |            151.08° |
| 225°               |            120.53° |
| 270°               |            123.76° |
| 315°               |             20.63° |

**Finding:** The decoder performs better for directions near 0° and 315° than for directions between 135° and 270°.

**Possible explanations:** Uneven preferred-direction representation, differences in neuronal response strength, and orientation-versus-direction selectivity.

**Visualization:** `results/figures/direction_errors.png`

**Next steps:** Investigate direction selectivity and compare the population vector approach against an alternative decoding method.
