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

## Experiment 4 — Confusion Matrix and Normalization

Today I looked at which directions the decoder was getting right and wrong. I made a confusion matrix and found that it was predicting 0° a lot, even when the actual direction was different.

I also tried normalizing the neurons' responses to see if that would improve accuracy.

**Results:**

- Original decoder: 27.73%
- Baseline-corrected decoder: 29.41%
- Normalized decoder: 21.85%

Normalization actually made the decoder worse. This might mean that neurons with stronger responses are giving us useful information, and scaling them down makes predictions less accurate.

I also had to fix some Python errors and handle one prediction that returned NaN.

**Next:** Try a different decoding method and see if we can get better accuracy, especially for directions between 135° and 270°.

## Experiment 5 — Building a Neural Network Decoder

Today I built my first neural network using PyTorch to predict stimulus direction from neural activity.

Instead of using the population vector method, I trained a multilayer perceptron (MLP) using data from all 207 recorded neurons.

The model has 207 input features, 64 hidden neurons, and 8 output classes representing directions from 0° to 315°.

**Method:** Used normalized ΔF/F responses as inputs and trained the model for 100 epochs using cross-entropy loss and the Adam optimizer.

**Results:**

- Training accuracy: 86.61%
- Test accuracy: 60.00%
- Population vector baseline: 29.41%
- Random guessing: 12.50%

The neural network performed much better than the population vector decoder, suggesting that it can learn relationships between neural activity and stimulus direction that the simpler method misses.

I also created a confusion matrix to see which directions the model struggled with.

One thing I noticed was that training accuracy was much higher than testing accuracy, which could mean the model was overfitting.

**Next:** Investigate overfitting and experiment with ways to improve the model's performance on unseen data.

## Experiment 6 — Improving the Neural Network

Today I focused on understanding overfitting and improving the neural network.

First, I divided the data into 382 training samples, 96 validation samples, and 120 testing samples. I also fixed the normalization process so that it only used information from the training data.

I plotted training and validation loss over 100 epochs. Both decreased steadily, but training loss was lower than validation loss.

I then experimented with two regularization techniques: dropout and weight decay.

**Results:**

| Model                | Training accuracy | Validation accuracy |
| -------------------- | ----------------: | ------------------: |
| Original MLP         |            88.74% |              63.54% |
| Dropout (20%)        |            87.70% |              63.54% |
| Weight decay (0.001) |            90.31% |              65.62% |

Dropout didn't improve validation accuracy, but weight decay increased it slightly to 65.62%.

The improvement was only about 2 percentage points, so I can't say for sure that weight decay is consistently better without running more experiments.

I also added a random seed to make the experiments more reproducible and saved the trained model in `models/visual_cortex_decoder.pth`.

**What I learned:** A lower training loss doesn't always mean a model will perform better on new data. Using validation data helps compare models without relying on test performance.

**Next:** Test the model with different random seeds, save the normalization parameters, and create a function that takes neural activity and predicts the stimulus direction.

## Experiment 7 — Dropout and Early Stopping

Today I continued improving the neural network by experimenting with dropout, longer training, and early stopping.

I used the same 382 training samples, 96 validation samples, and 120 testing samples. I also kept the learning rate at 0.001 and weight decay at 0.001.

First, I compared the model with and without 30% dropout over 100 epochs.

**Results:**

| Model                    | Training accuracy | Validation accuracy | Test accuracy |
| ------------------------ | ----------------: | ------------------: | ------------: |
| No dropout (100 epochs)  |            90.31% |              65.62% |        58.33% |
| Dropout 30% (100 epochs) |            88.74% |              63.54% |        57.50% |
| Dropout 30% (300 epochs) |            99.74% |              75.00% |        66.67% |

Adding dropout initially made accuracy slightly worse. However, increasing training to 300 epochs improved the model's performance significantly.

I also implemented early stopping, which saves the model weights with the lowest validation loss and stops training if validation loss doesn't improve for 30 epochs.

**Final results:**

- Best validation loss: 0.8078
- Training accuracy: 99.74%
- Validation accuracy: 75.00%
- Test accuracy: 66.67%
- Training completed all 300 epochs.

The loss graph showed that training loss continued decreasing while validation loss started flattening around epoch 250. This suggests the model is beginning to overfit, even with dropout and weight decay.

My best test accuracy so far is **66.67%**, compared to 60.00% from my original neural network.

I also saved the best model weights in `results/best_decoder.pt`.

**What I learned:** Training for more epochs can improve performance, but eventually the model starts fitting the training data much better than the validation data. Early stopping helps preserve the best model instead of automatically using the weights from the final epoch.

**Next:** Compare different model settings using validation results, test with multiple random seeds, and save the normalization parameters so the model can make predictions on new neural activity.

### Experiment 8 — Regularization and Model Comparison

Today I trained a logistic regression model to compare against my PyTorch MLP. Logistic regression achieved 62.5% test accuracy, slightly higher than the original MLP's 60%.

I also updated the MLP with dropout (0.3), weight decay (0.001), and early stopping. The updated model stopped at epoch 144 and achieved 98.69% training accuracy, 70.83% validation accuracy, and 61.67% test accuracy.

The model still seems to overfit because training accuracy is much higher than test accuracy. Logistic regression performed slightly better on this test set, suggesting that a simpler model can already extract useful information from the neural activity.

Next, I want to investigate which neurons contribute most to decoding accuracy and whether reducing the number of input features improves generalization.

### Experiment 9 — Neuron Importance Analysis

Investigated which neurons contribute most to decoding visual stimulus direction using permutation importance.

I shuffled each neuron's activity and measured how much the MLP's validation accuracy decreased. Neurons 84, 80, and 67 had the highest importance scores.

I then trained logistic regression models using different numbers of neurons.

| Neurons | Training Accuracy | Validation Accuracy |
| ------- | ----------------- | ------------------- |
| 10      | 34.29%            | 35.42%              |
| 25      | 53.93%            | 45.83%              |
| 50      | 70.42%            | 55.21%              |
| 100     | 88.74%            | 67.71%              |
| 207     | 97.91%            | 72.92%              |

**Observations:**

Validation accuracy increased as more neurons were included, suggesting that visual stimulus information is distributed across the population rather than concentrated in a few individual neurons.

The gap between training and validation accuracy also increased with more neurons, suggesting overfitting.

Because neuron importance was calculated using validation data, these results are exploratory and need further evaluation using training-only feature selection.

**Figures saved:**

- `results/figures/neuron_importance.png`
- `results/figures/neuron_count_accuracy.png`

**Next steps:** Investigate whether PCA can reduce the number of neural features while preserving decoding accuracy. Compare PCA with Oja's learning rule to explore biologically inspired dimensionality reduction.
