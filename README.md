# A Comparative Study Between Three Neuronal Models on Ex Vivo, Intracellular, Electrophysiological Datasets

---

## Table of contents:

[1. Description](#1-description)

[2. Computational Tools Used](#2-computational-tools-used)

[3. Summary](#3-summary)

---

### 1. Description

**A comparative study between three neuronal models on Ex Vivo, Intracellular, Electrophysiological Datasets** is the dissertation for a MSc in Computer Science (Conversion).

The code included is a sample used for the project. Some edits and updates have been made since submission.

---

This study simulates voltage traces, and compares three neuronal models:

- Leaky Integrate-and-Fire (LIF)
- Adaptive Exponential Integrate-and-Fire (AdEx)
- Hodgkin-Huxley (HH)

The following stimulus types were explored:

1. Short Square: Triple
2. Long Square
3. Ramp

Each model was evaluated with each stimulus type as collections of sweeps, and as an
individual sweep, and measured against the ex vivo sample.

The means of assessments included:

- Mean Squared Error (MSE) scores
- Matches for first and last spikes
- The number spikes produced
- Spike time matches with the target trace

All experiments were subjected to at least two tests to assess the variance in numerical results and visual traces.
Where results did not match expectations or were too varied to reach a definitive conclusion, further experiments were run.

---

### 2. Computational Tools Used

- Python
- Brian2
- Allen SDK
- NumPy
- Pandas
- Matplotlib

---

### 3. Summary

1. For stimulus types that result in uncomplicated response patterns, such as the 'Short Square: Triple' stimulus in this study, all models are able to replicate the target trace.

2. For stimulus types that result in some adaptation in their response or delayed accelerating, such as ’Long Square’ or ’Ramp’ in this study, the AdEx model returns the most accurate results for number of spikes and spike times.

3. For stimulus types that result in unexpected sensitivity in repsonse to increased input current, such as the ’Ramp’ stimulus in this study, all models fail to respond aligned with the target trace. As the threshold and reset variables are able to be adjusted in the LIF and AdEx models, it is possible to match spikes and spike times, although not voltage traces. AdEx returns the most accurate results in this form.

4. Sweeps with a narrow ISI should be separated for more accurate results.

5. MSE scores alone cannot be relied upon as an accurate measure for comparison. However, they can be considered an important evaluation tool once the point at which more accuracy has been noted.

7. The program may need to be run several times to check that best fit parameters have been found for the optimised parameters.
