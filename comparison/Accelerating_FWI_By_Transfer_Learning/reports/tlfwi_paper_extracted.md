# TL-FWI Paper Extracted Notes

Source PDF: `comparison/Accelerating_FWI_By_Transfer_Learning/[FWI, NN] TL-FWI.pdf`

Extraction note: `PDFExtracter` was not available in this Codex environment, so the PDF text was extracted locally with `pypdf` and summarized into this Markdown file. Page numbers below refer to the PDF page order.

## Bibliographic Information

| Item | Extracted information |
|:---|:---|
| Title | Accelerating Full Waveform Inversion By Transfer Learning |
| Authors | Divya Shyam Singh, Leon Herrmann, Qing Sun, Tim Burchner, Felix Dietrich, Stefan Kollmannsberger |
| Institutions | Technical University of Munich; Bauhaus-Universitat Weimar |
| Keywords | transfer learning, neural network, deep learning, adjoint optimization, Full waveform inversion |
| Code/data availability | PyTorch implementation is provided via Zenodo DOI `10.5281/zenodo.13150916` |
| PDF metadata | Created with LaTeX/hyperref on 2024-08-02 |

## One-Paragraph Summary

The paper proposes transfer learning NN-based FWI, where a U-Net is first pretrained in a supervised way to map the first-iteration conventional FWI adjoint gradient to the true density scaling field. The pretrained U-Net is then used as the neural parameterization for FWI, and its weights are further optimized using the waveform misfit. The method is compared against conventional FWI, NN-based FWI without pretraining, and conventional FWI initialized by the pretrained NN output. Across synthetic 2D damage examples, transfer learning NN-based FWI generally converges faster and reconstructs density defects with fewer artifacts, although difficult out-of-distribution cases can favor NN-based FWI without pretraining.

## Core Idea

The unknown material field is represented by a dimensionless density scaling function `gamma(x)`. Instead of directly optimizing every grid value, NN-based FWI uses a neural network to generate `gamma`. Transfer learning improves this by using a pretrained U-Net as the generator.

The key pretraining task is:

- Input: adjoint gradient from the first conventional FWI iteration, computed at an initially undamaged field.
- Output: true density scaling function.
- Loss: supervised mean squared error over generated training samples.
- Downstream task: use the pretrained U-Net inside FWI and update its weights using the waveform misfit gradient.

Important distinction:

- `epochs_pretrain = 100` in the paper/code means U-Net supervised pretraining epochs.
- Downstream FWI iterations are separate. In the paper's main comparison, the methods are typically run for 35 FWI iterations.
- In our `scripts/run_salt_fwi.py`, `--epochs 100` means FWI optimizer iterations, not pretrained U-Net epochs.

## Physical and Numerical Model

Pages: 3-4

The paper uses a scalar wave equation with density scaling:

- Density is represented as `rho(x) = gamma(x) * rho0`.
- Wave speed is kept fixed as `c(x) = c0`.
- The inversion target is `gamma(x)`, where void/damage regions have very small values and undamaged regions are near 1.

Numerical setup:

| Parameter | Value |
|:---|:---|
| Domain | `0.1 m x 0.05 m` |
| Reference simulation grid | `512 x 256` |
| FWI grid | `256 x 128` |
| Reference time steps | `2000` |
| FWI time steps | `1000` |
| Reference `dt` | `3e-8 s` |
| FWI `dt` | `6e-8 s` |
| Undamaged density | `2700 kg/m^3` |
| Wave speed | `6000 m/s` |
| Source | two-cycle sine burst |
| Central frequency | `500 kHz` |
| Sources | 4 sources along the top edge |
| Sensors | 24 sensors along the top edge |
| Sensor spacing | 6 grid points |
| Source/sensor layout | distributed symmetrically about the center |

The reference simulation and FWI grid differ to avoid inverse crime.

## Compared Methods

Pages: 4-7

The paper compares four methods:

| Method | Description |
|:---|:---|
| Conventional FWI | Directly optimizes grid coefficients of `gamma` using adjoint gradients. |
| NN-based FWI | Optimizes NN weights; the NN generates `gamma`. Uses adjoint gradient with automatic differentiation. |
| Conventional FWI with NN initial guess | Uses the pretrained U-Net prediction only as an initial model, then performs conventional FWI. |
| Transfer learning NN-based FWI | Starts from pretrained U-Net weights and continues optimizing the U-Net weights using waveform misfit. |

## NN-Based FWI

Pages: 5-6

NN-based FWI parameterizes `gamma` by a neural network:

- Network input: random noise tensor.
- Network output: density scaling field.
- Gradient computation is hybrid:
  - adjoint method gives gradient with respect to `gamma`;
  - automatic differentiation maps that gradient to NN parameters.
- Optimizer: Adam.

The generator network maps a random input tensor of size `128 x 8 x 4` to an output field equivalent to the finite-difference grid size.

## Transfer Learning Workflow

Pages: 5-7

The transfer learning workflow has two phases.

Pretraining phase:

1. Generate training samples with synthetic damage fields.
2. For each sample, compute observed wavefields.
3. Run the first conventional FWI gradient computation at an initially undamaged `gamma0 = 1`.
4. Train a U-Net to map that first adjoint gradient to the true `gamma`.

Downstream FWI phase:

1. Compute the initial adjoint gradient for the test case.
2. Feed it to the pretrained U-Net.
3. Use the U-Net output as the generated `gamma`.
4. Minimize waveform misfit by updating U-Net weights.

The paper emphasizes that using the adjoint gradient as input is more expensive than using only observed wavefields because an adjoint computation is needed for each training sample, but it improves initialization quality.

## Pretraining Details

Pages: 6, 19-20

| Item | Value |
|:---|:---|
| Network | U-Net convolutional NN |
| Training samples used for final comparison | 800 |
| Pretraining epochs used for final comparison | 100 |
| Batch size | 80 for 800 samples |
| Loss | MSE between predicted and true `gamma` |
| Optimizer | RMSprop |
| Hardware reported | Nvidia Quadro RTX 8000 |
| Reported pretraining time | about 240 seconds |

The paper reports a parameter study:

- Pretraining epochs tested: 50, 100, 200, 300, 500.
- Pretraining sample counts tested: 200, 300, 400, 500, 800.
- 100 pretraining epochs performed best for the downstream FWI task.
- More pretraining samples improved downstream performance, with 800 samples used for final comparisons.

## Hyperparameters

Page: 19

Hyperparameters for the four downstream methods:

| Hyperparameter | Transfer learning NN-FWI | Conventional FWI with NN initial guess | NN-based FWI | Conventional FWI |
|:---|---:|---:|---:|---:|
| Learning rate | `5e-4` | `5e-2` | `5e-4` | `8e-2` |
| Gradient clipping | `1e-5` | `1e-5` | `5e-5` | `1e-5` |
| Cost scaling | `1e10` | `1e12` | `1e8` | `1e12` |

Learning-rate scheduler:

- polynomial decay of form `(beta * epoch + 1)^alpha`;
- `alpha = -0.5`;
- `beta = 0.2`.

Pretraining hyperparameters:

| Hyperparameter | Value |
|:---|---:|
| Learning rate | `8e-4` |
| Gradient clipping | `5e-5` |
| Scheduler alpha | `-0.5` |
| Scheduler beta | `0.2` |
| Batch size | number of samples divided by 10 |

## Network Architectures

Pages: 17-18

### Generator Network for NN-Based FWI

- Purpose: generate `gamma` from random input.
- Input tensor: `128 x 8 x 4`.
- Output tensor: approximately `1 x 256 x 128`.
- Total parameters: `526,252`.
- Main operations:
  - repeated upsampling;
  - 2D convolution;
  - PReLU activations;
  - final adaptive sigmoid.

### U-Net for Transfer Learning

- Purpose: map first-iteration adjoint gradient to `gamma`.
- Input tensor: `1 x 256 x 128`.
- Output tensor: `1 x 256 x 128`.
- Total parameters: `784,039`.
- Main operations:
  - 2D convolution;
  - batch normalization;
  - PReLU;
  - max pooling;
  - upsampling;
  - skip connections;
  - final sigmoid.

## Damage Data

Pages: 7-8

Training data:

- Synthetic 2D damage cases.
- Training samples contain a single ellipsoid-shaped damage.
- Ellipse parameters include axes, center coordinates, and rotation angle.
- Damage/void regions use a very small density scaling value.
- Undamaged regions use values near 1.

Evaluation data:

- Four visual case studies with increasing complexity.
- 100 additional test cases for averaged quantitative comparison.
- The 100 test cases are not part of the U-Net pretraining set.

## Results by Case

Pages: 8-15

### Case 1: Single Elliptical Damage

- Conventional FWI finds the damage in roughly 10-15 iterations but has artifacts.
- NN-based FWI reconstructs the damage within about 10 iterations and with fewer artifacts.
- Conventional FWI with NN initial guess improves over conventional FWI.
- Transfer learning NN-based FWI reconstructs the damage accurately within about 5 iterations and performs best.

### Case 2: Rectangular and Circular Damage

- Conventional FWI reconstructs approximate shape in about 20-25 iterations but produces artifacts.
- NN-based FWI recovers both damages within about 20 iterations.
- Conventional FWI with NN initial guess improves the reconstruction but struggles with the circular damage.
- Transfer learning NN-based FWI gives an almost perfect reconstruction within about 10 iterations.

### Case 3: Three Holes

- Conventional FWI recovers only part of the damage and has many artifacts.
- NN-based FWI recovers one damage but struggles with the others.
- Conventional FWI with NN initial guess behaves similarly to NN-based FWI without pretraining.
- Transfer learning NN-based FWI recovers all hole locations within about 20 iterations and has negligible artifacts.

### Case 4: Difficult Out-of-Distribution Multi-Damage Case

- This case contains multiple intersecting and circular damages and differs strongly from training data.
- Transfer learning NN-based FWI performs poorly at 35 iterations and still fails to recover the full shape even after 100 iterations.
- NN-based FWI without pretraining performs best in this difficult case.
- The paper uses this example to show that transfer learning is not always best when the pretrained initial guess is too inaccurate.

## Average over 100 Test Cases

Pages: 14-15

Setup:

- 100 test cases.
- 35 FWI iterations per method.
- Metrics:
  - waveform cost;
  - MSE between predicted `gamma` and true `gamma`.

Observed trends:

- Transfer learning NN-based FWI reduces the cost fastest up to about 20 iterations.
- After 35 iterations, NN-based FWI and transfer learning NN-based FWI have comparable cost.
- Transfer learning NN-based FWI has the lowest MSE.
- NN-based FWI has the second-best MSE.
- Conventional FWI and conventional FWI with NN initial guess can show increasing MSE due to reconstruction artifacts.

Interpretation:

- NN parameterization appears to reduce high-frequency artifacts.
- The paper attributes this to neural network spectral bias.

## Bad Initial Guess Experiment

Pages: 15-16

The paper tests whether transfer learning FWI can recover from a deliberately bad initial guess.

Findings:

- With a bad initial guess, transfer learning FWI may require many more iterations.
- In the revisited Case 2 example, it recovers the damage after roughly 70 iterations.
- For more complex cases, recovery quality can remain worse than with a good initial guess.

## Limitations

Page: 16

The paper lists several limitations:

- Hyperparameter tuning is time-intensive.
- Important hyperparameters include learning rate, scheduler, gradient clipping, cost scaling, and number of pretraining samples.
- Training data selection is critical.
- Too-similar training data can overfit and reduce generalization.
- The method becomes harder to tune as domain size and damage complexity increase.
- Transfer learning can fail for strongly out-of-distribution damage patterns.

## Conclusion

Page: 16

Main conclusions:

- Supervised pretraining can improve NN-based FWI.
- The U-Net is pretrained to map first-iteration conventional FWI gradients to true density scaling fields.
- Pretraining reduces dependence on random NN initialization.
- Transfer learning NN-based FWI improves convergence speed and reconstruction quality compared with conventional FWI.
- The method also outperforms conventional FWI initialized with the same pretrained NN prediction, which suggests that continuing to optimize the NN parameterization is important.
- The authors note that future work should investigate additional regularization and whether conventional FWI with regularization can reach similar quality.

## Information Relevant to Our TL-FWI Implementation

The paper's actual TL-FWI method is not simply conventional FWI for 100 iterations. It includes:

1. U-Net supervised pretraining using first-iteration adjoint gradients.
2. Loading pretrained U-Net weights.
3. Using the adjoint gradient as the U-Net input for a target case.
4. Updating U-Net weights during FWI.

Our current `scripts/run_salt_fwi.py` differs:

- It uses the Zenodo finite-difference and adjoint code.
- It directly optimizes a `gamma_pred` tensor with Adam.
- It does not load `model_Unet_pretrained_800`.
- It does not use `TransferLearningFWI.py`.
- Its `--epochs` argument is the number of downstream FWI iterations.

Therefore, our current experiment is best described as a Zenodo-solver FWI adapter for Efficient-style Salt data, not a complete reproduction of the paper's transfer learning NN-FWI pipeline.

## Extraction Checklist

- Title/authors extracted: yes.
- Abstract summarized: yes.
- Methodology summarized: yes.
- Numerical configuration extracted: yes.
- Compared methods extracted: yes.
- Hyperparameters extracted: yes.
- Pretraining details extracted: yes.
- Architectures summarized: yes.
- Results and limitations summarized: yes.
- Full verbatim paper text copied: no.
