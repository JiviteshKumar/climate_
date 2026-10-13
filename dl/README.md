# Deep Learning laboratory — Experiments 02, 03, 07, 08

Jivitesh Kumar · Roll No. 16010423041 · Batch A2 · KJSSE, Semester VII

**Dataset:** [PerceptiLabs Weather-Analysis](https://github.com/PerceptiLabs/Weather-Analysis)
(Multi-class Weather Dataset) — 1123 RGB images, four classes: cloud 300, rain 214,
shine 252, sunrise 357. This is the image corpus for the Climate Intelligence use case
(KJS-CES-01): classifying sky condition is the vision counterpart of region-wise
heatwave severity.

## Run

```bash
pip install tensorflow-cpu scikit-learn matplotlib pillow
set WEATHER_DIR=<path to the cloned Weather-Analysis folder>
python exp2_ffnn.py && python exp3_dnn.py && python exp7_rnn.py && python exp8_lstm.py
```

## Files

| File | Experiment | What it does |
| --- | --- | --- |
| `dl_common.py` | shared | Loading, feature extraction, class-balanced splits |
| `exp2_ffnn.py` | 02 | Feedforward network written from scratch in NumPy, forward pass only |
| `exp3_dnn.py` | 03 | Deep Keras network; activation and hyperparameter sweeps |
| `exp7_rnn.py` | 07 | SimpleRNN over scan-line sequences |
| `exp8_lstm.py` | 08 | LSTM vs SimpleRNN vs GRU, plus a direct gradient-flow measurement |

## Results

| Experiment | Model | Test accuracy | Note |
| --- | --- | --- | --- |
| 02 | From-scratch FFNN (untrained) | 24.04 % | Matches chance (25 %); Keras cross-check agrees to 3.4e-08 |
| 03 | Deep NN, 1.81 M params | **84.71 %** | ELU best of four activations; 88.24 % without dropout but overfitting |
| 07 | SimpleRNN, 13.5 K params | 79.41 % | 134× fewer parameters than the dense model |
| 08 | LSTM, 53.8 K params | **84.12 %** | GRU 86.47 %; LSTM loses only 0.59 pts at T=64 vs RNN's 4.71 |

### Sequence framing

Experiments 07 and 08 read each photograph as an ordered sequence of 32 horizontal scan
lines, top to bottom. This is meaningful for weather imagery because the discriminative
signal is the vertical structure of the scene — sky to horizon to ground.

### Vanishing gradient, measured

Doubling the sequence length to 64 steps costs the SimpleRNN **4.71 points** of accuracy
and the LSTM only **0.59**. Measuring the gradient of the loss with respect to the input
at each time step: the SimpleRNN decays ~6 orders of magnitude from the last step back to
the first, the LSTM ~3 — the LSTM delivers **401× more gradient** to step 0.
