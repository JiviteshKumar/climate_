# -*- coding: utf-8 -*-
"""
DL Experiment 02 -- Feed Forward Neural Network, implemented from scratch.

Architecture (as required by the activity sheet):
    input layer (8)  ->  hidden layer 1 (6, tanh)
                     ->  hidden layer 2 (5, tanh)
                     ->  output layer  (4, softmax)

Only FORWARD propagation is implemented, which is what the activity specifies;
the weights are created as sample matrices of size M x N, where M is the number
of neurons in the layer and N the number of neurons in the next layer.

Jivitesh Kumar | Roll No. 16010423041 | Batch A2 | KJSSE, Sem VII
Run:  python exp2_ffnn.py
"""

import os
import numpy as np
from dl_common import (handcrafted_features, CLASSES, banner, class_table,
                       stratified_split, SEED)

np.set_printoptions(precision=4, suppress=True, linewidth=110)
OUT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------- activation --
def tanh(z):
    return np.tanh(z)


def softmax(z):
    """Numerically stable softmax over the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


banner("STEP 1  Import libraries and load the dataset")
print(" numpy %s" % np.__version__)
print(" Dataset: PerceptiLabs Weather-Analysis (Multi-class Weather Dataset)")
class_table()

banner("STEP 2  Build the input feature vector for each image")
F, y, names = handcrafted_features()
print("\n Feature matrix : %s" % (F.shape,))
print(" Features       : %s" % ", ".join(names))

# standardise so that every input signal is on a comparable scale
mu, sd = F.mean(axis=0), F.std(axis=0)
Xs = (F - mu) / sd
tr, va, te = stratified_split(y)
print(" Split          : %d train / %d val / %d test" % (len(tr), len(va), len(te)))

print("\n Class-mean of each feature (raw units), which is what the network sees:")
print(" %-10s %s" % ("CLASS", " ".join("%9s" % n[:9] for n in names)))
print(" " + "-" * 94)
for i, c in enumerate(CLASSES):
    print(" %-10s %s" % (c, " ".join("%9.3f" % v for v in F[y == i].mean(axis=0))))

# ------------------------------------------------------------------ weights --
banner("STEP 3  Create the sample weight matrices")
rng = np.random.default_rng(SEED)
N_IN, N_H1, N_H2, N_OUT = len(names), 6, 5, len(CLASSES)

# Xavier / Glorot scaling keeps the tanh units inside their useful range
W1 = rng.normal(0, np.sqrt(1.0 / N_IN), (N_IN, N_H1));  b1 = np.zeros(N_H1)
W2 = rng.normal(0, np.sqrt(1.0 / N_H1), (N_H1, N_H2));  b2 = np.zeros(N_H2)
W3 = rng.normal(0, np.sqrt(1.0 / N_H2), (N_H2, N_OUT)); b3 = np.zeros(N_OUT)

for nm, W, b in [("W1 (input -> hidden 1)", W1, b1),
                 ("W2 (hidden 1 -> hidden 2)", W2, b2),
                 ("W3 (hidden 2 -> output)", W3, b3)]:
    print(" %-28s shape %-9s bias %s" % (nm, str(W.shape), b.shape))
print("\n W1 =")
print(W1)

# ------------------------------------------------------------ forward pass --
def forward(x, verbose=False):
    """Propagate one input signal through the network to the output layer."""
    z1 = x @ W1 + b1;  a1 = tanh(z1)      # step 5: hidden layer 1, tanh
    z2 = a1 @ W2 + b2; a2 = tanh(z2)      # step 6: hidden layer 2, tanh
    z3 = a2 @ W3 + b3                     # step 7: output layer, linear
    p = softmax(z3)                       # step 8: softmax probabilities
    if verbose:
        print("   z1 (weighted sum, layer 1) =", z1)
        print("   a1 = tanh(z1)              =", a1)
        print("   z2 (weighted sum, layer 2) =", z2)
        print("   a2 = tanh(z2)              =", a2)
        print("   z3 (weighted sum, output)  =", z3)
        print("   p  = softmax(z3)           =", p)
        print("   sum(p) = %.6f" % p.sum())
    return p


banner("STEP 4  Forward propagate one input signal, layer by layer")
i0 = int(te[0])
x0 = Xs[i0]
print(" Sample image index %d, true class '%s'" % (i0, CLASSES[y[i0]]))
print(" Standardised input signal x =", x0)
print()
p0 = forward(x0, verbose=True)
print("\n Predicted class: %s  (p = %.4f)" % (CLASSES[int(p0.argmax())], p0.max()))

banner("STEP 5  Forward propagate a batch and read the softmax output")
batch = te[:8]
P = forward(Xs[batch])
print(" %-6s %-10s %s %-10s" % ("IDX", "TRUE", "  ".join("%8s" % c for c in CLASSES), "PREDICTED"))
print(" " + "-" * 74)
for k, i in enumerate(batch):
    print(" %-6d %-10s %s %-10s" % (i, CLASSES[y[i]],
                                    "  ".join("%8.4f" % v for v in P[k]),
                                    CLASSES[int(P[k].argmax())]))

banner("STEP 6  Verify the implementation against Keras")
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras import layers, models
    ref = models.Sequential([
        layers.Input(shape=(N_IN,)),
        layers.Dense(N_H1, activation="tanh"),
        layers.Dense(N_H2, activation="tanh"),
        layers.Dense(N_OUT, activation="softmax"),
    ])
    ref.layers[0].set_weights([W1, b1])
    ref.layers[1].set_weights([W2, b2])
    ref.layers[2].set_weights([W3, b3])
    P_keras = ref.predict(Xs[batch], verbose=0)
    err = np.abs(P - P_keras).max()
    print(" TensorFlow %s" % tf.__version__)
    print(" Maximum absolute difference between the from-scratch forward pass")
    print(" and the equivalent Keras model : %.3e" % err)
    print(" VERDICT: %s" % ("MATCH - the manual implementation is correct"
                            if err < 1e-5 else "MISMATCH"))
except Exception as e:                                   # pragma: no cover
    print(" Keras cross-check skipped:", e)

banner("STEP 7  Behaviour of the untrained network")
P_all = forward(Xs)
pred = P_all.argmax(axis=1)
acc = (pred == y).mean()
print(" Accuracy with random (untrained) weights : %.2f %%" % (100 * acc))
print(" Accuracy expected from random guessing   : %.2f %%" % (100.0 / len(CLASSES)))
print(" Mean maximum softmax probability          : %.4f" % P_all.max(axis=1).mean())
print(" Entropy of the mean output distribution   : %.4f nats (max %.4f)"
      % (-(P_all.mean(axis=0) * np.log(P_all.mean(axis=0))).sum(), np.log(len(CLASSES))))
print("""
 The network is close to chance and its output distribution is close to uniform.
 That is the correct result: forward propagation alone only evaluates the
 function the current weights happen to represent. The weights carry no
 information about the data until backpropagation adjusts them, which is the
 subject of Experiment 01 and is applied in Experiment 03.""")

np.savez(os.path.join(OUT, "exp2_weights.npz"),
         W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, mu=mu, sd=sd)
banner("DONE")
print(" Sample weights written to exp2_weights.npz")
