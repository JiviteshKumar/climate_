# -*- coding: utf-8 -*-
"""
DL Experiment 08 -- LSTM network.

Same scan-line sequence framing as Experiment 07, so that the LSTM can be
compared directly against the SimpleRNN on identical data. Two things are
measured: accuracy at the sequence length used in Experiment 07, and what
happens to both models when the sequence is made twice as long, which is where
the vanishing-gradient weakness of the simple RNN should appear.

Jivitesh Kumar | Roll No. 16010423041 | Batch A2 | KJSSE, Sem VII
Run:  python exp8_lstm.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from dl_common import load_images, stratified_split, CLASSES, banner, class_table, SEED

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)
PAL = ["#0b5fa5", "#e2620d", "#12805c", "#8b1c34"]


def sequences(size):
    X, y = load_images(size=size, verbose=False)
    return X.reshape(len(X), size, size * 3), y


def build(kind, T, F, units=64, lr=1e-3):
    cell = {"SimpleRNN": layers.SimpleRNN, "LSTM": layers.LSTM, "GRU": layers.GRU}[kind]
    m = models.Sequential([
        layers.Input(shape=(T, F)),
        cell(units, return_sequences=True, name="rec_1"),
        cell(units // 2, name="rec_2"),
        layers.Dropout(0.25),
        layers.Dense(len(CLASSES), activation="softmax", name="output"),
    ], name="weather_%s" % kind.lower())
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def train(kind, Xs, y, tr, va, te, epochs=80):
    tf.keras.utils.set_random_seed(SEED)
    T, F = Xs.shape[1], Xs.shape[2]
    m = build(kind, T, F)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                          restore_best_weights=True)
    h = m.fit(Xs[tr], y[tr], validation_data=(Xs[va], y[va]), epochs=epochs,
              batch_size=32, verbose=0, callbacks=[es])
    acc_tr = (m.predict(Xs[tr], verbose=0).argmax(1) == y[tr]).mean()
    acc_te = (m.predict(Xs[te], verbose=0).argmax(1) == y[te]).mean()
    loss_te, _ = m.evaluate(Xs[te], y[te], verbose=0)
    return m, h, acc_tr, acc_te, loss_te


banner("STEP 1  Load and pre-process the dataset")
print(" TensorFlow %s  |  Keras %s" % (tf.__version__, tf.keras.__version__))
class_table()
X32, y = sequences(32)
tr, va, te = stratified_split(y)
print("\n Sequence tensor : %s  (%d steps x %d features)" % (X32.shape, X32.shape[1], X32.shape[2]))
print(" Split           : %d train / %d val / %d test" % (len(tr), len(va), len(te)))

banner("STEP 2  Develop and train the LSTM")
lstm, hl, ltr, lte, lloss = train("LSTM", X32, y, tr, va, te)
lstm.summary(print_fn=lambda s: print(" " + s))
print("\n Epochs run        : %d" % len(hl.history["loss"]))
print(" Training accuracy : %.2f %%" % (100 * ltr))
print(" Test accuracy     : %.2f %%" % (100 * lte))
print(" Test loss         : %.4f" % lloss)

pte = lstm.predict(X32[te], verbose=0).argmax(1)
print("\n" + classification_report(y[te], pte, target_names=CLASSES, digits=3))
cm = confusion_matrix(y[te], pte)
print(" Confusion matrix (rows = true, cols = predicted)")
print(" %-10s %s" % ("", " ".join("%8s" % c for c in CLASSES)))
for i, c in enumerate(CLASSES):
    print(" %-10s %s" % (c, " ".join("%8d" % v for v in cm[i])))

banner("STEP 3  LSTM against SimpleRNN and GRU, sequence length 32")
rows, hists = [], {"LSTM": hl.history}
for kind in ["SimpleRNN", "GRU"]:
    m, h, a_tr, a_te, lo = train(kind, X32, y, tr, va, te)
    rows.append((kind, m.count_params(), a_tr, a_te, lo, len(h.history["loss"])))
    hists[kind] = h.history
rows.append(("LSTM", lstm.count_params(), ltr, lte, lloss, len(hl.history["loss"])))
rows.sort(key=lambda r: -r[3])
print(" %-11s %10s %10s %10s %9s %8s" % ("MODEL", "PARAMS", "TRAIN ACC", "TEST ACC", "TEST LOSS", "EPOCHS"))
print(" " + "-" * 64)
for k, p, a_tr, a_te, lo, ep in rows:
    print(" %-11s %10d %9.2f%% %9.2f%% %9.4f %8d" % (k, p, 100 * a_tr, 100 * a_te, lo, ep))

banner("STEP 4  The vanishing gradient: doubling the sequence length")
print(" The same images are re-read at 64 x 64, giving 64 time steps instead of 32.")
print(" The information the model needs is still the top-to-bottom structure, so")
print(" the dependency it must carry is now twice as long.\n")
X64, y64 = sequences(64)
tr2, va2, te2 = stratified_split(y64)
long_rows = []
for kind in ["SimpleRNN", "LSTM"]:
    m, h, a_tr, a_te, lo = train(kind, X64, y64, tr2, va2, te2)
    long_rows.append((kind, a_te, lo, len(h.history["loss"])))
print(" %-11s %12s %12s %9s" % ("MODEL", "T=32 TEST", "T=64 TEST", "CHANGE"))
print(" " + "-" * 50)
base = {k: a for k, _, _, a, _, _ in rows}
for kind, a_te, lo, ep in long_rows:
    b = base[kind]
    print(" %-11s %11.2f%% %11.2f%% %+8.2f pts" % (kind, 100 * b, 100 * a_te, 100 * (a_te - b)))

banner("STEP 5  Measuring the gradient directly")
print(" Gradient of the loss with respect to the input at each time step,")
print(" averaged over a batch. A simple RNN should show the signal decaying as")
print(" it is propagated back towards step 0; the LSTM should not.\n")
grad_curves = {}
for kind in ["SimpleRNN", "LSTM"]:
    tf.keras.utils.set_random_seed(SEED)
    m = build(kind, X64.shape[1], X64.shape[2])
    xb = tf.convert_to_tensor(X64[tr2[:64]])
    yb = tf.convert_to_tensor(y64[tr2[:64]])
    with tf.GradientTape() as tape:
        tape.watch(xb)
        loss = tf.keras.losses.sparse_categorical_crossentropy(yb, m(xb, training=True))
        loss = tf.reduce_mean(loss)
    g = tape.gradient(loss, xb).numpy()
    norms = np.linalg.norm(g, axis=2).mean(axis=0)
    grad_curves[kind] = norms
    ratio = norms[0] / max(norms[-1], 1e-12)
    print(" %-11s  grad norm at step 0 = %.3e   at step 63 = %.3e   ratio = %.4f"
          % (kind, norms[0], norms[-1], ratio))
rnn_r = grad_curves["SimpleRNN"][0] / max(grad_curves["SimpleRNN"][-1], 1e-12)
ls_r = grad_curves["LSTM"][0] / max(grad_curves["LSTM"][-1], 1e-12)
print("\n The simple RNN retains %.2f %% of its final-step gradient at step 0;" % (100 * rnn_r))
print(" the LSTM retains %.2f %%." % (100 * ls_r))

# ------------------------------------------------------------------ figures --
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for i, (k, h) in enumerate(hists.items()):
    ax[0].plot(h["val_accuracy"], color=PAL[i], lw=2, label=k)
ax[0].set_title("Validation accuracy, T = 32", fontsize=11, weight="bold")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy"); ax[0].legend(frameon=False, fontsize=9)
xs = np.arange(2); w = 0.36
ax[1].bar(xs - w / 2, [base["SimpleRNN"], base["LSTM"]], w, color=PAL[0], label="T = 32", edgecolor="white")
ax[1].bar(xs + w / 2, [long_rows[0][1], long_rows[1][1]], w, color=PAL[1], label="T = 64", edgecolor="white")
ax[1].set_xticks(xs); ax[1].set_xticklabels(["SimpleRNN", "LSTM"])
ax[1].set_ylabel("Test accuracy"); ax[1].set_ylim(0, 1.0)
ax[1].set_title("Effect of doubling sequence length", fontsize=11, weight="bold")
ax[1].legend(frameon=False, fontsize=9)
for i, (k, v) in enumerate(grad_curves.items()):
    ax[2].semilogy(v / v.max(), color=PAL[i], lw=2, label=k)
ax[2].set_xlabel("Time step"); ax[2].set_ylabel("Normalised gradient norm (log)")
ax[2].set_title("Gradient flow back through time", fontsize=11, weight="bold")
ax[2].legend(frameon=False, fontsize=9)
for a_ in ax:
    a_.grid(alpha=.25); a_.set_axisbelow(True)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
fig.suptitle("LSTM compared with SimpleRNN and GRU on scan-line sequences",
             fontsize=12.5, weight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e8_compare.png"), dpi=165); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(hl.history["accuracy"], color=PAL[0], lw=2, label="train")
ax[0].plot(hl.history["val_accuracy"], color=PAL[1], lw=2, label="validation")
ax[0].set_title("LSTM accuracy", fontsize=11, weight="bold"); ax[0].set_ylabel("Accuracy")
ax[1].plot(hl.history["loss"], color=PAL[0], lw=2, label="train")
ax[1].plot(hl.history["val_loss"], color=PAL[1], lw=2, label="validation")
ax[1].set_title("LSTM loss", fontsize=11, weight="bold"); ax[1].set_ylabel("Loss")
for a_ in ax:
    a_.set_xlabel("Epoch"); a_.grid(alpha=.25); a_.set_axisbelow(True); a_.legend(frameon=False, fontsize=9)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
fig.suptitle("LSTM training history (test accuracy %.1f %%)" % (100 * lte),
             fontsize=12.5, weight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e8_lstm.png"), dpi=165); plt.close(fig)

np.savez(os.path.join(OUT, "exp8_results.npz"), cm=cm, rows=np.array(rows, dtype=object),
         long_rows=np.array(long_rows, dtype=object),
         grad_rnn=grad_curves["SimpleRNN"], grad_lstm=grad_curves["LSTM"])
banner("DONE")
print(" Figures: e8_compare.png, e8_lstm.png")
