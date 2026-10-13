# -*- coding: utf-8 -*-
"""
DL Experiment 07 -- Recurrent Neural Network.

The Weather-Analysis corpus is image data, so to apply a sequence model each
photograph is read as an ordered sequence of horizontal scan lines, from the top
of the frame to the bottom. That ordering is meaningful for weather imagery: the
discriminative signal is the vertical structure of the scene, the gradient from
sky to horizon to ground. A sunrise is a strong warm-to-dark gradient down the
frame, an overcast sky is flat and bright throughout, and rain is dark and
low-contrast. Each row is therefore one time step and the RNN reads the image
the way the eye scans a horizon.

Jivitesh Kumar | Roll No. 16010423041 | Batch A2 | KJSSE, Sem VII
Run:  python exp7_rnn.py
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
SIZE = 32
PAL = ["#0b5fa5", "#e2620d", "#12805c", "#8b1c34"]

banner("STEP 1  Import libraries and load the dataset")
print(" TensorFlow %s  |  Keras %s" % (tf.__version__, tf.keras.__version__))
class_table()
X, y = load_images(size=SIZE)

banner("STEP 2  Pre-process: turn each image into a sequence")
TIMESTEPS, FEATURES = SIZE, SIZE * 3
Xs = X.reshape(len(X), TIMESTEPS, FEATURES)
print(" Image tensor      : %s" % (X.shape,))
print(" Sequence tensor   : %s" % (Xs.shape,))
print(" Time steps        : %d  (one per image row, top to bottom)" % TIMESTEPS)
print(" Features per step : %d  (%d pixels x 3 colour channels)" % (FEATURES, SIZE))

tr, va, te = stratified_split(y)
Xtr, Xva, Xte = Xs[tr], Xs[va], Xs[te]
ytr, yva, yte = y[tr], y[va], y[te]
print(" Split             : %d train / %d val / %d test" % (len(tr), len(va), len(te)))

banner("STEP 3  Visualise the sequence representation")
lum = (0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]).mean(axis=2)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
for i, c in enumerate(CLASSES):
    prof = lum[y == i].mean(axis=0)
    ax[0].plot(range(TIMESTEPS), prof, color=PAL[i], lw=2.2, label=c)
    sd = lum[y == i].std(axis=0)
    ax[0].fill_between(range(TIMESTEPS), prof - sd, prof + sd, color=PAL[i], alpha=.10)
ax[0].set_xlabel("Time step (image row, top to bottom)")
ax[0].set_ylabel("Mean row luminance")
ax[0].set_title("Class-mean row-luminance sequence", fontsize=11, weight="bold")
ax[0].legend(frameon=False, fontsize=9)
sat = (X.max(axis=3) - X.min(axis=3)).mean(axis=2)
for i, c in enumerate(CLASSES):
    ax[1].plot(range(TIMESTEPS), sat[y == i].mean(axis=0), color=PAL[i], lw=2.2, label=c)
ax[1].set_xlabel("Time step (image row, top to bottom)")
ax[1].set_ylabel("Mean row saturation")
ax[1].set_title("Class-mean row-saturation sequence", fontsize=11, weight="bold")
for a_ in ax:
    a_.grid(alpha=.25); a_.set_axisbelow(True)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
fig.suptitle("Each image read as a 32-step sequence of scan lines", fontsize=12.5, weight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e7_sequences.png"), dpi=165); plt.close(fig)
print(" Saved figures/e7_sequences.png")
print("\n The four classes separate visibly in the sequence domain, which is what")
print(" makes a recurrent model applicable to this dataset.")

banner("STEP 4  Develop and train the RNN")


def build_rnn(units=64, lr=1e-3):
    m = models.Sequential([
        layers.Input(shape=(TIMESTEPS, FEATURES)),
        layers.SimpleRNN(units, activation="tanh", return_sequences=True, name="rnn_1"),
        layers.SimpleRNN(units // 2, activation="tanh", name="rnn_2"),
        layers.Dropout(0.25),
        layers.Dense(len(CLASSES), activation="softmax", name="output"),
    ], name="weather_rnn")
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


model = build_rnn()
model.summary(print_fn=lambda s: print(" " + s))
es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                      restore_best_weights=True)
hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=80,
                 batch_size=32, verbose=0, callbacks=[es])
print("\n Epochs run : %d" % len(hist.history["loss"]))

banner("STEP 5  Predictions on the training and the test data")
ptr = model.predict(Xtr, verbose=0).argmax(axis=1)
pte = model.predict(Xte, verbose=0).argmax(axis=1)
acc_tr, acc_te = (ptr == ytr).mean(), (pte == yte).mean()
loss_te, _ = model.evaluate(Xte, yte, verbose=0)
print(" Training accuracy : %.2f %%" % (100 * acc_tr))
print(" Test accuracy     : %.2f %%" % (100 * acc_te))
print(" Test loss         : %.4f" % loss_te)
print(" Generalisation gap: %+.2f points" % (100 * (acc_tr - acc_te)))
print("\n" + classification_report(yte, pte, target_names=CLASSES, digits=3))
cm = confusion_matrix(yte, pte)
print(" Confusion matrix (rows = true, cols = predicted)")
print(" %-10s %s" % ("", " ".join("%8s" % c for c in CLASSES)))
for i, c in enumerate(CLASSES):
    print(" %-10s %s" % (c, " ".join("%8d" % v for v in cm[i])))

banner("STEP 6  Plot the predictions")
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
ax[0].plot(hist.history["accuracy"], color=PAL[0], lw=2, label="train")
ax[0].plot(hist.history["val_accuracy"], color=PAL[1], lw=2, label="validation")
ax[0].set_title("Accuracy", fontsize=11, weight="bold"); ax[0].set_ylabel("Accuracy")
ax[1].plot(hist.history["loss"], color=PAL[0], lw=2, label="train")
ax[1].plot(hist.history["val_loss"], color=PAL[1], lw=2, label="validation")
ax[1].set_title("Loss", fontsize=11, weight="bold"); ax[1].set_ylabel("Loss")
for a_ in ax[:2]:
    a_.set_xlabel("Epoch"); a_.grid(alpha=.25); a_.set_axisbelow(True)
    a_.legend(frameon=False, fontsize=9)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
w = 0.38
xs = np.arange(len(CLASSES))
tr_acc = [(ptr[ytr == i] == i).mean() for i in range(len(CLASSES))]
te_acc = [(pte[yte == i] == i).mean() for i in range(len(CLASSES))]
ax[2].bar(xs - w / 2, tr_acc, w, color=PAL[0], label="train", edgecolor="white")
ax[2].bar(xs + w / 2, te_acc, w, color=PAL[1], label="test", edgecolor="white")
ax[2].set_xticks(xs); ax[2].set_xticklabels(CLASSES, rotation=20, ha="right")
ax[2].set_ylabel("Per-class recall"); ax[2].set_ylim(0, 1.05)
ax[2].set_title("Predictions by class", fontsize=11, weight="bold")
ax[2].legend(frameon=False, fontsize=9); ax[2].grid(axis="y", alpha=.25); ax[2].set_axisbelow(True)
for s in ("top", "right"): ax[2].spines[s].set_visible(False)
fig.suptitle("Simple RNN on scan-line sequences (test accuracy %.1f %%)" % (100 * acc_te),
             fontsize=12.5, weight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e7_rnn.png"), dpi=165); plt.close(fig)

np.savez(os.path.join(OUT, "exp7_results.npz"), cm=cm, acc_tr=acc_tr, acc_te=acc_te,
         loss_te=loss_te, hist=np.array([hist.history["val_accuracy"]], dtype=object),
         epochs=len(hist.history["loss"]))
banner("DONE")
print(" Figures: e7_sequences.png, e7_rnn.png")
