# -*- coding: utf-8 -*-
"""
DL Experiment 03 -- Deep Neural Network.

Builds a deep fully-connected network on the Weather-Analysis image dataset,
then studies the effect of the activation function and of the hyperparameters.

Jivitesh Kumar | Roll No. 16010423041 | Batch A2 | KJSSE, Sem VII
Run:  python exp3_dnn.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from dl_common import load_images, stratified_split, CLASSES, banner, class_table, SEED

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)
SIZE = 48
PAL = ["#0b5fa5", "#e2620d", "#12805c", "#8b1c34", "#c98a04"]

banner("STEP 1  Load the dataset")
print(" TensorFlow %s  |  Keras %s" % (tf.__version__, tf.keras.__version__))
class_table()
X, y = load_images(size=SIZE)
print("\n Image tensor : %s" % (X.shape,))
tr, va, te = stratified_split(y)
Xtr, Xva, Xte = X[tr], X[va], X[te]
ytr, yva, yte = y[tr], y[va], y[te]
print(" Split        : %d train / %d val / %d test" % (len(tr), len(va), len(te)))

banner("STEP 2  Define the input shape")
INPUT_SHAPE = (SIZE, SIZE, 3)
N_FEAT = int(np.prod(INPUT_SHAPE))
print(" Input shape to the network : %s" % (INPUT_SHAPE,))
print(" Flattened input dimension  : %d  (%d x %d x 3)" % (N_FEAT, SIZE, SIZE))
print(" Output units               : %d  (one per class, softmax)" % len(CLASSES))


def build(hidden=(256, 128, 64), act="relu", lr=1e-3, dropout=0.3, l2=1e-4):
    """Steps 3 and 4: model with hidden layers, output layer and hyperparameters."""
    m = models.Sequential(name="weather_dnn")
    m.add(layers.Input(shape=INPUT_SHAPE))
    m.add(layers.Flatten())
    for i, h in enumerate(hidden):
        m.add(layers.Dense(h, activation=act,
                           kernel_regularizer=regularizers.l2(l2),
                           name="dense_%d" % (i + 1)))
        if dropout:
            m.add(layers.Dropout(dropout, name="drop_%d" % (i + 1)))
    m.add(layers.Dense(len(CLASSES), activation="softmax", name="output"))
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def run(tag, epochs=60, verbose=0, **kw):
    tf.keras.utils.set_random_seed(SEED)
    m = build(**kw)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                          restore_best_weights=True)
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
              batch_size=32, verbose=verbose, callbacks=[es])
    te_loss, te_acc = m.evaluate(Xte, yte, verbose=0)
    return m, h, te_acc, te_loss


banner("STEP 3-4  Build the model and print it")
model = build()
model.summary(print_fn=lambda s: print(" " + s))
print("\n Hyperparameters chosen")
for k, v in [("Hidden layers", "256 -> 128 -> 64"), ("Activation (hidden)", "ReLU"),
             ("Activation (output)", "Softmax"), ("Optimizer", "Adam"),
             ("Learning rate", "0.001"), ("Batch size", "32"),
             ("Dropout", "0.30"), ("L2 weight decay", "1e-4"),
             ("Loss", "sparse_categorical_crossentropy"), ("Max epochs", "60 (early stopping)")]:
    print("   %-22s %s" % (k, v))

banner("STEP 5  Effect of the activation function")
acts, arows, hists = ["relu", "tanh", "sigmoid", "elu"], [], {}
for a in acts:
    m, h, acc, loss = run(a, act=a)
    ep = len(h.history["loss"])
    arows.append((a, acc, loss, max(h.history["val_accuracy"]), ep))
    hists[a] = h.history
    print("   %-9s test acc %6.2f %%   test loss %.4f   best val %6.2f %%   epochs %d"
          % (a, 100 * acc, loss, 100 * max(h.history["val_accuracy"]), ep))
best_act = max(arows, key=lambda r: r[1])[0]
print("\n Best activation: %s" % best_act.upper())

banner("STEP 6  Effect of the hyperparameters")
hrows = []
grid = [("learning rate", "lr", [1e-2, 1e-3, 1e-4]),
        ("depth / width", "hidden", [(128,), (256, 128), (256, 128, 64), (512, 256, 128, 64)]),
        ("dropout", "dropout", [0.0, 0.3, 0.6])]
for gname, key, vals in grid:
    print("\n %s" % gname.upper())
    for v in vals:
        m, h, acc, loss = run("%s=%s" % (key, v), **{key: v, "act": best_act})
        gap = max(h.history["accuracy"]) - max(h.history["val_accuracy"])
        hrows.append((gname, str(v), acc, loss, gap))
        print("   %-22s test acc %6.2f %%   test loss %.4f   train-val gap %+.3f"
              % (str(v), 100 * acc, loss, gap))

banner("STEP 7  Final model, trained and evaluated")
model, hist, test_acc, test_loss = run("final", act=best_act, epochs=80, verbose=0)
print(" Final test accuracy : %.2f %%" % (100 * test_acc))
print(" Final test loss     : %.4f" % test_loss)
print(" Epochs run          : %d" % len(hist.history["loss"]))
pred = model.predict(Xte, verbose=0).argmax(axis=1)
print("\n" + classification_report(yte, pred, target_names=CLASSES, digits=3))
cm = confusion_matrix(yte, pred)
print(" Confusion matrix (rows = true, cols = predicted)")
print(" %-10s %s" % ("", " ".join("%8s" % c for c in CLASSES)))
for i, c in enumerate(CLASSES):
    print(" %-10s %s" % (c, " ".join("%8d" % v for v in cm[i])))

# ------------------------------------------------------------------ figures --
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, a in enumerate(acts):
    ax[0].plot(hists[a]["val_accuracy"], color=PAL[i], lw=1.8, label=a)
    ax[1].plot(hists[a]["val_loss"], color=PAL[i], lw=1.8, label=a)
ax[0].set_title("Validation accuracy by activation", fontsize=11, weight="bold")
ax[1].set_title("Validation loss by activation", fontsize=11, weight="bold")
for a_ in ax:
    a_.set_xlabel("Epoch"); a_.grid(alpha=.25); a_.set_axisbelow(True); a_.legend(frameon=False, fontsize=9)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
ax[0].set_ylabel("Accuracy"); ax[1].set_ylabel("Loss")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e3_activations.png"), dpi=165); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist.history["accuracy"], color=PAL[0], lw=2, label="train")
ax[0].plot(hist.history["val_accuracy"], color=PAL[1], lw=2, label="validation")
ax[1].plot(hist.history["loss"], color=PAL[0], lw=2, label="train")
ax[1].plot(hist.history["val_loss"], color=PAL[1], lw=2, label="validation")
ax[0].set_title("Accuracy", fontsize=11, weight="bold"); ax[0].set_ylabel("Accuracy")
ax[1].set_title("Loss", fontsize=11, weight="bold"); ax[1].set_ylabel("Loss")
for a_ in ax:
    a_.set_xlabel("Epoch"); a_.grid(alpha=.25); a_.set_axisbelow(True); a_.legend(frameon=False, fontsize=9)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
fig.suptitle("Final deep neural network, training history", fontsize=12.5, weight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e3_history.png"), dpi=165); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > cm.max() * .55 else "#101828", weight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion matrix, test set (acc %.1f %%)" % (100 * test_acc),
             fontsize=11.5, weight="bold")
fig.colorbar(im, fraction=0.045)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e3_confusion.png"), dpi=165); plt.close(fig)

np.savez(os.path.join(OUT, "exp3_results.npz"),
         acts=np.array(arows, dtype=object), hyper=np.array(hrows, dtype=object),
         cm=cm, test_acc=test_acc, test_loss=test_loss)
banner("DONE")
print(" Figures: e3_activations.png, e3_history.png, e3_confusion.png")
