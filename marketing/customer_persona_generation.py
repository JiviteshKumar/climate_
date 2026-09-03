# -*- coding: utf-8 -*-
"""
AI-Based Customer Persona Generation from Marketing Data
Digital Marketing Laboratory, Experiment 08
Jivitesh Kumar | Roll No. 16010423041 | Batch A2 | KJSSE, Sem VII

Pipeline
--------
    load  ->  explore  ->  clean  ->  select features  ->  scale
          ->  choose K (elbow + silhouette)  ->  K-Means
          ->  visualise  ->  profile clusters  ->  write personas

Run:  python customer_persona_generation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

PALETTE = ["#0b5fa5", "#e2620d", "#12805c", "#8b1c34", "#c98a04"]


def line(title):
    print("\n" + "=" * 74)
    print(" " + title)
    print("=" * 74)


# ---------------------------------------------------------------------------
# STEP 1-2 : build and load the marketing dataset
# ---------------------------------------------------------------------------
def build_dataset(path):
    """Synthesise a marketing dataset with five latent buying behaviours.

    The generator deliberately injects duplicate rows and missing values so
    that the cleaning step in the pipeline does real work rather than being a
    formality.
    """
    # (label, n, age, income k INR/yr, spending score, purchases/yr, visits/mo)
    groups = [
        ("careful_young",  46, (22, 4.5), (340,  60), (28, 9),  (4, 1.5), (6, 2.0)),
        ("premium_loyal",  38, (41, 7.0), (1450, 210), (81, 8),  (22, 4.0), (19, 4.0)),
        ("high_income_low",34, (46, 8.0), (1520, 230), (22, 8),  (5, 2.0), (7, 2.5)),
        ("impulsive_young",44, (26, 4.0), (410,  70), (79, 9),  (17, 3.5), (23, 5.0)),
        ("steady_mid",     48, (38, 6.5), (880,  120), (51, 8),  (12, 3.0), (13, 3.5)),
    ]
    rows = []
    cities = ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane", "Delhi", "Bengaluru", "Hyderabad"]
    for label, n, age, inc, spend, freq, vis in groups:
        for _ in range(n):
            rows.append({
                "Age":              int(np.clip(rng.normal(*age), 18, 70)),
                "Gender":           rng.choice(["Male", "Female"], p=[0.48, 0.52]),
                "City":             rng.choice(cities),
                "AnnualIncome_kINR": int(np.clip(rng.normal(*inc), 180, 2600)),
                "SpendingScore":    int(np.clip(rng.normal(*spend), 1, 100)),
                "PurchaseFrequency": int(np.clip(rng.normal(*freq), 0, 40)),
                "WebsiteVisits":    int(np.clip(rng.normal(*vis), 0, 45)),
                "_truth":           label,
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    df.insert(0, "CustomerID", ["CUST%04d" % (i + 1) for i in range(len(df))])

    # inject 6 duplicate records and 9 missing values
    dupes = df.sample(6, random_state=RANDOM_STATE)
    df = pd.concat([df, dupes], ignore_index=True)
    for col, k in [("AnnualIncome_kINR", 4), ("SpendingScore", 3), ("Age", 2)]:
        idx = rng.choice(df.index, size=k, replace=False)
        df.loc[idx, col] = np.nan

    df.drop(columns=["_truth"]).to_csv(path, index=False)
    return path


DATA = os.path.join(OUT, "customer_marketing_data.csv")
if not os.path.exists(DATA):
    build_dataset(DATA)

line("STEP 2  Load the marketing dataset")
df = pd.read_csv(DATA)
print("Loaded : %s" % os.path.basename(DATA))
print("Shape  : %d rows x %d columns" % df.shape)
print("\nFirst five records:")
print(df.head().to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 3 : explore
# ---------------------------------------------------------------------------
line("STEP 3  Explore the dataset")
print("Columns and data types:")
print(df.dtypes.to_string())
print("\nDescriptive statistics for the numeric attributes:")
print(df.describe().round(2).to_string())

# ---------------------------------------------------------------------------
# STEP 4 : clean
# ---------------------------------------------------------------------------
line("STEP 4  Clean the dataset")
print("Duplicate rows found      : %d" % df.duplicated().sum())
print("Missing values per column :")
print(df.isna().sum().to_string())

before = len(df)
df = df.drop_duplicates()
print("\nRemoved %d duplicate rows (%d -> %d)." % (before - len(df), before, len(df)))

num_cols = ["Age", "AnnualIncome_kINR", "SpendingScore", "PurchaseFrequency", "WebsiteVisits"]
for c in num_cols:
    if df[c].isna().any():
        med = df[c].median()
        n = int(df[c].isna().sum())
        df[c] = df[c].fillna(med)
        print("Filled %d missing value(s) in %-18s with the median (%.1f)." % (n, c, med))
df[num_cols] = df[num_cols].astype(int)
print("\nMissing values remaining  : %d" % int(df.isna().sum().sum()))
print("Clean dataset shape       : %d rows x %d columns" % df.shape)

# ---------------------------------------------------------------------------
# STEP 5-6 : select features and scale
# ---------------------------------------------------------------------------
line("STEP 5-6  Select features and normalise with StandardScaler")
X = df[num_cols].copy()
print("Features selected for segmentation: %s" % ", ".join(num_cols))
print("\nRanges before scaling:")
print(X.agg(["min", "max"]).to_string())

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print("\nAfter StandardScaler  ->  mean = %.3f, standard deviation = %.3f"
      % (Xs.mean(), Xs.std()))
print("Scaling matters because AnnualIncome_kINR spans hundreds while")
print("SpendingScore spans 1-100; without it, income alone would drive the clusters.")

# ---------------------------------------------------------------------------
# STEP 7a : choose K
# ---------------------------------------------------------------------------
line("STEP 7a  Choose the number of clusters K")
ks, inertias, sils = range(2, 11), [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xs, km.labels_))
print(" K   inertia (WCSS)   silhouette")
print(" " + "-" * 34)
for k, i, s in zip(ks, inertias, sils):
    mark = "   <-- selected" if s == max(sils) else ""
    print(" %-3d %13.1f %11.3f%s" % (k, i, s, mark))
K = list(ks)[int(np.argmax(sils))]
print("\nThe elbow in the inertia curve and the highest silhouette score both")
print("indicate K = %d." % K)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
ax[0].plot(list(ks), inertias, "o-", color=PALETTE[0], lw=2, ms=6)
ax[0].axvline(K, color=PALETTE[3], ls="--", lw=1.4)
ax[0].set_title("Elbow method", fontsize=11, weight="bold")
ax[0].set_xlabel("Number of clusters, K"); ax[0].set_ylabel("Inertia (WCSS)")
ax[1].plot(list(ks), sils, "o-", color=PALETTE[1], lw=2, ms=6)
ax[1].axvline(K, color=PALETTE[3], ls="--", lw=1.4)
ax[1].set_title("Silhouette score", fontsize=11, weight="bold")
ax[1].set_xlabel("Number of clusters, K"); ax[1].set_ylabel("Mean silhouette")
for a in ax:
    a.grid(alpha=.25); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.suptitle("Choosing K for customer segmentation", fontsize=12.5, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_elbow.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------------------
# STEP 7b : K-Means
# ---------------------------------------------------------------------------
line("STEP 7b  Apply K-Means clustering")
km = KMeans(n_clusters=K, n_init=25, random_state=RANDOM_STATE).fit(Xs)
df["Cluster"] = km.labels_
print("K-Means converged in %d iterations." % km.n_iter_)
print("Final inertia      : %.1f" % km.inertia_)
print("Silhouette score   : %.3f" % silhouette_score(Xs, km.labels_))
print("\nCustomers per cluster:")
print(df["Cluster"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# STEP 8 : visualise
# ---------------------------------------------------------------------------
line("STEP 8  Visualise the clusters")
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
for c in range(K):
    m = df["Cluster"] == c
    ax[0].scatter(df.loc[m, "AnnualIncome_kINR"], df.loc[m, "SpendingScore"],
                  s=34, alpha=.82, color=PALETTE[c], label="Cluster %d" % c,
                  edgecolors="white", linewidths=.6)
    ax[1].scatter(df.loc[m, "Age"], df.loc[m, "PurchaseFrequency"],
                  s=34, alpha=.82, color=PALETTE[c], edgecolors="white", linewidths=.6)
cent = scaler.inverse_transform(km.cluster_centers_)
ax[0].scatter(cent[:, 1], cent[:, 2], marker="X", s=210, c="black",
              edgecolors="white", linewidths=1.6, zorder=5, label="Centroid")
ax[0].set_xlabel("Annual income (thousand INR)"); ax[0].set_ylabel("Spending score (1-100)")
ax[0].set_title("Income vs spending score", fontsize=11, weight="bold")
ax[1].set_xlabel("Age (years)"); ax[1].set_ylabel("Purchases per year")
ax[1].set_title("Age vs purchase frequency", fontsize=11, weight="bold")
for a in ax:
    a.grid(alpha=.25); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9.5, ncol=K + 1,
           loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Customer segments identified by K-Means (K = %d)" % K,
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(FIG, "fig_clusters.png"), dpi=170)
plt.close(fig)
print("Saved figures/fig_elbow.png and figures/fig_clusters.png")

# ---------------------------------------------------------------------------
# STEP 9 : profile each cluster
# ---------------------------------------------------------------------------
line("STEP 9  Cluster profiles")
prof = df.groupby("Cluster")[num_cols].mean().round(1)
prof["Customers"] = df["Cluster"].value_counts().sort_index()
prof["Share_%"] = (prof["Customers"] / len(df) * 100).round(1)
prof["TopCity"] = df.groupby("Cluster")["City"].agg(lambda s: s.mode().iat[0])
print(prof.to_string())

fig, ax = plt.subplots(figsize=(9.5, 4.2))
norm = (prof[num_cols] - prof[num_cols].min()) / (prof[num_cols].max() - prof[num_cols].min())
w, xs = 0.15, np.arange(len(num_cols))
for c in range(K):
    ax.bar(xs + c * w, norm.loc[c], width=w, color=PALETTE[c],
           label="Cluster %d" % c, edgecolor="white", linewidth=.7)
ax.set_xticks(xs + w * (K - 1) / 2)
ax.set_xticklabels(["Age", "Income", "Spending\nscore", "Purchase\nfrequency", "Website\nvisits"],
                   fontsize=9.5)
ax.set_ylabel("Relative level (min-max scaled)")
ax.set_title("Cluster profile comparison across the five segmentation features",
             fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=9, ncol=K)
ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_profiles.png"), dpi=170)
plt.close(fig)
print("\nSaved figures/fig_profiles.png")

# ---------------------------------------------------------------------------
# STEP 10 : generate personas
# ---------------------------------------------------------------------------
line("STEP 10  Generated customer personas")

# Rank each cluster against the others rather than against an absolute cut-off,
# so that a cluster sitting exactly on a threshold is not mislabelled.
def _rank01(col):
    """Rank the clusters on one attribute, scaled so the lowest is 0 and the
    highest is 1. pandas' pct=True divides by n, which puts the middle of five
    clusters at 0.6 rather than 0.5, so the scaling is done explicitly."""
    r = prof[col].rank(method="average")
    return (r - 1.0) / (len(prof) - 1.0)


inc_rank = _rank01("AnnualIncome_kINR")
spd_rank = _rank01("SpendingScore")


def name_persona(c):
    r = prof.loc[c]
    i, sp = inc_rank.loc[c], spd_rank.loc[c]
    hi_i, lo_i = i >= 0.6, i <= 0.4
    hi_s, lo_s = sp >= 0.6, sp <= 0.4
    if hi_i and hi_s:   base = "Premium Loyalist"
    elif hi_i and lo_s: base = "Cautious Affluent"
    elif lo_i and hi_s: base = "Impulsive Explorer"
    elif lo_i and lo_s: base = "Budget Minimalist"
    else:               base = "Steady Mainstream"
    age = r["Age"]
    band = "Young" if age < 32 else ("Mid-career" if age < 45 else "Senior")
    return "%s %s" % (band, base)

PERSONA_LIB = {
    "Premium Loyalist":  ("High income and high spending, buys often and visits the site regularly.",
                          "Retain and grow basket size.",
                          "Fears missing a better deal elsewhere; dislikes generic offers.",
                          "Email, loyalty app push, WhatsApp",
                          "Tiered loyalty programme, early access to launches, personalised bundles."),
    "Cautious Affluent": ("High income but low spending score; browses without converting.",
                          "Be convinced the product is worth it.",
                          "Price-to-value doubt; no urgency to buy.",
                          "Retargeting display, long-form email, YouTube",
                          "Comparison content, reviews and warranty messaging; retargeting with proof."),
    "Impulsive Explorer": ("Modest income but high spending score; frequent, impulsive purchases.",
                          "Discover something new and affordable.",
                          "Budget ceiling; abandons cart on high shipping.",
                          "Instagram, YouTube Shorts, SMS",
                          "Flash sales, EMI and BNPL options, influencer and reel-led discovery."),
    "Steady Mainstream": ("Mid income and mid spending; predictable, repeat purchases.",
                          "Reliable value with no surprises.",
                          "Switches on small price differences.",
                          "Email newsletter, Google Search, Facebook",
                          "Subscription and replenishment offers, bundle discounts, referral scheme."),
    "Budget Minimalist": ("Low income and low spending; infrequent visits and purchases.",
                          "Meet a specific need at the lowest cost.",
                          "Price sensitive; ignores full-price messaging.",
                          "Google Search, SMS, seasonal email",
                          "Entry-price range, festive discounting, free-shipping thresholds."),
}

personas = []
for c in range(K):
    r = prof.loc[c]
    full = name_persona(c)
    key = " ".join(full.split()[1:])
    behaviour, goal, pain, channels, strategy = PERSONA_LIB[key]
    p = {
        "Cluster": c,
        "Persona": full,
        "Size": "%d customers (%.1f%%)" % (r["Customers"], r["Share_%"]),
        "AgeGroup": "%d years (mean)" % r["Age"],
        "Income": "INR %.0fk per year (mean)" % r["AnnualIncome_kINR"],
        "Spending": "score %.0f / 100" % r["SpendingScore"],
        "Frequency": "%.1f purchases per year, %.1f site visits per month"
                     % (r["PurchaseFrequency"], r["WebsiteVisits"]),
        "TopCity": r["TopCity"],
        "Behaviour": behaviour,
        "Goal": goal,
        "PainPoint": pain,
        "Channels": channels,
        "Strategy": strategy,
    }
    personas.append(p)
    print("\n" + "-" * 74)
    print(" CLUSTER %d  |  %s" % (c, full.upper()))
    print("-" * 74)
    print(" Segment size    : %s" % p["Size"])
    print(" Age group       : %s" % p["AgeGroup"])
    print(" Income level    : %s" % p["Income"])
    print(" Spending        : %s" % p["Spending"])
    print(" Frequency       : %s" % p["Frequency"])
    print(" Top location    : %s" % p["TopCity"])
    print(" Behaviour       : %s" % behaviour)
    print(" Goal            : %s" % goal)
    print(" Pain point      : %s" % pain)
    print(" Preferred chan. : %s" % channels)
    print(" Marketing plan  : %s" % strategy)

pd.DataFrame(personas).to_csv(os.path.join(OUT, "generated_personas.csv"), index=False)
df.to_csv(os.path.join(OUT, "customers_with_clusters.csv"), index=False)

line("DONE")
print("Artefacts written to %s" % OUT)
print("  customer_marketing_data.csv    input dataset")
print("  customers_with_clusters.csv    dataset with the assigned cluster label")
print("  generated_personas.csv         the %d personas generated" % K)
print("  figures/fig_elbow.png          elbow and silhouette plots")
print("  figures/fig_clusters.png       cluster scatter plots")
print("  figures/fig_profiles.png       cluster profile comparison")
