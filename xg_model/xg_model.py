"""
Expected Goals (xG) model + player style clustering
Data: StatsBomb Open Data - Premier League 2015/2016 (all 380 matches)

Part A (supervised):   predict P(goal | shot) with Logistic Regression vs XGBoost
Part B (unsupervised): cluster outfield players by style with PCA + K-Means

Run:  pip install -r ../requirements.txt
      python xg_model.py
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, silhouette_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID, SEASON_ID = 2, 27          # Premier League 2015/2016
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
RANDOM_STATE = 42

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Data loading (downloaded once, then cached on disk)
# ----------------------------------------------------------------------------
def fetch_json(url, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    data = requests.get(url, timeout=60).json()
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def load_season():
    matches = fetch_json(f"{BASE_URL}/matches/{COMPETITION_ID}/{SEASON_ID}.json",
                         f"{DATA_DIR}/matches.json")
    match_ids = [m["match_id"] for m in matches]
    print(f"Matches in season: {len(match_ids)}")

    def get_events(mid):
        events = fetch_json(f"{BASE_URL}/events/{mid}.json", f"{DATA_DIR}/events_{mid}.json")
        for e in events:
            e["match_id"] = mid
        return events

    with ThreadPoolExecutor(max_workers=16) as pool:
        all_events = [e for evs in pool.map(get_events, match_ids) for e in evs]
    print(f"Total events: {len(all_events):,}")
    return all_events


# ----------------------------------------------------------------------------
# 2. Part A - xG model
# ----------------------------------------------------------------------------
GOAL_X, GOAL_Y = 120.0, 40.0      # StatsBomb pitch is 120 x 80, goal centre at (120, 40)
POST_HALF_WIDTH = 4.0             # goal is 8 yards wide


def build_shot_table(events):
    rows = []
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        shot = e["shot"]
        if shot["type"]["name"] == "Penalty":       # penalties are a separate problem (~76% goals)
            continue
        x, y = e["location"]
        dx, dy = GOAL_X - x, GOAL_Y - y
        distance = np.hypot(dx, dy)
        # angle between the two posts as seen from the shot location
        a1 = np.arctan2(GOAL_Y - POST_HALF_WIDTH - y, dx)
        a2 = np.arctan2(GOAL_Y + POST_HALF_WIDTH - y, dx)
        angle = abs(a2 - a1)
        body = shot["body_part"]["name"]
        rows.append({
            "match_id": e["match_id"],
            "player": e["player"]["name"],
            "distance": distance,
            "angle": np.degrees(angle),
            "x": x,
            "y_abs_from_centre": abs(dy),
            "body_part": "Head" if body == "Head" else ("Foot" if "Foot" in body else "Other"),
            "shot_type": shot["type"]["name"],                 # Open Play / Free Kick / Corner
            "technique": shot["technique"]["name"],            # Normal, Volley, Half Volley, ...
            "under_pressure": int(e.get("under_pressure", False)),
            "first_time": int(shot.get("first_time", False)),
            "one_on_one": int(shot.get("one_on_one", False)),
            "from_counter": int(e["play_pattern"]["name"] == "From Counter"),
            "statsbomb_xg": shot["statsbomb_xg"],              # benchmark only, NOT a feature
            "goal": int(shot["outcome"]["name"] == "Goal"),
        })
    return pd.DataFrame(rows)


NUM_FEATURES = ["distance", "angle", "x", "y_abs_from_centre",
                "under_pressure", "first_time", "one_on_one", "from_counter"]
CAT_FEATURES = ["body_part", "shot_type", "technique"]


def evaluate(name, y_true, p):
    return {"model": name,
            "ROC-AUC": roc_auc_score(y_true, p),
            "Brier": brier_score_loss(y_true, p),
            "LogLoss": log_loss(y_true, p)}


def run_xg_model(shots):
    print(f"\n=== PART A: xG model ===")
    print(f"Non-penalty shots: {len(shots):,} | goal rate: {shots.goal.mean():.1%}")

    X = shots[NUM_FEATURES + CAT_FEATURES]
    y = shots["goal"]

    # Split by MATCH, so shots from the same game never appear in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=shots["match_id"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])

    logreg = Pipeline([("prep", preprocess),
                       ("clf", LogisticRegression(max_iter=1000))])
    logreg.fit(X_train, y_train)
    p_lr = logreg.predict_proba(X_test)[:, 1]

    xgb = Pipeline([("prep", preprocess),
                    ("clf", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                                          subsample=0.8, colsample_bytree=0.8,
                                          min_child_weight=5, eval_metric="logloss",
                                          random_state=RANDOM_STATE))])
    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]

    baseline = np.full(len(y_test), y_train.mean())      # "every shot = average goal rate"
    results = pd.DataFrame([
        evaluate("Baseline (constant)", y_test, baseline),
        evaluate("Logistic Regression", y_test, p_lr),
        evaluate("XGBoost", y_test, p_xgb),
        evaluate("StatsBomb xG (benchmark)", y_test, shots["statsbomb_xg"].iloc[test_idx]),
    ])
    print(results.round(4).to_string(index=False))

    # --- Calibration plot: does predicted xG match the real goal frequency? ---
    plt.figure(figsize=(6, 6))
    for name, p in [("Logistic Regression", p_lr), ("XGBoost", p_xgb)]:
        frac, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Predicted xG"); plt.ylabel("Actual goal rate")
    plt.title("Calibration on test matches"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/calibration.png", dpi=150); plt.close()

    # --- SHAP: which features drive the XGBoost predictions? ---
    X_test_enc = pd.DataFrame(xgb.named_steps["prep"].transform(X_test),
                              columns=xgb.named_steps["prep"].get_feature_names_out())
    explainer = shap.TreeExplainer(xgb.named_steps["clf"])
    shap_values = explainer.shap_values(X_test_enc)
    shap.summary_plot(shap_values, X_test_enc, show=False, max_display=12)
    plt.title("SHAP - feature impact on goal probability")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=150); plt.close()

    # Refit on all shots so every shot in the season gets an xG value for Part B
    xgb.fit(X, y)
    shots["xg"] = xgb.predict_proba(X)[:, 1]

    top = (shots.groupby("player").agg(shots=("goal", "size"), goals=("goal", "sum"), xG=("xg", "sum"))
           .assign(goals_minus_xG=lambda d: d.goals - d.xG)
           .sort_values("xG", ascending=False).head(10).round(2))
    print("\nTop 10 players by our xG (non-penalty):")
    print(top.to_string())
    return shots


# ----------------------------------------------------------------------------
# 3. Part B - player style clustering
# ----------------------------------------------------------------------------
def minutes_played(events):
    """Minutes per player per match, from Starting XI + Substitution events."""
    df_end = {}
    starters, subs_on, subs_off = [], [], {}
    for e in events:
        mid = e["match_id"]
        df_end[mid] = max(df_end.get(mid, 90), e["minute"])
        t = e["type"]["name"]
        if t == "Starting XI":
            for p in e["tactics"]["lineup"]:
                starters.append((mid, p["player"]["name"]))
        elif t == "Substitution":
            subs_off[(mid, e["player"]["name"])] = e["minute"]
            subs_on.append((mid, e["substitution"]["replacement"]["name"], e["minute"]))

    minutes = {}
    for mid, name in starters:
        minutes[name] = minutes.get(name, 0) + subs_off.get((mid, name), df_end[mid])
    for mid, name, start in subs_on:
        end = subs_off.get((mid, name), df_end[mid])
        minutes[name] = minutes.get(name, 0) + (end - start)
    return pd.Series(minutes, name="minutes")


def build_player_table(events, shots):
    stats = {}
    positions = {}

    def add(player, key, value=1):
        stats.setdefault(player, {}).setdefault(key, 0)
        stats[player][key] += value

    for e in events:
        if "player" not in e:
            continue
        p = e["player"]["name"]
        t = e["type"]["name"]
        if "position" in e:
            positions.setdefault(p, {}).setdefault(e["position"]["name"], 0)
            positions[p][e["position"]["name"]] += 1

        if t == "Pass":
            pas = e["pass"]
            add(p, "passes")
            if "outcome" not in pas:                       # no outcome = completed
                add(p, "passes_completed")
            if pas.get("shot_assist") or pas.get("goal_assist"):
                add(p, "key_passes")
            if pas.get("cross"):
                add(p, "crosses")
            if pas["length"] > 30:
                add(p, "long_passes")
            # progressive: moves the ball at least 10 yards closer to the opponent's goal
            if pas["end_location"][0] - e["location"][0] >= 10:
                add(p, "progressive_passes")
        elif t == "Carry":
            if e["carry"]["end_location"][0] - e["location"][0] >= 10:
                add(p, "progressive_carries")
        elif t == "Dribble" and e["dribble"]["outcome"]["name"] == "Complete":
            add(p, "dribbles")
        elif t == "Shot":
            add(p, "shots")
        elif t == "Interception":
            add(p, "interceptions")
        elif t == "Clearance":
            add(p, "clearances")
        elif t == "Pressure":
            add(p, "pressures")
        elif t == "Duel" and e["duel"]["type"]["name"] == "Tackle":
            add(p, "tackles")
        elif t == "Ball Recovery":
            add(p, "recoveries")

    df = pd.DataFrame(stats).T.fillna(0)
    df["xg"] = shots.groupby("player")["xg"].sum()
    df["xg"] = df["xg"].fillna(0)
    df = df.join(minutes_played(events))
    df["main_position"] = pd.Series({p: max(c, key=c.get) for p, c in positions.items()})

    df = df[(df.minutes >= 900) & (df.main_position != "Goalkeeper")].copy()

    df["pass_completion"] = df.passes_completed / df.passes
    count_cols = ["passes", "key_passes", "crosses", "long_passes", "progressive_passes",
                  "progressive_carries", "dribbles", "shots", "xg", "interceptions",
                  "clearances", "pressures", "tackles", "recoveries"]
    for c in count_cols:
        df[c + "_p90"] = df[c] / df.minutes * 90
    features = [c + "_p90" for c in count_cols] + ["pass_completion"]
    return df, features


def run_clustering(events, shots):
    print(f"\n=== PART B: player style clustering ===")
    players, features = build_player_table(events, shots)
    print(f"Outfield players with >= 900 minutes: {len(players)}")

    X = StandardScaler().fit_transform(players[features])

    # Choose k with the silhouette score
    scores = {}
    for k in range(3, 9):
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    print("Silhouette by k:", {k: round(s, 3) for k, s in scores.items()})
    # The best silhouette is usually k=3 (roughly defence / midfield / attack), which is
    # too coarse to be interesting. So pick the best k among 4-8 for finer roles.
    best_k = max((k for k in scores if k >= 4), key=scores.get)
    print(f"Chosen k = {best_k}")

    km = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    players["cluster"] = km.fit_predict(X)

    # Describe each cluster by its strongest features (z-scores of the centroid)
    centroids = pd.DataFrame(km.cluster_centers_, columns=features)
    for c in range(best_k):
        members = players[players.cluster == c]
        top_feats = centroids.loc[c].sort_values(ascending=False).head(3).index.tolist()
        examples = members.sort_values("minutes", ascending=False).index[:6].tolist()
        print(f"\nCluster {c} ({len(members)} players) | high in: {', '.join(top_feats)}")
        print(f"  most common position: {members.main_position.mode()[0]}")
        print(f"  examples: {', '.join(examples)}")

    # 2D view with PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    print(f"\nPCA: 2 components explain {pca.explained_variance_ratio_.sum():.1%} of variance")

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=players.cluster, cmap="tab10", s=25, alpha=0.8)
    for name in players.sort_values("minutes", ascending=False).index[:3 * best_k:3]:
        i = players.index.get_loc(name)
        plt.annotate(name.split()[-1], coords[i], fontsize=7)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Premier League 2015/16 - player styles (PCA + K-Means)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/player_clusters.png", dpi=150); plt.close()

    players[["main_position", "minutes", "cluster"] + features].round(3).to_csv(
        f"{OUT_DIR}/player_clusters.csv")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    events = load_season()
    shots = build_shot_table(events)
    shots = run_xg_model(shots)
    run_clustering(events, shots)
    print(f"\nPlots saved in {OUT_DIR}")
