"""
Match outcome predictor (Home win / Draw / Away win)
Data: top 5 European leagues, seasons 2005/06 - 2025/26
      (football-data.co.uk results, mirrored at github.com/datasets/football-datasets)

Pipeline:
  1. Pre-match features computed ONLY from past matches (no leakage):
     - Elo rating for every team, updated after each match
     - rolling form over the last 5 matches: points, goals, shots, shots on target
  2. Time-based split: train on 2005/06 - 2021/22, test on 2022/23 - 2025/26
  3. Models: class-prior baseline, Elo-only Logistic Regression, full Logistic Regression,
     Random Forest, XGBoost (tuned with TimeSeriesSplit cross-validation)
  4. Metrics: accuracy, log loss and Ranked Probability Score (RPS, the standard
     metric for football forecasts because it respects the order H > D > A)

Run:  pip install -r ../requirements.txt
      python match_predictor.py
"""

import os
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets"
LEAGUES = ["premier-league", "la-liga", "bundesliga", "serie-a", "ligue-1"]
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2005, 2026)]   # 0506 ... 2526
TEST_FROM_SEASON = 2022                     # seasons starting in 2022 or later = test set
RANDOM_STATE = 42

LABELS = ["H", "D", "A"]                    # class order used everywhere
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}


# ----------------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------------
def load_matches():
    frames = []
    for league in LEAGUES:
        for season in SEASONS:
            path = os.path.join(DATA_DIR, f"{league}_{season}.csv")
            if not os.path.exists(path):
                pd.read_csv(f"{BASE_URL}/{league}/season-{season}.csv").to_csv(path, index=False)
            df = pd.read_csv(path)
            df["league"] = league
            df["season_start"] = 2000 + int(season[:2])
            frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST"])
    df = df.sort_values(["Date", "league", "HomeTeam"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} matches | {df.league.nunique()} leagues | "
          f"{df.season_start.min()}/{df.season_start.min() + 1} - "
          f"{df.season_start.max()}/{df.season_start.max() + 1}")
    return df


# ----------------------------------------------------------------------------
# 2. Feature engineering (strictly chronological)
# ----------------------------------------------------------------------------
ELO_START, ELO_K, ELO_HOME_ADV = 1500.0, 20.0, 60.0
SEASON_REGRESSION = 0.25                    # pull ratings 25% back to the mean each summer
FORM_WINDOW = 5


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def build_features(df):
    elo = defaultdict(lambda: ELO_START)
    history = defaultdict(lambda: deque(maxlen=FORM_WINDOW))   # per team: last 5 match stats
    last_season = {}

    def form(team):
        h = history[team]
        if not h:
            return [np.nan] * 6
        arr = np.array(h)
        return list(arr.mean(axis=0))   # points, goals for/against, shots for/against, SoT diff

    rows = []
    for m in df.itertuples(index=False):
        home, away = (m.league, m.HomeTeam), (m.league, m.AwayTeam)

        # new season -> regress ratings toward the mean (squads change over the summer)
        for team in (home, away):
            if last_season.get(team) not in (None, m.season_start):
                elo[team] = ELO_START + (1 - SEASON_REGRESSION) * (elo[team] - ELO_START)
            last_season[team] = m.season_start

        # ---- features: everything here is known BEFORE kick-off ----
        h_form, a_form = form(home), form(away)
        rows.append([elo[home], elo[away], elo[home] + ELO_HOME_ADV - elo[away], *h_form, *a_form])

        # ---- update state with the result (used only for future matches) ----
        hg, ag = m.FTHG, m.FTAG
        result_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        exp_home = expected_score(elo[home] + ELO_HOME_ADV, elo[away])
        margin = np.log(abs(hg - ag) + 1) + 1            # bigger wins move ratings more
        delta = ELO_K * margin * (result_home - exp_home)
        elo[home] += delta
        elo[away] -= delta

        pts_home = 3 if hg > ag else (1 if hg == ag else 0)
        pts_away = 3 if ag > hg else (1 if hg == ag else 0)
        history[home].append([pts_home, hg, ag, m.HS, m.AS, m.HST - m.AST])
        history[away].append([pts_away, ag, hg, m.AS, m.HS, m.AST - m.HST])

    form_names = ["points", "goals_for", "goals_against", "shots_for", "shots_against", "sot_diff"]
    cols = (["elo_home", "elo_away", "elo_diff"]
            + [f"home_{n}_l5" for n in form_names] + [f"away_{n}_l5" for n in form_names])
    feats = pd.DataFrame(rows, columns=cols, index=df.index)
    for n in form_names:
        feats[f"diff_{n}_l5"] = feats[f"home_{n}_l5"] - feats[f"away_{n}_l5"]
    return pd.concat([df, feats], axis=1)


# ----------------------------------------------------------------------------
# 3. Evaluation helpers
# ----------------------------------------------------------------------------
def ranked_probability_score(y_true, proba):
    """Mean RPS for ordered outcomes H < D < A. Lower is better (0 = perfect)."""
    onehot = np.eye(3)[y_true]
    cum_pred, cum_true = np.cumsum(proba, axis=1), np.cumsum(onehot, axis=1)
    return np.mean(np.sum((cum_pred - cum_true)[:, :-1] ** 2, axis=1) / 2)


def evaluate(name, y_true, proba):
    return {"model": name,
            "accuracy": accuracy_score(y_true, proba.argmax(axis=1)),
            "log_loss": log_loss(y_true, proba, labels=[0, 1, 2]),
            "RPS": ranked_probability_score(y_true, proba)}


# ----------------------------------------------------------------------------
# 4. Training
# ----------------------------------------------------------------------------
def main():
    df = build_features(load_matches())
    df["target"] = df["FTR"].map(LABEL_TO_INT)

    # Drop the first season (Elo still warming up) and matches where a team has no history yet
    feature_cols = [c for c in df.columns if c.startswith(("elo_", "home_", "away_", "diff_"))]
    df = df[df.season_start > 2005].dropna(subset=feature_cols + ["target"])
    train, test = df[df.season_start < TEST_FROM_SEASON], df[df.season_start >= TEST_FROM_SEASON]
    X_train, y_train = train[feature_cols], train["target"].values
    X_test, y_test = test[feature_cols], test["target"].values
    print(f"Train: {len(train):,} matches (2006/07 - {TEST_FROM_SEASON - 1}/{TEST_FROM_SEASON % 100})"
          f" | Test: {len(test):,} matches ({TEST_FROM_SEASON}/{(TEST_FROM_SEASON + 1) % 100} onward)")
    print("Train outcome distribution:",
          {l: f"{(y_train == i).mean():.1%}" for l, i in LABEL_TO_INT.items()})

    results, probas = [], {}

    # Baseline: always predict the training-set frequencies (H ~45%, D ~26%, A ~29%)
    prior = np.bincount(y_train, minlength=3) / len(y_train)
    probas["Baseline (class priors)"] = np.tile(prior, (len(y_test), 1))

    # Elo only - shows how much a single well-designed feature captures
    elo_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elo_lr.fit(X_train[["elo_diff"]], y_train)
    probas["LogReg (Elo only)"] = elo_lr.predict_proba(X_test[["elo_diff"]])

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    lr.fit(X_train, y_train)
    probas["LogReg (all features)"] = lr.predict_proba(X_test)

    rf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=20,
                                n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    probas["Random Forest"] = rf.predict_proba(X_test)

    # XGBoost tuned with time-series CV (each fold validates on matches AFTER its training data)
    grid = GridSearchCV(
        XGBClassifier(objective="multi:softprob", n_estimators=300, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="mlogloss", random_state=RANDOM_STATE),
        param_grid={"max_depth": [2, 3, 4], "learning_rate": [0.02, 0.05]},
        cv=TimeSeriesSplit(n_splits=4), scoring="neg_log_loss", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"XGBoost best params (TimeSeriesSplit CV): {grid.best_params_}")
    xgb = grid.best_estimator_
    probas["XGBoost (tuned)"] = xgb.predict_proba(X_test)

    for name, p in probas.items():
        results.append(evaluate(name, y_test, p))
    results = pd.DataFrame(results)
    print("\nTest set results (lower log loss / RPS = better):")
    print(results.round(4).to_string(index=False))
    results.round(4).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

    best_name = results.loc[results.RPS.idxmin(), "model"]
    best_pred = probas[best_name].argmax(axis=1)
    print(f"\nBest model by RPS: {best_name}")
    print("Predicted class distribution:",
          {l: int((best_pred == i).sum()) for l, i in LABEL_TO_INT.items()},
          "(draws are rarely the single most likely outcome, a known property of football)")

    # ---- Plot 1: confusion matrix of the best model ----
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, display_labels=["Home", "Draw", "Away"],
                                            normalize="true", values_format=".0%", ax=ax,
                                            colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix - {best_name}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150); plt.close()

    # ---- Plot 2: XGBoost feature importance ----
    imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values().tail(12)
    plt.figure(figsize=(7, 5))
    imp.plot.barh(color="#3b6ea5")
    plt.title("XGBoost feature importance (gain)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150); plt.close()

    # ---- Plot 3: does a bigger Elo gap really mean more home wins? ----
    test_view = test.assign(p_home=probas[best_name][:, 0], home_win=(y_test == 0).astype(int))
    test_view["elo_bin"] = pd.qcut(test_view.elo_diff, 10)
    g = test_view.groupby("elo_bin", observed=True)[["p_home", "home_win"]].mean()
    plt.figure(figsize=(7, 4.5))
    x = range(len(g))
    plt.plot(x, g.home_win, "o-", label="Actual home win rate")
    plt.plot(x, g.p_home, "s--", label="Predicted P(home win)")
    plt.xticks(x, [f"{int(i.mid)}" for i in g.index], rotation=45)
    plt.xlabel("Elo difference (home - away, incl. home advantage), decile midpoint")
    plt.ylabel("Probability"); plt.legend()
    plt.title("Predicted vs actual home win rate by Elo gap")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "elo_calibration.png"), dpi=150); plt.close()

    print(f"\nPlots saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
