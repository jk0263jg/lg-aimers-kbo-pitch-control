"""
검증 실험 모음 — 개발 과정에서 내린 설계 결정의 근거를 다시 돌려볼 수 있게 정리한 스크립트.

    python analysis/experiments.py --exp gametype
    python analysis/experiments.py --exp all --sample 0.1     # 빠른 동작 확인용(점수는 의미 없음)

[검증 방식: 시간 순서를 지킨 rolling-origin]
  val=2024 폴드 : 2019~2023 로 학습 → 2024 예측
  val=2023 폴드 : 2019~2022 로 학습 → 2023 예측
  → 실제 평가가 '미래 시즌(2025)' 예측이므로, 무작위 분할이 아니라 과거→미래 분할만 사용한다.

[점수 2종류 — 헷갈리기 쉬우니 구분]
  raw 점수      : 모델 출력 그대로 Brier Skill Score 계산 (실제 리더보드와 같은 방식)
  판별력 점수   : 예측 평균을 실제 평균에 맞춘 뒤 계산 (oracle shift).
                  '평균이 얼마나 맞았나' 효과를 제거하고 모델의 구분 능력만 비교할 때 사용.
                  ※ 실전에서는 정답 평균을 알 수 없으므로 달성 불가능한 상한선이다.

실험 목록
  baseline        : 대회 베이스라인(RandomForest) vs 피처 엔지니어링 + LightGBM
  gametype        : game_type 피처 포함/제외 (시즌 간 부호가 뒤집히는 피처)
  season_shift    : 시즌 드리프트 보정(DELTA) 효과
  regularization  : 잎 노드 최소 샘플 수/L2 정규화 강도
  extra_features  : 피처를 14개 더 추가하면 좋아지는가
"""

import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                              # train.py
sys.path.insert(0, os.path.join(ROOT, "submission"))  # script.py
import train as T          # noqa: E402  (학습 스크립트의 상수·함수 재사용)
import script as inference  # noqa: E402  (추론과 같은 피처 생성 함수)

ID_COL, TARGET = T.ID_COL, T.TARGET_COL


# ----------------------------------------------------------------------------
# 점수 함수
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bss(y, p):
    """Brier Skill Score = 100000 * (1 - Brier / (r(1-r))).  대회 산식과 동일(하한 0 클리핑만 생략)."""
    y = np.asarray(y, dtype=float)
    r = y.mean()
    return 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def oracle_shift(p, y):
    """예측 평균이 실제 평균과 같아지도록 로짓 상수를 더한다 (판별력만 비교하기 위한 상한선)."""
    z = T.logit(p)
    d = brentq(lambda d: sigmoid(z + d).mean() - float(np.mean(y)), -5, 5)
    return sigmoid(z + d)


# ----------------------------------------------------------------------------
# 공통 유틸
# ----------------------------------------------------------------------------
def make_cfg(train, keep_game_type=False):
    """제출 모델과 같은 설정. keep_game_type=True 면 game_type 을 범주형 피처로 되살린다."""
    cfg = T.build_cfg(train)
    if keep_game_type:
        raw = [c for c in train.columns if c not in (ID_COL, TARGET)]
        new = ["hand_combo", "count_diff", "walk_threat", "pitcher_rate_smoothed", "li_x_pitcher_rate"]
        cfg["FEATS"] = raw + new
        cfg["CATS"] = cfg["CATS"] + ["game_type"]
        cfg["CAT_LEVELS"]["game_type"] = sorted(train["game_type"].astype(str).unique().tolist())
        cfg["DROP"] = []
    return cfg


def fit(X, y, cats, seed=42, **overrides):
    params = dict(T.LGB_PARAMS)
    params.update(overrides)
    m = lgb.LGBMClassifier(random_state=seed, **params)
    m.fit(X, y, categorical_feature=cats)
    return m


def masks(train, val_season):
    return train["season"] < val_season, train["season"] == val_season


def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ----------------------------------------------------------------------------
# 실험 1. 베이스라인 vs 개선 모델
# ----------------------------------------------------------------------------
def exp_baseline(train, folds):
    header("baseline: 대회 베이스라인(RF) vs 피처 엔지니어링 + LightGBM  [val=2024]")
    tr, va = masks(train, 2024)
    y_tr, y_va = train.loc[tr, TARGET], train.loc[va, TARGET].values

    # (a) 대회 베이스라인과 같은 설정: 원본 47개 컬럼 + 순서형 인코딩 + 중앙값 대치 + 얕은 RF
    feats = [c for c in train.columns if c not in (ID_COL, TARGET)]
    cats = ["top_bottom", "game_type", "base_state"]
    nums = [c for c in feats if c not in cats]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
        ("num", SimpleImputer(strategy="median"), nums),
    ])
    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=42))])
    t0 = time.time()
    rf.fit(train.loc[tr, feats], y_tr)
    print(f"[베이스라인 RF           ] raw={bss(y_va, rf.predict_proba(train.loc[va, feats])[:, 1]):8.1f}"
          f"  ({time.time() - t0:.0f}s)")
    del rf

    # (b) 피처 엔지니어링 + LightGBM (game_type 은 아직 포함한 상태)
    cfg = make_cfg(train, keep_game_type=True)
    X = inference.build_features(train, cfg)
    t0 = time.time()
    m = fit(X[tr], y_tr, cfg["CATS"])
    print(f"[피처엔지니어링 + LightGBM] raw={bss(y_va, m.predict_proba(X[va])[:, 1]):8.1f}"
          f"  ({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------
# 실험 2. game_type 포함/제외
# ----------------------------------------------------------------------------
def exp_gametype(train, folds):
    header("gametype: game_type 피처 포함 vs 제외  (raw 점수)")
    for keep in (True, False):
        cfg = make_cfg(train, keep_game_type=keep)
        X = inference.build_features(train, cfg)
        for val in folds:
            tr, va = masks(train, val)
            t0 = time.time()
            m = fit(X[tr], train.loc[tr, TARGET], cfg["CATS"])
            p = m.predict_proba(X[va])[:, 1]
            tag = "game_type 포함" if keep else "game_type 제외"
            print(f"[{tag}] val={val}  raw={bss(train.loc[va, TARGET].values, p):9.1f}"
                  f"  ({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------
# 실험 3. 시즌 드리프트 보정(DELTA)
# ----------------------------------------------------------------------------
def exp_season_shift(train, folds):
    header("season_shift: 3시드 평균 모델에 DELTA 보정을 적용  (game_type 제외)")
    cfg = make_cfg(train)
    X = inference.build_features(train, cfg)
    for val in folds:
        tr, va = masks(train, val)
        y_va = train.loc[va, TARGET].values

        # DELTA 는 '학습 구간의 데이터만' 으로 만든다 (검증 시즌 정답은 보지 않음)
        ref_fit, ref_hold = train["season"] < val - 1, train["season"] == val - 1
        ref = fit(X[ref_fit], train.loc[ref_fit, TARGET], cfg["CATS"], seed=T.SEEDS[0])
        ref_predmean = float(ref.predict_proba(X[ref_hold])[:, 1].mean())
        sm = train.loc[tr].groupby("season")[TARGET].mean()
        slope, icpt = np.polyfit(sm.index.astype(float), sm.values, 1)
        target_mean = float(np.clip(slope * val + icpt, 0.35, 0.65))
        delta = float(T.logit(target_mean) - T.logit(ref_predmean))

        p = np.mean([fit(X[tr], train.loc[tr, TARGET], cfg["CATS"], seed=s).predict_proba(X[va])[:, 1]
                     for s in T.SEEDS], axis=0)
        print(f"val={val}  외삽 평균={target_mean:.4f}  실제 평균={y_va.mean():.4f}  DELTA={delta:+.5f}")
        print(f"   보정 없음      raw={bss(y_va, p):8.1f}   (예측 평균 {p.mean():.4f})")
        ps = sigmoid(T.logit(p) + delta)
        print(f"   + DELTA 보정   raw={bss(y_va, ps):8.1f}   (예측 평균 {ps.mean():.4f})")
        print(f"   (oracle 상한)  raw={bss(y_va, oracle_shift(p, y_va)):8.1f}")


# ----------------------------------------------------------------------------
# 실험 4. 정규화 강도
# ----------------------------------------------------------------------------
def exp_regularization(train, folds):
    header("regularization: min_child_samples / reg_lambda  (판별력 점수)")
    cfg = make_cfg(train)
    X = inference.build_features(train, cfg)
    grid = [("현재 제출본 (mcs=200, l2=0)", dict(min_child_samples=200, reg_lambda=0)),
            ("mcs=2000,  l2=100", dict(min_child_samples=2000, reg_lambda=100)),
            ("mcs=10000, l2=100", dict(min_child_samples=10000, reg_lambda=100))]
    for tag, params in grid:
        row = []
        for val in folds:
            tr, va = masks(train, val)
            m = fit(X[tr], train.loc[tr, TARGET], cfg["CATS"], **params)
            y_va = train.loc[va, TARGET].values
            row.append(f"val={val}: {bss(y_va, oracle_shift(m.predict_proba(X[va])[:, 1], y_va)):8.1f}")
        print(f"{tag:<30} " + "   ".join(row))


# ----------------------------------------------------------------------------
# 실험 5. 피처 추가
# ----------------------------------------------------------------------------
def _shrink(rate, n, prior, k):
    n = n.fillna(0)
    return ((rate * n).fillna(0) + prior * k).div(n + k).astype("float32")


def add_extra_features(train, X):
    """shrinkage 를 다른 지표로 확장 + 최근 폼 이탈도 + 매치업 + 구종 다양성 등 14개.
    (사전분포 평균은 전체 train 기준 — 피처 추가 효과를 보기 위한 근사)"""
    X = X.copy()
    mean = lambda c: float(train[c].mean())
    pn, bn = train["asof_pitcher_n"], train["asof_batter_n"]
    X["p_reverse_sm"] = _shrink(train["asof_pitcher_reverse_rate"], pn, mean("asof_pitcher_reverse_rate"), 200)
    X["p_middle_sm"] = _shrink(train["asof_pitcher_middle_rate"], pn, mean("asof_pitcher_middle_rate"), 200)
    X["p_ball_sm"] = _shrink(train["asof_pitcher_ball_rate"], pn, mean("asof_pitcher_ball_rate"), 200)
    X["b_success_sm"] = _shrink(train["asof_batter_success_rate"], bn, mean("asof_batter_success_rate"), 100)
    X["b_middle_sm"] = _shrink(train["asof_batter_middle_rate"], bn, mean("asof_batter_middle_rate"), 100)
    for k in (1, 3, 5):
        X[f"form_dev{k}"] = (train[f"asof_pitcher_prev{k}_game_success_rate"] - X["pitcher_rate_smoothed"]).astype("float32")
    X["matchup_edge"] = (X["pitcher_rate_smoothed"] - X["b_success_sm"]).astype("float32")
    X["mid_dev1"] = (train["asof_pitcher_prev1_game_middle_rate"] - X["p_middle_sm"]).astype("float32")
    mix = np.clip(train[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                         "asof_pitcher_offspeed_rate"]].fillna(1 / 3).values, 1e-6, 1)
    X["mix_entropy"] = (-(mix * np.log(mix)).sum(axis=1)).astype("float32")
    X["mix_fb_dev"] = (train["asof_pitcher_fastball_rate"] - mean("asof_pitcher_fastball_rate")).astype("float32")
    cc = train["balls_before"].astype(str) + "-" + train["strikes_before"].astype(str)
    X["count_cat"] = pd.Categorical(cc, categories=sorted(cc.unique()))
    X["exp_log"] = np.log1p(train["asof_pitcher_n"].fillna(0)).astype("float32")
    return X


def exp_extra_features(train, folds):
    header("extra_features: 51개(현재) vs 65개(+14개)  (판별력 점수)")
    cfg = make_cfg(train)
    X1 = inference.build_features(train, cfg)
    X2 = add_extra_features(train, X1)
    for tag, X, cats in [("51개 (현재)", X1, cfg["CATS"]), ("65개 (+14개)", X2, cfg["CATS"] + ["count_cat"])]:
        row = []
        for val in folds:
            tr, va = masks(train, val)
            m = fit(X[tr], train.loc[tr, TARGET], cats)
            y_va = train.loc[va, TARGET].values
            row.append(f"val={val}: {bss(y_va, oracle_shift(m.predict_proba(X[va])[:, 1], y_va)):8.1f}")
        print(f"{tag:<14} " + "   ".join(row))


EXPERIMENTS = {
    "baseline": exp_baseline,
    "gametype": exp_gametype,
    "season_shift": exp_season_shift,
    "regularization": exp_regularization,
    "extra_features": exp_extra_features,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--exp", default="all", choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--folds", default="2023,2024", help="검증 시즌(쉼표 구분)")
    ap.add_argument("--sample", type=float, default=1.0, help="행 샘플링 비율(동작 확인용)")
    args = ap.parse_args()

    train = T.downcast(pd.read_csv(args.data, encoding="utf-8-sig"))
    if args.sample < 1.0:
        train = train.sample(frac=args.sample, random_state=0).reset_index(drop=True)
        print(f"[주의] 행 {args.sample:.0%} 샘플링 — 동작 확인용, 점수는 의미 없음")
    folds = [int(v) for v in args.folds.split(",")]
    print(f"rows={len(train):,}  folds={folds}")

    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    for name in names:
        EXPERIMENTS[name](train, folds)


if __name__ == "__main__":
    main()
