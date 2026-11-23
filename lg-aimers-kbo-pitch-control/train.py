"""
학습 스크립트 — submission/model/ 아래에 추론에 필요한 파일을 전부 만든다.

    python train.py --data data/train.csv --out submission/model

생성물
  lgbm_seed{0,1,2}.txt : LightGBM 네이티브 텍스트 모델 (3개 시드)
  config.json          : 피처 목록 / 범주 목록 / 시즌 드리프트 보정값(DELTA) 등

[이 스크립트가 하는 일 — 큰 흐름]
  1) train.csv 로드
  2) 피처 생성  → submission/script.py 의 build_features() 를 그대로 import 해서 사용
                  (학습과 추론이 같은 함수를 쓰므로 '학습-추론 불일치(train/serve skew)'가 생길 수 없음)
  3) DELTA 산출 → 2025 시즌 리그 평균을 학습 데이터 추세로 외삽하고, 모델 예측 평균과의 차이를 로짓 상수로 저장
  4) 최종 학습 → 전체 데이터로 3개 시드 학습 후 저장
"""

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

# 추론 코드와 피처 생성 로직을 공유한다 (같은 함수를 쓰는 것이 핵심)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submission"))
import script as inference  # noqa: E402  (submission/script.py)

ID_COL = "row_id"
TARGET_COL = "control_success"

# --- 설계 상수 -------------------------------------------------------------
GLOBAL_MEAN = 0.5236            # train.csv 전체 평균 제구 성공률 (EDA 값, 고정 상수)
SHRINK_K = 200                  # 경험적 베이즈 축소 강도: 투구 200개 분량의 사전분포
DROP_FEATURES = ["game_type"]   # 시즌 간 부호가 뒤집히는 비정상 피처 → 제거
CATS = ["top_bottom", "base_state", "hand_combo"]
SEEDS = [42, 202, 777]
TARGET_SEASON = 2025            # 평가 시즌

# 참고: subsample 은 subsample_freq 가 0 이라 실제 행 샘플링에는 적용되지 않는다.
#       (시드 간 차이는 colsample_bytree 의 열 샘플링에서만 생김) — 제출본과 동일하게 유지.
LGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=-1,
    n_jobs=-1,
)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def downcast(df):
    """메모리 절약(float64→float32, int64→작은 정수). 제출본 학습 때와 같은 수치 정밀도 유지."""
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "int64":
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def build_cfg(train):
    """추론 때 그대로 쓰일 설정(config.json 내용)을 만든다. 모델 학습 전에 확정되는 값들."""
    feats_raw = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    new_feats = ["hand_combo", "count_diff", "walk_threat",
                 "pitcher_rate_smoothed", "li_x_pitcher_rate"]
    feats = [c for c in feats_raw if c not in DROP_FEATURES] + new_feats

    # 범주 목록을 학습 데이터에서 고정 → 추론 때 새 값이 와도 컬럼 구조가 깨지지 않음
    hand = (train["pitcher_hand"].astype(str) + "_" + train["batter_hand"].astype(str))
    levels = {
        "top_bottom": sorted(train["top_bottom"].astype(str).unique().tolist()),
        "base_state": sorted(train["base_state"].astype(str).unique().tolist()),
        "hand_combo": sorted(hand.unique().tolist()),
    }
    return dict(FEATS=feats, CATS=CATS, CAT_LEVELS=levels, DROP=DROP_FEATURES,
                GLOBAL_MEAN=GLOBAL_MEAN, K=SHRINK_K)


def fit_model(X, y, seed):
    m = lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)
    m.fit(X, y, categorical_feature=CATS)
    return m


def compute_delta(train, X, cfg):
    """시즌 드리프트 보정값 DELTA 를 '학습 데이터만으로' 계산한다.

    1) 마지막 시즌(2024)을 뺀 데이터로 참조 모델을 학습 → 2024 예측 평균(ref_predmean)
    2) 시즌별 실제 평균에 직선을 맞춰 2025 평균을 외삽(target_mean)
    3) DELTA = logit(target_mean) - logit(ref_predmean)
       → 추론 때 모든 예측의 로짓에 이 상수를 더한다 (테스트 데이터는 전혀 참조하지 않음)
    """
    last = int(train["season"].max())
    fit_mask = train["season"] < last
    hold_mask = train["season"] == last

    ref = fit_model(X[fit_mask], train.loc[fit_mask, TARGET_COL], seed=SEEDS[0])
    ref_predmean = float(ref.predict_proba(X[hold_mask])[:, 1].mean())

    season_mean = train.groupby("season")[TARGET_COL].mean()
    slope, intercept = np.polyfit(season_mean.index.astype(float), season_mean.values, 1)
    target_mean = float(np.clip(slope * TARGET_SEASON + intercept, 0.35, 0.65))

    delta = float(logit(target_mean) - logit(ref_predmean))
    print("시즌별 평균 제구 성공률:")
    print(season_mean.round(4).to_string())
    print(f"{TARGET_SEASON} 외삽 목표 평균={target_mean:.4f} | "
          f"{last} 예측 평균(참조모델)={ref_predmean:.4f} | DELTA={delta:+.5f}")
    return delta, target_mean, ref_predmean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--out", default="submission/model")
    args = ap.parse_args()

    t0 = time.time()
    train = downcast(pd.read_csv(args.data, encoding="utf-8-sig"))
    print(f"train: {train.shape}  시즌 {train['season'].min()}~{train['season'].max()}  "
          f"({time.time() - t0:.0f}s)")

    cfg = build_cfg(train)
    X = inference.build_features(train, cfg)      # ← 추론 때와 완전히 같은 함수
    y = train[TARGET_COL]
    print(f"features={X.shape[1]}  dropped={DROP_FEATURES}")

    print("\n=== DELTA 산출 ===")
    delta, target_mean, ref_predmean = compute_delta(train, X, cfg)

    print("\n=== 최종 모델 학습 (전체 데이터, 3 seed) ===")
    os.makedirs(args.out, exist_ok=True)
    model_files = []
    for i, seed in enumerate(SEEDS):
        t1 = time.time()
        m = fit_model(X, y, seed)
        fn = f"lgbm_seed{i}.txt"
        m.booster_.save_model(os.path.join(args.out, fn))  # pickle 대신 네이티브 텍스트 포맷
        model_files.append(fn)
        print(f"  seed {seed}: {time.time() - t1:.0f}s → {fn}")

    cfg.update(DELTA=delta, target_mean_2025=target_mean,
               ref_predmean_2024=ref_predmean, MODEL_FILES=model_files)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"\n저장 완료 → {args.out}  (총 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
