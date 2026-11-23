"""
LG Aimers 9기 온라인 해커톤 (Phase 2) — 투구 제구 성공 확률 예측
추론 스크립트 (평가 서버가 자동 실행)

[모델 저장 방식]
  LightGBM 네이티브 텍스트 포맷(.txt)만 사용합니다. pickle/joblib 을 쓰지 않으므로
  로컬 빌드 환경과 평가 서버의 numpy/pandas/python 버전이 달라도 로드가 실패하지 않습니다.

[설계 요약]
 1) 피처 엔지니어링
    - hand_combo            : 투수-타자 좌우 조합 (조합별 성공률 격차 최대 4.7%p)
    - pitcher_rate_smoothed : asof_pitcher_success_rate 의 경험적 베이즈 축소(shrinkage).
                              표본이 적은 투수의 극단값을 리그 평균 쪽으로 당겨 평균회귀를 보정
    - li_x_pitcher_rate     : 상황 중요도 x 투수 제구력 상호작용
    - count_diff, walk_threat : 볼카운트 압박 (3볼에서 성공률이 일관되게 하락)
 2) game_type 제거
    - gain 기준 1위 피처였으나 2022 -> 2023 사이 수준이 역전되는 비정상(non-stationary) 피처.
      2023 시즌 검증에서 점수를 크게 악화시켜 제거함.
 3) 시즌 드리프트 보정 (DELTA)
    - 리그 평균 제구 성공률이 2019(0.565) -> 2024(0.486) 로 단조 하락.
    - 학습 데이터의 시즌별 평균에 선형 추세를 적합해 2025 값을 외삽하고,
      그 목표 평균에 맞도록 로짓 공간에서 상수 DELTA 를 더함.
    - DELTA 는 학습 시점에 확정된 상수(config.json)이며 평가 데이터를 전혀 참조하지 않음.
      각 행에 독립적으로 적용되므로 '평가 데이터 행 간 정보 사용 금지' 규칙을 준수함.
 4) 3개 시드 LightGBM 평균 — 분산 감소
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

TEST_DIR = "./data"
MODEL_DIR = "./model"
OUT_DIR = "./output"

TEST_PATH = os.path.join(TEST_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

# 모델 로드 실패 등 최악의 경우에도 submission.csv 는 반드시 생성되도록 하기 위한 값
FALLBACK_P = 0.4747  # 2025 리그 평균 제구 성공률 외삽치


# =======================
# 데이터 로드
# =======================
def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}"
        )
    return df


# =======================
# 전처리 (학습과 완전히 동일)
# =======================
def build_features(df, cfg):
    """학습 때와 동일한 피처 생성. 행 간 정보를 일절 사용하지 않음."""
    df = df.copy()
    gm = float(cfg["GLOBAL_MEAN"])
    k = float(cfg["K"])

    df["hand_combo"] = (
        df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    )
    df["count_diff"] = (df["balls_before"] - df["strikes_before"]).astype("int8")
    df["walk_threat"] = (df["balls_before"] == 3).astype("int8")

    # 경험적 베이즈 축소: 표본 수가 적을수록 리그 평균 쪽으로 당겨짐
    n = df["asof_pitcher_n"].fillna(0)
    succ = df["asof_pitcher_success_rate"] * n
    df["pitcher_rate_smoothed"] = ((succ.fillna(0) + gm * k) / (n + k)).astype("float32")

    df["li_x_pitcher_rate"] = (
        df["li"] * df["asof_pitcher_success_rate"].fillna(gm)
    ).astype("float32")

    # 범주형은 학습 때와 동일한 카테고리 목록을 고정 사용 (미지의 값은 결측 처리)
    for c in cfg["CATS"]:
        df[c] = pd.Categorical(df[c].astype(str), categories=cfg["CAT_LEVELS"][c])

    missing = [c for c in cfg["FEATS"] if c not in df.columns]
    if missing:
        raise ValueError(f"필요한 피처가 test 데이터에 없음: {missing}")

    return df[cfg["FEATS"]]


# =======================
# 시즌 드리프트 보정
# =======================
def apply_prior_shift(p, delta):
    """로짓 공간에서 상수 delta 를 더해 예측 분포를 이동.
    delta 는 학습 시점에 결정된 상수이며 각 행에 독립적으로 적용됨."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) + float(delta)
    return 1.0 / (1.0 + np.exp(-z))


# =======================
# 제출 파일 생성
# =======================
def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================
def main():
    sub = None
    try:
        print("Load config...")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f" features={len(cfg['FEATS'])}  DELTA={cfg['DELTA']:+.5f}")

        print("Load test data...")
        test = load_test(TEST_PATH)
        sub = load_sample_submission(SAMPLE_SUB_PATH)
        print(f" test={len(test)}  submission={len(sub)}")

        print("Load models (LightGBM native text)...")
        import lightgbm as lgb

        boosters = [
            lgb.Booster(model_file=os.path.join(MODEL_DIR, fn))
            for fn in cfg["MODEL_FILES"]
        ]
        print(f" models={len(boosters)}")

        print("Build features...")
        ids = test[ID_COL].tolist()
        X = build_features(test, cfg)
        print(f" features={X.shape[1]}")

        print("Inference (seed ensemble)...")
        if len(X):
            preds = np.mean([b.predict(X) for b in boosters], axis=0)
            preds = apply_prior_shift(preds, cfg["DELTA"])
            preds = np.clip(preds, 1e-6, 1 - 1e-6)
        else:
            preds = np.array([])
        mean_str = f"{float(np.mean(preds)):.4f}" if len(preds) else "n/a"
        print(f" preds={len(preds)}  mean={mean_str}")

        print("Build submission...")
        sub = merge_predictions(sub, ids, preds)
        save_submission(OUT_PATH, sub)
        print(f"Saved: {OUT_PATH} (rows={len(sub)})")

    except Exception as e:
        # 어떤 경우에도 submission.csv 는 생성되도록 함 (미생성은 제출 오류로 처리되므로)
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        try:
            if sub is None:
                sub = load_sample_submission(SAMPLE_SUB_PATH)
            sub[TARGET_COL] = FALLBACK_P
            save_submission(OUT_PATH, sub)
            print(f"[FALLBACK] 상수 {FALLBACK_P} 로 {OUT_PATH} 생성 (rows={len(sub)})")
        except Exception as e2:
            print(f"[FATAL] fallback 실패: {e2}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
