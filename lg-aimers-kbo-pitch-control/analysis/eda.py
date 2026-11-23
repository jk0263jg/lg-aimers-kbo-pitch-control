"""
EDA 스크립트 — README 에 들어가는 그림을 전부 다시 만들어 docs/images/ 에 저장한다.

    python analysis/eda.py --data data/train.csv --out docs/images

[이 스크립트가 답하려는 질문]
  Q1. 볼카운트에 따라 제구 성공률이 달라지는가?        → count_heatmap.png
  Q2. 투수·타자 손 조합이 영향을 주는가?               → handedness.png
  Q3. 과거 성공률(asof_*)을 그대로 믿어도 되는가?      → calibration.png   (평균 회귀 확인)
  Q4. 이닝이 진행되면 성공률이 달라지는가?             → inning_trend.png
  Q5. 시즌이 바뀌어도 분포가 유지되는가?               → season_drift.png, game_type_flip.png
      (이 대회에서 가장 중요했던 질문: 미래 시즌을 예측해야 하므로 '시간에 따른 안정성'이 핵심)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET = "control_success"


def setup_font():
    """한글 폰트를 찾아 적용 (Windows/Mac/Linux 어디서든 동작하도록 후보를 순회)."""
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                  "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans KR"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


def count_heatmap(t, out):
    """볼-스트라이크 카운트별 성공률. 3볼에서 성공률이 일관되게 낮아지는지 확인."""
    pv = t.pivot_table(index="strikes_before", columns="balls_before", values=TARGET, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pv.values, cmap="RdYlGn", vmin=0.49, vmax=0.54)
    ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
    ax.set_xlabel("볼 개수"); ax.set_ylabel("스트라이크 개수")
    ax.set_title("볼카운트별 제구 성공률")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(im, label="성공률")
    save(fig, out, "count_heatmap.png")
    print(pv.round(4).to_string(), "\n")


def handedness(t, out):
    """투수·타자 손 조합별 성공률 (손 값은 익명화되어 1/2)."""
    g = t.groupby(["pitcher_hand", "batter_hand"])[TARGET].agg(["mean", "count"]).reset_index()
    labels = [f"투수{p}·타자{b}" for p, b in zip(g["pitcher_hand"], g["batter_hand"])]
    avg = t[TARGET].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, g["mean"], color=["#e74c3c" if v < avg - 0.02 else "#2ecc71" for v in g["mean"]])
    ax.axhline(avg, color="gray", ls="--", lw=1, label=f"전체 평균 {avg:.3f}")
    ax.set_ylim(0.45, 0.56); ax.set_ylabel("성공률"); ax.legend()
    ax.set_title("투수·타자 손 조합별 제구 성공률")
    for b, v in zip(bars, g["mean"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center")
    save(fig, out, "handedness.png")
    print(g.round(4).to_string(index=False), "\n")


def calibration(t, out):
    """asof_pitcher_success_rate 를 확률로 보고 실제 성공률과 비교 → 평균 회귀 여부 확인."""
    v = t.dropna(subset=["asof_pitcher_success_rate"]).copy()
    v["bucket"] = pd.qcut(v["asof_pitcher_success_rate"], 10, duplicates="drop")
    c = v.groupby("bucket", observed=True).agg(pred=("asof_pitcher_success_rate", "mean"),
                                               actual=(TARGET, "mean"))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo, hi = min(c.min()) - 0.02, max(c.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="완벽한 보정")
    ax.plot(c["pred"], c["actual"], "o-", color="#3498db", label="asof_pitcher_success_rate")
    ax.set_xlabel("과거 성공률 (피처 값)"); ax.set_ylabel("실제 성공률")
    ax.set_title("과거 성공률의 보정 확인 (평균 회귀)")
    ax.legend()
    save(fig, out, "calibration.png")
    print(c.round(4).to_string(), "\n")


def inning_trend(t, out):
    g = t[t["inning"] <= 9].groupby("inning")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g.index, g.values, "o-", color="#9b59b6")
    ax.set_xticks(range(1, 10)); ax.set_xlabel("이닝"); ax.set_ylabel("성공률")
    ax.set_title("이닝별 제구 성공률")
    save(fig, out, "inning_trend.png")
    print(g.round(4).to_string(), "\n")


def season_drift(t, out):
    """시즌별 리그 평균. 학습 구간(2019~2023)이 아닌 마지막 시즌까지 포함해 직선 외삽."""
    g = t.groupby("season")[TARGET].mean()
    slope, intercept = np.polyfit(g.index.astype(float), g.values, 1)
    nxt = int(g.index.max()) + 1
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(g.index, g.values, "o-", color="#c0392b", lw=2, ms=8, label="시즌별 실제 평균")
    xs = np.array([g.index.min(), nxt])
    ax.plot(xs, slope * xs + intercept, "--", color="gray", label="선형 추세 외삽")
    ax.scatter([nxt], [slope * nxt + intercept], color="#3498db", zorder=5, label=f"{nxt} 외삽값")
    ax.set_xlabel("시즌"); ax.set_ylabel("제구 성공률"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("리그 평균 제구 성공률의 시즌별 하락")
    save(fig, out, "season_drift.png")
    print(g.round(4).to_string())
    print(f"외삽 {nxt}: {slope * nxt + intercept:.4f}  (기울기 {slope:+.4f}/년)\n")


def game_type_flip(t, out):
    """game_type 별 시즌 평균. F 가 R 보다 높다가 2023부터 역전되는지 확인."""
    pv = t.pivot_table(index="season", columns="game_type", values=TARGET, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for col, color in zip(pv.columns, ["#e67e22", "#2c3e50"]):
        ax.plot(pv.index, pv[col], "o-", lw=2, color=color, label=f"game_type={col}")
    ax.set_xlabel("시즌"); ax.set_ylabel("제구 성공률"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("game_type별 성공률: 시즌 간 관계가 뒤집힘")
    save(fig, out, "game_type_flip.png")
    print(pv.round(4).to_string(), "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--out", default="docs/images")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    setup_font()
    cols = ["season", "inning", "balls_before", "strikes_before", "pitcher_hand",
            "batter_hand", "game_type", "asof_pitcher_success_rate", TARGET]
    t = pd.read_csv(args.data, encoding="utf-8-sig", usecols=cols)
    print(f"rows={len(t):,}  평균 성공률={t[TARGET].mean():.4f}\n")

    for fn in (count_heatmap, handedness, calibration, inning_trend, season_drift, game_type_flip):
        fn(t, args.out)


if __name__ == "__main__":
    main()
