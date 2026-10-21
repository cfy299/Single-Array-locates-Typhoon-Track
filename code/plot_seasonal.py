# -*- coding: utf-8 -*-
"""
按季度 + 全时段总图
"""
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

BASE    = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_R = 6371.0
CFY_LON, CFY_LAT = 118.0, 36.0
K_CALIB = 7.8135
V_TY    = 300.0
R2_MIN  = 0.10

plt.rcParams["font.family"] = ["Palatino Linotype", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def gc_dist(lo1, la1, lo2, la2):
    p1, p2 = np.deg2rad(la1), np.deg2rad(la2)
    dl = np.deg2rad(np.array(lo2) - np.array(lo1))
    a = np.sin((p2-p1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def project(lo0, la0, baz, d_km):
    R = EARTH_R; d = d_km / R; b = np.deg2rad(baz)
    la0r, lo0r = np.deg2rad(la0), np.deg2rad(lo0)
    la2 = np.arcsin(np.sin(la0r)*np.cos(d) + np.cos(la0r)*np.sin(d)*np.cos(b))
    lo2 = lo0r + np.arctan2(np.sin(b)*np.sin(d)*np.cos(la0r),
                            np.cos(d) - np.sin(la0r)*np.sin(la2))
    return float(np.rad2deg(lo2)), float(np.rad2deg(la2))


def parse_cma(fp):
    typhoons = []
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        if l.startswith("66666"):
            p = l.split()
            t_id, t_name = p[1], p[7]
            data = []
            i += 1
            while i < len(lines) and not lines[i].startswith("66666"):
                pp = lines[i].split()
                if len(pp) >= 6:
                    try:
                        dt = datetime.strptime(pp[0], "%Y%m%d%H")
                        data.append({"dt": dt, "lat": int(pp[2])/10.,
                                     "lon": int(pp[3])/10., "wind": int(pp[5])})
                    except Exception:
                        pass
                i += 1
            if data:
                typhoons.append({"id": t_id, "name": t_name, "data": data})
        else:
            i += 1
    return typhoons


# ── 数据 ──
cfy = pd.read_csv(BASE / "AmpGrad_baz.csv")
cfy["date"] = pd.to_datetime(cfy["date"])
cfy = cfy[cfy["array"] == "CFY"].set_index("date")[["baz_deg", "r2", "grad_per_km"]]

typhoons = (parse_cma(BASE/"2017、2018年台风路径数据"/"CH2017BST.txt") +
            parse_cma(BASE/"2017、2018年台风路径数据"/"CH2018BST.txt"))


def predict(ty, q0, q1):
    """对台风 ty 在窗口 [q0,q1] 内做物理外推预测; 返回 (real_df, pred_df, valid_dates)."""
    days_in = sorted({pd.Timestamp(p["dt"].strftime("%Y-%m-%d")) for p in ty["data"]
                      if q0 <= p["dt"] <= q1})
    valid = []
    for d in days_in:
        if d not in cfy.index: continue
        r2 = float(cfy.loc[d, "r2"]); g = float(cfy.loc[d, "grad_per_km"])
        if r2 < R2_MIN or g <= 0: continue
        valid.append(d)
    if len(valid) < 2:
        return None, None, None

    g0 = float(cfy.loc[valid[0], "grad_per_km"])
    d_start = float(np.clip(K_CALIB / g0, 500, 4500))
    daily = []
    for d in valid:
        baz = float(cfy.loc[d, "baz_deg"])
        days_off = (d - valid[0]).days
        dist = max(d_start - V_TY * days_off, 400)
        lo, la = project(CFY_LON, CFY_LAT, baz, dist)
        daily.append({"date": d, "lon": lo, "lat": la})
    daily_df = pd.DataFrame(daily)

    real6h = sorted([p for p in ty["data"]
                     if p["dt"].hour % 6 == 0
                     and valid[0] <= pd.Timestamp(p["dt"].strftime("%Y-%m-%d")) <= valid[-1]],
                    key=lambda x: x["dt"])
    if not real6h:
        return None, None, None

    real_df = pd.DataFrame([{"datetime": p["dt"], "lon": p["lon"],
                             "lat": p["lat"], "wind": p["wind"]}
                            for p in real6h])
    t_src = (daily_df["date"] - daily_df["date"].min()).dt.total_seconds().values
    pred6h = []
    for p in real6h:
        ts = (pd.Timestamp(p["dt"]) - daily_df["date"].min()).total_seconds()
        if ts < 0 or ts > t_src[-1]: continue
        lo = float(np.interp(ts, t_src, daily_df["lon"].values))
        la = float(np.interp(ts, t_src, daily_df["lat"].values))
        pred6h.append({"datetime": p["dt"], "lon": lo, "lat": la})
    pred_df = pd.DataFrame(pred6h)
    if pred_df.empty:
        return None, None, None
    real_df = real_df[real_df["datetime"].isin(pred_df["datetime"])].reset_index(drop=True)
    return real_df, pred_df, valid


QUARTERS = {
    "2017_Q4": (datetime(2017,10,1), datetime(2017,12,31, 23, 59)),
    "2018_Q1": (datetime(2018, 1,1), datetime(2018, 3,31, 23, 59)),
    "2018_Q3": (datetime(2018, 7,1), datetime(2018, 7,31, 23, 59)),
}

QUARTER_LABEL = {
    "cn": {"2017_Q4": "2017年第四季度",
           "2018_Q1": "2018年第一季度",
           "2018_Q3": "2018年第三季度（7月）"},
    "en": {"2017_Q4": "Q4 2017",
           "2018_Q1": "Q1 2018",
           "2018_Q3": "Q3 2018 (Jul)"},
}

# 调色板 (循环使用)
PALETTE = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD",
           "#FF7F0E", "#17BECF", "#8C564B", "#E377C2"]


def predict_all(q0, q1):
    """返回该窗口所有可预测的预测路线 [(pred_df, ), ...].
    注: 与 CMA 哪条台风的对应关系仅用于内部数据筛选, 不展示给用户."""
    routes = []
    for ty in typhoons:
        real_df, pred_df, valid = predict(ty, q0, q1)
        if pred_df is None: continue
        if len(pred_df) < 2: continue
        routes.append(pred_df)
    return routes


def setup_basemap(ax, ext, label_step, tick_fs=9):
    m = Basemap(projection="cyl", llcrnrlon=ext[0], urcrnrlon=ext[1],
                llcrnrlat=ext[2], urcrnrlat=ext[3], resolution="i", ax=ax)
    m.drawmapboundary(fill_color="#E0F2F7")
    m.fillcontinents(color="#F5E6D3", lake_color="#E0F2F7", alpha=0.6)
    m.drawcoastlines(linewidth=0.9, color="black")
    m.drawcountries(linewidth=0.5, color="gray", linestyle=":")
    m.drawmeridians(np.arange(int(ext[0]), int(ext[1])+1, label_step),
                    labels=[0,0,0,1], linewidth=0.4,
                    color="gray", dashes=[4,4], alpha=0.4, fontsize=tick_fs)
    m.drawparallels(np.arange(int(ext[2]), int(ext[3])+1, label_step),
                    labels=[1,0,0,0], linewidth=0.4,
                    color="gray", dashes=[4,4], alpha=0.4, fontsize=tick_fs)
    return m


# ───── 每季度单图 ─────
def draw_one_quarter(qkey: str, lang: str):
    q0, q1 = QUARTERS[qkey]
    routes = predict_all(q0, q1)

    fig, ax = plt.subplots(figsize=(12, 10))

    if not routes:
        ax.text(0.5, 0.5, "No data" if lang == "en" else "无数据",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="gray")
        ax.text(0.025, 0.97, QUARTER_LABEL[lang][qkey],
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="black", lw=0.6, alpha=0.95))
        suffix = "_EN" if lang == "en" else "_CN"
        out = OUT_DIR / f"quarter_{qkey}{suffix}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"[OK] {out}"); return

    all_lo, all_la = [], []
    for p in routes:
        all_lo += list(p["lon"]); all_la += list(p["lat"])
    rng_lo = max(all_lo) - min(all_lo); rng_la = max(all_la) - min(all_la)
    mg_lo = max(rng_lo * 0.18, 4); mg_la = max(rng_la * 0.18, 3)
    ext = [min(all_lo) - mg_lo, max(all_lo) + mg_lo,
           max(min(all_la) - mg_la, 0), max(all_la) + mg_la]

    m = setup_basemap(ax, ext, 5)

    route_lab = "Route" if lang == "en" else "轨迹"
    for i, pred_df in enumerate(routes):
        color = PALETTE[i % len(PALETTE)]
        xp, yp = m(pred_df["lon"].values, pred_df["lat"].values)
        t0 = pred_df.iloc[0]["datetime"]; t1 = pred_df.iloc[-1]["datetime"]
        label = f"{route_lab} {i+1}  {t0.strftime('%m-%d')} ~ {t1.strftime('%m-%d')}"
        ax.plot(xp, yp, "-", color=color, linewidth=2.2, alpha=0.92,
                zorder=5, label=label)
        ax.scatter(xp, yp, s=42, color=color, edgecolors="black",
                   linewidths=0.5, zorder=6)
        # 每 24h (4 个 6h 点) 标时间
        step = 4
        for j in range(0, len(pred_df), step):
            row = pred_df.iloc[j]
            lx, ly = m(float(row["lon"]), float(row["lat"]) + 0.35)
            ax.text(lx, ly, row["datetime"].strftime("%m/%d %Hh"),
                    fontsize=7, ha="center", va="bottom",
                    color=color, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.13", fc="white",
                              ec=color, lw=0.4, alpha=0.88))

    ax.text(0.025, 0.97, QUARTER_LABEL[lang][qkey],
            transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.32", fc="lightyellow",
                      ec="black", lw=0.6, alpha=0.95), zorder=10)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92,
              title=("Predicted routes" if lang == "en" else "预测轨迹"),
              title_fontsize=10)

    plt.tight_layout()
    suffix = "_EN" if lang == "en" else "_CN"
    out = OUT_DIR / f"quarter_{qkey}{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ───── 全时段总图 ─────
def draw_total(lang: str):
    q0 = datetime(2017, 10, 1); q1 = datetime(2018, 7, 31, 23, 59)
    routes = predict_all(q0, q1)

    fig, ax = plt.subplots(figsize=(15, 11))
    if not routes:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        out = OUT_DIR / f"total_typhoon_prediction_{lang.upper()}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); return

    all_lo, all_la = [], []
    for p in routes:
        all_lo += list(p["lon"]); all_la += list(p["lat"])
    mg_lo, mg_la = 4, 4
    ext = [min(all_lo) - mg_lo, max(all_lo) + mg_lo,
           max(min(all_la) - mg_la, 0), max(all_la) + mg_la]
    m = setup_basemap(ax, ext, 5, tick_fs=13)

    route_lab = "Route" if lang == "en" else "轨迹"
    for i, pred_df in enumerate(routes):
        color = PALETTE[i % len(PALETTE)]
        xp, yp = m(pred_df["lon"].values, pred_df["lat"].values)
        t0 = pred_df.iloc[0]["datetime"]; t1 = pred_df.iloc[-1]["datetime"]
        label = f"{route_lab} {i+1}  {t0.strftime('%Y-%m-%d')} ~ {t1.strftime('%m-%d')}"
        ax.plot(xp, yp, "-", color=color, linewidth=2.0, alpha=0.92,
                zorder=5, label=label)
        ax.scatter(xp, yp, s=34, color=color, edgecolors="black",
                   linewidths=0.4, zorder=6)
        step = 4
        for j in range(0, len(pred_df), step):
            row = pred_df.iloc[j]
            lx, ly = m(float(row["lon"]), float(row["lat"]) + 0.3)
            ax.text(lx, ly, row["datetime"].strftime("%Y-%m-%d"),
                    fontsize=8.5, ha="center", va="bottom",
                    color=color, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec=color, lw=0.4, alpha=0.85))

    leg_title = "Predicted routes" if lang == "en" else "预测轨迹"
    ax.legend(loc="lower left", fontsize=10.5, framealpha=0.92,
              ncol=2, title=leg_title, title_fontsize=12)

    plt.tight_layout()
    suffix = "_EN" if lang == "en" else "_CN"
    out = OUT_DIR / f"total_typhoon_prediction{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


for lang in ["cn", "en"]:
    for qkey in QUARTERS.keys():
        draw_one_quarter(qkey, lang)
    draw_total(lang)
