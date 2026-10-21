# -*- coding: utf-8 -*-
"""
AMPIL 高质量段 6h 预测轨迹 — 不同台风移速 V_TY 的变体图
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
R2_MIN  = 0.15

# 需要绘制的移速 (km/day)
SPEEDS = [240, 270, 285, 300, 315, 330, 360]

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
    with open(fp, "r", encoding="utf-8") as f: lines = f.readlines()
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        if l.startswith("66666"):
            p = l.split(); t_id, t_name = p[1], p[7]; data = []; i += 1
            while i < len(lines) and not lines[i].startswith("66666"):
                pp = lines[i].split()
                if len(pp) >= 6:
                    try:
                        dt = datetime.strptime(pp[0], "%Y%m%d%H")
                        data.append({"dt": dt, "lat": int(pp[2])/10.,
                                     "lon": int(pp[3])/10., "wind": int(pp[5])})
                    except Exception: pass
                i += 1
            if data: typhoons.append({"id": t_id, "name": t_name, "data": data})
        else: i += 1
    return typhoons


# ── 数据 ──
cfy = pd.read_csv(BASE / "AmpGrad_baz.csv")
cfy["date"] = pd.to_datetime(cfy["date"])
cfy = cfy[cfy["array"] == "CFY"].set_index("date")[["baz_deg", "r2", "grad_per_km"]]

typhoons = parse_cma(BASE / "2017、2018年台风路径数据" / "CH2018BST.txt")
ampil = next(t for t in typhoons if t["name"] == "AMPIL")

real6h = sorted([p for p in ampil["data"]
                 if p["dt"].hour % 6 == 0
                 and datetime(2018, 7, 18) <= p["dt"] <= datetime(2018, 7, 21, 23)],
                key=lambda x: x["dt"])
real_crop = pd.DataFrame([{"datetime": p["dt"], "lon": p["lon"],
                           "lat": p["lat"], "wind": p["wind"]} for p in real6h])

GOOD_DAYS = [pd.Timestamp(f"2018-07-{d}") for d in [18, 19, 20, 21]]
g0 = float(cfy.loc[GOOD_DAYS[0], "grad_per_km"])
d_start = float(np.clip(K_CALIB / g0, 500, 4000))
print(f"起点距离 d0 = {d_start:.0f} km  (g0 = {g0:.6f})")


def predict_with_speed(v_ty):
    """给定 V_TY, 返回 6h 预测 DataFrame."""
    daily = []
    for d in GOOD_DAYS:
        if d not in cfy.index: continue
        baz = float(cfy.loc[d, "baz_deg"]); r2 = float(cfy.loc[d, "r2"])
        g = float(cfy.loc[d, "grad_per_km"])
        if r2 < R2_MIN or g <= 0: continue
        days = (d - GOOD_DAYS[0]).days
        dist = max(d_start - v_ty * days, 400)
        lo, la = project(CFY_LON, CFY_LAT, baz, dist)
        daily.append({"date": d, "lon": lo, "lat": la, "dist": dist})
    daily_df = pd.DataFrame(daily)

    t_src = (daily_df["date"] - daily_df["date"].min()).dt.total_seconds().values
    pred = []
    for p in real6h:
        ts = (pd.Timestamp(p["dt"]) - daily_df["date"].min()).total_seconds()
        lo = float(np.interp(ts, t_src, daily_df["lon"].values))
        la = float(np.interp(ts, t_src, daily_df["lat"].values))
        pred.append({"datetime": p["dt"], "lon": lo, "lat": la, "wind": p["wind"]})
    return pd.DataFrame(pred), daily_df


# 统一视野: 覆盖所有速度下的预测点 + 真实轨迹
all_lo = list(real_crop["lon"]); all_la = list(real_crop["lat"])
pred_cache = {}
for v in SPEEDS:
    pdf, ddf = predict_with_speed(v)
    pred_cache[v] = pdf
    all_lo += list(pdf["lon"]); all_la += list(pdf["lat"])
    errs = gc_dist(pdf["lon"].values, pdf["lat"].values,
                   real_crop["lon"].values, real_crop["lat"].values)
    print(f"V_TY={v:3d} km/d: 6h 误差 均值={errs.mean():.0f} "
          f"中位={np.median(errs):.0f} 最小={errs.min():.0f} km  "
          f"终点距离={ddf.iloc[-1]['dist']:.0f} km")

cx, cy = np.mean(all_lo), np.mean(all_la); mg = 12
ext = [cx - mg, cx + mg, max(cy - 9, 0), cy + 9]


def draw_speed(v_ty, lang):
    pred_df = pred_cache[v_ty]
    cbar_label = "Wind speed (m/s)" if lang == "en" else "风速 (m/s)"

    fig, ax = plt.subplots(figsize=(11, 9))
    m = Basemap(projection="cyl", llcrnrlon=ext[0], urcrnrlon=ext[1],
                llcrnrlat=ext[2], urcrnrlat=ext[3], resolution="i", ax=ax)
    m.drawmapboundary(fill_color="#E0F2F7")
    m.fillcontinents(color="#F5E6D3", lake_color="#E0F2F7", alpha=0.6)
    m.drawcoastlines(linewidth=1.0, color="black")
    m.drawcountries(linewidth=0.6, color="gray", linestyle=":")
    m.drawmeridians(np.arange(int(ext[0]), int(ext[1]) + 1, 4),
                    labels=[0, 0, 0, 1], linewidth=0.5, color="gray",
                    dashes=[4, 4], alpha=0.4, fontsize=10)
    m.drawparallels(np.arange(int(ext[2]), int(ext[3]) + 1, 4),
                    labels=[1, 0, 0, 0], linewidth=0.5, color="gray",
                    dashes=[4, 4], alpha=0.4, fontsize=10)

    x, y = m(pred_df["lon"].values, pred_df["lat"].values)
    ax.plot(x, y, "-", color="gray", linewidth=2.5, alpha=0.5, zorder=4)
    sc = ax.scatter(x, y, c=real_crop["wind"].values, cmap="jet", s=110,
                    edgecolors="black", linewidths=0.7, zorder=5,
                    vmin=15, vmax=35)
    step = 4
    for i in range(0, len(pred_df), step):
        row = pred_df.iloc[i]
        lx, ly = m(float(row["lon"]), float(row["lat"]) + 0.5)
        ax.text(lx, ly, row["datetime"].strftime("%m/%d"), fontsize=9,
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          alpha=0.85, ec="black", lw=0.5), zorder=6)

    fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.7, label=cbar_label)
    plt.tight_layout()
    suffix = "_EN" if lang == "en" else "_CN"
    out = OUT_DIR / f"AMPIL_HighQ_Pred_V{v_ty}{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


for v in SPEEDS:
    draw_speed(v, "cn")
    draw_speed(v, "en")
