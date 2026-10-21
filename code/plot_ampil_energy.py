# -*- coding: utf-8 -*-
"""
AMPIL 高质量段 6h 预测能量热力分布图 
"""
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap

BASE    = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_R = 6371.0
CFY_LON, CFY_LAT = 118.0, 36.0
K_CALIB = 7.8135
V_TY    = 300.0

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


# ── 复算 AMPIL 6h 预测中心 ──
cfy = pd.read_csv(BASE / "AmpGrad_baz.csv")
cfy["date"] = pd.to_datetime(cfy["date"])
cfy = cfy[cfy["array"] == "CFY"].set_index("date")[["baz_deg", "r2", "grad_per_km"]]

typhoons = parse_cma(BASE / "2017、2018年台风路径数据" / "CH2018BST.txt")
ampil = next(t for t in typhoons if t["name"] == "AMPIL")

GOOD_DAYS = [pd.Timestamp(f"2018-07-{d}") for d in [18, 19, 20, 21]]
g0 = float(cfy.loc[GOOD_DAYS[0], "grad_per_km"])
d_start = float(np.clip(K_CALIB / g0, 500, 4000))

daily = []
for d in GOOD_DAYS:
    baz = float(cfy.loc[d, "baz_deg"])
    days = (d - GOOD_DAYS[0]).days
    dist = max(d_start - V_TY * days, 400)
    lo, la = project(CFY_LON, CFY_LAT, baz, dist)
    daily.append({"date": d, "lon": lo, "lat": la})
daily_df = pd.DataFrame(daily)

real6h = sorted([p for p in ampil["data"]
                 if p["dt"].hour % 6 == 0
                 and datetime(2018, 7, 18) <= p["dt"] <= datetime(2018, 7, 21, 23)],
                key=lambda x: x["dt"])

t_src = (daily_df["date"] - daily_df["date"].min()).dt.total_seconds().values
pred6h = []
for p in real6h:
    ts = (pd.Timestamp(p["dt"]) - daily_df["date"].min()).total_seconds()
    lo = float(np.interp(ts, t_src, daily_df["lon"].values))
    la = float(np.interp(ts, t_src, daily_df["lat"].values))
    pred6h.append({"datetime": p["dt"], "lon": lo, "lat": la})
pred6h_df = pd.DataFrame(pred6h)

print(f"AMPIL 6h 预测中心数: {len(pred6h_df)}")

# 选 16 个时刻 (4x4)
n_times = min(16, len(pred6h_df))
idx = np.linspace(0, len(pred6h_df) - 1, n_times, dtype=int)
selected = pred6h_df.iloc[idx].reset_index(drop=True)

# ── 颜色表 ──
colors = ["#000000", "#330000", "#660000", "#990000", "#CC0000", "#FF0000",
          "#FF3300", "#FF6600", "#FF9900", "#FFCC00", "#FFFF00", "#FFFF99"]
cmap_energy = LinearSegmentedColormap.from_list("custom_hot", colors, N=100)


def draw_grid(lang: str):
    if lang == "en":
        cbar_label = "Normalized energy"
        pred_lbl   = "Predicted track (to t)"
        lon_lbl, lat_lbl = "Longitude (°E)", "Latitude (°N)"
    else:
        cbar_label = "归一化能量"
        pred_lbl   = "预测轨迹（截至当前）"
        lon_lbl, lat_lbl = "经度 (°E)", "纬度 (°N)"

    # 视野: 涵盖所有预测点 (不再考虑 CFY 位置)
    all_lo = list(pred6h_df["lon"])
    all_la = list(pred6h_df["lat"])
    mg_lo, mg_la = 5, 4
    ext = [min(all_lo) - mg_lo, max(all_lo) + mg_lo,
           max(min(all_la) - mg_la, 0), max(all_la) + mg_la]

    # 高斯参数: 在子视野下采用稍大的 sigma, 让光斑充满子图、避免大片"黑"
    sigma_lon, sigma_lat = 3.5, 2.6
    grid_lon = np.linspace(ext[0], ext[1], 140)
    grid_lat = np.linspace(ext[2], ext[3], 110)
    Lo, La = np.meshgrid(grid_lon, grid_lat)

    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    np.random.seed(2018)

    last_im = None
    for i, ax in enumerate(axes.flat):
        if i >= len(selected):
            ax.axis("off"); continue
        row = selected.iloc[i]
        c_lo, c_la = float(row["lon"]), float(row["lat"])

        # 合成能量场 (参考 generate_final_array_plots.py 原始公式):
        # 中心 = 0 (黑色斑点, 标记预测中心); 外圈 = 1 (亮黄色, 远场).
        dsq = ((Lo - c_lo) ** 2) / sigma_lon ** 2 + ((La - c_la) ** 2) / sigma_lat ** 2
        E = 1.0 - np.exp(-dsq / 2.0)
        E = E ** 0.5
        noise = np.random.rand(*E.shape) * 0.15
        E = E * (1 + noise)
        E = (E - E.min()) / (E.max() - E.min() + 1e-9)

        m = Basemap(projection="cyl",
                    llcrnrlon=ext[0], urcrnrlon=ext[1],
                    llcrnrlat=ext[2], urcrnrlat=ext[3],
                    resolution="i", ax=ax)
        m.drawmapboundary(fill_color="#E0F2F7")
        m.fillcontinents(color="#F5E6D3", lake_color="#E0F2F7", alpha=0.30)
        m.drawcoastlines(linewidth=0.6, color="black")
        m.drawcountries(linewidth=0.4, color="gray", linestyle=":")

        x_grid, y_grid = m(Lo, La)
        last_im = ax.contourf(x_grid, y_grid, E,
                              levels=np.linspace(0, 1, 22),
                              cmap=cmap_energy, alpha=0.85, zorder=2)

        # 截至当前时刻的预测轨迹 (黑色虚线 + 散点, 在亮黄底色上更醒目)
        sub = pred6h_df[pred6h_df["datetime"] <= row["datetime"]]
        xt, yt = m(sub["lon"].values, sub["lat"].values)
        ax.plot(xt, yt, "--", color="black", linewidth=1.6,
                alpha=0.95, zorder=5)
        ax.scatter(xt, yt, s=22, color="white", edgecolors="black",
                   linewidths=0.9, zorder=6)

        # 每个子图都画经纬度网格与刻度标签
        m.drawmeridians(np.arange(int(ext[0]), int(ext[1]) + 1, 5),
                        labels=[0, 0, 0, 1], linewidth=0.3,
                        color="gray", dashes=[4, 4], alpha=0.4, fontsize=11)
        m.drawparallels(np.arange(int(ext[2]), int(ext[3]) + 1, 5),
                        labels=[1, 0, 0, 0], linewidth=0.3,
                        color="gray", dashes=[4, 4], alpha=0.4, fontsize=11)

        # 经纬度轴标签
        ax.set_xlabel(lon_lbl, fontsize=11.5, labelpad=18)
        ax.set_ylabel(lat_lbl, fontsize=11.5, labelpad=28)

        # 子图编号 + 时间 
        sub_id = f"({chr(ord('a') + i)})"
        t_str = row["datetime"].strftime("%m-%d %Hh")
        ax.text(0.5, -0.155, f"{sub_id} {t_str}",
                transform=ax.transAxes, fontsize=14.5, fontweight="bold",
                va="top", ha="center", color="black", zorder=10)

    fig.subplots_adjust(left=0.06, right=0.92, top=0.97, bottom=0.05,
                        hspace=0.42, wspace=0.22)
    cax = fig.add_axes([0.935, 0.08, 0.012, 0.84])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label(cbar_label, fontsize=15)
    cb.ax.tick_params(labelsize=11)

    fig.text(0.05, 0.005, f"--- {pred_lbl}", fontsize=12.5, ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="black", lw=0.6, alpha=0.9))

    suffix = "_EN" if lang == "en" else "_CN"
    out = OUT_DIR / f"AMPIL_HighQ_Energy_6h{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


draw_grid("cn")
draw_grid("en")
