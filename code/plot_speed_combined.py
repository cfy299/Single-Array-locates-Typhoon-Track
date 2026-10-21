# -*- coding: utf-8 -*-
"""
AMPIL 高质量段 6h —— 多速度轨迹合并小图 + 真实轨迹 / 多速度对比大图
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
# 时间标签参考轨迹 
REF_SPEED = 300

# 多速度小图的配色: 7 种高对比色, 与 SPEEDS 一一对应 (便于逐条区分)
SPEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2"]

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
            p = l.split(); t_id, t_name = p[1], p[7]; data = []; i += 1
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
    """给定 V_TY, 返回 6h 预测 DataFrame"""
    daily = []
    for d in GOOD_DAYS:
        if d not in cfy.index:
            continue
        baz = float(cfy.loc[d, "baz_deg"]); r2 = float(cfy.loc[d, "r2"])
        g = float(cfy.loc[d, "grad_per_km"])
        if r2 < R2_MIN or g <= 0:
            continue
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


# ── 统一视野: 真实轨迹 + 所有移速预测点 ──
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

# 统一视野
cx0 = float(np.mean(list(real_crop["lon"]) + list(pred_cache[300]["lon"])))
cy0 = float(np.mean(list(real_crop["lat"]) + list(pred_cache[300]["lat"])))
ext = [cx0 - 12, cx0 + 12, max(cy0 - 9, 0), cy0 + 9]
print(f"统一视野 ext = [{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.4f}, {ext[3]:.4f}]  "
      f"({ext[1]-ext[0]:.1f}° x {ext[3]-ext[2]:.1f}°)")
print(f"  真实轨迹 lon[{real_crop['lon'].min():.1f},{real_crop['lon'].max():.1f}] "
      f"lat[{real_crop['lat'].min():.1f},{real_crop['lat'].max():.1f}]")


def setup_map(ax):
    """按统一 ext 画底图."""
    m = Basemap(projection="cyl", llcrnrlon=ext[0], urcrnrlon=ext[1],
                llcrnrlat=ext[2], urcrnrlat=ext[3], resolution="i", ax=ax)
    m.drawmapboundary(fill_color="#E0F2F7")
    m.fillcontinents(color="#F5E6D3", lake_color="#E0F2F7", alpha=0.6)
    m.drawcoastlines(linewidth=1.0, color="black")
    m.drawcountries(linewidth=0.6, color="gray", linestyle=":")
    m.drawmeridians(np.arange(ext[0], ext[1] + 1, 4),
                    labels=[0, 0, 0, 1], linewidth=0.5, color="gray",
                    dashes=[4, 4], alpha=0.4, fontsize=10)
    m.drawparallels(np.arange(ext[2], ext[3] + 1, 4),
                    labels=[1, 0, 0, 0], linewidth=0.5, color="gray",
                    dashes=[4, 4], alpha=0.4, fontsize=10)
    return m


def draw_real(ax, m, lang, fs=9, s=110, lw=2.5):
    """真实 6h 轨迹 (灰线 + 风速 jet 散点 + 时间标签)."""
    x, y = m(real_crop["lon"].values, real_crop["lat"].values)
    ax.plot(x, y, "-", color="gray", linewidth=lw, alpha=0.5, zorder=4)
    sc = ax.scatter(x, y, c=real_crop["wind"].values, cmap="jet", s=s,
                    edgecolors="black", linewidths=0.7, zorder=5, vmin=15, vmax=35)
    for i in range(0, len(real_crop), 4):
        row = real_crop.iloc[i]
        lx, ly = m(float(row["lon"]), float(row["lat"]) + 0.5)
        ax.text(lx, ly, row["datetime"].strftime("%m/%d"), fontsize=fs,
                ha="center", va="bottom", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          alpha=0.85, ec="black", lw=0.5))
    return sc


def draw_speeds(ax, m, lang, fs=9, s=110, lw=2.5):
    """多速度预测轨迹: 每条一个高对比颜色, 返回图例句柄."""
    handles = []
    for v, col in zip(SPEEDS, SPEED_COLORS):
        pdf = pred_cache[v]
        x, y = m(pdf["lon"].values, pdf["lat"].values)
        h, = ax.plot(x, y, "-", color=col, linewidth=lw, alpha=0.95,
                     zorder=5, label=f"{v} km/day")
        # 终点: 带黑边的圆点, 便于辨认每条线末端
        ax.scatter([x[-1]], [y[-1]], s=s * 0.6, color=col,
                   edgecolors="black", linewidths=0.9, zorder=8)
        handles.append(h)
    # 时间标签只沿参考移速画一条, 避免 7 条重复
    ref = pred_cache[REF_SPEED]
    for i in range(0, len(ref), 4):
        row = ref.iloc[i]
        lx, ly = m(float(row["lon"]), float(row["lat"]) + 0.5)
        ax.text(lx, ly, row["datetime"].strftime("%m/%d"), fontsize=fs,
                ha="center", va="bottom", fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          alpha=0.85, ec="black", lw=0.5))
    return handles


def draw_pred_wind(ax, m, s=110):
    """7 条预测轨迹的 6h 点按真实风速着色 (与原变体图一致); 返回色标 mappable."""
    sc = None
    for v in SPEEDS:
        pdf = pred_cache[v]
        x, y = m(pdf["lon"].values, pdf["lat"].values)
        sc = ax.scatter(x, y, c=real_crop["wind"].values, cmap="jet",
                        s=s * 0.32, edgecolors="black", linewidths=0.45,
                        zorder=6, vmin=15, vmax=35)
    return sc


# ───── 图 1: 多速度合并小图 ─────
def draw_combined(lang):
    fig, ax = plt.subplots(figsize=(11, 9))
    m = setup_map(ax)
    handles = draw_speeds(ax, m, lang, fs=9, s=110, lw=2.0)
    leg_title = "移速 V_TY" if lang == "cn" else "Speed V_TY"
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.92,
              title=leg_title, title_fontsize=10)
    plt.tight_layout()
    suffix = "_EN" if lang == "en" else "_CN"
    out = OUT_DIR / f"AMPIL_HighQ_SpeedVariants{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ───── 图 2: 真实轨迹 vs 多速度 1x2 大图 (两子图共用 ext) ─────
def draw_pair(lang):
    # 显式坐标轴: 两个地图子图矩形完全相同; 各自带一条独立的风速色标
    fig = plt.figure(figsize=(23, 9))
    MW, MH = 0.393, 0.753            # 宽高比 = 24°/18° = 1.333, 与地图范围一致
    ax_l = fig.add_axes([0.022, 0.135, MW, MH])
    ax_r = fig.add_axes([0.022 + MW + 0.082, 0.135, MW, MH])

    # 左: 真实轨迹 (风速着色)
    m_l = setup_map(ax_l)
    sc_l = draw_real(ax_l, m_l, lang, fs=9, s=110, lw=2.5)

    # 右: 7 条移速轨迹 (点按真实风速着色, 不画真实轨迹)
    m_r = setup_map(ax_r)
    sc_r = draw_pred_wind(ax_r, m_r, s=110)
    handles = draw_speeds(ax_r, m_r, lang, fs=9, s=110, lw=2.0)
    leg_title = "移速 V_TY" if lang == "cn" else "Speed V_TY"
    ax_r.legend(handles=handles, loc="lower left", fontsize=8.5,
                framealpha=0.92, title=leg_title, title_fontsize=9.5)

    # 风速色标: 每个子图各一条 (独立窄轴, 不挤占地图)
    cbar_lab = "风速 (m/s)" if lang == "cn" else "Wind speed (m/s)"
    cax_l = fig.add_axes([0.022 + MW + 0.014, 0.28, 0.010, 0.44])
    fig.colorbar(sc_l, cax=cax_l).set_label(cbar_lab)
    cax_r = fig.add_axes([0.022 + 2*MW + 0.082 + 0.014, 0.28, 0.010, 0.44])
    fig.colorbar(sc_r, cax=cax_r).set_label(cbar_lab)

    suffix = "_EN" if lang == "en" else "_CN"
    out = OUT_DIR / f"AMPIL_HighQ_Real_vs_Speed{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


for lang in ("cn", "en"):
    draw_combined(lang)
    draw_pair(lang)
