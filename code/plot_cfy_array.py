# -*- coding: utf-8 -*-
"""
CFY 台阵布局图
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFY_stations = {
    'AH.HBE': [116.0, 31.8], 'HE.HNS': [113.6, 34.7], 'HE.LUX': [112.5, 34.5],
    'JS.GAY': [119.4, 32.3], 'JS.PX': [118.0, 34.7], 'JS.PZ': [117.2, 34.2],
    'JS.XIY': [118.8, 32.1], 'LN.DL2': [121.6, 38.9], 'LN.GUS': [124.8, 41.7],
    'LN.HSH': [125.3, 41.3], 'LN.SHS': [123.5, 41.8], 'LN.SUZ': [124.9, 40.0],
    'LN.WFD': [122.0, 42.0], 'SD.ANQ': [119.2, 36.1], 'SD.BHC': [117.5, 37.4],
    'SD.BSH': [117.0, 36.8], 'SD.CHAL': [119.2, 36.7], 'SD.CHAY': [119.3, 36.3],
    'SD.CHD': [116.3, 35.4], 'SD.CHQ': [118.3, 36.7], 'SD.CSH': [117.8, 36.2],
    'SD.DSD': [117.5, 36.0], 'SD.DSH': [116.7, 35.6], 'SD.DYG': [118.6, 37.4],
    'SD.JCZ': [116.3, 36.4], 'SD.JIN': [118.1, 35.4], 'SD.LAY': [117.8, 36.9],
    'SD.LIS': [119.5, 37.5], 'SD.QUF': [116.6, 35.6], 'SD.RCH': [117.0, 37.3],
    'SD.RSH': [117.8, 35.0], 'SD.TAT': [117.1, 36.2], 'SD.WSH': [119.2, 37.2],
    'SD.YTA': [121.4, 37.5], 'SD.ZHY': [118.0, 36.8],
}

# 按网络分组着色
NETWORK_COLOR = {
    "SD": "#D62728",   # 山东 - 红
    "JS": "#1F77B4",   # 江苏 - 蓝
    "LN": "#2CA02C",   # 辽宁 - 绿
    "HE": "#FF7F0E",   # 河北/河南 - 橙
    "AH": "#9467BD",   # 安徽 - 紫
}

CENTER_LON, CENTER_LAT = 118.0, 36.0  # CFY 阵列中心

plt.rcParams["font.family"] = ["Palatino Linotype", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def draw(lang: str):
    if lang == "en":
        net_label = {"SD": "SD (Shandong)", "JS": "JS (Jiangsu)",
                     "LN": "LN (Liaoning)", "HE": "HE (Hebei/Henan)",
                     "AH": "AH (Anhui)"}
    else:
        net_label = {"SD": "SD (山东)", "JS": "JS (江苏)",
                     "LN": "LN (辽宁)", "HE": "HE (河北/河南)",
                     "AH": "AH (安徽)"}

    fig, ax = plt.subplots(figsize=(11, 10))

    lons = [v[0] for v in CFY_stations.values()]
    lats = [v[1] for v in CFY_stations.values()]
    mg = 1.8
    ext = [min(lons)-mg, max(lons)+mg, min(lats)-mg, max(lats)+mg]

    m = Basemap(projection="cyl",
                llcrnrlon=ext[0], urcrnrlon=ext[1],
                llcrnrlat=ext[2], urcrnrlat=ext[3],
                resolution="i", ax=ax)
    m.drawmapboundary(fill_color="#E0F2F7")
    m.fillcontinents(color="#F5E6D3", lake_color="#E0F2F7", alpha=0.7)
    m.drawcoastlines(linewidth=1.0, color="black")
    m.drawcountries(linewidth=0.8, color="gray")
    m.drawmeridians(np.arange(int(ext[0]), int(ext[1])+1, 2),
                    labels=[0,0,0,1], linewidth=0.4, color="gray",
                    dashes=[4,4], alpha=0.5, fontsize=14)
    m.drawparallels(np.arange(int(ext[2]), int(ext[3])+1, 2),
                    labels=[1,0,0,0], linewidth=0.4, color="gray",
                    dashes=[4,4], alpha=0.5, fontsize=14)

    by_net = {}
    for name, (lo, la) in CFY_stations.items():
        net = name.split(".")[0]
        by_net.setdefault(net, []).append((lo, la, name))

    for net, pts in by_net.items():
        color = NETWORK_COLOR.get(net, "#7F7F7F")
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x, y = m(xs, ys)
        ax.scatter(x, y, s=130, c=color, marker="^",
                   edgecolors="black", linewidths=0.9,
                   zorder=5, label=net_label.get(net, net))

    for name, (lo, la) in CFY_stations.items():
        x, y = m(lo, la + 0.10)
        short = name.split(".")[1]
        ax.text(x, y, short, fontsize=10, ha="center", va="bottom",
                color="black", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12",
                          fc="white", ec="none", alpha=0.55))

    ax.legend(loc="lower right", fontsize=13, framealpha=0.92,
              title=("Networks" if lang=="en" else "台网"),
              title_fontsize=14)

    plt.tight_layout()
    suffix = "_EN" if lang == "en" else "_CN"
    out = OUT_DIR / f"CFY_array_layout{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


draw("cn")
draw("en")
