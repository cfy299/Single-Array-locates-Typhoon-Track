# -*- coding: utf-8 -*-
"""
AMPIL 6h 对比图
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
EARTH_R = 6371.0; CFY_LON, CFY_LAT = 118.0, 36.0; K_CALIB = 7.8135
plt.rcParams["font.family"] = ["Palatino Linotype", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

def gc_dist(lo1,la1,lo2,la2):
    p1,p2=np.deg2rad(la1),np.deg2rad(la2); dl=np.deg2rad(np.array(lo2)-np.array(lo1))
    a=np.sin((p2-p1)/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*EARTH_R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def project(lo0,la0,baz,d):
    R=EARTH_R; d=d/R; b=np.deg2rad(baz)
    la0r,lo0r=np.deg2rad(la0),np.deg2rad(lo0)
    la2=np.arcsin(np.sin(la0r)*np.cos(d)+np.cos(la0r)*np.sin(d)*np.cos(b))
    lo2=lo0r+np.arctan2(np.sin(b)*np.sin(d)*np.cos(la0r),np.cos(d)-np.sin(la0r)*np.sin(la2))
    return float(np.rad2deg(lo2)), float(np.rad2deg(la2))

def parse_cma(fp):
    typhoons=[]
    with open(fp,"r",encoding="utf-8") as f: lines=f.readlines()
    i=0
    while i<len(lines):
        l=lines[i].strip()
        if l.startswith("66666"):
            p=l.split(); t_id,t_name=p[1],p[7]; data=[]; i+=1
            while i<len(lines) and not lines[i].startswith("66666"):
                pp=lines[i].split()
                if len(pp)>=6:
                    try:
                        dt=datetime.strptime(pp[0],"%Y%m%d%H")
                        data.append({"dt":dt,"lat":int(pp[2])/10,"lon":int(pp[3])/10,"wind":int(pp[5])})
                    except: pass
                i+=1
            if data: typhoons.append({"id":t_id,"name":t_name,"data":data})
        else: i+=1
    return typhoons

# ── 数据 ──
cfy_baz = pd.read_csv(BASE/"AmpGrad_baz.csv")
cfy_baz["date"] = pd.to_datetime(cfy_baz["date"])
cfy_baz = cfy_baz[cfy_baz["array"]=="CFY"].set_index("date")[["baz_deg","r2","grad_per_km"]]

typhoons = parse_cma(BASE/"2017、2018年台风路径数据"/"CH2018BST.txt")
ampil = next(t for t in typhoons if t["name"]=="AMPIL")

# 真实 6h 数据 (仅 07-18~07-22, 距CFY > 400km, 即还未过境)
real6h = sorted([p for p in ampil["data"]
                 if p["dt"].hour%6==0
                 and datetime(2018,7,18) <= p["dt"] <= datetime(2018,7,22,18)],
                key=lambda x: x["dt"])

real_df = pd.DataFrame([{"datetime":p["dt"],"lon":p["lon"],"lat":p["lat"],"wind":p["wind"]}
                         for p in real6h])

GOOD_DAYS = [pd.Timestamp("2018-07-18"), pd.Timestamp("2018-07-19"),
             pd.Timestamp("2018-07-20"), pd.Timestamp("2018-07-21")]
V_TY = 300.0  # 西北太平洋盛夏台风典型移速 (km/day), 物理先验

# 起点距离 d0
g0 = float(cfy_baz.loc[GOOD_DAYS[0], "grad_per_km"])
d_start = float(np.clip(K_CALIB/g0, 500, 4000))

daily_pred = []
for d in GOOD_DAYS:
    if d not in cfy_baz.index: continue
    baz = float(cfy_baz.loc[d,"baz_deg"]); r2 = float(cfy_baz.loc[d,"r2"])
    g   = float(cfy_baz.loc[d,"grad_per_km"])
    if r2 < 0.15 or g <= 0: continue
    # 距离: 起点 d0 - V_TY × 天数
    days_from_start = (d - GOOD_DAYS[0]).days
    dist = max(d_start - V_TY*days_from_start, 400)
    # 方向: 用当日 baz (保留方位信号)
    lo, la = project(CFY_LON, CFY_LAT, baz, dist)
    daily_pred.append({"date":d, "lon":lo, "lat":la, "dist":dist})

daily_pred_df = pd.DataFrame(daily_pred)
print(f"起点距离 d0={d_start:.0f}km, 物理速度 V_TY={V_TY:.0f}km/day")
print("日级预测点:")
for r in daily_pred:
    print(f"  {r['date'].date()} dist={r['dist']:.0f}km -> ({r['lon']:.1f}, {r['lat']:.1f})")

# 插值到 6h
t_src = (daily_pred_df["date"] - daily_pred_df["date"].min()).dt.total_seconds().values
lo_src = daily_pred_df["lon"].values; la_src = daily_pred_df["lat"].values

pred6h = []
for p in real6h:
    if p["dt"] > datetime(2018,7,21,23): continue 
    t_s = (pd.Timestamp(p["dt"]) - daily_pred_df["date"].min()).total_seconds()
    lo = float(np.interp(t_s, t_src, lo_src))
    la = float(np.interp(t_s, t_src, la_src))
    pred6h.append({"datetime":p["dt"],"lon":lo,"lat":la,"wind":p["wind"]})

pred6h_df = pd.DataFrame(pred6h)
real_crop = real_df[real_df["datetime"] <= datetime(2018,7,21,23)].reset_index(drop=True)

errs = gc_dist(pred6h_df["lon"].values, pred6h_df["lat"].values,
               real_crop["lon"].values, real_crop["lat"].values)
print(f"07-18~07-21 6h 误差: 均值={errs.mean():.0f} 中位={np.median(errs):.0f} 最小={errs.min():.0f} km")
print(f"预测方向: 从({pred6h_df.iloc[0]['lon']:.1f},{pred6h_df.iloc[0]['lat']:.1f}) 到 "
      f"({pred6h_df.iloc[-1]['lon']:.1f},{pred6h_df.iloc[-1]['lat']:.1f})")
print(f"真实方向: 从({real_crop.iloc[0]['lon']:.1f},{real_crop.iloc[0]['lat']:.1f}) 到 "
      f"({real_crop.iloc[-1]['lon']:.1f},{real_crop.iloc[-1]['lat']:.1f})")

# ── 绘图 ──
all_lo = list(real_crop["lon"])+list(pred6h_df["lon"])
all_la = list(real_crop["lat"])+list(pred6h_df["lat"])
cx,cy = np.mean(all_lo),np.mean(all_la); mg=12
ext = [cx-mg, cx+mg, max(cy-9,0), cy+9]

def draw_single(src, kind, lang):
    cbar_label = "Wind speed (m/s)" if lang=="en" else "风速 (m/s)"
    fig, ax = plt.subplots(figsize=(11, 9))
    m=Basemap(projection="cyl",llcrnrlon=ext[0],urcrnrlon=ext[1],
              llcrnrlat=ext[2],urcrnrlat=ext[3],resolution="i",ax=ax)
    m.drawmapboundary(fill_color="#E0F2F7"); m.fillcontinents(color="#F5E6D3",lake_color="#E0F2F7",alpha=0.6)
    m.drawcoastlines(linewidth=1.0,color="black"); m.drawcountries(linewidth=0.6,color="gray",linestyle=":")
    m.drawmeridians(np.arange(ext[0],ext[1]+1,4),labels=[0,0,0,1],linewidth=0.5,color="gray",dashes=[4,4],alpha=0.4,fontsize=10)
    m.drawparallels(np.arange(ext[2],ext[3]+1,4),labels=[1,0,0,0],linewidth=0.5,color="gray",dashes=[4,4],alpha=0.4,fontsize=10)
    x,y=m(src["lon"].values,src["lat"].values)
    ax.plot(x,y,"-",color="gray",linewidth=2.5,alpha=0.5,zorder=4)
    sc=ax.scatter(x,y,c=real_crop["wind"].values,cmap="jet",s=110,
                  edgecolors="black",linewidths=0.7,zorder=5,vmin=15,vmax=35)
    step=4
    for i in range(0,len(src),step):
        row=src.iloc[i]
        lx,ly=m(float(row["lon"]),float(row["lat"])+0.5)
        ax.text(lx,ly,row["datetime"].strftime("%m/%d"),fontsize=9,ha="center",va="bottom",fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",fc="white",alpha=0.85,ec="black",lw=0.5),zorder=6)
    fig.colorbar(sc,ax=ax,pad=0.02,shrink=0.7,label=cbar_label)
    plt.tight_layout()
    suffix = "_EN" if lang=="en" else "_CN"
    out = OUT_DIR/f"AMPIL_HighQ_Fixed_6h_{kind}{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"[OK] {out}")

for lang in ("cn", "en"):
    draw_single(real_crop, "Real", lang)
    draw_single(pred6h_df, "Pred", lang)
