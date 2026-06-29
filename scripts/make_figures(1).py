#!/usr/bin/env python3
"""
Generate the three figures for the Koide note from the sweep output.
Run koide_sweep.py first (it writes sweep_data.npz). Then:
    python make_figures.py
Produces: fig1_sphere.pdf, fig2_hist.pdf, fig3_null.pdf
Requires matplotlib (plus the NumPy used by the sweep).
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"figure.dpi":150})

NULL_SEED = 20240628   # same measure as koide_sweep geometric null

# ---------- FIG 1: spherical triangle S^2_+/S_3 with Koide band ----------
def fig1():
    n=np.array([1,1,1])/np.sqrt(3); u=np.array([1,-1,0])/np.sqrt(2); w=np.cross(n,u)
    proj=lambda P: np.stack([P@u,P@w],1)
    rng=np.random.default_rng(1)
    P=np.abs(rng.normal(size=(120000,3))); P/=np.linalg.norm(P,axis=1,keepdims=True)
    XY=proj(P); e2=P[:,0]*P[:,1]+P[:,0]*P[:,2]+P[:,1]*P[:,2]
    fig,ax=plt.subplots(figsize=(4.8,4.4))
    def arc(a,b,m=200):
        ts=np.linspace(0,1,m)[:,None]; pts=a*(1-ts)+b*ts; pts/=np.linalg.norm(pts,axis=1,keepdims=True)
        return proj(pts)
    for i,j in [(0,1),(1,2),(0,2)]:
        A=arc(np.eye(3)[i],np.eye(3)[j]); ax.plot(A[:,0],A[:,1],"k",lw=1.3)
    band=(e2>0.24)&(e2<0.26)
    ax.scatter(XY[~band,0],XY[~band,1],s=0.5,c="#d2dae4",alpha=0.30,rasterized=True)
    ax.scatter(XY[band,0],XY[band,1],s=1.1,c="#d1495b",alpha=0.55,rasterized=True,label=r"Koide band $e_2\!\approx\!1/4$")
    Vx=proj(np.eye(3)); dem=proj(n[None,:])[0]
    ax.plot(dem[0],dem[1],"o",ms=8,mfc="#2e7d32",mec="k",label=r"democratic ($e_2{=}1$)")
    ax.plot(Vx[:,0],Vx[:,1],"s",ms=7,mfc="#1f4e79",mec="k",label=r"vertices ($e_2{=}0$)")
    xph=np.sqrt(np.array([0.51100,105.658,1776.86])); xph/=np.linalg.norm(xph)
    pph=proj(xph[None,:])[0]
    ax.plot(pph[0],pph[1],"*",ms=16,mfc="gold",mec="k",label="physical leptons",zorder=5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="upper center",fontsize=7,frameon=False,ncol=2,bbox_to_anchor=(0.5,1.13))
    plt.tight_layout(); plt.savefig("fig1_sphere.pdf",bbox_inches="tight"); plt.close()

# ---------- FIG 2 & 3 from sweep_data.npz ----------
def fig23():
    d=np.load("sweep_data.npz"); e2_g,e2_u=d["e2_gauss"],d["e2_unif"]
    if "e2_null" in d:                       # use the SAME null as the sweep tables
        e2_null=d["e2_null"]
    else:                                    # fallback: recompute with the sweep's seed
        rng=np.random.default_rng(NULL_SEED)
        P=np.abs(rng.normal(size=(2_000_000,3))); P/=np.linalg.norm(P,axis=1,keepdims=True)
        e2_null=P[:,0]*P[:,1]+P[:,0]*P[:,2]+P[:,1]*P[:,2]
    # FIG 2
    fig,ax=plt.subplots(figsize=(5.2,3.2)); bins=np.linspace(0,1,80)
    ax.hist(e2_g,bins=bins,color="#4c72b0",alpha=0.75,label="Gaussian coeff.",density=True)
    ax.hist(e2_u,bins=bins,color="#dd8452",alpha=0.5,label="Uniform coeff.",density=True)
    ax.axvspan(0.22,0.28,color="#d1495b",alpha=0.18,label="Koide band")
    ax.axvline(0.25,color="#d1495b",lw=1.2,ls="--")
    ax.axvline(1.0,color="#2e7d32",lw=1); ax.axvline(0.0,color="#1f4e79",lw=1)
    ax.text(0.25,ax.get_ylim()[1]*0.92,r"$e_2{=}1/4$",color="#d1495b",ha="center",fontsize=8)
    ax.set_xlabel(r"attractor $e_2$"); ax.set_ylabel("density")
    ax.set_title("Distribution of attractors: peaks at democratic / vertices",fontsize=9)
    ax.legend(fontsize=7,frameon=False); plt.tight_layout(); plt.savefig("fig2_hist.pdf"); plt.close()
    # FIG 3
    wins=[(0.22,0.28),(0.23,0.27),(0.24,0.26),(0.245,0.255)]
    labels=["0.22-0.28","0.23-0.27","0.24-0.26","0.245-0.255"]
    freq=lambda a,wv:np.mean((a>wv[0])&(a<wv[1]))
    null=[freq(e2_null,wv) for wv in wins]; fg=[freq(e2_g,wv) for wv in wins]; fu=[freq(e2_u,wv) for wv in wins]
    xp=np.arange(len(wins)); ww=0.26
    fig,ax=plt.subplots(figsize=(5.4,3.2))
    ax.bar(xp-ww,[100*x for x in null],ww,color="#888",label="geometric null")
    ax.bar(xp,[100*x for x in fg],ww,color="#4c72b0",label="Gaussian coeff.")
    ax.bar(xp+ww,[100*x for x in fu],ww,color="#dd8452",label="Uniform coeff.")
    ax.set_xticks(xp); ax.set_xticklabels(labels,fontsize=8)
    ax.set_ylabel("frequency in band (%)"); ax.set_xlabel(r"Koide window in $e_2$")
    ax.set_title("Koide is under-populated vs geometric null (all windows)",fontsize=9)
    ax.legend(fontsize=7,frameon=False); plt.tight_layout(); plt.savefig("fig3_null.pdf"); plt.close()

if __name__=="__main__":
    fig1(); fig23(); print("wrote fig1_sphere.pdf, fig2_hist.pdf, fig3_null.pdf")
