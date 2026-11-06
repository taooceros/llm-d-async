#!/usr/bin/env python3
"""Offline closed-form fleet accounting; repository inputs are read-only."""
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/autoscale_theory2/FleetCostTheory/numba')
import json, re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import maximum_filter1d
from scipy.optimize import least_squares
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('/tmp/autoscale_theory2/FleetCostTheory')
ROOT = Path('/home/hongtao/hetero-wt/CostModel/results/autoscale-cost/d9c83f3c')
TRACE = Path('/home/hongtao/.cache/pm-render/traces/chutes_gemma3_27b_d300plus.parquet')
DAY=86400; ND=67; C=.7; c=8; DAYS=63
TGRID=np.array([0,30,60,120,180,300,450,600,900,1200,1800,2700,3600,5400,7200])
inc=np.ones(ND,dtype=bool); inc[[3,4,5,66]]=False
mask=np.repeat(inc,DAY)
ip=np.r_[0,np.cumsum(mask,dtype=np.int64)]
FAC=c/3600/DAYS

@njit(cache=True)
def intervals(q):
    # Demand episodes of each ordered replica tier; the permanent base is omitted.
    starts=np.full(int(q.max())+1,-1,np.int64)
    a=[]; b=[]; v=[]
    prev=1
    for t in range(len(q)):
        cur=q[t]
        if cur>prev:
            for k in range(prev+1,cur+1): starts[k]=t
        elif cur<prev:
            for k in range(cur+1,prev+1):
                a.append(starts[k]); b.append(t); v.append(k)
                starts[k]=-1
        prev=cur
    for k in range(2,len(starts)):
        if starts[k]>=0:
            a.append(starts[k]); b.append(len(q)); v.append(k)
    return np.array(a),np.array(b),np.array(v)

@njit(cache=True)
def gaps(q):
    low=np.full(int(q.max())+1,-1,np.int64)
    seen=np.zeros(len(low),np.bool_)
    a=[]; b=[]; v=[]
    prev=1
    for t in range(len(q)):
        cur=q[t]
        if cur>prev:
            for k in range(prev+1,cur+1):
                if low[k]>=0 and seen[k]:
                    a.append(low[k]); b.append(t); v.append(k)
                seen[k]=True; low[k]=-1
        elif cur<prev:
            for k in range(cur+1,prev+1): low[k]=t
        prev=cur
    for k in range(2,len(low)):
        if low[k]>=0 and seen[k]:
            a.append(low[k]); b.append(len(q)); v.append(k)
    return np.array(a),np.array(b),np.array(v)

def gi(g,T):
    a,b,_=g
    return float((ip[np.minimum(b,a+T)]-ip[a]).sum()*FAC)

def ns(g,s):
    a,b,_=g; loc=a+int(s)
    ok=(loc<b)&(loc<len(mask)); loc=loc[ok]
    return float(mask[loc].sum()/DAYS)

def nreturn(g,T):
    a,b,_=g; ok=(b<len(mask))&(b-a>=T)
    return float(mask[b[ok]].sum()/DAYS)

def account(q,q0,T,R):
    held=q if T<=1 else maximum_filter1d(q,size=T,origin=(T-1)//2,mode='nearest')
    a,b,v=intervals(held)
    d=np.zeros(len(q)+1,np.int16)
    ar=np.minimum(a+int(round(R)),b)
    np.add.at(d,ar,1); np.add.at(d,b,-1)
    ready=1+np.cumsum(d[:-1],dtype=np.int16)
    boot=held-ready
    def cost(x): return float(x[mask].sum()*FAC)
    ans={'total':cost(held),'base':cost(np.minimum(ready,q0)),
         'head':cost(np.maximum(np.minimum(ready,q)-q0,0)),
         'keep':cost(np.maximum(ready-q,0)),'boot':cost(boot),
         'up_per_included_day':float(mask[a].sum()/DAYS),
         'up_per_67day':float(len(a)/67),
         'canceled_fraction':float(np.mean(b-a<R)),
         'ready_replica_days':float(ready[mask].sum()/DAY/DAYS)}
    return ans

def parse(s):
    h,T=re.search(r'h=([\d.]+)C T=(\d+)s',s).groups()
    return float(h),int(T)

def fitlog(x,y,base):
    f=lambda p:base+p[0]*np.log1p(x/p[1])-y
    r=least_squares(f,[100,80],bounds=([0,.01],[10000,10000]))
    return r.x

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':200,'savefig.dpi':200})
tab=pq.read_table(TRACE,columns=['s','it','ot'])
s,it,ot=[tab[n].to_numpy(zero_copy_only=False) for n in ['s','it','ot']]
valid=np.isfinite(it)&np.isfinite(ot)&(ot>=1)
s=np.sort(s[valid],kind='stable')-300*DAY
thin=np.random.Generator(np.random.PCG64(20261025)).random(len(s))<.35
results={}
fig_gap,ax_gap=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
fig_cost,ax_cost=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
fig_head,ax_head=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for col,(label,ts) in enumerate([('x0.35',s[thin]),('raw',s)]):
    print('START',label,flush=True)
    pub=json.loads((ROOT/f'analysis-gemma-{label}.json').read_text())
    cnt=np.bincount(ts.astype(np.int64),minlength=len(mask))[:len(mask)]
    cum=np.r_[0,np.cumsum(cnt)]
    ticks=np.arange(len(mask))
    rate=(cum[ticks]-cum[np.maximum(0,ticks-60)])/60
    q0=np.maximum(1,np.ceil(rate/C-1e-9)).astype(np.int16)
    g0=gaps(q0)
    ss=np.unique(np.geomspace(1,7200,160).astype(int))
    n0=np.array([ns(g0,t) for t in ss])
    fit=(ss>=120)&(ss<=1800)
    slope,inter=np.polyfit(np.log(ss[fit]),np.log(n0[fit]),1)
    err=np.sqrt(np.mean((np.log(n0[fit])-np.polyval([slope,inter],np.log(ss[fit])))**2))
    rec={'requests_included':int(cnt[mask].sum()),'requests_full':len(ts),
         'N_fit_120_1800':{'alpha':-slope,'K':float(np.exp(inter)),'log_rmse':float(err)},
         'N0':{str(t):ns(g0,t) for t in [0,30,60,120,300,600,1800,3600,7200]},
         'bare_gap_saving_120_600':gi(g0,600)-gi(g0,120), 'policies':{},'rise':{}}
    allg={0:g0}; allq={0:q0}
    for target in [95,99]:
        pair=pub['pairs']['standby'][f'reactive@{target}.0']
        for who,R in [('pm',63),('stock',377)]:
            row=pair[who]; h,T=parse(row['cfg'])
            if h not in allq:
                allq[h]=np.maximum(1,np.ceil(rate/C+h-1e-9)).astype(np.int16)
                allg[h]=gaps(allq[h])
            q=allq[h]; g=allg[h]
            pred=account(q,q0,T,R)
            pred['ideal_keep']=gi(g,max(T-1,0))
            pred['gap_up_included_day']=nreturn(g,T)
            pred['gap_boot_uncanceled']=nreturn(g,T)*c*R/3600
            pred['ideal_head']=float((q-q0)[mask].sum()*FAC)
            pred['T']=T;pred['h_C']=h;pred['R']=R
            pred['sim']=row
            # Renewal objective J/c = idle seconds + K_seconds * reacquisitions.
            iv=np.array([gi(g,int(t))*3600/c for t in TGRID])
            nv=np.array([nreturn(g,int(t)) for t in TGRID])
            Klo=0.; Khi=np.inf; j=int(np.where(TGRID==T)[0][0])
            for k in range(len(TGRID)):
                if nv[k]>nv[j]: Klo=max(Klo,(iv[j]-iv[k])/(nv[k]-nv[j]))
                elif nv[k]<nv[j]: Khi=min(Khi,(iv[k]-iv[j])/(nv[j]-nv[k]))
            pred['renewal_K_interval_for_selected_T']=[float(Klo),float(Khi)]
            pred['money_only_T']=int(TGRID[np.argmin(iv+R*nv)])
            rec['policies'][f'{target}/{who}']=pred
    h95=rec['policies']['95/pm']['h_C'];g95=allg[h95]
    rec['policy_gap_saving_120_600']=gi(g95,599)-gi(g95,119)
    rec['gap_grid']={'T':TGRID.tolist(),'idle_chip_h_day':[gi(g95,int(t)) for t in TGRID],
                     'restart_day':[nreturn(g95,int(t)) for t in TGRID]}
    Bs=np.array([15,30,45,60,63,90,120,180,240,265,300,377,390,480,581,600,750,900,1800,3600])
    for B in Bs:
        future=maximum_filter1d(rate,size=int(B)+1,origin=-((int(B)+1)//2),mode='nearest')
        rise=np.maximum(future-rate,0)/C
        rec['rise'][str(B)]={f'q{p}':float(np.percentile(rise[mask],p)) for p in [90,95,99,99.9]}
    rec['rise_exponents']={}
    for p in [95,99]:
        y=np.array([rec['rise'][str(B)][f'q{p}'] for B in Bs])
        keep=(Bs>=60)&(Bs<=900)
        beta,lnA=np.polyfit(np.log(Bs[keep]),np.log(y[keep]),1)
        rec['rise_exponents'][str(p)]={'beta_60_900':float(beta),'A':float(np.exp(lnA)),
            'beta_63_377':float(np.log(rec['rise']['377'][f'q{p}']/rec['rise']['63'][f'q{p}'])/np.log(377/63))}
        ax_head[col].loglog(Bs,y,'o-',label=f'{p}th percentile, slope {beta:.2f}')
    # Gap-integral cost fit, calibrated only to B=0,60,390,900; remaining 11 points are held out.
    curve=pub['boot_curve']; bx=np.array([r['B'] for r in curve],float)
    cy=np.array([r['hull@95.0'] for r in curve]); base=cy[0]
    train=np.isin(bx,[0,60,390,900])
    # Horizontal policy multiplier a and vertical factor z; finite 60-s smoothing avoids singularity.
    # G(aB)=the exact empirical idle-gap integral; a absorbs the selected keep-alive/SLO tradeoff.
    gap_a,gap_b,_=g95
    # Interpolation on every second is exact for this integer-second trace.
    gs=np.arange(0,7201)
    gv=np.array([ns(g95,int(t)) for t in gs])
    intg=np.r_[0,np.cumsum(gv[:-1])]*c/3600
    def G(x):return np.interp(x,gs,intg)
    fitfun=lambda p:base+p[0]*G(p[1]*bx[train])-cy[train]
    opt=least_squares(fitfun,[1,1.5],bounds=([0,.01],[10,8]))
    z,a=opt.x
    pred=base+z*G(a*bx)
    lp=fitlog(bx[train],cy[train],base)
    logpred=base+lp[0]*np.log1p(bx/lp[1])
    rec['cost_fit']={'base':float(base),'gap_z':float(z),'gap_a':float(a),
        'train_B':[0,60,390,900],'heldout_rmse':float(np.sqrt(np.mean((pred[~train]-cy[~train])**2))),
        'all_max_abs':float(np.max(np.abs(pred-cy))),
        'log_K':float(lp[0]),'log_tau':float(lp[1]),
        'log_heldout_rmse':float(np.sqrt(np.mean((logpred[~train]-cy[~train])**2))),
        'rows':[{'R':float(b),'sim':float(y),'gap_pred':float(p),'error':float(p-y),'log_pred':float(l)} for b,y,p,l in zip(bx,cy,pred,logpred)],
        'scenarios':{}}
    for name in ['pm-standby','stock-standby','pm-new','stock-new']:
        row=pub['scenario_points'][name]; B=row['B_mean']
        pp=float(base+z*G(a*B)); yy=row['hull@95.0']
        rec['cost_fit']['scenarios'][name]={'R':B,'pred':pp,'sim':yy,'error':pp-yy}
    rec['recovered_share']={}
    for target in [95,99]:
        v0=curve[0][f'hull@{target}.0']
        vs=pub['scenario_points']['stock-standby'][f'hull@{target}.0']
        vp=pub['scenario_points']['pm-standby'][f'hull@{target}.0']
        rec['recovered_share'][str(target)]=(vs-vp)/(vs-v0)
    predsc=rec['cost_fit']['scenarios']
    rec['cost_fit']['predicted_recovered_share']=(predsc['stock-standby']['pred']-predsc['pm-standby']['pred'])/(predsc['stock-standby']['pred']-base)
    # HPA accounting from already-computed simulator rows; no claimed fluid attainment prediction.
    rec['hpa90']=pub['pairs']['standby']['hpa-fast@90.0']
    results[label]=rec
    ax_gap[col].loglog(ss,n0,lw=2,label=r'Demand $\lceil\lambda/C\rceil$')
    ax_gap[col].loglog(ss,[ns(g95,t) for t in ss],ls='--',label=f'95% policy (+{h95}C)')
    ax_gap[col].loglog(ss,np.exp(inter)*ss**slope,ls=':',color='black',label=f'Fit: $s^{{-{(-slope):.2f}}}$ (120–1800 s)')
    ax_gap[col].axvspan(120,600,color='#cccccc',alpha=.4)
    ax_gap[col].set(title=f'Chutes {label}',xlabel='Idle-gap threshold s (s)',ylabel='Gaps surviving per day N(s)')
    ax_gap[col].legend(fontsize=8);ax_gap[col].grid(alpha=.2,which='both')
    xx=np.linspace(0,900,600)
    ax_cost[col].plot(xx,base+z*G(a*xx),label='Gap-integral fit (4 anchors)',color='#2166ac',lw=2)
    ax_cost[col].scatter(bx,cy,label='CostModel 95% hull',color='#333333',s=23,zorder=3)
    for name,marker,color in [('pm-standby','o','#1b9e77'),('stock-standby','s','#d95f02'),('pm-new','^','#1b9e77'),('stock-new','D','#d95f02')]:
        r=predsc[name]
        ax_cost[col].scatter(r['R'],r['sim'],marker=marker,color=color,s=70,zorder=4,label=name.replace('-',' '))
    ax_cost[col].set(title=f'Chutes {label}',xlabel='Recovery R (s; simulator labels it B)',ylabel='Chip-hours / included day')
    ax_cost[col].grid(alpha=.2);ax_cost[col].legend(fontsize=8)
    ax_head[col].set(title=f'Chutes {label}',xlabel='Recovery R (s)',ylabel='Forward maximum rate rise (replicas)')
    ax_head[col].legend(fontsize=8);ax_head[col].grid(alpha=.2,which='both')
    print(label,'keep bare/policy/adjusted/sim',rec['bare_gap_saving_120_600'],rec['policy_gap_saving_120_600'],rec['policies']['95/stock']['keep']-rec['policies']['95/pm']['keep'],rec['policies']['95/stock']['sim']['keep']-rec['policies']['95/pm']['sim']['keep'],flush=True)
    print(label,'cost fit',rec['cost_fit'],flush=True)
(OUT/'results.json').write_text(json.dumps(results,indent=2))
fig_gap.savefig(OUT/'gap_spectrum.png');fig_cost.savefig(OUT/'cost_vs_boot.png');fig_head.savefig(OUT/'headroom_growth.png')
print('WROTE',OUT/'results.json',flush=True)
