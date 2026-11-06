#!/usr/bin/env python3
"""Request-free accounting of HPA allocation episodes, not a queue/SLO simulator."""
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/autoscale_theory2/FleetCostTheory/numba')
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from numba import njit
P=Path('/tmp/autoscale_theory2/FleetCostTheory')
ROOT=Path('/home/hongtao/hetero-wt/CostModel/results/autoscale-cost/d9c83f3c')
@njit(cache=True)
def calc(rate,mask,u,R):
    n=1; active=1; front=0; back=0; acq=0; cancel=0
    events=np.zeros(1000000,np.int64)
    sums=np.zeros(6,np.float64)
    metric=0.; rec=1;acq_in=0
    for t in range(1,len(rate)):
        while front<back and events[front]<=t:
            active+=1;front+=1
        if t%10==0: metric=rate[t]
        if t%15==0:
            raw=metric/(u*.7)
            if abs(raw/max(n,1)-1)<=.1:rec=n
            else:rec=max(1,int(np.ceil(raw-1e-9)))
            rec=min(rec,160)
            want=min(rec,max(2*n,n+4)) if rec>n else rec
            if want>n:
                for _ in range(want-n):
                    events[back]=t+R;back+=1;acq+=1
                    if mask[t]:acq_in+=1
            elif want<n:
                rem=n-want
                cut=min(rem,back-front)
                back-=cut;cancel+=cut;active-=rem-cut
            n=want
        q0=max(1,min(160,int(np.ceil(rate[t]/.7-1e-9))))
        if mask[t]:
            sums[0]+=n;sums[1]+=n-active;sums[2]+=min(active,q0)
            sums[3]+=max(0,min(active,rec)-q0);sums[4]+=max(0,active-max(rec,q0))
    return sums*8/3600/63,acq/67,acq_in/63,cancel/max(acq,1)

z=pq.read_table('/home/hongtao/.cache/pm-render/traces/chutes_gemma3_27b_d300plus.parquet',columns=['s','it','ot'])
s,it,ot=[z[k].to_numpy(zero_copy_only=False) for k in ['s','it','ot']]
s=np.sort(s[np.isfinite(it)&np.isfinite(ot)&(ot>=1)],kind='stable')-300*86400
sel=np.random.Generator(np.random.PCG64(20261025)).random(len(s))<.35
mask=np.ones(67,dtype=bool);mask[[3,4,5,66]]=False;mask=np.repeat(mask,86400)
ans={}
for label,ts in [('x0.35',s[sel]),('raw',s)]:
    cnt=np.bincount(ts.astype(np.int64),minlength=len(mask))[:len(mask)]
    cum=np.r_[0,np.cumsum(cnt)];t=np.arange(len(mask))
    rate=(cum[t]-cum[np.maximum(0,t-60)])/60
    pair=json.loads((ROOT/f'analysis-gemma-{label}.json').read_text())['pairs']['standby']['hpa-fast@90.0']
    ans[label]={}
    for who,R in [('pm',63),('stock',377)]:
        sim=pair[who];u=float(sim['cfg'].split('=')[1])
        a,n,ni,cf=calc(rate,mask,u,R)
        ans[label][who]={'u':u,'R':R,'prediction':dict(zip(['total','boot','base','head','keep','unused'],a.tolist())),
           'acq_day_67':n,'acq_day_63':ni,'canceled_fraction':cf,'sim':sim}
    pm=ans[label]['pm']['prediction'];st=ans[label]['stock']['prediction']
    ans[label]['saving_pred']=st['total']-pm['total']
    ans[label]['saving_pct_pred']=100*(st['total']-pm['total'])/st['total']
    ans[label]['saving_sim']=pair['saving_chip_h_day']
    ans[label]['saving_pct_sim']=pair['saving_pct']
(P/'hpa_results.json').write_text(json.dumps(ans,indent=2))
print(json.dumps(ans,indent=2))
