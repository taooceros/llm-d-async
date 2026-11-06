#!/usr/bin/env python3
"""Offline burst theory checks. All source repositories are read-only."""
import json, gzip, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
LL=Path('/home/hongtao/hetero-wt/LlmdAutoscale/results/llmd-autoscale')
HA=Path('/home/hongtao/hetero-wt/AutoscaleBurst/results/autoscale-burst/4daec1af')
C=.7; MU=1.121; TAU=.7882183790206909-.162; DT=1.
def load(p):
    p=Path(p)
    if not p.exists(): p=Path(str(p)+'.gz')
    op=gzip.open if p.suffix=='.gz' else open
    with op(p,'rt') as f:
        return [json.loads(x) for x in f if x.strip()] if '.jsonl' in p.name else json.load(f)
def metric(s,prefix): return sum(float(v) for k,v in s.items() if k.startswith(prefix+'{') or k==prefix)
def recovery(t,q,start,min_t=None):
    high=False; low=None
    for ti,qi in zip(t,q):
        if ti<start:continue
        if qi>2: high=True; low=None
        elif high and (min_t is None or ti>=min_t):
            low=ti if low is None else low
            if ti-low>=10:return low-start
    return None

def collect():
    numbers=load(LL/'paper/numbers.json'); runs=[]
    for arm,a in numbers['arms'].items():
        for rid in a['runs']:
            p=LL/'4d97ec1a/runs'/rid;m=load(p/'metrics.json');cl=load(p/'loadgen/client.jsonl');ss=load(p/'loadgen/samples.jsonl')
            t0=float(np.median([r['send_at']-r['sched_t']-r.get('lateness_s',0) for r in cl]))
            ts=np.array([s['t']-t0 for s in ss]);q=np.array([metric(s['m'],'llm_d_epp_flow_control_queue_size') for s in ss]);n=np.array([metric(s['m'],'inference_pool_ready_pods') for s in ss])
            ups=m['scale_ups']; trigger=ups[0]['trigger_condition_t'] if ups else 0
            if m['trace']=='b' and ups:trigger=ups[0]['decision_t']
            r={'rid':rid,'arm':arm,'kind':'llmd','trace':load(p/'loadgen/trace.json'),'clients':cl,'t':ts,'q':q,'n':n,'trigger':trigger,'R':ups[0]['first_served_client_t']-trigger if ups else 0,'bw':m['burst_window_s'],'measA':m['burst']['slo_attainment_pct'],'measQ':m['queue_max'],'up':ups,'down':m['scale_downs'],'N':m['max_replicas']}
            r['measDrain']=recovery(ts,q,trigger,trigger+r['R'])
            # Prometheus independently checks ready-pod samples and queue maxima.
            for key,name in [('promN','ready_pods'),('promQ','epp_queue')]:
                z=load(p/f'prom/{name}.json')['data']['result'];r[key]=max([float(v) for s in z for t,v in s.get('values',[])] or [0])
            runs.append(r)
    summary=load(HA/'summary.json')
    for arm in ['pm-gcs','stock']:
        a=summary['arms']['a'][arm]
        for mr in a['per_run']:
            rid=mr['rid'];p=HA/'runs'/rid;rec=load(p/'run.json');t0=rec['client_start_at'];ss=load(p/'loadgen/router/samples.jsonl');ds=load(p/'loadgen/router/decisions.jsonl');tr=[d['t']-t0 for d in ds if d['kind']=='trigger'];
            ts=np.array([s['t']-t0 for s in ss]);q=np.array([s['queue_depth'] for s in ss]);n=np.array([s['active'] for s in ss]);r={'rid':rid,'arm':'hand/'+arm,'kind':'hand','trace':load(p/'loadgen/trace.json'),'clients':load(p/'loadgen/client.jsonl'),'t':ts,'q':q,'n':n,'trigger':tr[0],'R':mr['recovery_first_served_s'],'bw':[421.7142857142857,2811.4285714285716],'measA':mr['burst_slo_attainment_pct'],'measQ':mr['queue_max'],'measDrain':mr['queue_recovery_s'],'up':[],'down':[],'N':4}
            runs.append(r)
    return runs

def fluid(r,mu=MU,bin_s=10.,n_override=None):
    req=r['trace']['requests']; at=np.array([x['t'] for x in req]);end=math.ceil(max(at))+1000
    t=np.arange(end+1,dtype=float);edges=np.arange(0,end+bin_s,bin_s);counts=np.histogram(at,edges)[0];lam=np.repeat(counts/bin_s,int(bin_s))[:len(t)]
    if len(lam)<len(t):lam=np.pad(lam,(0,len(t)-len(lam)))
    n=np.interp(t,r['t'],r['n'],left=r['n'][0],right=r['n'][-1]);n=np.maximum(1,np.rint(n)) if n_override is None else n_override(t)
    # Reflected cumulative net input, NOT the non-draining integral of positive excess.
    net=np.r_[0,np.cumsum((lam[:-1]-mu*n[:-1])*DT)];q=net-np.minimum.accumulate(net)
    service=np.r_[0,np.cumsum(mu*n[:-1]*DT)]
    # Virtual wait: future capacity must clear the queue ahead (FIFO fluid).
    finish=np.interp(service+q,service,t,right=t[-1]);w=finish-t
    burst=(at>=r['bw'][0])&(at<r['bw'][1]);wait=np.interp(at,t,w);miss=float(np.mean(wait[burst]>TAU))
    ceiling=.946 if r['kind']=='llmd' else 1.
    predA=100*ceiling*(1-miss)
    idx=(t>=r['bw'][0])&(t<=r['bw'][1]);qt=float(np.max(q[idx])); drain=recovery(t,q,r['trigger'],r['trigger']+r['R'])
    return {'predA':predA,'predQ':qt,'predDrain':drain,'queueMiss':miss,'t':t,'q':q,'n':n,'lam':lam,'w':w}

def summary(runs):
    rows=[]
    for arm in dict.fromkeys(r['arm'] for r in runs):
        rs=[r for r in runs if r['arm']==arm];row={'arm':arm,'n':len(rs)}
        for k in ['R','measA','measQ','measDrain','predA','predQ','predDrain','predA_C','predQ_C']:
            vals=[r[k] for r in rs if r.get(k) is not None];row[k]=float(np.median(vals)) if vals else None
        for name in ['A','Q','Drain']:
            row['error'+name]=row['pred'+name]-row['meas'+name] if row.get('pred'+name) is not None and row.get('meas'+name) is not None else None
        rows.append(row)
    return rows

def main():
    runs=collect()
    for r in runs:
        f=fluid(r);r.update({k:v for k,v in f.items() if not isinstance(v,np.ndarray)})
        fc=fluid(r,C);r['predA_C']=fc['predA'];r['predQ_C']=fc['predQ'];r['series']=f
        b=[x for x in r['clients'] if r['bw'][0]<=x['sched_t']<r['bw'][1]]
        qm=np.mean([(x.get('queue_s') or 0)>TAU or x.get('status')!=200 for x in b]);r['queueDecompA']=100*(.946 if r['kind']=='llmd' else 1.)*(1-qm)
        r['actualQueueMiss']=float(qm)
    rows=summary(runs)
    probe=load('/home/hongtao/hetero-wt/Autoscale/results/gke-autoscale/campaign-5962c75/samples.jsonl')
    pd={}
    for arm in sorted({s['arm'] for s in probe}):
        ss=[s for s in probe if s['arm']==arm];pd[arm]={'n':len(ss),'derived_examples':ss[0].get('derived')}
    # Smooth comparative figure; per-run dots retain measurement spread.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
    fig,ax=plt.subplots(figsize=(7.0,4.1))
    for kind,mark in [('llmd','o'),('hand','s')]:
        rr=[r for r in runs if r['kind']==kind and r['R']>0]
        ax.scatter([r['measA'] for r in rr],[r['predA'] for r in rr],marker=mark,label=kind+'; calibrated fluid',alpha=.8)
    ax.plot([25,100],[25,100],'k--',lw=1);ax.set(xlabel='Measured burst SLO attainment (%)',ylabel='Predicted attainment (%)',xlim=(25,100),ylim=(25,100));ax.legend(frameon=False);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'attainment_check.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9.0,5.5),sharex=True)
    for ax,rid in zip(axes.flat,['a-s-pm-r1','a-s-stock-r1','a-pm-gcs-r1','a-stock-r1']):
        r=next(r for r in runs if r['rid']==rid);f=r['series'];ax.plot(r['t'],r['q'],label='Measured queue',lw=1.2,color='#222222');ax.plot(f['t'],f['q'],label='Fluid, μ = 1.121',lw=1.2,color='#2677b5');ax.axvline(r['trigger']+r['R'],ls=':',color='#cb5735',label='First new replica');ax.set_title(rid);ax.set_xlim(400,1250);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8);axes[1,0].set_xlabel('Time since replay start (s)');axes[1,1].set_xlabel('Time since replay start (s)');axes[0,0].set_ylabel('Waiting requests');axes[1,0].set_ylabel('Waiting requests');fig.tight_layout();fig.savefig(OUT/'backlog_check.png');plt.close(fig)
    payload={'parameters':{'C':C,'mu':MU,'tau_queue':TAU,'rate_bin_s':10},'arms':rows,'runs':[{k:v for k,v in r.items() if k not in ['trace','clients','t','q','n','series']} for r in runs],'probe':pd}
    (OUT/'results.json').write_text(json.dumps(payload,indent=2,default=float))
    for row in rows: print(json.dumps(row))
if __name__=='__main__':main()
