#!/usr/bin/env python3
"""Deterministic checks and model tables; Python 3 + numpy, no scipy."""
import json, math
from pathlib import Path
from statistics import NormalDist
import numpy as np

ROOT = Path(__file__).resolve().parent
BOOTS = [779.4, 284.9, 50.2, 350.0, 385.0, 419.0, 18.8, 27.5]
C = 16

def sinusoid_overhead(B, P, a=0.5):
    b = min(B, P)
    return a * (b / P + math.sin(math.pi*b/P) / math.pi)

def numeric_sinusoid(B, P, a=0.5, n=32768, k=1024):
    # Independently sample demand over each forward window, then integrate.
    t = (np.arange(n)+0.5)*P/n
    maximum = np.full(n, -np.inf)
    for u in np.linspace(0, min(B,P), k+1):
        maximum = np.maximum(maximum, 1+a*np.cos(2*np.pi*(t+u)/P))
    return float(np.mean(maximum)-1)

def main():
    rng = np.random.default_rng(20261020)
    checks = []
    def record(name, simulated, closed, tolerance):
        error = abs(simulated-closed)
        checks.append(dict(name=name, simulation=float(simulated), closed=float(closed), absolute_error=float(error), tolerance=tolerance, passed=bool(error<=tolerance)))
        assert error<=tolerance, checks[-1]

    # Layer demand: [0,10) has 2 instances, [10,20) has 1,
    # [20,25) has 3, [25,40) zero, [40,50) has 1. B=4.
    # Merge left-expanded intervals independently at each capacity level.
    intervals = [[(0,25),(40,50)],[(0,10),(20,25)],[(20,25)]]
    B=4.0
    exact=0.0
    for layer in intervals:
        end=-math.inf
        for start,stop in layer:
            left=start-B
            exact += max(0,stop-max(left,end))
            end=max(end,stop)
    dt=0.001
    t=np.arange(-B,50,dt)+dt/2
    envelope=np.zeros_like(t)
    for start,stop,value in [(0,10,2),(10,20,1),(20,25,3),(40,50,1)]:
        envelope=np.maximum(envelope, value*((t>=start-B)&(t<stop)))
    record('finite envelope: instance-seconds',float(envelope.sum()*dt),exact,1e-8)
    # Linear ramp d(t)=2+0.03t, all t through T+B; no boundary startup.
    T=100.; B=50.2; rate=0.03
    t=(np.arange(100000)+0.5)*T/100000
    delta=np.max(np.array([2+rate*(t+u) for u in np.linspace(0,B,20)]),axis=0)-(2+rate*t)
    record('linear ramp overhead: instance-seconds',float(delta.mean()*T),rate*B*T,1e-8)
    for B in BOOTS:
        record(f'diurnal fractional overhead B={B}',numeric_sinusoid(B,86400),sinusoid_overhead(B,86400),2e-8)
    for B in [50.2,284.9,779.4]:
        record(f'600s periodic fractional overhead B={B}',numeric_sinusoid(B,600,k=4096),sinusoid_overhead(B,600),1e-7)

    # Exact conditional bridge maxima eliminate time-grid missed crossings.
    n=150000; steps=128; dt=1/steps
    x=np.zeros(n); peak=np.zeros(n)
    for _ in range(steps):
        y=x+rng.normal(0,math.sqrt(dt),n)
        bridge=(x+y+np.sqrt((y-x)**2-2*dt*np.log(rng.random(n))))/2
        peak=np.maximum(peak,bridge); x=y
    eps=.01; z=NormalDist().inv_cdf(1-eps/2)
    record('Brownian maximum 99% quantile',float(np.quantile(peak,.99)),z,.035)
    record('Brownian path violation probability',float(np.mean(peak>z)),eps,.0012)
    record('Brownian terminal violation probability',float(np.mean(x>NormalDist().inv_cdf(.99))),eps,.0012)
    for B in BOOTS:
        # sigma=0.1 instances/sqrt(second); same normalized paths for paired scaling.
        h=.1*math.sqrt(B)*z
        record(f'diffusion headroom B={B}',float(np.quantile(.1*math.sqrt(B)*peak,.99)),h,.0035*math.sqrt(B))

    # Idle gaps: clairvoyant optimum versus timeout-B reactive policy.
    gaps=rng.exponential(300,1000000)
    for B in BOOTS:
        oracle=np.minimum(gaps,B)
        online=oracle+B*(gaps>B)
        oracle_mean=300*(-math.expm1(-B/300))
        online_mean=oracle_mean+B*math.exp(-B/300)
        record(f'oracle idle mean seconds B={B}',float(oracle.mean()),oracle_mean,.65)
        record(f'timeout idle+restart mean seconds B={B}',float(online.mean()),online_mean,.65)
        assert np.all(online<=2*oracle+1e-12)
        # Simulate an isolated cold start by summing billed nonserving slices.
        widths=np.diff(np.linspace(0,B,1001))
        record(f'one boot chip-hours B={B}',float(sum(C*widths)/3600),C*B/3600,1e-12)
    # Near-worst-case online/offline ratio for deterministic gap just above B.
    B=50.2; gap=B+1e-6
    record('ski-rental worst ratio', (min(gap,B)+B*(gap>B))/min(gap,B),2.,1e-12)

    table=[]
    for B in BOOTS:
        table.append(dict(B=B, boot_chip_hours=C*B/3600, long_timeout_cycle_chip_hours=2*C*B/3600,
            diffusion_headroom_instances=.1*z*math.sqrt(B), diffusion_ratio_vs_50_2=math.sqrt(B/50.2),
            diurnal_overhead_pct=100*sinusoid_overhead(B,86400),
            diurnal_chip_hours_overhead_for_mean10=16*10*24*sinusoid_overhead(B,86400),
            ten_minute_overhead_pct=100*sinusoid_overhead(B,600)))
    savings=[]
    for old,new in [(779.4,50.2),(284.9,50.2),(350,27.5),(419,18.8),(385,23.15)]:
        old_s=sinusoid_overhead(old,86400);new_s=sinusoid_overhead(new,86400)
        savings.append(dict(old_B=old,new_B=new, boot_saved_chip_hours=C*(old-new)/3600,
            boot_waste_reduction_pct=100*(1-new/old), diffusion_headroom_reduction_pct=100*(1-math.sqrt(new/old)),
            diurnal_total_saving_pct=100*(old_s-new_s)/(1+old_s)))
    result=dict(seed=20261020,assumptions=dict(chips=16,epsilon=.01,sigma_instances_per_sqrt_second=.1,diurnal_P_seconds=86400,relative_amplitude=.5), checks=checks,table=table,savings=savings)
    (ROOT/'theory_results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['| B (s) | Boot chip-h | Long-gap timeout cycle chip-h | Diffusion headroom (instances) | Diurnal overhead (%) | 10-min overhead (%) |','|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['B']:.1f} | {r['boot_chip_hours']:.4f} | {r['long_timeout_cycle_chip_hours']:.4f} | {r['diffusion_headroom_instances']:.4f} | {r['diurnal_overhead_pct']:.6f} | {r['ten_minute_overhead_pct']:.4f} |")
    (ROOT/'theory_tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(checks=len(checks),passed=all(c['passed'] for c in checks),table=table,savings=savings),indent=2))

if __name__=='__main__':
    main()
