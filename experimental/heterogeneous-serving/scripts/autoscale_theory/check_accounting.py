#!/usr/bin/env python3
"""Independent state-timeline and derived-denominator numerical checks."""
import json
from pathlib import Path
import numpy as np
from theory import numeric_sinusoid, sinusoid_overhead
ROOT=Path(__file__).resolve().parent

def main():
    checks=[]
    def check(name,simulation,closed,tol=1e-8):
        error=abs(simulation-closed)
        checks.append(dict(name=name,simulation=float(simulation),closed=float(closed),absolute_error=float(error),tolerance=tol,passed=bool(error<=tol)))
        assert error<=tol,checks[-1]
    # Generate a reactive ready/idle/off/boot timeline. Each return trigger
    # is relative to the previous actual service completion, NOT an external
    # raw arrival gap that might overlap boot or queued service.
    B=4.; gaps=[2.,7.,4.,12.]; time=0.; segments=[]
    def add(state,duration):
        nonlocal time
        segments.append((time,time+duration,state));time+=duration
    add('boot',B);add('service',10.)
    for G in gaps:
        add('ready_idle',min(G,B))
        if G>B:
            add('off',G-B);add('boot',B)
        add('service',10.)
    dt=.001;t=np.arange(0,time,dt)+dt/2
    nonserving=np.zeros(t.size)
    for start,stop,state in segments:
        if state in ['boot','ready_idle']:
            nonserving[(t>=start)&(t<stop)]=1
    check('timeout exact accounting: instance-seconds',nonserving.sum()*dt,B*(1+sum(G>B for G in gaps))+sum(min(G,B) for G in gaps))
    # Envelopes with both separate and merged level intervals.
    intervals=[[(0,25),(40,50)],[(0,10),(20,25)],[(20,25)]]
    for B in [4.,12.]:
        t=np.arange(-B,50,dt)+dt/2;allocated=np.zeros(t.size)
        for start,stop,value in [(0,10,2),(10,20,1),(20,25,3),(40,50,1)]:
            allocated=np.maximum(allocated,value*((t>=start-B)&(t<stop)))
        # First boot B per level, plus min(gap,B) for each ready-idle gap.
        useful=55.
        gap_overhead=sum(min(layer[i+1][0]-layer[i][1],B) for layer in intervals for i in range(len(layer)-1))
        closed=useful+3*B+gap_overhead
        check(f'offline envelope vs boot+gap accounting B={B}',allocated.sum()*dt,closed)
        check(f'offline overhead H-H0 B={B}',(allocated.sum()*dt-useful)*16/3600,(3*B+gap_overhead)*16/3600)
    # Confirm ratios are percentages of the intended total, not of excess.
    a=.5;P=86400.
    old=numeric_sinusoid(779.4,P);new=numeric_sinusoid(50.2,P)
    exactold=sinusoid_overhead(779.4,P);exactnew=sinusoid_overhead(50.2,P)
    check('daily total-saving fraction',(old-new)/(1+old),(exactold-exactnew)/(1+exactold))
    check('daily saving vs static peak',(a-old)/(1+a),(a-exactold)/(1+a))
    check('B=0 saving vs static peak', (1.5-1)/1.5,a/(1+a))
    result=dict(checks=checks,passed=all(c['passed'] for c in checks))
    (ROOT/'accounting_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
