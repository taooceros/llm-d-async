#!/usr/bin/env python3
"""Derive tables and hold-out checks from analyze_fleet.py output."""
import json
from pathlib import Path
import numpy as np
P=Path('/tmp/autoscale_theory2/FleetCostTheory')
r=json.loads((P/'results.json').read_text())
summary={}
for label,d in r.items():
    rows=[]
    for target in [95,99]:
        p=d['policies'][f'{target}/pm'];s=d['policies'][f'{target}/stock']
        for term in ['total','base','head','keep','boot']:
            pred=s[term]-p[term];sim=s['sim']['chip_h_day' if term=='total' else term]-p['sim']['chip_h_day' if term=='total' else term]
            rows.append({'target':target,'term':term,'pred':pred,'sim':sim,'error':pred-sim,'relative_error_pct':100*(pred-sim)/sim})
    T=np.array(d['gap_grid']['T']); I=np.array(d['gap_grid']['idle_chip_h_day'])*3600/8
    N=np.array(d['gap_grid']['restart_day']); opts=[]
    for who,R in [('pm',63),('stock',377)]:
        J=I+3*R*N; pred=int(T[np.argmin(J)])
        opts.append({'who':who,'R':R,'effective_reacquisition_seconds':3*R,'T_pred_grid':pred,'T_measured':d['policies']['95/'+who]['T'],
                     'powerlaw_T_pred':d['N_fit_120_1800']['alpha']*3*R})
    rec={'delta_tables':rows,'T_star_with_gamma3':opts,'headroom_points':{str(B):d['rise'][str(B)] for B in [63,377,265,581]},
         'hpa90':{},'cost_anchor_rows':[x for x in d['cost_fit']['rows'] if x['R'] in [0,60,390,900]]}
    h=d['hpa90'];hs=h['stock'];hp=h['pm']
    # Counterfactual with identical boot cost removed: explains how much of 90% gap is acquisition.
    rec['hpa90']={'stock':hs,'pm':hp,'boot_saving':hs['boot']-hp['boot'], 'head_saving':hs['head']-hp['head'],
        'rest_delta':hs['base']-hp['base']+hs['drain']-hp['drain'], 'single_saving_pct':h['saving_pct'],'hull_saving_pct':h['hull']['saving_pct'],
        'stock_naive_uncanceled_boot':hs['acq_per_day']*8*377/3600,'pm_naive_uncanceled_boot':hp['acq_per_day']*8*63/3600}
    summary[label]=rec
(P/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
