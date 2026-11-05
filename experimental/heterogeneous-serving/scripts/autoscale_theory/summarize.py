#!/usr/bin/env python3
"""Regenerate compact boot-time comparison tables from the full replay JSON."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
res=json.loads((ROOT/'trace_results.json').read_text())
evs={ev['bin_s']:ev for ev in res['evaluations']}
by={dt:{r['boot_s']:r for r in ev['boots']} for dt,ev in evs.items()}
lines=['| B (s) | 10 s bins, fluid chip-h | 10 s bins, integer chip-h | 60 s bins, fluid chip-h | 60 s bins, integer chip-h |','|---:|---:|---:|---:|---:|']
for b in [0,779.4,284.9,50.2,350,385,419,18.8,27.5]:
    values=[evs[dt]['baselines']['B0'][k]['chip_hours'] if b==0 else by[dt][b][k]['chip_hours_incl_pre'] for dt in [10,60] for k in ['continuous','integer']]
    lines.append('| '+f'{b:g}'+' | '+' | '.join(f'{v:,.2f}' for v in values)+' |')
lines+=['','| Comparison (s) | Bin size | Fluid chip-h saved | Fluid total saving | Integer chip-h saved | Integer total saving |','|---|---:|---:|---:|---:|---:|']
comparisons=[]
for old,new in [(779.4,50.2),(284.9,50.2),(350,27.5),(419,18.8)]:
    for dt in [10,60]:
        vals=[]
        row=dict(old_B=old,new_B=new,bin_s=dt)
        for kind in ['continuous','integer']:
            before=by[dt][old][kind]['chip_hours_incl_pre'];after=by[dt][new][kind]['chip_hours_incl_pre']
            saved=before-after;frac=saved/before
            row[kind]=dict(saved_chip_hours=saved,saved_fraction=frac)
            vals.extend([f'{saved:,.2f}',f'{100*frac:.3f}%'])
        comparisons.append(row)
        lines.append('| '+f'{old:g} → {new:g}'+' | '+str(dt)+' s | '+' | '.join(vals)+' |')
text='\n'.join(lines)+'\n'
(ROOT/'trace_comparisons.md').write_text(text)
(ROOT/'trace_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
print(text)
