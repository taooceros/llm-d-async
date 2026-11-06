from analyze import *

def main():
 runs=collect();out={'bounds':[],'flaps':[],'probes':{},'validation':{}}
 for r in runs:
  if r['R']<=0:continue
  f=fluid(r);at=np.array([x['t'] for x in r['trace']['requests']]);bw=(at>=r['bw'][0])&(at<r['bw'][1]);T=r['trigger']+r['R'];w=np.interp(at,f['t'],f['q'])/MU
  doomed=bw&(at<T-TAU)&(w>TAU)
  ceiling=.946 if r['kind']=='llmd' else 1.
  cr=[x for x in r['clients'] if r['bw'][0]<=x['sched_t']<r['bw'][1]]
  actual_doom=sum(x['sched_t']+.7882183790206909<T and (x.get('status')!=200 or (x.get('ttft_s') or 1e9)>.7882183790206909) for x in cr)
  actual_qdoom=sum(x['sched_t']+TAU<T and (x.get('queue_s') or 0)>TAU for x in cr)
  # Earliest sustained observed EPP queue (>2 for 10 s), not isolated preburst spikes.
  high=None;onset=None
  for ti,qi in zip(r['t'],r['q']):
   if ti<r['trigger']-100:continue
   high=(ti if high is None else high) if qi>2 else None
   if high is not None and ti-high>=10: onset=high;break
  out['bounds'].append({'rid':r['rid'],'arm':r['arm'],'trigger':r['trigger'],'T':T,'R':r['R'],'headstart':onset-r['trigger'] if onset is not None else None,'queue_only_ceiling_pct':100*(1-doomed.sum()/bw.sum()),'ceiling_with_warm_factor_pct':100*ceiling*(1-doomed.sum()/bw.sum()),'doomed_requests':int(doomed.sum())})
  out['bounds'][-1].update({'empirical_pre_ready_ttft_misses':actual_doom,'empirical_ceiling_pct':100*(1-actual_doom/len(cr)),'empirical_queue_only_ceiling_pct':100*(1-actual_qdoom/len(cr)),'empirical_warm_adjusted_queue_ceiling_pct':100*ceiling*(1-actual_qdoom/len(cr))})
  if r['kind']!='llmd' or r['N']!=4 or r['arm'] not in ['a/s-pm','a/s-stock','a/n-pm','a/n-stock']:continue
  seen=set()
  for up in r['up']:
   g=up['group']
   if g not in seen: seen.add(g);continue
   start=up['decision_t'];end=up.get('first_served_client_t');downs=[d for d in r['down'] if d['t']<start and d['from']==4 and d['to']==3];down=downs[-1]['t'] if downs else start
   finish=min(end if end is not None else r['bw'][1],r['bw'][1]);boot=(up['engine_ready_t']-up['ray_joined_t']) if up.get('engine_ready_t') is not None and up.get('ray_joined_t') is not None else None
   rr=[x for x in r['clients'] if down<=x['sched_t']<finish];qm=sum((x.get('queue_s') or 0)>TAU for x in rr);miss=sum(x.get('status')!=200 or (x.get('ttft_s') or 1e9)>.7882183790206909 or (x.get('tpot_s') or 0)>.06634625338868454 for x in rr)
   # Same arrivals; removing observed late scale-down isolates its fluid prediction.
   def noflap(t):
    n=np.maximum(1,np.rint(np.interp(t,r['t'],r['n'])));n[(t>=down)&(t<finish)]=4;return n
   nf=fluid(r,n_override=noflap)
   out['flaps'].append({'rid':r['rid'],'arm':r['arm'],'down':down,'retrigger':start,'first_served':end,'boot_s':boot,'boot_chip_s':8*boot if boot is not None else None,'off_to_retrigger_s':start-down,'unavailable_in_window_s':finish-down,'requests_during_unavailability':len(rr),'observed_misses_during_unavailability':miss,'observed_queue_misses_during_unavailability':qm,'predicted_extra_queue_misses':(nf['predA']-f['predA'])/100/ceiling*bw.sum(),'incremental_held_chip_s':0 if '/s-' in r['arm'] else None})
 probe=load('/home/hongtao/hetero-wt/Autoscale/results/gke-autoscale/campaign-5962c75/samples.jsonl')
 for arm in ['2x4-od','2x4-spot']:
  ss=[p for p in probe if p['arm']==arm];z={'n':len(ss),'status':[s['status'] for s in ss]}
  for key in ['node_registered','node_ready','image_pulled','jax_devices','first_compute']:
   vals=[s['derived'][key] for s in ss if s['derived'].get(key) is not None];z[key]={'median':float(np.median(vals)),'p90':float(np.quantile(vals,.9)),'max':max(vals),'values':vals}
  out['probes'][arm]=z
 val=load('/home/hongtao/hetero-wt/CostModel/results/autoscale-cost/d9c83f3c/validation.json')
 for arm in ['pm-gcs','stock']:
  ss=[x for x in val if x['arm']==arm];out['validation'][arm]={}
  for key in ['burst_viol_pct','max_queue','queue_recovery_s']:
   out['validation'][arm][key]={'sim':float(np.median([s['sim'][key] for s in ss])),'real':float(np.median([s['real'][key] for s in ss]))}
 # Common delay injected into all ready transitions: a provisioning-tail sensitivity, not new measured runs.
 r=next(x for x in runs if x['rid']=='a-n-pm-r3');base_n=np.maximum(1,np.rint(np.interp(np.arange(4000.),r['t'],r['n'])));out['tail_sensitivity']=[]
 for delta in [0,40,80,120]:
  def shifted(t):return np.maximum(1,np.rint(np.interp(t-delta,r['t'],r['n'],left=1,right=r['n'][-1])))
  f=fluid(r,n_override=shifted);out['tail_sensitivity'].append({'extra_F_s':delta,'predA':f['predA'],'predQ':f['predQ']})
 rows=load(OUT/'results.json')['arms'];pairs=[('a/s-pm','a/s-stock','Standby'),('a/n-pm','a/n-stock','New spot'),('a/s-pm-stab120','a/s-stock-stab120','120-s stabilization'),('hand/pm-gcs','hand/stock','1-s controller')]
 effects=[]
 for pm,stock,label in pairs:
  p=next(x for x in rows if x['arm']==pm);s=next(x for x in rows if x['arm']==stock);effects.append({'label':label,'measured_pp':p['measA']-s['measA'],'predicted_pp':p['predA']-s['predA']})
 out['effects']=effects
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4)
 ax.bar(x-.18,[s['measured_pp'] for s in effects],.36,label='Measured',color='#333333');ax.bar(x+.18,[s['predicted_pp'] for s in effects],.36,label='Fluid prediction',color='#2677b5')
 ax.set_xticks(x,[s['label'] for s in effects]);ax.set_ylabel('TPUFlex attainment gain (percentage points)');ax.set_ylim(0,24);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'attainment_gain.png',dpi=200);plt.close(fig)
 (OUT/'extended.json').write_text(json.dumps(out,indent=2,default=float))
 for arm in dict.fromkeys(x['arm'] for x in out['bounds']):
  z=[x for x in out['bounds'] if x['arm']==arm];print(arm,'ceiling',round(np.median([x['ceiling_with_warm_factor_pct'] for x in z]),2),'headstart',round(np.median([x['headstart'] for x in z if x['headstart'] is not None]),2))
 print('probes',json.dumps(out['probes']));print('flaps',json.dumps(out['flaps']));print('validation',out['validation']);print('tail',out['tail_sensitivity'])
if __name__=='__main__':main()
