from analyze import *
runs=collect();r=next(r for r in runs if r['rid']=='a-s-stock-r1');mask=(r['t']>=600)&(r['t']<=r['trigger']+r['R'])
res=[]
for mu in np.arange(1.08,1.241,.002):
 f=fluid(r,mu);pred=np.interp(r['t'][mask],f['t'],f['q']);obs=r['q'][mask];offset=max(0,float(np.mean(pred-obs)));err=float(np.mean((np.maximum(0,pred-offset)-obs)**2));err0=float(np.mean((pred-obs)**2));res.append((err,mu,offset,err0))
print('best offset',min(res));print('best no offset',min(res,key=lambda z:z[3]))
for mu in [1.121,1.15,1.16,min(res,key=lambda z:z[3])[1]]:
 print('MU',mu)
 for arm in dict.fromkeys(x['arm'] for x in runs):
  rs=[x for x in runs if x['arm']==arm];f=[fluid(x,mu) for x in rs]
  print(arm,round(np.median([x['predQ'] for x in f]),1),round(np.median([x['predA'] for x in f]),1))
