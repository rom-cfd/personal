import numpy as np


def evaluate(true, pred, current, element_matrix, floor=1e-8):
    y,p = true[:,1:],pred[:,1:]
    finite = np.all(np.isfinite(pred),axis=1)
    d = p-y
    valid = np.isfinite(d)
    major = y >= 1e-3
    minor = ~major
    def rmse(mask):
        a=d[mask & valid]
        return float(np.sqrt(np.mean(a*a))) if a.size else float('nan')
    rel = np.abs(d)/np.maximum(y,floor)
    # Capped log-relative score remains finite but explicitly penalizes invalid rows.
    penalties=np.where(np.isfinite(rel),np.log1p(np.minimum(rel,1e12)),np.log1p(1e12))
    mass=np.abs(p.sum(axis=1)-1)
    element=np.max(np.abs((p-current[:,1:1+y.shape[1]])@element_matrix.T),axis=1)
    out=dict(rmse=rmse(np.ones_like(major)), major_rmse=rmse(major), minor_rmse=rmse(minor),
             relative_log_mean=float(penalties.mean()),relative_p50=float(np.nanmedian(rel)),
             relative_p95=float(np.nanquantile(rel,.95)),
             invalid_row_fraction=float(1-finite.mean()), negative_fraction=float(np.mean(p<0)),
             negative_row_fraction=float(np.mean(np.any(p<0,axis=1))),
             mass_sum_mae=float(np.nanmean(mass)),mass_sum_max=float(np.nanmax(mass)),
             elemental_mae=float(np.nanmean(element)),elemental_max=float(np.nanmax(element)),
             temperature_rmse=float(np.sqrt(np.nanmean((pred[:,0]-true[:,0])**2))))
    out['selection_score']=out['relative_log_mean']+10*out['major_rmse']+10*out['mass_sum_mae']+100*out['invalid_row_fraction']
    return out


def distributions(true,pred,species):
    rows=[]
    edges=[0,1e-16,1e-12,1e-8,1e-4,1e-3,1.01]
    for j,s in enumerate(species):
        d=pred[:,j+1]-true[:,j+1]
        for lo,hi in zip(edges[:-1],edges[1:]):
            m=(true[:,j+1]>=lo)&(true[:,j+1]<hi)
            if m.any():
                a=np.abs(d[m]); rel=a/np.maximum(true[m,j+1],1e-8)
                rows.append(dict(species=str(s),lower=lo,upper=hi,count=int(m.sum()),
                                 finite_count=int(np.isfinite(a).sum()),
                                 mae=float(np.nanmean(a)),rmse=float(np.sqrt(np.nanmean(a*a))),
                                 absolute_p95=float(np.nanquantile(a,.95)),relative_p95=float(np.nanquantile(rel,.95))))
    return rows
