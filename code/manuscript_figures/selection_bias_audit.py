import pandas as pd, numpy as np, sys
from scipy import stats
d=pd.read_csv(sys.argv[1])
d['grp']=np.where(d.STATUS=='insufficient_followup','excluded','included')
d['year']=pd.to_datetime(d.INDEX_DATE).dt.year
d['fu_y']=d.FOLLOWUP_DAYS_FROM_INDEX/365.25
for c in ['ABETA42','PTAU','ABETA40']:
    d.loc[~d.BIOMARKER_AVAILABLE_AS_OF_INDEX.astype(bool),c]=np.nan
rows=[]
def cont(c,lab):
    a=d.loc[d.grp=='included',c].dropna();b=d.loc[d.grp=='excluded',c].dropna()
    p=stats.mannwhitneyu(a,b).pvalue
    sd=np.sqrt((a.var()+b.var())/2);smd=(a.mean()-b.mean())/sd
    rows.append([lab,f"{a.median():.1f} [{a.quantile(.25):.1f}–{a.quantile(.75):.1f}] (n={len(a)})",f"{b.median():.1f} [{b.quantile(.25):.1f}–{b.quantile(.75):.1f}] (n={len(b)})",f"{smd:.2f}",f"{p:.3g}"])
def cat(c,lab):
    t=pd.crosstab(d[c],d.grp);p=stats.chi2_contingency(t)[1]
    for k in t.index:
        rows.append([f"{lab}: {k}",f"{t.loc[k,'included']} ({100*t.loc[k,'included']/t['included'].sum():.1f}%)",f"{t.loc[k,'excluded']} ({100*t.loc[k,'excluded']/t['excluded'].sum():.1f}%)","",f"{p:.3g}"])
cat('PHASE','ADNI phase at index');cont('year','Index year');cont('fu_y','Diagnostic follow-up (years)')
d['bio']=d.BIOMARKER_AVAILABLE_AS_OF_INDEX.astype(bool);cat('bio','CSF biomarker available as of index')
cont('ABETA42','ABETA42 (pg/mL)');cont('PTAU','PTAU (pg/mL)')
o=pd.DataFrame(rows,columns=['Characteristic','Included (n=911)','Excluded, insufficient follow-up (n=842)','SMD','p'])
o.to_csv('selection_bias_audit_911_vs_842.csv',index=False);print(o.to_string())
