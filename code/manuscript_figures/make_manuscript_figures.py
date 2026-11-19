import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
T='/home/claude/s47/AUAF_Stage4_7_Complete_Package/tables/'
m=pd.read_csv(T+'stage47_model_metrics_bootstrap_CI_all_conditions.csv')
m=m[(m.cohort=='full_test')&(m.missingness==0)]
order=['XGBoost','CatBoost','ExtraTrees','EarlyFusion','LateFusion','AttentionFusion','AUAF']
col={'XGBoost':'#4C78A8','CatBoost':'#4C78A8','ExtraTrees':'#4C78A8','EarlyFusion':'#59A14F','LateFusion':'#59A14F','AttentionFusion':'#59A14F','AUAF':'#E15759'}
tasks=[('diagnosis_multiclass','(a) CN/MCI/AD'),('diagnosis_binary','(b) AD vs non-AD'),('mci_conversion','(c) 36-month MCI-to-AD')]
def bars(metric,fn,ylab):
    f,ax=plt.subplots(1,3,figsize=(13,4))
    for a,(t,ttl) in zip(ax,tasks):
        d=m[(m.task==t)&(m.metric==metric)].set_index('model').loc[order]
        y=d.mean_across_5_seeds.values;lo=y-d.bootstrap_ci95_low.values;hi=d.bootstrap_ci95_high.values-y
        a.bar(range(7),y,yerr=[lo,hi],color=[col[k] for k in order],capsize=3)
        a.set_xticks(range(7));a.set_xticklabels(order,rotation=45,ha='right',fontsize=8);a.set_title(ttl,fontsize=10)
        a.set_ylim(max(0,(y-lo).min()-0.05),min(1,(y+hi).max()+0.02));a.set_ylabel(ylab)
    plt.tight_layout();plt.savefig(fn,dpi=200);plt.close()
bars('roc_auc','fig/fig5_rocauc_ci.png','ROC-AUC (mean, 95% CI)')
c=pd.read_csv(T+'stage47_calibration_pre_post_bootstrap_CI.csv');c=c[(c.cohort=='full_test')&(c.missingness==0)]
f,ax=plt.subplots(2,3,figsize=(13,7))
for j,met in enumerate(['brier','ece']):
  for i,(t,ttl) in enumerate(tasks):
    d=c[(c.task==t)&(c.metric==met)].set_index('model').loc[order];x=np.arange(7);a=ax[j,i]
    a.bar(x-0.2,d.pre_mean_across_seeds,0.4,label='Before TS',color='#BAB0AC');a.bar(x+0.2,d.post_mean_across_seeds,0.4,label='After TS',color=[col[k] for k in order])
    a.set_xticks(x);a.set_xticklabels(order,rotation=45,ha='right',fontsize=8);a.set_title(f'{ttl} – {met.upper() if met=="ece" else "Brier"}',fontsize=9)
    if i==0 and j==0:a.legend(fontsize=8)
plt.tight_layout();plt.savefig('fig/fig8_calibration.png',dpi=200);plt.close()
