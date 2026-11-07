from pathlib import Path
import json,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Regenerate the current study displays from saved results.')
parser.add_argument('--output',type=Path,default=ROOT/'outputs/displays')
args=parser.parse_args()
W=args.output.resolve();R=ROOT/'results'
for folder in ['figures','tables']:(W/folder).mkdir(parents=True,exist_ok=True)
read=lambda name:json.loads((R/name).read_text(encoding='utf-8'))
NAVY='#304d70';ORANGE='#b66a2e';TEAL='#18847b';GREY='#78828c'
COL=[NAVY,ORANGE,TEAL]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def save(fig,name):
    for ext in ['pdf','png','svg']: fig.savefig(W/'figures'/f'{name}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
def panel(ax,label):ax.text(-.12,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=10)
def intervals(ax,rows,y,key='estimate',color=TEAL):
    x=np.array([r[key]for r in rows]);q=np.array([r['ci']for r in rows]);ax.errorbar(x,y,xerr=[x-q[:,0],q[:,1]-x],fmt='o',color=color,capsize=2,ms=4,lw=1)
def f(x,d=4):return f'{x:.{d}f}'
def ci(q,d=4):return '['+f(q[0],d)+', '+f(q[1],d)+']'
def table(name,caption,label,cols,header,rows,note=''):
    s='\\begin{table}[H]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\small\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'
    s+='\n'.join(' & '.join(row)+r' \\'for row in rows)
    s+='\n\\bottomrule\n\\end{tabular}\n'
    if note:s+='\\par\\smallskip\n\\begin{minipage}{\\textwidth}\\footnotesize '+note+'\\end{minipage}\n'
    s+='\\end{table}\n';s=s.replace('5 × 8 s',r'$5\times8$ s').replace('10 × 4 s',r'$10\times4$ s');(W/'tables'/name).write_text(s,encoding='utf-8')

# Original estimates and original intervals are read directly, never recomputed.
old=json.loads((ROOT/'inputs/summary_results.json').read_text())
old['PRIMARY']['participant_rank_reproducibility']['F5']=old['PRESPECIFIED_DESCRIPTIVE_SECONDARY']['F5']['participant_rank_reproducibility']
fig,axs=plt.subplots(1,2,figsize=(6.85,2.7));fig.subplots_adjust(left=.09,right=.98,bottom=.27,top=.9,wspace=.70)
for key,label,col,marker in [('PRIMARY','Primary (N = 531)',NAVY,'o'),('DIRECTIONAL_REPLICATION','Replication (N = 59)',ORANGE,'s')]:
    vals=[old[key]['participant_rank_reproducibility'][g]for g in ['C','F5','F10']]
    axs[0].plot(range(3),vals,marker,ls='none',color=col,label=label,ms=4)
axs[0].set_xticks(range(3),['Contiguous','Distributed\n5 × 8 s','Distributed\n10 × 4 s']);axs[0].set_ylim(.79,.99);axs[0].set_ylabel('Rank reproducibility ($\\widetilde{\\rho}$)');axs[0].legend(frameon=False,loc='lower right',fontsize=7)
for y,(key,label,col) in enumerate([('DIRECTIONAL_REPLICATION','Replication\nN = 59',ORANGE),('PRESPECIFIED_SENSITIVITY','Complete support\nN = 516',GREY),('PRIMARY','Primary\nN = 531',NAVY)]):
    v=old[key];e=v['delta_z_F10_minus_C'];q=v['percentile_95_interval'];axs[1].errorbar(e,y,xerr=[[e-q[0]],[q[1]-e]],fmt='o',color=col,capsize=3,ms=4)
axs[1].set_yticks([0,1,2],['Replication\nN = 59','Complete support\nN = 516','Primary\nN = 531']);axs[1].set_ylim(-.4,2.4);axs[1].axvline(0,color=GREY,ls='--',lw=.8);axs[1].set_xlabel('10 × 4 s minus contiguous ($\\Delta z$)')
for j,ax in enumerate(axs):panel(ax,['(a)','(b)'][j])
save(fig,'figure2_rank_results')

fig,axs=plt.subplots(2,2,figsize=(6.85,4.6));fig.subplots_adjust(left=.085,right=.99,top=.92,bottom=.12,hspace=.72,wspace=.4)
for row,(cohort,label) in enumerate([('primary531','Primary, N = 531'),('replication59','Replication, N = 59')]):
    with np.load(R/f'agreement_{cohort}.npz')as z:p=z['participant_medians_pp']
    for j,lab in enumerate(['Contiguous','5 × 8 s','10 × 4 s']):
        x=np.sort(p[:,j]);axs[row,0].step(x,np.arange(1,len(x)+1)/len(x),where='post',color=COL[j],label=lab,lw=1.2)
    axs[row,0].set(xlabel='Median absolute A/B difference (pp)',ylabel='Cumulative fraction',title=label)
    if row==0:axs[row,0].legend(frameon=False,loc='lower right')
    delta=p[:,2]-p[:,0];axs[row,1].hist(delta,bins=25 if len(p)>100 else 14,color=TEAL,alpha=.85)
    axs[row,1].yaxis.set_major_locator(MaxNLocator(integer=True,steps=[1,2,5,10]))
    axs[row,1].axvline(0,color=GREY,ls='--',lw=.9);axs[row,1].set(xlabel='Change in median absolute A/B difference (pp)\n10 × 4 s minus contiguous',ylabel='Participants')
    v=read(f'agreement_{cohort}.json')['contrasts'][1]
    axs[row,1].set_title(f'Mean {v["estimate_pp"]:.2f} pp\n95% interval {ci(v["ci"],2)}',fontsize=8)
    for j in range(2):panel(axs[row,j],['(a)','(b)','(c)','(d)'][row*2+j])
save(fig,'figure3_marker_agreement')

post=read('pre_post_summary.json');r=post['conditions']
fig,axs=plt.subplots(1,2,figsize=(6.85,3.1));fig.subplots_adjust(left=.085,right=.98,top=.81,bottom=.25,wspace=.72)
names=['Within\npre','Within\npost','Pre/post\nA','Pre/post\nB'];x=np.arange(4)
for j,g in enumerate(['Contiguous','5 × 8 s','10 × 4 s']):
    rows=[r[h*3+j]for h in range(4)];e=np.array([v['rho']for v in rows]);q=np.array([v['rho_ci']for v in rows]);axs[0].errorbar(x+(j-1)*.15,e,yerr=[e-q[:,0],q[:,1]-e],fmt='o',color=COL[j],label=g,capsize=2,ms=3,lw=1)
axs[0].set_xticks(x,names);axs[0].set_ylabel('Rank correlation ($\\widetilde{\\rho}$)');axs[0].axvline(1.5,color=GREY,ls=':',lw=1);axs[0].set_ylim(.74,.99)
intervals(axs[1],post['contrasts'][:4],[3,2,1,0]);axs[1].set_yticks([3,2,1,0],['Within pre','Within post','Pre/post A','Pre/post B']);axs[1].set_ylim(-.5,3.5);axs[1].axvline(0,color=GREY,ls='--',lw=.8);axs[1].set_xlabel('10 × 4 s minus contiguous ($\\Delta z$)')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.0))
for j,ax in enumerate(axs):panel(ax,['(a)','(b)'][j])
save(fig,'figure4_pre_post')

norm=read('normalization_summary.json')
fig,ax=plt.subplots(figsize=(6.5,2.6));fig.subplots_adjust(left=.27,right=.94,top=.94,bottom=.25)
intervals(ax,norm['contrasts'],[2,1,0],key='delta_z');ax.set_yticks([2,1,0],['Relative alpha','Absolute alpha','Remaining 1–30-Hz power']);ax.set_ylim(-.5,2.5);ax.axvline(0,color=GREY,ls='--',lw=.8);ax.set_xlabel('10 × 4 s minus contiguous ($\\Delta z$)');save(fig,'figure_s4_normalization')

pair=read('pairing_summary.json');feas=read('pairing_feasibility.json')['rows']
fig,axs=plt.subplots(1,2,figsize=(6.5,2.8));fig.subplots_adjust(left=.10,right=.98,top=.81,bottom=.24,wspace=.6)
for geom,label,col in [('C','Contiguous',NAVY),('F10','10 × 4 s',TEAL)]:
    vals=[v['mean_distance_s']for v in feas if v['regime']==geom];axs[0].plot(range(3),vals,'o',color=col,label=label,ms=4)
axs[0].set_xticks(range(3),['Original','Nearer','Farther']);axs[0].set_ylabel('Mean nearest-cell distance (s)');axs[0].set_ylim(0,60)
intervals(axs[1],pair['contrasts'][:3],[2,1,0]);axs[1].set_yticks([2,1,0],['Original','Nearer','Farther']);axs[1].set_ylim(-.5,2.5);axs[1].axvline(0,color=GREY,ls='--',lw=.8);axs[1].set_xlabel('10 × 4 s minus contiguous ($\\Delta z$)')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,frameon=False,ncol=2,loc='upper center')
for j,ax in enumerate(axs):panel(ax,['(a)','(b)'][j])
save(fig,'figure_s5_pairing')

# Supplementary tables, generated from unrounded saved result artifacts.
with (R/'agreement_summary.csv').open(encoding='utf-8-sig')as stream:means=list(csv.DictReader(stream))
rows=[]
for cohort,label in [('primary531','Primary (531)'),('replication59','Replication (59)')]:
    v=read(f'agreement_{cohort}.json');m=[a for a in means if a['cohort']==cohort]
    for j,g in enumerate(['Contiguous','5 × 8 s','10 × 4 s']):
        delta='---' if j==0 else f(v['contrasts'][j-1]['estimate_pp'],2)+' '+ci(v['contrasts'][j-1]['ci'],2)
        frac='---' if j==0 else f(100*v['contrasts'][j-1]['fraction_participants_smaller'],1)+r'\%'
        rows.append([label if j==0 else '',g,f(float(m[j]['mean_participant_median_pp']),2)+' '+ci([float(m[j]['ci_low']),float(m[j]['ci_high'])],2),delta,frac])
table('table_s5_agreement.tex','Within-participant absolute A/B differences. Values and contrasts are in percentage points; brackets contain 95\\% participant-bootstrap intervals.','tab:s-agreement',r'@{}llrrr@{}',r'Cohort ($N$) & Geometry & Mean median & Difference from contiguous & Smaller difference',rows,r'Each participant contributes the median absolute difference across 200 draws. The last column gives the percentage of participants whose median difference was smaller than under contiguous sampling; it is descriptive.')

rows=[]
for k,name in enumerate(['Within pre, A/B','Within post, A/B','Pre/post, A','Pre/post, B']):
    values=post['conditions'][3*k:3*k+3];con=post['contrasts'][k]
    rows.append([name,'524',*[f(v['rho'])for v in values],f(con['estimate']),ci(con['ci'])])
boundary=read('pre_post_boundary_summary.json')
for k,name in enumerate(['Within pre, annotation exclusion','Within post, annotation exclusion','Pre/post A, annotation exclusion']):
    values=boundary['conditions'][2*k:2*k+2];con=boundary['contrasts'][k]
    rows.append([name,'513',f(values[0]['rho']),'---',f(values[1]['rho']),f(con['estimate']),ci(con['ci'])])
v=post['contrasts'][-1]
table('table_s6_prepost.tex','Rank correlations within and across recordings. The geometry contrast is 10 × 4 s distributed sampling minus contiguous sampling on the mean Fisher scale.','tab:s-prepost',r'@{}p{0.29\textwidth}rrrrrr@{}',r'Comparison & $N$ & Contiguous & 5 × 8 s & 10 × 4 s & $\Delta z$ & 95\% interval',rows,'Correlations are back-transformed mean Fisher values. All comparisons in the 524-person sample share participant-bootstrap resamples. The pre/post A contrast minus the within-pre contrast was $'+f(v['estimate'])+'$ (95\\% interval '+ci(v['ci'])+'). Annotation exclusion used the union of affected pre/post participants; 5 × 8 s sampling was not recomputed for that sensitivity analysis.')

rows=[]
for k,name in enumerate(['Relative alpha','Absolute alpha','Remaining 1--30-Hz power']):
    v=norm['contrasts'][k];c=norm['conditions'][2*k:2*k+2]
    rows.append([name,*[f(a['rho'])for a in c],f(v['delta_z']),ci(v['ci'])])
note='All rows use the same 516 participants and samples. Differences from the relative-alpha contrast: absolute alpha $'+f(norm['contrasts'][1]['difference_from_relative'])+'$ '+ci(norm['contrasts'][1]['difference_ci'])+'; remaining power $'+f(norm['contrasts'][2]['difference_from_relative'])+'$ '+ci(norm['contrasts'][2]['difference_ci'])+'. These are paired contrasts, not equivalence tests.'
table('table_s7_normalization.tex','Spectral normalization comparison. Intervals are 95\\% participant-bootstrap intervals for the geometry contrast.','tab:s-normalization',r'@{}lrrrr@{}',r'Feature & Contiguous $\widetilde\rho$ & 10 × 4 s $\widetilde\rho$ & $\Delta z$ & 95\% interval',rows,note)

rows=[]
for k,(name,lab) in enumerate([('Original','original'),('Nearer','near'),('Farther','far')]):
    con=pair['contrasts'][k];d=[v['mean_distance_s']for v in feas if v['pairing']==lab]
    rows.append([name,f(d[0],2),f(d[1],2),f(con['estimate']),ci(con['ci'])])
changes='; '.join(name+' $'+f(v['estimate'])+'$ '+ci(v['ci'])for name,v in zip(['nearer minus original','farther minus original','farther minus nearer'],pair['contrasts'][3:]))
table('table_s8_pairing.tex','Re-pairing of fixed samples in 516 participants. Distances are mean symmetric nearest-cell distances, in seconds.','tab:s-pairing',r'@{}lrrrr@{}',r'Pairing & Contiguous distance & 10 × 4 s distance & $\Delta z$ & 95\% interval',rows,'Paired differences between geometry contrasts (95\\% intervals): '+changes+'. All original A and B samples were retained once per pairing arrangement, with no within-pair overlap.')

reporting=(ROOT/'templates/table_s9_reporting.tex').read_text(encoding='utf-8').replace(r'\begin{table*}[!t]',r'\begin{table}[H]').replace(r'\end{table*}',r'\end{table}')
(W/'tables/table_s9_reporting.tex').write_text(reporting,encoding='utf-8')
import runpy
runpy.run_path(str(Path(__file__).with_name('build_sampling_display.py')), run_name='__main__')
print('Seven figures and five supplementary tables generated from existing results.')
