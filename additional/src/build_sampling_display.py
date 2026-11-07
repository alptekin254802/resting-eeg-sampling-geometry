"""Compact sampling schematic and relabelled marker-displacement plot; no analysis."""
from pathlib import Path
import json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Regenerate sampling and signed-shift displays.')
parser.add_argument('--output',type=Path,default=ROOT/'outputs/displays')
W=parser.parse_args().output.resolve()/'figures'
W.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'svg.fonttype':'none'})
with np.load(ROOT/'inputs/scientific_results.npz',allow_pickle=False) as z:
    example=z['sensitivity_schedule'][0,:,0].copy()
    displacement=z['secondary_participant_displacement'].copy()
# The first complete-support participant/draw is the same archived draw previously shown.
fig=plt.figure(figsize=(3.31,2.25))
ax=fig.add_axes([.33,.23,.65,.73])
colors=['#304d70','#b66a2e','#18847b'];groups=[(5.3,4.7),(3.2,2.6),(1.1,.5)]
for j,(ya,yb) in enumerate(groups):
    partitions=[np.arange(42)] if j==0 else np.array_split(np.arange(42),[0,5,10][j])
    for k,cells in enumerate(partitions):
        ax.add_patch(Rectangle((8+4*cells[0],yb-.25),4*len(cells),ya-yb+.5,
           facecolor='#e9eef1' if k%2==0 else '#f8f9fa',edgecolor='none'))
    for arm,y in enumerate([ya,yb]):
        selected=example[j,arm];assert len(set(selected))==10
        for cell in selected:ax.add_patch(Rectangle((8+4*int(cell),y-.18),4,.36,
           facecolor=colors[j],edgecolor='white',linewidth=.22))
        ax.text(5,y,['A','B'][arm],ha='right',va='center',fontsize=7)
    assert set(example[j,0]).isdisjoint(example[j,1])
    ax.text(-.50,(ya+yb)/2,['Contiguous','5 × 8 s','10 × 4 s'][j],
      transform=ax.get_yaxis_transform(),va='center',ha='left',fontsize=8)
ax.set(xlim=(8,176),ylim=(-.1,5.9),xticks=[8,60,120,176],yticks=[],xlabel='Time from recording onset (s)')
ax.spines['left'].set_visible(False);ax.tick_params(axis='x',pad=2,length=2)
for ext in ['pdf','png','svg']:fig.savefig(W/f'figure1_sampling_geometry.{ext}',dpi=300,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
plt.rcParams.update({'font.size':10.2,'axes.labelsize':10.2,'xtick.labelsize':9.8,'ytick.labelsize':9.8,'legend.fontsize':9.8})
s=json.loads((ROOT/'inputs/summary_results.json').read_text())
mean=s['PRESPECIFIED_DESCRIPTIVE_SECONDARY']['absolute_displacement']['cohort_mean']
fig,ax=plt.subplots(figsize=(6.4,3.4),layout='constrained')
counts,edges,patches=ax.hist(displacement,bins=32,color='#6C8EBF',edgecolor='white',linewidth=.5)
expected,expected_edges=np.histogram(displacement,bins=32)
assert np.array_equal(counts,expected) and np.array_equal(edges,expected_edges)
ax.axvline(0,color='#555555',ls='--',lw=1,label='Zero')
ax.axvline(mean,color='#A65E2E',lw=1.3,label='Cohort mean')
ax.set(xlabel='Signed shift in logit relative alpha\n(distributed 10 × 4 s minus contiguous sampling)',ylabel='Participants')
ax.legend(frameon=False)
for ext in ['pdf','png','svg']:fig.savefig(W/f'figure_s1_participant_displacement.{ext}',dpi=300,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
