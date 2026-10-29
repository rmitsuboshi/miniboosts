import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/miniboosts-mpl')
import json, platform, subprocess, time, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
import sklearn
import miniboosts as mb

OUT = Path(__file__).resolve().parent
def spirals(n, seed):
    rng = np.random.default_rng(seed)
    y = np.tile([-1, 1], n//2)
    t = rng.uniform(.08, 1., n)
    theta = 2.2*np.pi*t + (y == 1)*np.pi
    X = np.c_[t*np.cos(theta), t*np.sin(theta)]
    X += rng.normal(0, .15, X.shape)
    order = rng.permutation(n)
    return X[order], y[order]

X, y = spirals(160, 73)
y_clean = y.copy()
flipped = np.random.default_rng(997).choice(len(y), 20, replace=False)
y[flipped] *= -1
X_test, y_test = spirals(2000, 173)
lo, hi = np.array([-1.25, -1.25]), np.array([1.25, 1.25])
xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 180), np.linspace(lo[1], hi[1], 180))
grid = np.c_[xx.ravel(), yy.ravel()]
classes = [mb.AdaBoostClassifier, mb.AdaBoostVClassifier, mb.MadaBoostClassifier,
           mb.LPBoostClassifier, mb.ERLPBoostClassifier, mb.CorrectiveERLPBoostClassifier,
           mb.MLPBoostClassifier, mb.SmoothBoostClassifier, mb.SoftBoostClassifier,
           mb.TotalBoostClassifier, mb.GraphSeparationBoostingClassifier]
fig, axes = plt.subplots(3, 4, figsize=(16, 14), sharex=True, sharey=True)
colors = ['#2476b8', '#d55e35']
bg = ListedColormap(['#daeaf6', '#f9e0d7'])
records = []
arrays = {'X': X, 'y': y, 'X_test': X_test, 'y_test': y_test, 'grid': grid, 'y_clean': y_clean, 'flipped_rows': flipped}

def points(ax):
    for label, color in zip([-1, 1], colors):
        ax.scatter(*X[y == label].T, s=16, color=color, edgecolor='white', linewidth=.6, zorder=4)
    ax.scatter(*X[flipped].T, s=48, facecolors='none', edgecolors='#333333', linewidth=.7, zorder=5)
    ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_aspect('equal')
    ax.grid(alpha=.15); ax.spines[['top', 'right']].set_visible(False)

points(axes.flat[0])
axes.flat[0].set_title('Input: spirals + label noise', weight='bold', fontsize=12)
axes.flat[0].text(.03,.03, 'Rings: 20 flipped training labels',
                  transform=axes.flat[0].transAxes, fontsize=9)
for ax, cls in zip(list(axes.flat)[1:], classes):
    name=cls.__name__.replace('Classifier','')
    params={'estimator':mb.DecisionTreeClassifier(max_depth=2), 'max_iterations':60}
    available=cls().get_params()
    if 'tolerance' in available: params['tolerance']=.03
    if 'nu' in available: params['nu']=16.
    if cls is mb.SmoothBoostClassifier: params.update(gamma=.05,kappa=.1)
    record={'algorithm':name,'parameters':{k:(str(v) if k=='estimator' else v) for k,v in params.items()}}
    print('START',name,flush=True)
    start=time.perf_counter()
    try:
        model=cls(**params)
        model.fit(X,y)
        record['fit_seconds']=time.perf_counter()-start
        scores=model.decision_function(grid)
        predictions=model.predict(grid)
        assert np.isfinite(scores).all()
        assert np.array_equal(predictions,np.where(scores>=0,1,-1))
        train_acc=float(np.mean(model.predict(X)==y))
        test_acc=float(np.mean(model.predict(X_test)==y_test))
        record.update(status='ok',train_accuracy=train_acc,test_accuracy=test_acc,
                      n_iter=int(model.n_iter_),stop_reason=model.stop_reason_,
                      score_min=float(scores.min()),score_max=float(scores.max()))
        arrays[name+'_scores']=scores
        ax.pcolormesh(xx,yy,(scores.reshape(xx.shape)>=0).astype(int),cmap=bg,vmin=0,vmax=1,shading='auto',rasterized=True)
        if scores.min()<0<scores.max():
            ax.contour(xx,yy,scores.reshape(xx.shape),levels=[0],colors=['#333333'],linewidths=1.15)
        suffix = ' (cap)' if model.stop_reason_ == 'max_iterations' else ''
        ax.set_title(f'{name}\ntrain {train_acc:.1%} | test {test_acc:.1%}\n{model.n_iter_} rounds{suffix}',fontsize=10,weight='bold')
        print('OK',name,record,flush=True)
    except Exception as error:
        record.update(status='error',error=f'{type(error).__name__}: {error}')
        ax.set_title(name+'\nFAILED',fontsize=11,color='#a82929',weight='bold')
        ax.text(.5,.5,record['error'][:180].replace(': ', ':\n').replace(' with ', '\nwith '),ha='center',va='center',wrap=True,transform=ax.transAxes,fontsize=9)
        print('ERROR',name,record['error'],flush=True)
    points(ax)
    records.append(record)
    (OUT/'results.json').write_text(json.dumps(records,indent=2))
fig.suptitle('Harder still: overlapping spirals + 12.5% label noise',fontsize=21,weight='bold',y=.98)
fig.text(.5,.938,'Rust decision trees (max_depth=2) | max_iterations=60 | black line: decision_function(X)=0',ha='center',fontsize=12)
fig.text(.5,.025,'160 training points; 20 labels flipped (ringed). 2,000 independent test points without label flips; coordinate noise SD=0.15.\nTolerance=0.03 where applicable; nu=16 for capped methods; SmoothBoost gamma=0.05, kappa=0.1. Same learner and limits for every algorithm.\n(cap) = reached the 60-round limit. Equal tolerances do not imply equal accuracy; this is one dataset, not a ranking.',ha='center',fontsize=10,color='#444444')
fig.subplots_adjust(top=.89,bottom=.09,hspace=.36,wspace=.13)
fig.savefig(OUT/'decision-boundaries.png',dpi=180,facecolor='white')
fig.savefig(OUT/'decision-boundaries.pdf',facecolor='white')
np.savez_compressed(OUT/'data-and-scores.npz',**arrays)
metadata={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
          'matplotlib':matplotlib.__version__,'miniboosts':mb.__version__,
          'module_path':mb.__file__,'platform':platform.platform(),'machine':platform.machine(),
          'processor':platform.processor(),'cpu_count':os.cpu_count(),
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
          'build':'maturin release wheel, local uncommitted Python bindings',
          'timing':'wall time of Python fit only, includes validation/copies and Python/Rust callbacks; excludes prediction',
          'seeds':{'train':73,'test':173,'label_flips':997},'split':'independently generated test sample, labels not flipped; training labels flipped at 20 fixed random indices; no parameter selection on test', 'noise_sd':.15, 'training_label_flip_fraction':.125,
          'RAYON_NUM_THREADS':os.getenv('RAYON_NUM_THREADS','default')}
(OUT/'environment.json').write_text(json.dumps(metadata,indent=2))
print('SAVED',OUT/'decision-boundaries.png',flush=True)

# Compare signs, not score magnitudes: algorithms have different score scales.
ok = [r['algorithm'] for r in records if r['status'] == 'ok']
votes = np.stack([arrays[name+'_scores'] >= 0 for name in ok])
p = votes.mean(axis=0)
minority = np.minimum(p, 1-p)
pair = np.mean(votes[:, None, :] != votes[None, :, :], axis=2)
fig2, (ax1, ax2) = plt.subplots(1,2,figsize=(15,6.6),gridspec_kw={'width_ratios':[1,1.3]})
m=ax1.pcolormesh(xx,yy,100*minority.reshape(xx.shape),cmap='magma',vmin=0,vmax=50,shading='auto')
ax1.set_aspect('equal'); ax1.set_title('Where do the algorithms disagree?')
fig2.colorbar(m,ax=ax1,shrink=.8,label='Minority prediction (%)')
for label,color in zip([-1,1],colors):
    ax1.scatter(*X[y==label].T,s=13,color=color,edgecolor='white',linewidth=.3)
names=[name.replace('GraphSeparationBoosting','GraphSep').replace('CorrectiveERLPBoost','CorrectiveERLP') for name in ok]
m2=ax2.imshow(pair*100,cmap='Blues',vmin=0,vmax=max(1,pair.max()*100))
ax2.set_xticks(range(len(ok)),names,rotation=50,ha='right',fontsize=8)
ax2.set_yticks(range(len(ok)),names,fontsize=9)
for i in range(len(ok)):
    for j in range(len(ok)):
        ax2.text(j,i,f'{pair[i,j]*100:.1f}',ha='center',va='center',fontsize=8,color='white' if pair[i,j] > .55*pair.max() else 'black')
ax2.set_title('Pairwise prediction disagreement (%)')
fig2.suptitle('Same points and weak learner, different boundaries',fontsize=17,weight='bold')
fig2.text(.5,.015,f'{len(ok)} successful fits only. Fractions use the uniform plotted grid, not the test distribution.\nDark on the left = agreement; bright = split predictions. Disagreement is not an accuracy measure.',ha='center',fontsize=10)
fig2.tight_layout(rect=(0,.06,1,.95))
fig2.savefig(OUT/'disagreement.png',dpi=180,facecolor='white')
(OUT/'disagreement.json').write_text(json.dumps({'algorithms':ok,'pairwise_fraction':pair.tolist(),'grid_fraction_with_any_disagreement':float(np.mean(minority>0))},indent=2))
