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
X, labels = make_moons(n_samples=80, noise=0.13, random_state=42)
y = 2*labels-1
X_test, labels_test = make_moons(n_samples=400, noise=0.13, random_state=142)
y_test = 2*labels_test-1
lo = X.min(axis=0)-.4
hi = X.max(axis=0)+.4
xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 160))
grid = np.c_[xx.ravel(), yy.ravel()]
classes = [mb.AdaBoostClassifier, mb.AdaBoostVClassifier, mb.MadaBoostClassifier,
           mb.LPBoostClassifier, mb.ERLPBoostClassifier, mb.CorrectiveERLPBoostClassifier,
           mb.MLPBoostClassifier, mb.SmoothBoostClassifier, mb.SoftBoostClassifier,
           mb.TotalBoostClassifier, mb.GraphSeparationBoostingClassifier]
fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True, sharey=True)
colors = ['#2476b8', '#d55e35']
bg = ListedColormap(['#daeaf6', '#f9e0d7'])
records = []
arrays = {'X': X, 'y': y, 'X_test': X_test, 'y_test': y_test, 'grid': grid}

def points(ax):
    for label, color in zip([-1, 1], colors):
        ax.scatter(*X[y == label].T, s=27, color=color, edgecolor='white', linewidth=.6, zorder=4)
    ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_aspect('equal')
    ax.grid(alpha=.15); ax.spines[['top', 'right']].set_visible(False)

points(axes.flat[0])
axes.flat[0].set_title('Input: 80 two-moon points', weight='bold', fontsize=12)
axes.flat[0].text(.03,.03, 'Blue = -1    Orange = +1\nSame points; depth-2 trees',
                  transform=axes.flat[0].transAxes, fontsize=9)
for ax, cls in zip(list(axes.flat)[1:], classes):
    name=cls.__name__.replace('Classifier','')
    params={'estimator':mb.DecisionTreeClassifier(max_depth=2), 'max_iterations':100}
    available=cls().get_params()
    if 'tolerance' in available: params['tolerance']=.05
    if 'nu' in available: params['nu']=5.
    if cls is mb.SmoothBoostClassifier: params.update(gamma=.1,kappa=.1)
    record={'algorithm':name,'parameters':{k:(str(v) if k=='estimator' else v) for k,v in params.items()}}
    print('START',name,flush=True)
    start=time.perf_counter()
    try:
        model=cls(**params)
        try:
            model.fit(X,y)
        except RuntimeError as error:
            if cls is not mb.SoftBoostClassifier:
                raise
            record['initial_failure'] = {'nu': 5.0, 'error': str(error)}
            print('INITIAL FAILURE',name,str(error),'RETRY nu=1',flush=True)
            params['nu'] = 1.0
            record['parameters']['nu'] = 1.0
            model = cls(**params)
            start=time.perf_counter()
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
        suffix = ' | nu=1 (nu=5 failed)' if 'initial_failure' in record else ''
        ax.set_title(f'{name}\ntrain {train_acc:.1%} | test {test_acc:.1%}\n{model.n_iter_} rounds{suffix}',fontsize=10,weight='bold')
        print('OK',name,record,flush=True)
    except Exception as error:
        record.update(status='error',error=f'{type(error).__name__}: {error}')
        ax.set_title(name+'\nFAILED',fontsize=11,color='#a82929',weight='bold')
        ax.text(.5,.5,record['error'][:220],ha='center',va='center',wrap=True,transform=ax.transAxes,fontsize=9)
        print('ERROR',name,record['error'],flush=True)
    points(ax)
    records.append(record)
    (OUT/'results.json').write_text(json.dumps(records,indent=2))
fig.suptitle('MiniBoosts called from Python: decision boundaries',fontsize=21,weight='bold',y=.98)
fig.text(.5,.938,'Rust decision trees (max_depth=2) | max_iterations=100 | black line: decision_function(X)=0',ha='center',fontsize=12)
fig.text(.5,.025,'Training points: seed 42, noise 0.13. Independent test set: 400 points, seed 142. Background shows predicted class.\nTolerance=0.05; capped methods nu=5 except SoftBoost nu=1 (nu=5: solver AlmostSolved). SmoothBoost gamma=0.1, kappa=0.1.',ha='center',fontsize=10,color='#444444')
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
          'seeds':{'train':42,'test':142},'split':'independently generated test sample; no parameter selection on test',
          'RAYON_NUM_THREADS':os.getenv('RAYON_NUM_THREADS','default')}
(OUT/'environment.json').write_text(json.dumps(metadata,indent=2))
print('SAVED',OUT/'decision-boundaries.png',flush=True)
