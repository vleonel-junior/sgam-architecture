from copy import deepcopy
from pathlib import Path
import warnings

import numpy as np
import zero
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from benchmark import lib

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

args, output = lib.load_config()
args['model']['random_state'] = args['seed']

try:
    zero.random.seed(args['seed'])
except AttributeError:
    try:
        zero.set_randomness(args['seed'])
    except AttributeError:
        pass

dataset_dir = lib.get_path(args['data']['path'])
stats = lib.load_json(output / 'stats.json')
stats.update({'dataset': dataset_dir.name, 'algorithm': Path(__file__).stem})

# Prepare data and model
D = lib.Dataset.from_dir(dataset_dir)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'ohe'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
assert isinstance(X, dict)

try:
    zero.random.seed(args['seed'])
except AttributeError:
    try:
        zero.set_randomness(args['seed'])
    except AttributeError:
        pass

Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')

# Prepare model kwargs
fit_kwargs = deepcopy(args.get("fit", {}))
model_kwargs = deepcopy(args.get("model", {}))

model_kwargs.setdefault('n_jobs', -1)

# Create model
if D.is_regression:
    model = ExplainableBoostingRegressor(**model_kwargs)
    predict = model.predict
else:
    model = ExplainableBoostingClassifier(**model_kwargs)
    if D.is_multiclass:
        predict = model.predict_proba
    else:
        predict = lambda x: model.predict_proba(x)[:, 1]

# Fit model
timer = zero.Timer()
timer.run()
print("Training EBM...")
model.fit(X[lib.TRAIN], Y[lib.TRAIN])

# Save model importances
try:
    explanation = model.explain_global()
    # InterpretML EBM: importances are in explanation.data(0)['scores'] per feature
    importances = [explanation.data(i)['scores'].mean() for i in range(len(explanation.data()['names']))]
    np.save(output / "feature_importances.npy", np.array(importances))
except Exception as e:
    print(f"Could not extract feature importances: {e}")
    # Fallback: save empty
    try:
        np.save(output / "feature_importances.npy", np.array([]))
    except Exception:
        pass

stats['metrics'] = {}
for part in X:
    p = predict(X[part])
    
    stats['metrics'][part] = lib.calculate_metrics(
        D.info['task_type'], Y[part], p, 'probs', y_info
    )
    np.save(output / f'p_{part}.npy', p)

stats['time'] = lib.format_seconds(timer())
lib.dump_stats(stats, output, True)
lib.backup_output(output)
