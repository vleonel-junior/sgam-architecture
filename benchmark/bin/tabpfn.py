from copy import deepcopy
from pathlib import Path
import warnings

import numpy as np
import zero
from tabpfn import TabPFNClassifier
try:
    from tabpfn import TabPFNRegressor
except ImportError:
    TabPFNRegressor = None

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
model_kwargs = deepcopy(args.get("model", {}))
model_kwargs.pop('random_state', None)  # TabPFN might not accept random_state natively depending on version

# Create model
if D.is_regression:
    if TabPFNRegressor is None:
        print("TabPFNRegressor not available in this version. Skipping.")
        import sys; sys.exit(0)
    model = TabPFNRegressor(**model_kwargs)
    predict = model.predict
else:
    model = TabPFNClassifier(**model_kwargs)
    if D.is_multiclass:
        predict = model.predict_proba
    else:
        predict = lambda x: model.predict_proba(x)[:, 1]

# Fit model
timer = zero.Timer()
timer.run()
print("Training TabPFN...")
# TabPFN is generally limited in size. If dataset is huge, this might fail or OOM.
# We assume the benchmark config passes subsets or appropriate datasets.
model.fit(X[lib.TRAIN], Y[lib.TRAIN])

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
