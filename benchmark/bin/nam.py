import math
import sys
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero

project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark import lib
try:
    from nam.config import defaults
    from nam.models.nam import NAM
    _NAM_AVAILABLE = True
except ImportError:
    try:
        from nam.models import NAM
        _NAM_AVAILABLE = True
    except ImportError:
        NAM = None
        _NAM_AVAILABLE = False

args, output = lib.load_config()

zero.random.seed(args['seed'])
dataset_dir = lib.get_path(args['data']['path'])
stats: ty.Dict[str, ty.Any] = {
    'dataset': dataset_dir.name,
    'algorithm': Path(__file__).stem,
    **lib.load_json(output / 'stats.json'),
}
timer = zero.Timer()
timer.run()

D = lib.Dataset.from_dir(dataset_dir)
X = D.build_X(
    normalization=args['data'].get('normalization', 'quantile'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'ohe'), # ohe for NAM as it handles continuous features well
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)

# Merge X_num and X_cat for NAM (needs flat continuous input)
if isinstance(X, tuple):
    X_num, X_cat = X
    X_merged = {}
    available_parts = X_num if X_num is not None else X_cat
    for part in available_parts:
        if X_num is not None and X_cat is not None:
            X_merged[part] = np.concatenate([X_num[part], X_cat[part]], axis=1)
        elif X_num is not None:
            X_merged[part] = X_num[part]
        else:
            X_merged[part] = X_cat[part]
else:
    X_merged = X

zero.random.seed(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')

X_tensors = lib.to_tensors(X_merged)
Y_tensors = lib.to_tensors(Y)
device = lib.get_device()

if device.type != 'cpu':
    X_tensors = {k: v.to(device) for k, v in X_tensors.items()}
    Y_tensors = {k: v.to(device) for k, v in Y_tensors.items()}

if not D.is_multiclass:
    Y_tensors = {k: v.float() for k, v in Y_tensors.items()}

train_size = D.size(lib.TRAIN)
batch_size = args['training']['batch_size']
epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)

n_features = X_merged['train'].shape[1]
d_out = D.info['n_classes'] if D.is_multiclass else 1

if not _NAM_AVAILABLE:
    print("NAM is not properly installed. Skipping.")
    sys.exit(0)

print("Building NAM Model...")
# NAM: one sub-network per feature
model_kwargs = args.get('model', {})
nam_hidden = model_kwargs.get('hidden_sizes', [64, 64])
nam_dropout = model_kwargs.get('dropout', 0.1)

model = NAM(
    config=None,
    name='NAM',
    num_inputs=n_features,
    num_units=nam_hidden[0] if isinstance(nam_hidden, list) else nam_hidden,
)

model = model.to(device)
stats['n_parameters'] = sum(p.numel() for p in model.parameters())

optimizer = lib.make_optimizer(
    args['training']['optimizer'],
    model.parameters(),
    args['training']['lr'],
    args['training']['weight_decay'],
)

loss_fn = (
    F.binary_cross_entropy_with_logits if D.is_binclass
    else F.cross_entropy if D.is_multiclass
    else F.mse_loss
)

stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
progress = zero.ProgressTracker(args['training']['patience'])
training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
checkpoint_path = output / 'checkpoint.pt'

def evaluate(parts):
    metrics = {}
    predictions = {}
    model.eval()
    with torch.no_grad():
        for part in parts:
            preds = []
            for idx in lib.IndexLoader(D.size(part), args['training']['eval_batch_size'], False, device):
                batch_x = X_tensors[part][idx]
                out = model(batch_x)
                # NAM can return (logits, ann_out) or just logits depending on version
                if isinstance(out, tuple):
                    out = out[0]
                preds.append(out)
            preds = torch.cat(preds).cpu().numpy()
            if D.is_binclass:
                preds = preds.squeeze(-1)
            predictions[part] = preds
            
            metrics[part] = lib.calculate_metrics(
                D.info['task_type'],
                Y[part],  # Y[part] is already np.ndarray
                preds,
                'logits' if not D.is_regression else 'probs',
                y_info,
            )
    return metrics, predictions

print("Training NAM...")
evaluate([lib.VAL, lib.TEST])

for batch in stream:
    model.train()
    optimizer.zero_grad()
    idx = batch
    
    y_hat = model(X_tensors['train'][idx])
    if isinstance(y_hat, tuple):
        y_hat = y_hat[0]
    if not D.is_multiclass:
        y_hat = y_hat.squeeze(-1)
        
    loss = loss_fn(y_hat, Y_tensors['train'][idx])
    loss.backward()
    optimizer.step()

    if stream.is_epoch_finish:
        metrics, predictions = evaluate([lib.VAL, lib.TEST])
        val_score = metrics[lib.VAL]['score']
        test_score = metrics[lib.TEST]['score']

        training_log[lib.TRAIN].append({'loss': loss.item()})
        training_log[lib.VAL].append({'score': val_score})
        training_log[lib.TEST].append({'score': test_score})

        if progress.update(val_score):
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            lib.dump_pickle(predictions, output / 'predictions.pickle')
            torch.save(model.state_dict(), checkpoint_path)

        if progress.stop:
            break

print('\n>>> Evaluation on best checkpoint...')
if checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path))
stats['metrics'], predictions = evaluate([lib.VAL, lib.TEST])
lib.dump_pickle(predictions, output / 'predictions.pickle')
stats['time'] = lib.format_seconds(timer())
lib.dump_json(stats, output / 'stats.json')
