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
from kan import KAN

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
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'ohe'), # OHE for pykan as it expects continuous inputs
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
# For pykan, we merge num and cat if they exist
X_merged = {}
for part in X:
    if isinstance(X[part], dict):
        # We assume X[part] is already merged by lib.Dataset if using OHE, but build_X might return a dict for num and cat
        pass

# Actually, if we use cat_policy='ohe', build_X returns merged numpy arrays if we don't separate them? 
# In sgam.py, it expects tuple. But let's check lib.Dataset. If it returns a tuple or dict.
# Usually lib.Dataset.build_X returns a dictionary {part: array} or a tuple of dicts (X_num, X_cat).
if isinstance(X, tuple):
    X_num, X_cat = X
    X_merged = {}
    for part in X_num:
        if X_num is not None and X_cat is not None:
            X_merged[part] = np.concatenate([X_num[part], X_cat[part]], axis=1)
        elif X_num is not None:
            X_merged[part] = X_num[part]
        elif X_cat is not None:
            X_merged[part] = X_cat[part]
else:
    X_merged = X

zero.random.seed(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')

X_tensors = {k: lib.to_tensors(v) for k, v in X_merged.items()}
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

loss_fn = (
    F.binary_cross_entropy_with_logits
    if D.is_binclass
    else F.cross_entropy
    if D.is_multiclass
    else F.mse_loss
)

n_features = X_merged['train'].shape[1]
d_out = D.info['n_classes'] if D.is_multiclass else 1

# KAN architecture: [input, hidden, output]
# By default, let's use [n_features, 64, d_out] or based on model config
hidden_dim = args.get('model', {}).get('hidden_dim', 64)
width = [n_features, hidden_dim, d_out]

model = KAN(width=width, grid=5, k=3, seed=args['seed']).to(device)

stats['n_parameters'] = sum(p.numel() for p in model.parameters())

optimizer = lib.make_optimizer(
    args['training']['optimizer'],
    model.parameters(),
    args['training']['lr'],
    args['training']['weight_decay'],
)

stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
progress = zero.ProgressTracker(args['training']['patience'])
training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
checkpoint_path = output / 'checkpoint.pt'

def evaluate(parts):
    metrics = {}
    predictions = {}
    # KAN library might not have .eval() or might not need it, but we call it safely
    if hasattr(model, 'eval'): model.eval()
    with torch.no_grad():
        for part in parts:
            preds = []
            for idx in lib.IndexLoader(D.size(part), args['training']['eval_batch_size'], False, device):
                batch_x = X_tensors[part][idx]
                out = model(batch_x)
                preds.append(out)
            preds = torch.cat(preds).cpu().numpy()
            if D.is_binclass:
                preds = preds.squeeze(-1)
            predictions[part] = preds
            
            metrics[part] = lib.calculate_metrics(
                D.info['task_type'],
                Y[part].numpy(),
                preds,
                predictions['train'] if part != 'train' else None,
                y_info,
            )
    return metrics, predictions

print("Training TabKAN...")
evaluate([lib.VAL, lib.TEST])

for batch in stream:
    if hasattr(model, 'train'): model.train()
    optimizer.zero_grad()
    idx = batch
    
    y_hat = model(X_tensors['train'][idx])
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
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), checkpoint_path)

        if progress.stop:
            break

print('\n>>> Evaluation on best checkpoint...')
if checkpoint_path.exists() and hasattr(model, 'load_state_dict'):
    model.load_state_dict(torch.load(checkpoint_path))
stats['metrics'], predictions = evaluate([lib.VAL, lib.TEST])
lib.dump_pickle(predictions, output / 'predictions.pickle')
stats['time'] = lib.format_seconds(timer())
lib.dump_json(stats, output / 'stats.json')
