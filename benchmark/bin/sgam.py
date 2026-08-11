# %%
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
from sgam.models.sgam_core import SGAM

# %%
args, output = lib.load_config()

# %%
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
    cat_policy=args['data'].get('cat_policy', 'indices'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
if not isinstance(X, tuple):
    X = (X, None)

zero.random.seed(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')
X = tuple(None if x is None else lib.to_tensors(x) for x in X)
Y = lib.to_tensors(Y)
device = lib.get_device()
if device.type != 'cpu':
    X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
    Y_device = {k: v.to(device) for k, v in Y.items()}
else:
    Y_device = Y
X_num, X_cat = X
if not D.is_multiclass:
    Y_device = {k: v.float() for k, v in Y_device.items()}

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

n_num_features = 0 if X_num is None else X_num['train'].shape[1]
categories = lib.get_categories(X_cat)

model = SGAM(
    n_num_features=n_num_features,
    categories=categories,
    d_out=D.info['n_classes'] if D.is_multiclass else 1,
    **args['model'],
).to(device)

if X_num is not None and hasattr(model.tokenizer, 'num_tokenizer') and model.tokenizer.num_tokenizer is not None:
    model.tokenizer.num_tokenizer.set_thresholds_from_data(X_num['train'])

stats['n_parameters'] = lib.get_n_parameters(model)
optimizer = lib.make_optimizer(
    args['training']['optimizer'],
    model.parameters(),
    args['training']['lr'],
    args['training']['weight_decay'],
)

stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
progress = zero.ProgressTracker(args['training']['patience'])
training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
timer = zero.Timer()
checkpoint_path = output / 'checkpoint.pt'

def print_epoch_info():
    print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())} | {output}')
    print(
        ' | '.join(
            f'{k} = {v}'
            for k, v in {
                'lr': lib.get_lr(optimizer),
                'batch_size': batch_size,
                'epoch_size': stats['epoch_size'],
                'n_parameters': stats['n_parameters'],
            }.items()
        )
    )

@torch.no_grad()
def evaluate(parts):
    model.eval()
    metrics = {}
    predictions = {}
    for part in parts:
        predictions[part] = (
            torch.cat(
                [
                    model(
                        None if X_num is None else X_num[part][idx],
                        None if X_cat is None else X_cat[part][idx],
                    )
                    for idx in lib.IndexLoader(
                        D.size(part),
                        args['training']['eval_batch_size'],
                        False,
                        device,
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        if D.is_binclass:
            predictions[part] = predictions[part].squeeze(-1)
        metrics[part] = lib.calculate_metrics(
            D.info['task_type'],
            Y[part].numpy(),
            predictions[part],
            predictions['train'] if part != 'train' else None,
            y_info,
        )
    for part in parts:
        for k, v in metrics[part].items():
            print(f'({part}) {k}: {v:.4f}')
    return metrics, predictions

print_epoch_info()
evaluate([lib.VAL, lib.TEST])

for batch in stream:
    model.train()
    optimizer.zero_grad()
    idx = batch
    y_hat = model(
        None if X_num is None else X_num['train'][idx],
        None if X_cat is None else X_cat['train'][idx],
    )
    if not D.is_multiclass:
        y_hat = y_hat.squeeze(-1)
    loss = loss_fn(y_hat, Y_device['train'][idx])
    loss.backward()
    optimizer.step()

    if stream.is_epoch_finish:
        print_epoch_info()
        metrics, predictions = evaluate([lib.VAL, lib.TEST])
        val_score = metrics[lib.VAL]['score']
        test_score = metrics[lib.TEST]['score']

        training_log[lib.TRAIN].append({'loss': loss.item()})
        training_log[lib.VAL].append({'score': val_score})
        training_log[lib.TEST].append({'score': test_score})

        if progress.update(val_score):
            print('*** Best epoch ***')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            lib.dump_pickle(predictions, output / 'predictions.pickle')
            torch.save(model.state_dict(), checkpoint_path)

        if progress.stop:
            break

print('\n>>> Running evaluation on best checkpoint...')
model.load_state_dict(torch.load(checkpoint_path))
stats['metrics'], predictions = evaluate([lib.VAL, lib.TEST])
lib.dump_pickle(predictions, output / 'predictions.pickle')
stats['time'] = lib.format_seconds(timer())
lib.dump_json(stats, output / 'stats.json')
