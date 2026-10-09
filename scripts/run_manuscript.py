"""Train and evaluate the manuscript models using explicit file partitions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ['DA_LGBM', 'ADANN', 'LightGBM', 'DSCNN', '1D_CNN', 'XGBoost',
          'Transformer_Encoder', 'ADANN_GRL', 'ADANN_MMD', 'ADANN_CORAL']


def parameters(model, fold, epochs):
    protocol = json.loads((ROOT / 'configs/manuscript_protocol.json').read_text())
    if model == 'DA_LGBM':
        params = json.loads((ROOT / 'configs/da_lgbm_loso.json').read_text())[str(fold)]
    else:
        mapping = json.loads((ROOT / 'configs/baseline_configs.json').read_text())[model]
        path = mapping['fold_parameter_files'].get(str(fold))
        if path:
            payload = json.loads((ROOT / path).read_text())
            params = payload.get('best_params', payload)
        else:
            raise ValueError(f'No saved parameter file for {model}, fold {fold}')
    params.update(protocol['augmentation'])
    params.pop('training_only', None)
    params.update(class_balanced_batches=True, auto_tune_gate_thresholds=False,
                  adann_conf_threshold=0.5, lgb_conf_threshold=0.5,
                  adann_feature_size=64)
    params['adann_epochs'] = min(int(params.get('adann_epochs', epochs)), epochs)
    params['n_epochs'] = min(int(params.get('n_epochs', epochs)), epochs)
    return params


def train(args):
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from reproducibility.splits import partition_indices
    from scripts.export_reproducibility import load_windows
    from experiments.awgn_robustness import (
        augment_with_subjects, train_one_fold, predict_labels, set_seeds,
    )

    files, labels, subjects, windows = load_windows(ROOT / 'datasets/gesture_csv')
    folds = args.folds or list(range(1, 7))
    records = []
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for fold in (folds if args.evaluation == 'loso' else ['IID']):
        manifest = ROOT / 'reproducibility/generated' / ('loso_seed42.csv' if args.evaluation == 'loso' else 'iid_seed42.csv')
        tr, va, te = partition_indices(manifest, files, fold)
        set_seeds(args.seed)
        params = parameters(args.model, int(fold) if fold != 'IID' else 1, args.epochs)
        Xtr, ytr, str_ = augment_with_subjects(windows[tr], labels[tr], subjects[tr], params)
        model_type = 'ADANN_LightGBM' if args.model == 'DA_LGBM' else args.model
        if model_type in {'ADANN_LightGBM', 'ADANN', 'LightGBM', 'DSCNN'}:
            creator, trained, scaler = train_one_fold(
                model_type, params, Xtr, ytr, str_, windows[va], labels[va],
                subjects[va], args.epochs, 'torch')
            predicted = predict_labels(model_type, creator, trained, windows[te], scaler)
        elif model_type.startswith('ADANN_'):
            from src.training.train_adann_domain_compare import AdannDomainCompareModelCreator
            mode = model_type.split('_', 1)[1].lower()
            params['domain_alignment_method'] = mode
            creator = AdannDomainCompareModelCreator(default_method=mode)
            wrapper = creator.create_model(params, arduino_mode=False)
            trained, _ = creator.train_model(wrapper.pytorch_model, Xtr, ytr, str_,
                windows[va], labels[va], subjects[va], params, return_history=False)
            predicted = creator.predict(trained, windows[te])
        else:
            from importlib import import_module
            modules = {'1D_CNN': ('train_cnn1d', 'Cnn1dModelCreator'),
                       'XGBoost': ('train_xgboost', 'XgboostModelCreator'),
                       'Transformer_Encoder': ('train_transformer', 'TransformerModelCreator')}
            module, cls = modules[model_type]
            creator = getattr(import_module('src.training.' + module), cls)()
            Xtr_f, scaler = creator.extract_and_scale_features(Xtr, fit=True)
            Xva_f, _ = creator.extract_and_scale_features(windows[va], scaler=scaler)
            Xte_f, _ = creator.extract_and_scale_features(windows[te], scaler=scaler)
            trained = creator.create_model(params, arduino_mode=False)
            if model_type == 'XGBoost':
                trained.fit(Xtr_f, ytr, eval_set=[(Xva_f, labels[va])], verbose=False)
            else:
                trained.fit(Xtr_f, ytr, validation_data=(Xva_f, labels[va]),
                            epochs=args.epochs, verbose=0)
            values = np.asarray(trained.predict(Xte_f))
            predicted = values.argmax(axis=1) if values.ndim == 2 else values
        predicted = np.asarray(predicted, dtype=int)
        cm = confusion_matrix(labels[te], predicted, labels=np.arange(11))
        np.savetxt(output / f'{args.model}_fold{fold}_seed{args.seed}_confusion_matrix.csv', cm, delimiter=',', fmt='%d')
        with (output / f'{args.model}_fold{fold}_seed{args.seed}_predictions.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['filename', 'true_label', 'predicted_label'])
            writer.writerows((files[i], int(labels[i]), int(p)) for i, p in zip(te, predicted))
        records.append({'fold':fold, 'seed':args.seed,
                        'accuracy':float(accuracy_score(labels[te], predicted)),
                        'macro_f1':float(f1_score(labels[te], predicted, labels=np.arange(11), average='macro', zero_division=0))})
        (output / f'{args.model}_fold{fold}_seed{args.seed}_parameters.json').write_text(json.dumps(params, indent=2))
    result = {'model':args.model, 'evaluation':args.evaluation, 'records':records,
              'mean_macro_f1':float(np.mean([r['macro_f1'] for r in records])),
              'mean_accuracy':float(np.mean([r['accuracy'] for r in records]))}
    (output / f'{args.model}_seed{args.seed}_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare', help='Generate feature arrays and file partitions')
    p = sub.add_parser('train', help='Train one model with manifest-defined partitions')
    p.add_argument('--model', choices=MODELS, default='DA_LGBM')
    p.add_argument('--evaluation', choices=['iid','loso'], default='loso')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--folds', nargs='+', type=int, choices=range(1,7))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--output_dir', default=str(ROOT / 'outputs/manuscript'))
    sub.add_parser('seeds', help='Run the five seeds with fixed LOSO partitions')
    p = sub.add_parser('robustness', help='Run controlled held-out-window disturbances')
    p.add_argument('experiment', choices=['awgn','jitter','drift'])
    args, extra = parser.parse_known_args()
    if args.action == 'prepare':
        command = [sys.executable, str(ROOT / 'scripts/export_reproducibility.py'), *extra]
    elif args.action == 'seeds':
        command = [sys.executable, str(ROOT / 'experiments/da_lgbm_seed_stability_loso.py'),
                   '--csv_dir', str(ROOT / 'datasets/gesture_csv'),
                   '--seeds', '42,123,2025,2026,3047', '--selection_mode', 'gated',
                   '--no_train_on_full_fold', '--fixed_hyperparams_path', str(ROOT / 'configs/da_lgbm_loso.json'),
                   '--split_manifest', str(ROOT / 'reproducibility/generated/loso_seed42.csv'),
                   '--output_dir', str(ROOT / 'outputs/manuscript/seeds'), *extra]
    elif args.action == 'robustness':
        command = [sys.executable, str(ROOT / 'scripts/run_robustness.py'), args.experiment, *extra]
    else:
        if extra:
            parser.error('Unrecognised training arguments: ' + ' '.join(extra))
        return train(args)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
