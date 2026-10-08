"""Simple project commands. Running this file without arguments shows results.

No training is started unless you explicitly choose the `train` command.
"""
import argparse
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def result_table():
    """Combine existing summaries without recalculating or changing any result."""
    import pandas as pd
    tables = []
    sources = [
        ('original', 'test', RESULTS / 'original/outputs/results.csv'),
        ('tuned', 'test', RESULTS / 'tuned/comparison.csv'),
        ('preprocessing', 'validation', RESULTS / 'preprocessing/outputs/validation_comparison.csv'),
    ]
    for experiment, split, path in sources:
        frame = pd.read_csv(path).rename(columns={'variant': 'model'})
        frame['experiment'], frame['split'] = experiment, split
        if 'selected_on_validation' not in frame:
            frame['selected_on_validation'] = False
        tables.append(frame[['experiment', 'split', 'model', 'accuracy_percent',
                             'precision_percent', 'recall_percent', 'f1_percent', 'selected_on_validation']])
    return pd.concat(tables, ignore_index=True)


def verify():
    """Inference-only checks: load all four models and compare saved predictions."""
    import gc
    import joblib
    import numpy as np
    import model
    from ensemble import variant_scores, PAPER_WEIGHTS, weighted_scores
    tf = model.setup_tensorflow()
    original = RESULTS / 'original'
    revised = RESULTS / 'preprocessing'
    cases = [(original, name, 'test', f'{name}_predictions.npy')
             for name in ['google', 'fasttext', 'subword']]
    cases.append((revised, 'fasttext', 'val', 'preserved_fasttext_validation_probabilities.npy'))
    for directory, name, split, prediction_file in cases:
        with model.load_array(directory / 'outputs/prepared.npz') as data:
            inputs = {'question1_input': data[f'x1_{split}'][:64],
                      'question2_input': data[f'x2_{split}'][:64]}
        network = model.load_model(directory / f'models/{name}_malstm.keras')
        # Match the batch size and traced predict path used for saved scores.
        # Different GPU batch shapes can select different numerical kernels.
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        samples = tf.data.Dataset.from_tensor_slices(inputs).batch(64).with_options(options)
        actual = network.predict(samples, verbose=0).reshape(-1)
        expected = model.load_array(directory / 'outputs' / prediction_file)[:64]
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
        empty = np.zeros((1, 20), dtype='int32')
        np.testing.assert_allclose(network([empty, empty], training=False).numpy(), 1)
        print(f'OK: {directory.name}/{name} matches saved predictions; empty inputs supported.')
        del network
        tf.keras.backend.clear_session()
        gc.collect()
    scores = np.column_stack([model.load_array(original / f'outputs/{name}_predictions.npy')
                              for name in ['google', 'fasttext', 'subword']])
    np.testing.assert_array_equal(weighted_scores(scores, PAPER_WEIGHTS),
                                  model.load_array(original / 'outputs/ensemble_predictions.npy'))
    tuned = RESULTS / 'tuned'
    choices = model.read_json(tuned / 'frozen_validation_choices.json')
    estimators = joblib.load(tuned / 'fitted_meta_models.joblib')
    for name, settings in choices['variants'].items():
        actual = variant_scores(name, settings, estimators, scores)
        np.testing.assert_array_equal(actual, model.load_array(tuned / f'{name}_probabilities.npy'))
    print('OK: original blend and all seven tuned variants reproduce saved probabilities exactly.')
    print('No fitting, training, or model updates were performed.')


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='results',
                        choices=['results', 'verify', 'prepare', 'train', 'blend', 'tune', 'compare'])
    parser.add_argument('--run-dir', type=Path, help='Experiment folder; use a new one for new training.')
    parser.add_argument('--base-dir', type=Path, default=RESULTS / 'original')
    parser.add_argument('--variant', choices=['original', 'preserved'], default='original')
    parser.add_argument('--model', choices=['google', 'fasttext', 'subword', 'all'], default='fasttext')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--sample-size', type=int, help='Optional small preparation sample (original variant only).')
    parser.add_argument('--output-dir', type=Path, help='New folder for ensemble tuning.')
    args = parser.parse_args()
    if args.command == 'results':
        print(result_table().to_string(index=False, float_format=lambda v: f'{v:.2f}'))
        return
    if args.command == 'verify':
        verify()
        return
    import model
    directory = 'preprocessing' if args.variant == 'preserved' or args.command == 'compare' else 'original'
    run_dir = (args.run_dir or RESULTS / directory).resolve()
    if args.command == 'prepare':
        if args.variant == 'preserved':
            if args.sample_size:
                parser.error('Preserved preprocessing reuses the baseline split, not a new sample.')
            model.prepare_preserved(args.base_dir.resolve(), run_dir)
        else:
            model.prepare(model.DATA, run_dir, args.sample_size)
    elif args.command == 'train':
        if args.epochs < 1:
            parser.error('--epochs must be positive.')
        names = ['google', 'fasttext', 'subword'] if args.model == 'all' else [args.model]
        if len(names) == 1:
            model.train(names[0], run_dir, epochs=args.epochs)
        else:
            # Separate processes release each model's GPU/RAM allocations.
            for name in names:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), 'train',
                                '--model', name, '--run-dir', str(run_dir), '--epochs', str(args.epochs)], check=True)
    elif args.command == 'compare':
        model.compare_validation(args.base_dir.resolve(), run_dir)
    else:
        import ensemble
        if args.command == 'blend':
            ensemble.ensemble(run_dir)
        else:
            ensemble.tune(run_dir, (args.output_dir or RESULTS / 'tuned').resolve())


if __name__ == '__main__':
    cli()
