"""Original weighted blend and validation-only ensemble tuning.

Use main.py blend or main.py tune. All seven tuning variants, their grids,
tie-breaking rules, and validation split are unchanged from the prior code.
"""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, log_loss, brier_score_loss)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model import SEED, THRESHOLD, setup_tensorflow, evaluate, comparison, read_json, load_array


# 1. Paper blend: no additional training.

def ensemble(run_dir):
    out = run_dir / 'outputs'
    data = load_array(out / 'prepared.npz')
    expected_hash = read_json(out / 'preprocessing.json')['tokenizer_sha256']
    probabilities = []
    for name in ['google', 'fasttext', 'subword']:
        rows = load_array(out / f'{name}_test_rows.npy')
        metadata = read_json(out / f'{name}_training.json')
        if not np.array_equal(rows, data['rows_test']) or metadata['tokenizer_sha256'] != expected_hash:
            raise ValueError(f'{name}: predictions are from a different split/vocabulary.')
        p = load_array(out / f'{name}_predictions.npy')
        if p.shape != data['y_test'].shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f'{name}: invalid predictions.')
        probabilities.append(p)
    pg, pf, ps = probabilities
    final = 0.33 * pg + 0.33 * pf + 0.34 * ps
    np.save(out / 'ensemble_predictions.npy', final)
    pd.DataFrame({'source_row': data['rows_test'], 'is_duplicate': data['y_test'],
                  'google_probability': pg, 'fasttext_probability': pf, 'subword_probability': ps,
                  'final_probability': final, 'final_prediction': (final > THRESHOLD).astype('int32')
                  }).to_csv(out / 'ensemble_predictions.csv', index=False)
    evaluate('ensemble', data['y_test'], final, out)
    comparison(out)


# 2. Optional extensions: fit on validation only, then freeze choices.

NAMES = ['google', 'fasttext', 'subword']
PAPER_WEIGHTS = [0.33, 0.33, 0.34]
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)


def weighted_scores(scores, weights):
    """Keep the original blend's operation order, including float32 rounding."""
    g, f, s = [float(w) for w in weights]
    return np.clip(g * scores[:, 0] + f * scores[:, 1] + s * scores[:, 2], 0, 1)


def best_threshold(labels, scores, thresholds=THRESHOLDS):
    """Find accuracy for every threshold without building a huge N x T array.

    side='right' implements the strict rule: score == threshold predicts 0.
    On equal accuracy prefer the threshold closest to the original 0.5.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if not np.issubdtype(scores.dtype, np.floating):
        scores = scores.astype('float64')
    order = np.argsort(scores, kind='stable')
    positives = np.r_[0, np.cumsum(labels[order])]
    # Match NumPy's score_array > Python-float behavior for float32 outputs.
    boundaries = np.searchsorted(scores[order], np.asarray(thresholds, dtype=scores.dtype), side='right')
    correct = boundaries - positives[boundaries] + positives[-1] - positives[boundaries]
    winner = max(range(len(thresholds)), key=lambda i: (correct[i], -abs(thresholds[i] - 0.5)))
    return float(thresholds[winner]), float(correct[winner] / len(labels))


def search_blend(scores, labels, thresholds=THRESHOLDS):
    # Include the exact paper weights as well as a 0.05 simplex grid. Zero
    # weights allow a weak/redundant member to be excluded, based on validation.
    weights = [PAPER_WEIGHTS] + [[i / 20, j / 20, (20 - i - j) / 20]
                               for i in range(21) for j in range(21 - i)]
    trials = []
    for w in weights:
        threshold, accuracy = best_threshold(labels, weighted_scores(scores, w), thresholds)
        trials.append({'weights': w, 'threshold': threshold, 'selection_accuracy': accuracy})
    winner = max(trials, key=lambda r: (
        r['selection_accuracy'], -abs(r['threshold'] - .5),
        -np.linalg.norm(np.array(r['weights']) - PAPER_WEIGHTS)))
    return dict(winner), trials


def logits(scores):
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(models, method, scores):
    if method == 'sigmoid':
        return np.column_stack([m.predict_proba(logits(scores[:, i]))[:, 1]
                                for i, m in enumerate(models)])
    return np.column_stack([m.predict(scores[:, i]) for i, m in enumerate(models)])


def choose_variants(fit_scores, fit_labels, selection_scores, selection_labels):
    """No test arrays are accepted by this function."""
    baseline = weighted_scores(selection_scores, PAPER_WEIGHTS)
    variants = {'original_blend': {'kind': 'raw', 'weights': PAPER_WEIGHTS,
                'threshold': .5, 'selection_accuracy': float(accuracy_score(selection_labels, baseline > .5))}}
    threshold, accuracy = best_threshold(selection_labels, baseline)
    variants['threshold_only'] = {'kind': 'raw', 'weights': PAPER_WEIGHTS,
                                 'threshold': threshold, 'selection_accuracy': accuracy}
    searches, estimators = {}, {}
    for name, thresholds in [('weights_only', np.array([.5])), ('weights_and_threshold', THRESHOLDS)]:
        chosen, searches[name] = search_blend(selection_scores, selection_labels, thresholds)
        variants[name] = {'kind': 'raw', **chosen}
    for method in ['sigmoid', 'isotonic']:
        fitted = []
        for i in range(3):
            if method == 'sigmoid':
                # A sigmoid of the clipped log-odds: a standard calibration map.
                estimator = LogisticRegression(C=1e6, max_iter=2000, random_state=SEED)
                estimator.fit(logits(fit_scores[:, i]), fit_labels)
            else:
                estimator = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
                estimator.fit(fit_scores[:, i], fit_labels)
            fitted.append(estimator)
        name = f'{method}_blend'
        chosen, searches[name] = search_blend(calibrate(fitted, method, selection_scores), selection_labels)
        variants[name] = {'kind': method, **chosen}
        estimators[name] = fitted
    stacking_trials = []
    stackers = []
    for regularization in [.01, .1, 1., 10.]:
        estimator = make_pipeline(StandardScaler(), LogisticRegression(
            C=regularization, max_iter=2000, random_state=SEED))
        estimator.fit(fit_scores, fit_labels)
        threshold, accuracy = best_threshold(selection_labels, estimator.predict_proba(selection_scores)[:, 1])
        stackers.append(estimator)
        stacking_trials.append({'C': regularization, 'threshold': threshold, 'selection_accuracy': accuracy})
    # Stable tie: closest to 0.5, then stronger regularization.
    best = max(range(len(stacking_trials)), key=lambda i: (
        stacking_trials[i]['selection_accuracy'], -abs(stacking_trials[i]['threshold'] - .5),
        -stacking_trials[i]['C']))
    variants['stacking'] = {'kind': 'stacking', **stacking_trials[best]}
    estimators['stacking'] = stackers[best]
    searches['stacking'] = stacking_trials
    # Stable tie favors the earlier, simpler approach. Test metrics cannot
    # affect this winner: they are not loaded until after choices are saved.
    winner = max(variants, key=lambda name: variants[name]['selection_accuracy'])
    return variants, estimators, searches, winner


def variant_scores(name, settings, estimators, scores):
    if settings['kind'] == 'stacking':
        return estimators[name].predict_proba(scores)[:, 1]
    if settings['kind'] != 'raw':
        scores = calibrate(estimators[name], settings['kind'], scores)
    return weighted_scores(scores, settings['weights'])


def validation_predictions(run_dir, out):
    tf = setup_tensorflow()
    with load_array(run_dir / 'outputs/prepared.npz') as data:
        x1, x2 = data['x1_val'], data['x2_val']
        labels, rows = data['y_val'], data['rows_val']
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    ds = tf.data.Dataset.from_tensor_slices({
        'question1_input': x1, 'question2_input': x2,
    }).batch(256).with_options(options).prefetch(1)
    predictions = []
    for name in NAMES:
        print(f'Generating validation predictions: {name}', flush=True)
        network = tf.keras.models.load_model(run_dir / f'models/{name}_malstm.keras', compile=False)
        # Keras does not serialize use_cudnn. Restore the training-time kernel
        # so all-masked questions also work inside predict's traced graph.
        network.get_layer('shared_lstm').use_cudnn = False
        p = network.predict(ds, verbose=0).reshape(-1)
        predictions.append(p)
        del network
        tf.keras.backend.clear_session()
    scores = np.column_stack(predictions)
    check_scores(scores, len(labels))
    np.savez(out / 'validation_predictions.npz', scores=scores, labels=labels, source_rows=rows)
    return scores, labels, rows


def check_scores(scores, count):
    if scores.shape != (count, 3) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Expected aligned, finite N x 3 probabilities in [0, 1].')


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tune(run_dir, out):
    original = run_dir / 'outputs'
    metadata = read_json(original / 'preprocessing.json')
    # A fresh subdirectory prevents accidental replacement of earlier results.
    out.mkdir(parents=True, exist_ok=False)
    provenance = {name: fingerprint(run_dir / f'models/{name}_malstm.keras') for name in NAMES}
    scores, labels, rows = validation_predictions(run_dir, out)
    fit, select = train_test_split(np.arange(len(labels)), test_size=.5,
                                  random_state=SEED, stratify=labels)
    np.savez(out / 'meta_split.npz', fit_rows=rows[fit], selection_rows=rows[select])
    print(f'Meta fitting: {len(fit)} pairs; parameter selection: {len(select)} pairs.', flush=True)
    variants, estimators, searches, winner = choose_variants(
        scores[fit], labels[fit], scores[select], labels[select])
    frozen = {'objective': 'accuracy', 'seed': SEED, 'model_order': NAMES,
              'meta_fit_rows': len(fit), 'selection_rows': len(select),
              'selected_variant': winner, 'variants': variants, 'searches': searches,
              'model_sha256': provenance, 'tokenizer_sha256': metadata['tokenizer_sha256'],
              'threshold_grid': THRESHOLDS.tolist(), 'weight_grid_step': .05,
              'refit_after_selection': False,
              'note': 'Exploratory extension. Original test metrics were previously inspected; no test-based parameter search.'}
    (out / 'frozen_validation_choices.json').write_text(json.dumps(frozen, indent=2))
    joblib.dump(estimators, out / 'fitted_meta_models.joblib')
    print(f'Validation-selected approach: {winner}. Choices saved before test evaluation.', flush=True)

    # TEST EVALUATION STARTS HERE. Do not select a winner from this comparison.
    with load_array(original / 'prepared.npz') as data:
        test_labels, test_rows = data['y_test'], data['rows_test']
    if np.intersect1d(rows, test_rows).size:
        raise ValueError('Validation/test overlap detected.')
    for name in NAMES:
        if not np.array_equal(load_array(original / f'{name}_test_rows.npy'), test_rows):
            raise ValueError(f'{name}: test prediction row order mismatch.')
        training = read_json(original / f'{name}_training.json')
        if training['tokenizer_sha256'] != metadata['tokenizer_sha256']:
            raise ValueError(f'{name}: tokenizer mismatch.')
    test_scores = np.column_stack([load_array(original / f'{name}_predictions.npy') for name in NAMES])
    check_scores(test_scores, len(test_labels))
    baseline = weighted_scores(test_scores, PAPER_WEIGHTS)
    np.testing.assert_allclose(baseline, load_array(original / 'ensemble_predictions.npy'), atol=1e-7, rtol=1e-6)
    predictions = pd.DataFrame({'source_row': test_rows, 'is_duplicate': test_labels})
    results = []
    baseline_correct = (baseline > .5) == test_labels
    for name, settings in variants.items():
        probabilities = variant_scores(name, settings, estimators, test_scores)
        predicted = (probabilities > settings['threshold']).astype('int32')
        correct = predicted == test_labels
        result = {'variant': name, 'selected_on_validation': name == winner,
                  'selection_accuracy_percent': 100 * settings['selection_accuracy'],
                  'threshold': settings['threshold'],
                  'accuracy_percent': 100 * accuracy_score(test_labels, predicted),
                  'precision_percent': 100 * precision_score(test_labels, predicted, zero_division=0),
                  'recall_percent': 100 * recall_score(test_labels, predicted, zero_division=0),
                  'f1_percent': 100 * f1_score(test_labels, predicted, zero_division=0),
                  'log_loss': log_loss(test_labels, np.clip(probabilities, 1e-7, 1-1e-7)),
                  'brier_score': brier_score_loss(test_labels, probabilities),
                  'fixed_baseline_errors': int(np.sum(~baseline_correct & correct)),
                  'introduced_errors': int(np.sum(baseline_correct & ~correct))}
        result['accuracy_change_pp'] = result['accuracy_percent'] - 100 * baseline_correct.mean()
        results.append(result)
        predictions[f'{name}_probability'] = probabilities
        predictions[f'{name}_prediction'] = predicted
        np.save(out / f'{name}_probabilities.npy', probabilities)
        np.save(out / f'{name}_confusion_matrix.npy', confusion_matrix(test_labels, predicted, labels=[0, 1]))
        report = classification_report(test_labels, predicted, labels=[0, 1],
                                       target_names=['non_duplicate', 'duplicate'], zero_division=0)
        (out / f'{name}_classification_report.txt').write_text(report)
    table = pd.DataFrame(results)
    table.to_csv(out / 'comparison.csv', index=False)
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    print(table[['variant', 'selected_on_validation', 'threshold', 'accuracy_percent',
                 'precision_percent', 'recall_percent', 'f1_percent', 'accuracy_change_pp']]
          .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'Results: {out}', flush=True)

