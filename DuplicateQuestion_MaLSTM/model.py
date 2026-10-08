"""MaLSTM in one place: settings, text preparation, embeddings, model, and training.

Read the numbered sections from top to bottom. Use main.py to run commands.
Saved diagnostics live in details.zip; the small readers below open them
directly, so inspecting existing results never requires extracting everything.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import io
import json
import mmap
import re
import struct
import unicodedata
import zipfile

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models.fasttext import ft_ngram_hashes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, ConfusionMatrixDisplay)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Settings: paper values and our documented implementation choices.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'quora_question_pairs.csv'
EMBEDDINGS = ROOT / 'embeddings'
MAX_LENGTH = 20  # Paper.
DIM = 300  # Paper: separate 300-dimensional experiments.
THRESHOLD = 0.5  # Paper: strictly greater than this threshold.
EPOCHS = 30  # Inferred from paper plots, not explicitly specified in text.
# Implementation choices: these are not specified by the paper.
SEED = 42
UNITS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.10
FILES = {
    'google': 'GoogleNews-vectors-negative300.bin',
    'fasttext': 'crawl-300d-2M.vec',
    'subword': 'fasttext_subword.bin',
}
REFERENCE = {
    'google': [81.77, 78.77, 69.93, 74.09],
    'fasttext': [82.77, 79.20, 70.58, 74.64],
    'subword': [82.57, 78.26, 70.29, 74.04],
    'ensemble': [91.14, 83.68, 81.17, 82.41],
}

RESULTS = ROOT / 'results'
_TF_READY = False


def setup_tensorflow():
    """Keep GPU allocations incremental and make training reproducible."""
    global _TF_READY
    if _TF_READY:
        return tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    _TF_READY = True
    return tf




def read_bytes(path):
    """Read a visible file or its archived copy in the same folder."""
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    with zipfile.ZipFile(path.parent / 'details.zip') as archive:
        return archive.read(path.name)


def read_json(path):
    return json.loads(read_bytes(path))


def load_array(path):
    path = Path(path)
    return np.load(path if path.exists() else io.BytesIO(read_bytes(path)), allow_pickle=False)


def load_model(path):
    """Load existing weights; retain native LSTM support for empty questions."""
    network = keras.models.load_model(path, compile=False)
    network.get_layer('shared_lstm').use_cudnn = False
    return network


# 2. Text preparation: original recipe and the separate preservation experiment.

def prepare(data_path, run_dir, sample_size=None):
    """Fit on both TRAINING question columns to avoid test-vocabulary leakage.

    Missing questions become empty strings. Porter stemming, alphabetic word
    tokenization, post-padding/truncation, and an OOV token are our choices.
    The held-out validation subset is also excluded from vocabulary fitting.
    """
    out = run_dir / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'prepared.npz').exists():
        raise FileExistsError(f'{out} already has prepared data. Use a new --run-dir.')
    df = pd.read_csv(data_path, usecols=['question1', 'question2', 'is_duplicate'])
    if not df.is_duplicate.isin([0, 1]).all():
        raise ValueError('is_duplicate must contain only 0 and 1, without missing labels.')
    source_rows = len(df)
    if sample_size is not None and sample_size < len(df):
        selected, _ = train_test_split(
            np.arange(len(df)), train_size=sample_size,
            random_state=SEED, stratify=df.is_duplicate,
        )
        df = df.iloc[selected].copy()
    row_ids = df.index.to_numpy()
    df = df.reset_index(drop=True)
    y = df.is_duplicate.to_numpy(dtype=np.int32)
    train, test = train_test_split(np.arange(len(df)), test_size=0.25,
                                   random_state=SEED, stratify=y)
    # Exactly the final 10% convention used by Keras validation_split=0.10.
    boundary = int(len(train) * (1 - VALIDATION_SPLIT))
    fit, val = train[:boundary], train[boundary:]
    try:
        stops = set(stopwords.words('english'))
    except LookupError:
        raise RuntimeError('Install NLTK data: python -m nltk.downloader stopwords') from None
    stemmer = PorterStemmer()

    @lru_cache(maxsize=100_000)
    def clean(text):
        return ' '.join(stemmer.stem(word) for word in re.findall(r'[a-z]+', text.lower())
                        if word not in stops)

    q1 = df.question1.fillna('').astype(str).map(clean).tolist()
    q2 = df.question2.fillna('').astype(str).map(clean).tolist()
    tokenizer = Tokenizer(oov_token='<OOV>', filters='', lower=False)
    tokenizer.fit_on_texts([q1[i] for i in fit] + [q2[i] for i in fit])
    x1, x2 = [pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LENGTH,
                            padding='post', truncating='post', dtype='int32')
              for texts in (q1, q2)]
    arrays = {}
    for name, indices in [('train', fit), ('val', val), ('test', test)]:
        arrays.update({f'x1_{name}': x1[indices], f'x2_{name}': x2[indices],
                       f'y_{name}': y[indices], f'rows_{name}': row_ids[indices]})
    np.savez(out / 'prepared.npz', **arrays)
    tokenizer_text = tokenizer.to_json()
    (out / 'tokenizer.json').write_text(tokenizer_text)
    df.iloc[test].assign(source_row=row_ids[test]).to_csv(out / 'test_pairs.csv', index=False)
    manifest = {
        'data_path': str(data_path.resolve()), 'source_rows': source_rows,
        'used_rows': len(df), 'train_rows_before_validation': len(train),
        'fit_rows': len(fit), 'validation_rows': len(val), 'test_rows': len(test),
        'vocabulary_size_including_padding': len(tokenizer.word_index) + 1,
        'seed': SEED, 'sequence_length': MAX_LENGTH,
        'tokenizer_fit': 'both question columns of fit subset only',
        'tokenizer_sha256': hashlib.sha256(tokenizer_text.encode()).hexdigest(),
        'stopwords': sorted(stops), 'nltk_version': nltk.__version__,
        'smoke_sample': sample_size,
    }
    (out / 'preprocessing.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'stopwords'}, indent=2))


TOKEN_PATTERN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)*|[^\w\s]|_", re.UNICODE)
NEGATION = {'no', 'nor', 'not', 'never', 'neither', 'without'}


def make_cleaner():
    stops = set(stopwords.words('english')) - NEGATION
    stemmer = PorterStemmer()

    @lru_cache(maxsize=100_000)
    def clean(text):
        # NFC preserves accents; normalize curly apostrophes for contractions.
        text = unicodedata.normalize('NFC', text).lower().replace('’', "'")
        for contraction, expanded in [("can't", 'can not'), ('cannot', 'can not'),
                                      ("won't", 'will not'), ("shan't", 'shall not'),
                                      ("ain't", 'is not')]:
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text)
        text = re.sub(r"n't\b", ' not', text)
        words = []
        for token in TOKEN_PATTERN.findall(text):
            if token in stops:
                continue
            # Keep Porter stemming for English words; do not stem numbers,
            # punctuation, or Unicode words with an English-only stemmer.
            words.append(stemmer.stem(token) if token.isascii() and token.isalpha() else token)
        return ' '.join(words)

    return clean, stops


def collision_counts(x1, x2, labels):
    identical = np.all(x1 == x2, axis=1)
    return {'pairs': len(labels), 'identical_input_pairs': int(identical.sum()),
            'identical_nonduplicate_pairs': int(np.sum(identical & (labels == 0))),
            'empty_question1': int(np.all(x1 == 0, axis=1).sum()),
            'empty_question2': int(np.all(x2 == 0, axis=1).sum())}


def prepare_preserved(base_dir, run_dir):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    out = run_dir / 'outputs'
    if (out / 'prepared.npz').exists():
        raise FileExistsError('Prepared data already exists. Use a new experiment directory.')
    out.mkdir(parents=True, exist_ok=True)
    baseline_meta = read_json(base_dir / 'outputs/preprocessing.json')
    data_path = baseline_meta['data_path']
    df = pd.read_csv(data_path, usecols=['question1', 'question2', 'is_duplicate'])
    if len(df) != baseline_meta['source_rows']:
        raise ValueError('Source dataset row count differs from the original run.')
    with load_array(base_dir / 'outputs/prepared.npz') as baseline:
        rows = {split: baseline[f'rows_{split}'] for split in ['train', 'val', 'test']}
        for split, indices in rows.items():
            np.testing.assert_array_equal(df.is_duplicate.to_numpy()[indices], baseline[f'y_{split}'])
        original_audit = {split: collision_counts(baseline[f'x1_{split}'], baseline[f'x2_{split}'],
                                                  baseline[f'y_{split}']) for split in ['train', 'val']}
    clean, stops = make_cleaner()
    # Process only rows from this experiment; this also supports the smoke split.
    used_rows = np.concatenate(list(rows.values()))
    question1 = df.loc[used_rows, 'question1'].fillna('').astype(str).map(clean)
    question2 = df.loc[used_rows, 'question2'].fillna('').astype(str).map(clean)
    tokenizer = Tokenizer(oov_token='<OOV>', filters='', lower=False)
    tokenizer.fit_on_texts(question1.loc[rows['train']].tolist() + question2.loc[rows['train']].tolist())
    arrays, audit = {}, {}
    for split, indices in rows.items():
        for key, questions in [('x1', question1), ('x2', question2)]:
            sequences = tokenizer.texts_to_sequences(questions.loc[indices].tolist())
            arrays[f'{key}_{split}'] = pad_sequences(sequences, maxlen=MAX_LENGTH,
                                                    padding='post', truncating='post', dtype='int32')
            if split != 'test':
                audit[f'{split}_{key}_truncated_questions'] = sum(len(s) > MAX_LENGTH for s in sequences)
        arrays[f'y_{split}'] = df.loc[indices, 'is_duplicate'].to_numpy(dtype='int32')
        arrays[f'rows_{split}'] = indices
        if split != 'test':
            audit[split] = {'original': original_audit[split], 'preserved': collision_counts(
                arrays[f'x1_{split}'], arrays[f'x2_{split}'], arrays[f'y_{split}'])}
    # Test sequences are prepared, but neither test metrics nor test collisions
    # are inspected by this experiment. Selection uses validation only.
    np.savez(out / 'prepared.npz', **arrays)
    tokenizer_text = tokenizer.to_json()
    (out / 'tokenizer.json').write_text(tokenizer_text)
    manifest = {**baseline_meta, 'baseline_directory': str(base_dir.resolve()),
                'preprocessing_variant': 'preserve_numbers_negation_unicode_symbols',
                'vocabulary_size_including_padding': len(tokenizer.word_index) + 1,
                'tokenizer_sha256': hashlib.sha256(tokenizer_text.encode()).hexdigest(),
                'stopwords': sorted(stops), 'retained_negation': sorted(NEGATION),
                'unicode_normalization': 'NFC', 'token_pattern': TOKEN_PATTERN.pattern,
                'unchanged': ['saved row split', 'length 20', 'post padding/truncation',
                              'training-only vocabulary', 'Porter stemming of ASCII words'],
                'test_evaluation': 'deferred'}
    (out / 'preprocessing.json').write_text(json.dumps(manifest, indent=2))
    (out / 'preprocessing_comparison.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


# 3. Pretrained embeddings: stream only the needed vocabulary.

def read_subword(path, word_index, matrix):
    """Read unquantized native fastText; average word and character n-grams.

    Memory mapping avoids loading the multi-GB input/output matrices. Gensim's
    hash routine implements fastText's UTF-8 n-gram hashing, including OOV words.
    Only IDs for our vocabulary are retained from the source dictionary.
    """
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as source:
        magic, version = struct.unpack_from('<2i', source, 0)
        if magic != 793712314 or version != 12:
            raise ValueError('Expected a native fastText version-12 .bin file.')
        args = struct.unpack_from('<12id', source, 8)
        dim, bucket, minn, maxn = args[0], args[8], args[9], args[10]
        offset = 8 + struct.calcsize('<12id')
        size, nwords, nlabels, ntokens, prune = struct.unpack_from('<3i2q', source, offset)
        offset += struct.calcsize('<3i2q')
        if dim != DIM or nlabels != 0 or prune >= 0 or bucket <= 0:
            raise ValueError('Expected an unpruned, unsupervised 300d subword model.')
        known = {}
        for index in range(size):
            end = source.find(b'\0', offset)
            if end < 0:
                raise ValueError('Truncated fastText dictionary.')
            word = source[offset:end].decode('utf-8', errors='backslashreplace')
            if word in word_index:
                known[word] = index
            offset = end + 1 + 9  # int64 count and int8 entry type.
        quantized = struct.unpack_from('<?', source, offset)[0]
        rows, columns = struct.unpack_from('<2q', source, offset + 1)
        offset += 17
        if quantized or columns != DIM or rows != nwords + bucket:
            raise ValueError('Unsupported or corrupt fastText input matrix.')
        vectors = np.ndarray((rows, columns), dtype='<f4', buffer=source, offset=offset)
        covered = 0
        for word, index in word_index.items():
            if word == '<OOV>':
                continue
            ids = [nwords + h for h in ft_ngram_hashes(word, minn, maxn, bucket)]
            if word in known:
                ids.insert(0, known[word])
            if ids:
                matrix[index] = vectors[ids].mean(axis=0)
                covered += 1
        del vectors  # Release the view before closing its mmap.
    return covered


def build_embedding(kind, word_index, embedding_dir, out):
    path = embedding_dir / FILES[kind]
    print(f'Extracting {kind} vectors from {path}', flush=True)
    # Fixed random OOV vectors preserve unknown-word distinctions; padding is 0.
    # This OOV initialization is an implementation choice, not a paper setting.
    matrix = np.random.default_rng(SEED).normal(0, 0.05, (len(word_index) + 1, DIM)).astype('float32')
    matrix[0] = 0
    covered = 0
    if kind == 'subword':
        covered = read_subword(path, word_index, matrix)
    elif kind == 'fasttext':
        with open(path, encoding='utf-8') as handle:
            count, dim = map(int, handle.readline().split())
            if dim != DIM:
                raise ValueError(f'Expected {DIM} dimensions, found {dim}')
            for line in handle:
                word, _, values = line.rstrip().partition(' ')
                if word in word_index:
                    vector = np.fromstring(values, sep=' ', dtype='float32')
                    if vector.shape != (DIM,):
                        raise ValueError(f'Malformed vector for {word!r}')
                    matrix[word_index[word]] = vector
                    covered += 1
    else:
        # GoogleNews word2vec binary: token, space, 300 little-endian float32s.
        with open(path, 'rb') as handle:
            count, dim = map(int, handle.readline().split())
            if dim != DIM:
                raise ValueError(f'Expected {DIM} dimensions, found {dim}')
            for _ in range(count):
                token = bytearray()
                while True:
                    char = handle.read(1)
                    if not char:
                        raise ValueError('Truncated Word2Vec file.')
                    if char == b' ':
                        break
                    if char != b'\n':
                        token.extend(char)
                word = token.decode('utf-8', errors='replace')
                values = handle.read(4 * DIM)
                if len(values) != 4 * DIM:
                    raise ValueError('Truncated Word2Vec vector.')
                if word in word_index:
                    matrix[word_index[word]] = np.frombuffer(values, dtype='<f4')
                    covered += 1
    np.save(out / f'{kind}_embedding.npy', matrix)
    stats = {'matched_or_synthesized': covered, 'vocabulary_tokens': len(word_index),
             'source': str(path.resolve()), 'source_size': path.stat().st_size,
             'source_mtime_ns': path.stat().st_mtime_ns,
             'oov_initialization': 'Normal(0, 0.05), seed 42; zero padding'}
    (out / f'{kind}_embedding.json').write_text(json.dumps(stats, indent=2))
    print(f'{kind}: {covered}/{len(word_index)} vectors matched or synthesized.', flush=True)
    return matrix


# 4. Shared Siamese LSTM and exponential Manhattan similarity.

@keras.utils.register_keras_serializable(package='MaLSTM')
class ManhattanSimilarity(keras.layers.Layer):
    """Serializable alternative to an anonymous Lambda layer."""
    def call(self, inputs):
        h1, h2 = inputs
        return tf.exp(-tf.reduce_sum(tf.abs(h1 - h2), axis=1, keepdims=True))


def build_model(embedding_matrix):
    if embedding_matrix.ndim != 2 or embedding_matrix.shape[1] != DIM:
        raise ValueError('The embedding matrix must have shape (vocabulary_size, 300).')
    q1 = keras.Input(shape=(MAX_LENGTH,), dtype='int32', name='question1_input')
    q2 = keras.Input(shape=(MAX_LENGTH,), dtype='int32', name='question2_input')
    embedding = keras.layers.Embedding(
        len(embedding_matrix), DIM, weights=[embedding_matrix], trainable=False,
        mask_zero=True, name='shared_embedding',
    )
    # The native implementation accepts entirely masked (empty) questions.
    # cuDNN rejects those sequences, so use the native kernel explicitly.
    lstm = keras.layers.LSTM(UNITS, dropout=0, recurrent_dropout=0,
                             use_cudnn=False, name='shared_lstm')
    similarity = ManhattanSimilarity(name='manhattan_similarity')([
        lstm(embedding(q1)), lstm(embedding(q2)),
    ])
    model = keras.Model([q1, q2], similarity, name='siamese_malstm')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5)])
    return model


# 5. Evaluation and one reusable training loop.

def evaluate(name, labels, probabilities, out):
    probabilities = np.asarray(probabilities).reshape(-1)
    if len(probabilities) != len(labels) or not np.isfinite(probabilities).all():
        raise ValueError('Predictions must be finite and aligned with labels.')
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Probabilities must lie in [0, 1].')
    predicted = (probabilities > THRESHOLD).astype('int32')
    metrics = {'model': name, 'accuracy': accuracy_score(labels, predicted),
               'precision': precision_score(labels, predicted, zero_division=0),
               'recall': recall_score(labels, predicted, zero_division=0),
               'f1': f1_score(labels, predicted, zero_division=0)}
    report = classification_report(labels, predicted, labels=[0, 1],
                                    target_names=['non_duplicate', 'duplicate'], zero_division=0)
    (out / f'{name}_classification_report.txt').write_text(report)
    (out / f'{name}_metrics.json').write_text(json.dumps(metrics, indent=2))
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    np.save(out / f'{name}_confusion_matrix.npy', cm)
    ConfusionMatrixDisplay(cm, display_labels=['non_duplicate', 'duplicate']).plot()
    plt.title(name)
    plt.tight_layout()
    plt.savefig(out / f'{name}_confusion_matrix.png', dpi=150)
    plt.close()
    print(f'\n{name}\n{report}', flush=True)
    return metrics


def comparison(out):
    rows = []
    for name, reference in REFERENCE.items():
        try:
            row = read_json(out / f'{name}_metrics.json')
        except (FileNotFoundError, KeyError):
            continue
        for metric, paper in zip(['accuracy', 'precision', 'recall', 'f1'], reference):
            row[f'{metric}_percent'] = 100 * row.pop(metric)
            row[f'paper_{metric}_percent'] = paper
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'results.csv', index=False)
    print(table.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    return table


def dataset(tf, data, split, training=False):
    ds = tf.data.Dataset.from_tensor_slices((
        {'question1_input': data[f'x1_{split}'], 'question2_input': data[f'x2_{split}']},
        data[f'y_{split}'].astype('float32')[:, None]))
    if training:
        ds = ds.shuffle(len(data[f'y_{split}']), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.batch(BATCH_SIZE).with_options(options).prefetch(1)


def train(kind, run_dir, embedding_dir=EMBEDDINGS, epochs=EPOCHS):
    """One training loop for original and preserved-text experiments.

    This runs only when explicitly requested with main.py train. Existing
    completed models cannot be overwritten. Preserved-text runs evaluate
    validation only, matching the original experiment protocol.
    """
    setup_tensorflow()
    out, models = run_dir / 'outputs', run_dir / 'models'
    metadata = read_json(out / 'preprocessing.json')
    validation_only = 'preprocessing_variant' in metadata
    if validation_only and kind != 'fasttext':
        raise ValueError('The preserved-text experiment currently trains FastText only.')
    models.mkdir(parents=True, exist_ok=True)
    if (models / f'{kind}_malstm.keras').exists():
        raise FileExistsError('Model already exists. Use a new --run-dir.')
    tokenizer = tokenizer_from_json((out / 'tokenizer.json').read_text())
    matrix = build_embedding(kind, tokenizer.word_index, embedding_dir, out)
    data = load_array(out / 'prepared.npz')
    if validation_only:
        initial = np.random.default_rng(SEED).normal(0, .05, matrix.shape).astype('float32')
        random_rows = np.all(matrix == initial, axis=1)
        random_rows[0] = False
        counts = np.bincount(np.concatenate([data['x1_train'].ravel(), data['x2_train'].ravel()]),
                             minlength=len(matrix))
        counts[0] = 0
        coverage = {'random_fallback_vocabulary_rows': int(random_rows.sum()),
                    'random_fallback_training_token_fraction': float(counts[random_rows].sum() / counts.sum())}
        (out / 'embedding_coverage.json').write_text(json.dumps(coverage, indent=2))
        del initial
    network = build_model(matrix)
    del matrix
    callbacks = [keras.callbacks.CSVLogger(str(out / f'{kind}_epochs.csv'))]
    if validation_only:
        callbacks.append(keras.callbacks.ModelCheckpoint(str(models / 'latest_epoch.keras')))
    history = network.fit(dataset(tf, data, 'train', True), validation_data=dataset(tf, data, 'val'),
                          epochs=epochs, shuffle=False, verbose=2, callbacks=callbacks)
    network.save(models / f'{kind}_malstm.keras')
    (out / f'{kind}_history.json').write_text(json.dumps(history.history, indent=2))
    for metric in ['accuracy', 'loss']:
        plt.figure()
        for key, label in [(metric, 'Training'), ('val_' + metric, 'Validation')]:
            plt.plot(range(1, epochs + 1), history.history[key], label=label)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{kind}: {metric}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f'{kind}_{metric}.png', dpi=150)
        plt.close()
    training_info = {'epochs': epochs, 'batch_size': BATCH_SIZE, 'seed': SEED,
                     'optimizer': 'Adam', 'learning_rate': LEARNING_RATE, 'loss': 'binary_crossentropy',
                     'units': UNITS, 'dropout': 0, 'recurrent_dropout': 0, 'embedding_trainable': False,
                     'tensorflow_version': tf.__version__, 'tokenizer_sha256': metadata['tokenizer_sha256'],
                     'smoke_sample': metadata['smoke_sample'],
                     'test_evaluation': 'deferred' if validation_only else 'completed'}
    if validation_only:
        training_info['preprocessing_variant'] = metadata['preprocessing_variant']
    (out / f'{kind}_training.json').write_text(json.dumps(training_info, indent=2))
    if not validation_only:
        probabilities = network.predict(dataset(tf, data, 'test').map(lambda x, y: x), verbose=0).reshape(-1)
        np.save(out / f'{kind}_predictions.npy', probabilities)
        np.save(out / f'{kind}_test_rows.npy', data['rows_test'])
        evaluate(kind, data['y_test'], probabilities, out)
        comparison(out)


def compare_validation(base_dir, run_dir):
    tf = setup_tensorflow()
    import pandas as pd
    out = run_dir / 'outputs'
    results = []
    expected_rows = None
    for name, source in [('original_fasttext', base_dir), ('preserved_fasttext', run_dir)]:
        with load_array(source / 'outputs/prepared.npz') as data:
            rows, labels = data['rows_val'], data['y_val']
            if expected_rows is None:
                expected_rows, expected_labels = rows, labels
            np.testing.assert_array_equal(rows, expected_rows)
            np.testing.assert_array_equal(labels, expected_labels)
            ds = dataset(tf, data, 'val').map(lambda x, y: x)
        network = tf.keras.models.load_model(source / 'models/fasttext_malstm.keras', compile=False)
        # Keras does not serialize this execution setting. Empty questions must
        # continue to work after loading, without modifying any learned weights.
        network.get_layer('shared_lstm').use_cudnn = False
        probabilities = network.predict(ds, verbose=0).reshape(-1)
        np.save(out / f'{name}_validation_probabilities.npy', probabilities)
        results.append(evaluate(name + '_validation', labels, probabilities, out))
        del network
        tf.keras.backend.clear_session()
    table = pd.DataFrame(results)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        table[metric + '_percent'] = 100 * table.pop(metric)
    table.to_csv(out / 'validation_comparison.csv', index=False)
    print('\nValidation comparison (test evaluation remains deferred):', flush=True)
    print(table.to_string(index=False), flush=True)
    print(f"Accuracy change: {table.accuracy_percent.iloc[1] - table.accuracy_percent.iloc[0]:+.3f} percentage points", flush=True)

