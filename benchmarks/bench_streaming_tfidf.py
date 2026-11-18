#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scale / cost benchmark for the streaming TF-IDF sharded index.

Runs the exact scenario requested in the design brief:

* vocabulary of 4096 features, 16 non-zero entries per document,
* 2048 documents per generator batch, appended as 4 / 16 / 64 batches,
* shardsize=256, chunksize=32,
* fixed-seed per-document generators (the whole corpus is never held in the test process),
* 16 retained queries per scale,
* full-vector and top-10 queries, a save/load round-trip and one more append,
* independent reference checks on the small scale, streaming checks on every scale,
* timing, extra Python peak allocation (tracemalloc), native RSS and mmap volume,
  disk size and rewrite accounting.

Everything measured starts at index construction. Dependency imports happen outside the
measurement. Run with:  python bench_streaming_tfidf.py --workdir DIR
"""

import argparse
import gc
import glob
import math
import os
import resource
import shutil
import time
import tracemalloc

import numpy as np

from gensim.models import TfidfModel
from gensim.similarities import Similarity


VOCAB = 4096
NNZ_PER_DOC = 16
BATCH_SIZE = 2048
SHARDSIZE = 256
CHUNKSIZE = 32
SCALES = [4, 16, 64]
N_QUERIES = 16
TOPN = 10
SEED = 20261117


# --------------------------------------------------------------------------- #
# Fixed-seed streamed data (no full corpus retained in the test process)
# --------------------------------------------------------------------------- #
def doc_generator(seed, n_docs, vocab=VOCAB, nnz=NNZ_PER_DOC):
    """Yield `n_docs` validated BoW documents from a local, reseedable RNG."""
    rng = np.random.default_rng(seed)
    for _ in range(n_docs):
        ids = np.sort(rng.choice(vocab, size=nnz, replace=False))
        # positive, finite real counts
        counts = rng.integers(low=1, high=10, size=nnz).astype(np.float64)
        yield [(int(i), float(c)) for i, c in zip(ids, counts)]


def query_generator(seed, n_queries, vocab=VOCAB):
    rng = np.random.default_rng(seed)
    queries = []
    for _ in range(n_queries):
        length = int(rng.integers(low=1, high=20))
        ids = np.sort(rng.choice(vocab, size=length, replace=False))
        counts = rng.integers(low=1, high=10, size=length).astype(np.float64)
        queries.append([(int(i), float(c)) for i, c in zip(ids, counts)])
    return queries


class RewriteMeter:
    """Track total bytes written under a directory via periodic file-size scans."""

    def __init__(self, root):
        self.root = root
        self.seen = {}
        self.bytes_written = 0

    def sample(self):
        current = {}
        for path in glob.glob(os.path.join(self.root, '*')):
            if os.path.isfile(path):
                current[path] = os.path.getsize(path)
        for path, size in current.items():
            prev = self.seen.get(path)
            if prev is None:
                self.bytes_written += size            # newly created file
            elif size > prev:
                self.bytes_written += size - prev     # grew (a rewrite)
        # deletions do not subtract: write amplification counts bytes produced
        self.seen = current
        return self.bytes_written


def rss_mib():
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mmap_mib():
    total = 0
    maps = f'/proc/{os.getpid()}/maps'
    if not os.path.exists(maps):
        return 0.0
    with open(maps) as fh:
        for line in fh:
            # count only mapped shard data files (numpy .npy side files), not anonymous memory
            if '.npy' in line and ('r' in line.split()[1]):
                parts = line.split()
                start, end = parts[0].split('-')
                total += int(end, 16) - int(start, 16)
    return total / (1024.0 ** 2)


# --------------------------------------------------------------------------- #
# Independent reference, computed without the incremental entry points
# --------------------------------------------------------------------------- #
def reference_from_stream(seed, total_docs):
    """Replay the same generator and build a one-shot model + plain sharded index."""
    dfs = {}
    num_nnz = 0
    docs = []  # the reference is only built for the small scale (4 batches = 8192 docs)
    for doc in doc_generator(seed, total_docs):
        docs.append(doc)
        num_nnz += len(doc)
        for termid, _ in doc:
            dfs[termid] = dfs.get(termid, 0) + 1
    model = TfidfModel(docs)
    assert model.dfs == dfs and model.num_docs == total_docs and model.num_nnz == num_nnz
    return docs, model


def streaming_reference_scores(model, raw_query, num_docs, seed0, n_batches):
    """Stream every indexed document once and yield independent cosine scores.

    The index corpus is regenerated from the same per-batch seed series; nothing beyond one
    document and one score vector is held at a time. This mirrors how the large scale can be
    checked without keeping the corpus in memory.
    """
    q = dict(model[raw_query])
    qnorm = math.sqrt(sum(v * v for v in q.values()))
    scores = np.empty(num_docs, dtype=np.float64)
    if qnorm == 0.0:
        return scores
    pos = 0
    for batch_no in range(n_batches):
        for doc in doc_generator(seed0 + batch_no, BATCH_SIZE):
            w = dict(model[doc])
            norm = math.sqrt(sum(v * v for v in w.values()))
            if norm == 0.0:
                scores[pos] = 0.0
            else:
                dot = sum(q[i] * w.get(i, 0.0) for i in q)
                scores[pos] = dot / (qnorm * norm)
            pos += 1
    assert pos == num_docs
    return scores


def run_scale(n_batches, workdir, verify_full):
    total_docs = n_batches * BATCH_SIZE
    scale_dir = os.path.join(workdir, f'scale_{n_batches}')
    os.makedirs(scale_dir, exist_ok=True)
    prefix = os.path.join(scale_dir, 'index')
    meter = RewriteMeter(scale_dir)

    rss_before = rss_mib()
    tracemalloc.start()
    t_start = time.perf_counter()

    # ---- measured: construction (empty) + all appends ----
    index = Similarity(prefix, None, num_features=VOCAB, shardsize=SHARDSIZE,
                       chunksize=CHUNKSIZE, tfidf=True)
    append_times = []
    for batch_no in range(n_batches):
        seed = SEED + batch_no
        t0 = time.perf_counter()
        index.add_documents(doc_generator(seed, BATCH_SIZE))
        append_times.append(time.perf_counter() - t0)
        meter.sample()
    assert len(index) == total_docs

    # ---- measured: full and top-10 queries over the retained query set ----
    queries = query_generator(SEED ^ 0x5A5A, N_QUERIES)
    gc.collect()
    t_q0 = time.perf_counter()
    full_results = [index[q] for q in queries]
    full_query_time = time.perf_counter() - t_q0
    meter.sample()

    index.num_best = TOPN
    t_q1 = time.perf_counter()
    top_results = [index[q] for q in queries]
    top_query_time = time.perf_counter() - t_q1
    index.num_best = None
    meter.sample()

    # ---- measured: save / load and one more append ----
    save_path = os.path.join(scale_dir, 'saved')
    t_s0 = time.perf_counter()
    index.save(save_path)
    save_time = time.perf_counter() - t_s0
    meter.sample()

    t_l0 = time.perf_counter()
    loaded = Similarity.load(save_path)
    load_time = time.perf_counter() - t_l0

    extra_seed = SEED + n_batches
    t_a0 = time.perf_counter()
    loaded.add_documents(doc_generator(extra_seed, BATCH_SIZE))
    post_load_append_time = time.perf_counter() - t_a0
    meter.sample()
    final_docs = total_docs + BATCH_SIZE
    assert len(loaded) == final_docs

    # queries still work after the post-load append
    post_sims = loaded[queries[0]]

    elapsed = time.perf_counter() - t_start
    py_current, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = rss_mib()
    mmap_after = mmap_mib()
    disk_bytes = sum(
        os.path.getsize(p) for p in glob.glob(os.path.join(scale_dir, '*')) if os.path.isfile(p)
    )

    # ---- correctness ----------------------------------------------------- #
    checks = {}
    if verify_full:
        # full independent rebuild over the same streamed corpus (small scale only)
        ref_docs, ref_model = reference_from_stream_series(SEED, n_batches)
        ref_index = Similarity(None, ref_model[ref_docs], num_features=VOCAB, shardsize=SHARDSIZE)
        worst = 0.0
        for q, got in zip(queries, full_results):
            expected = ref_index[ref_model[q]]
            worst = max(worst, float(np.abs(got - expected).max()))
        checks['full_rebuild_max_abs_diff'] = worst
        # top-10 consistency against the full vectors
        for q, got_top, full in zip(queries, top_results, full_results):
            rows = [r for r in np.nonzero(full > 1e-12)[0]]
            rows.sort(key=lambda r: (-full[r], r))
            expected_top = [(int(r), float(full[r])) for r in rows[:TOPN]]
            assert [r for r, _ in got_top] == [r for r, _ in expected_top]
        checks['topk_vs_full'] = 'ok'
    # every scale: streaming per-document check on two queries (no corpus retained)
    check_model = TfidfModel()
    for batch_no in range(n_batches):
        check_model.add_documents(doc_generator(SEED + batch_no, BATCH_SIZE))
    stream_worst = 0.0
    for q in queries[:2]:
        expected = streaming_reference_scores(check_model, q, total_docs, SEED, n_batches)
        # compare only the pre-save index window (total_docs rows)
        got = index[q]
        stream_worst = max(stream_worst, float(np.abs(got - expected).max()))
    checks['streaming_max_abs_diff'] = stream_worst
    checks['post_load_query_finite'] = bool(np.isfinite(post_sims).all())

    return {
        'n_batches': n_batches,
        'total_docs': final_docs,
        'shards': len(loaded.shards),
        'append_total_s': sum(append_times),
        'append_per_batch_s': sum(append_times) / n_batches,
        'full_query_s': full_query_time,
        'top10_query_s': top_query_time,
        'save_s': save_time,
        'load_s': load_time,
        'post_load_append_s': post_load_append_time,
        'wallclock_s': elapsed,
        'py_peak_mib': py_peak / (1024.0 ** 2),
        'rss_delta_mib': rss_after - rss_before,
        'rss_total_mib': rss_after,
        'mmap_mib': mmap_after,
        'disk_mib': disk_bytes / (1024.0 ** 2),
        'rewrite_mib': meter.bytes_written / (1024.0 ** 2),
        **checks,
    }


def reference_from_stream_series(seed0, n_batches):
    docs, model = [], TfidfModel()
    for batch_no in range(n_batches):
        batch = list(doc_generator(seed0 + batch_no, BATCH_SIZE))
        docs.extend(batch)
        model.add_documents(batch)
    return docs, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--scales', default=','.join(map(str, SCALES)))
    args = parser.parse_args()
    workdir = os.path.abspath(args.workdir)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)

    rows = []
    for n in map(int, args.scales.split(',')):
        print(f'\n=== scale: {n} batches ({n * BATCH_SIZE} docs, then +{BATCH_SIZE}) ===', flush=True)
        row = run_scale(n, workdir, verify_full=(n == SCALES[0]))
        rows.append(row)
        for key, value in row.items():
            if isinstance(value, float):
                print(f'  {key:32s} {value:12.4f}')
            else:
                print(f'  {key:32s} {value}')

    print('\n=== summary (CSV) ===')
    keys = list(rows[0].keys())
    print(','.join(keys))
    for row in rows:
        cells = []
        for k in keys:
            value = row.get(k, '')
            cells.append(f'{value:.6f}' if isinstance(value, float) else str(value))
        print(','.join(cells))


if __name__ == '__main__':
    main()
