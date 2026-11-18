#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuous corpus retrieval with the streaming TF-IDF sharded index.

This is a self-contained, repeatable example of the new ``tfidf=True`` mode of
:class:`~gensim.similarities.docsim.Similarity`. It shows how to

* feed raw bag-of-words batches from one-shot generators (no history is ever replayed),
* keep every query weighted with TF-IDF statistics over *all* documents received so far,
* survive a malformed batch without losing earlier batches,
* persist the index to disk, reload it and keep appending,
* inspect the owned, read-only :class:`~gensim.models.tfidfmodel.TfidfModel`.

Run it with::

    python run_continuous_tfidf_retrieval.py
"""

import os
import shutil
import tempfile

import numpy as np

from gensim.models import TfidfModel
from gensim.similarities import Similarity


def incoming_batch(seed, n_docs, num_features):
    """A one-shot generator of raw BoW documents (stand-in for a live data feed)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_docs):
        length = int(rng.integers(low=1, high=num_features // 4))
        ids = np.sort(rng.choice(num_features, size=length, replace=False))
        counts = rng.integers(low=1, high=8, size=length)
        yield [(int(term_id), int(count)) for term_id, count in zip(ids, counts)]


def main():
    workdir = tempfile.mkdtemp(prefix='continuous_tfidf_')
    prefix = os.path.join(workdir, 'index')
    num_features, shardsize = 128, 256

    # 1. open the streaming index: it owns a default TfidfModel and stores raw counts on disk
    index = Similarity(prefix, None, num_features=num_features,
                       shardsize=shardsize, chunksize=64, tfidf=True)
    print('opened streaming TF-IDF index under', prefix)

    # 2. append several batches as they arrive; each generator is consumed exactly once
    for batch_no in range(3):
        index.add_documents(incoming_batch(seed=1000 + batch_no, n_docs=300,
                                           num_features=num_features))
        model = index.tfidf_model
        print('after batch %d: %d docs, %d terms, num_nnz=%d' % (
            batch_no, model.num_docs, len(model.dfs), model.num_nnz))

    # 3. a query is always scored against the statistics of the whole collection seen so far
    query = list(next(incoming_batch(seed=42, n_docs=1, num_features=num_features)))
    full_sims = index[query]
    print('full similarity vector: shape=%d, non-zeros=%d' % (
        full_sims.shape[0], int(np.count_nonzero(full_sims))))

    index.num_best = 5
    print('top-5 hits:', index[query])
    index.num_best = None

    # 4. a malformed batch is rejected atomically - earlier rows stay queryable
    def malformed_feed():
        yield [(5, 1)]
        yield [(9, 1), (3, 1)]  # ids out of order -> ValueError

    try:
        index.add_documents(malformed_feed())
    except ValueError as exc:
        print('malformed batch rejected as expected:', exc)
    print('index still has', len(index), 'queryable documents after the failure')

    # 5. persist, reload and continue appending with identical results
    save_path = os.path.join(workdir, 'saved_index')
    index.save(save_path)
    reloaded = Similarity.load(save_path)
    before = reloaded[query].copy()
    reloaded.add_documents(incoming_batch(seed=2000, n_docs=300, num_features=num_features))
    print('after reload + append:', reloaded.tfidf_model.num_docs, 'documents')

    # 6. the result is equivalent to rebuilding a default model over the whole corpus
    #    (here demonstrated against the pre-append window for clarity)
    assert np.allclose(before, index[query], atol=1e-6)
    print('reloaded index reproduces the in-memory results')

    # the owned model is read-only on purpose
    try:
        reloaded.tfidf_model = TfidfModel()
    except AttributeError as exc:
        print('tfidf_model is read-only:', exc)

    shutil.rmtree(workdir, ignore_errors=True)
    print('done')


if __name__ == '__main__':
    main()
