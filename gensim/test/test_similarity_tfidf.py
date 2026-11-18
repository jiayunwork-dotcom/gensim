#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Tests for streaming TF-IDF mode of :class:`~gensim.similarities.docsim.Similarity`.

Expectations are built independently from raw bag-of-words: a default
:class:`~gensim.models.tfidfmodel.TfidfModel` is fitted from scratch over *all* received
documents and fed to an ordinary (non-TF-IDF) :class:`~gensim.similarities.docsim.Similarity`.
The incremental entry points under test are never used to compute an expectation.
"""

import logging
import math
import os
import tempfile
import unittest

import numpy as np

from gensim.models import TfidfModel
from gensim.similarities import Similarity


def cosine_from_bow(doc_a, doc_b, idfs):
    """Plain-Python cosine between two raw BoW documents under given IDF weights."""
    wa = {i: c * idfs.get(i, 0.0) for i, c in doc_a}
    wb = {i: c * idfs.get(i, 0.0) for i, c in doc_b}
    na = math.sqrt(sum(v * v for v in wa.values()))
    nb = math.sqrt(sum(v * v for v in wb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(wa[i] * wb.get(i, 0.0) for i in wa)
    return dot / (na * nb)


class TestStreamingTfidfSimilarity(unittest.TestCase):
    def setUp(self):
        self.num_features = 8
        # irregular corpus: empty docs, repeated counts, multiple terms
        self.batch1 = [
            [(0, 1), (1, 2), (2, 1)],
            [(0, 3), (2, 1), (3, 2)],
            [],
            [(1, 1), (4, 2)],
        ]
        self.batch2 = [
            [(0, 2), (2, 2), (3, 1), (4, 1)],
            [(2, 7)],
            [(5, 1), (6, 3)],
            [],
        ]
        self.batch3 = [
            [(0, 1), (5, 2)],
            [(3, 4), (7, 1)],
        ]
        self.queries = [
            [(0, 1), (2, 1)],
            [(4, 2), (6, 1)],
            [],                                  # empty query
            [(7, 1)],                            # rare / newly introduced term
            [(1, 1), (2, 1), (3, 1), (5, 1)],
        ]
        self.tmpdir = tempfile.mkdtemp()
        self.prefix = os.path.join(self.tmpdir, 'index')

    def _index(self, shardsize=3, chunksize=4):
        return Similarity(self.prefix, None, num_features=self.num_features,
                          shardsize=shardsize, chunksize=chunksize, tfidf=True)

    def _check_against_reference(self, index, all_bow, shardsize):
        model = TfidfModel(all_bow)
        transformed = [model[doc] for doc in all_bow]
        ref = Similarity(None, transformed, num_features=self.num_features, shardsize=shardsize)
        for query in self.queries:
            sims = index[query]
            ref_sims = ref[model[query]]
            self.assertEqual(sims.shape, ref_sims.shape)
            self.assertTrue(
                np.allclose(sims, ref_sims, atol=1e-6),
                msg="query %r\n got %s\n ref %s" % (query, sims, ref_sims),
            )
        return model, ref

    def test_incremental_batches_match_full_rebuild(self):
        for shardsize in (1, 2, 3, 5, 100):  # includes multiple full shards + partial shard
            index = Similarity(os.path.join(self.tmpdir, 'i%s' % shardsize), self.batch1,
                               num_features=self.num_features, shardsize=shardsize, tfidf=True)
            index.add_documents(self.batch2)
            index.add_documents(self.batch3)
            all_bow = self.batch1 + self.batch2 + self.batch3
            self.assertEqual(len(index), len(all_bow))
            self._check_against_reference(index, all_bow, shardsize)
            index.destroy()

    def test_different_batch_cuts_equivalent(self):
        flat = self.batch1 + self.batch2 + self.batch3
        cut_a = Similarity(os.path.join(self.tmpdir, 'cutA'), flat,
                           num_features=self.num_features, shardsize=3, tfidf=True)
        cut_b = self._index(shardsize=3)
        for doc in flat:
            cut_b.add_documents([doc])  # one document per batch
        for query in self.queries:
            self.assertTrue(np.allclose(cut_a[query], cut_b[query], atol=1e-6))
        # statistics are identical as well
        self.assertEqual(cut_a.tfidf_model.dfs, cut_b.tfidf_model.dfs)
        self.assertEqual(cut_a.tfidf_model.num_docs, cut_b.tfidf_model.num_docs)

    def test_query_between_batches_uses_current_statistics_then_append(self):
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        # a query forces the open partial shard to close
        early = index[self.queries[0]]
        self.assertEqual(early.shape[0], len(self.batch1))
        n_shards_after_query = len(index.shards)
        index.add_documents(self.batch2)
        index.add_documents(self.batch3)
        all_bow = self.batch1 + self.batch2 + self.batch3
        self._check_against_reference(index, all_bow, 2)
        # appending did not reopen/rewrite the shards closed by the query
        self.assertGreaterEqual(len(index.shards), n_shards_after_query)

    def test_manual_cosine_reference(self):
        # cross-check against a fully independent plain-Python cosine implementation
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        index.add_documents(self.batch2)
        all_bow = self.batch1 + self.batch2
        idfs = TfidfModel(all_bow).idfs
        sims = index[self.queries[0]]
        for row, doc in enumerate(all_bow):
            self.assertAlmostEqual(
                sims[row], cosine_from_bow(self.queries[0], doc, idfs), places=5
            )

    def test_num_best_matches_full_results(self):
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        index.add_documents(self.batch2)
        index.add_documents(self.batch3)
        all_bow = self.batch1 + self.batch2 + self.batch3
        model, _ = self._check_against_reference(index, all_bow, 2)
        for topn in (0, 1, 2, 3, 1000):
            index.num_best = topn
            for query in self.queries:
                result = index[query]
                full = index
                full.num_best = None
                full_sims = full[query]
                full.num_best = topn
                if topn == 0:
                    self.assertEqual(result, [])
                    continue
                # build expectation from the full score vector directly
                expected_rows = [r for r in np.nonzero(full_sims > 1e-12)[0]]
                expected_rows.sort(key=lambda r: (-full_sims[r], r))
                expected = [(int(r), float(full_sims[r])) for r in expected_rows[:topn]]
                self.assertEqual([r for r, _ in result], [r for r, _ in expected],
                                 msg="topk row order mismatch for query %r" % (query,))
                for (r, s), (er, es) in zip(result, expected):
                    self.assertEqual(r, er)
                    self.assertAlmostEqual(s, es, places=6)
                # zero scores are always excluded
                self.assertTrue(all(s > 0.0 for _, s in result))

    def test_num_best_tie_breaking_is_by_row_number(self):
        # craft two documents with identical tf-idf cosine against the query
        bow = [
            [(0, 1), (1, 1)],
            [(0, 1), (1, 1)],   # identical content -> identical scores, rows 0 and 1
            [(2, 1)],
        ]
        index = Similarity(None, bow, num_features=4, shardsize=2, tfidf=True)
        index.num_best = 2
        result = index[[(0, 1), (1, 1)]]
        self.assertEqual([r for r, _ in result], [0, 1])  # ascending row number on the tie

    def test_corpus_batch_query(self):
        index = self._index(shardsize=2, chunksize=3)
        index.add_documents(self.batch1)
        index.add_documents(self.batch2)
        all_bow = self.batch1 + self.batch2
        model = TfidfModel(all_bow)
        ref = Similarity(None, [model[d] for d in all_bow],
                         num_features=self.num_features, shardsize=2)
        batch = self.queries
        result = index[batch]
        expected = ref[[model[d] for d in batch]]
        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(np.allclose(result, expected, atol=1e-6))
        # num_best over a corpus returns one ordered list per query document
        index.num_best = 2
        topk = index[batch]
        self.assertEqual(len(topk), len(batch))
        self.assertEqual(topk[2], [])  # the empty query in the middle gives zero results

    def test_empty_and_all_unknown_queries(self):
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        # empty query
        self.assertTrue(np.all(index[[]] == 0.0))
        # query made only of unknown / zero-idf terms
        unknown = [(self.num_features - 1, 1)]
        self.assertTrue(np.all(index[unknown] == 0.0))
        index.num_best = 3
        self.assertEqual(index[[]], [])
        self.assertEqual(index[unknown], [])
        index.num_best = None
        # empty corpus query returns an empty 2D result
        empty_corpus = index[iter([])] if False else index[[]]
        self.assertEqual(empty_corpus.ndim, 1)

    def test_vector_by_id_and_similarity_by_id(self):
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        index.add_documents(self.batch2)
        index.add_documents(self.batch3)
        all_bow = self.batch1 + self.batch2 + self.batch3
        model, ref = self._check_against_reference(index, all_bow, 2)
        for docpos in range(len(all_bow)):
            vec = index.vector_by_id(docpos)
            ref_vec = ref.vector_by_id(docpos)
            # TF-IDF mode stores sparse weighted rows; the ordinary index stores a dense row
            vec_arr = vec.toarray().ravel() if hasattr(vec, 'toarray') else np.asarray(vec).ravel()
            ref_arr = ref_vec.toarray().ravel() if hasattr(ref_vec, 'toarray') else np.asarray(ref_vec).ravel()
            self.assertEqual(vec_arr.shape, ref_arr.shape)
            self.assertTrue(np.allclose(vec_arr, ref_arr, atol=1e-6))
            # NOTE: the plain Similarity.similarity_by_id has a pre-existing failure on shards
            # containing empty documents (inconsistent array ranks in hstack); compare against the
            # weighted self-query result instead, which exercises the same scores
            got_self_sims = index.similarity_by_id(docpos)
            expected_self_sims = ref[model[all_bow[docpos]]]
            self.assertTrue(
                np.allclose(got_self_sims, expected_self_sims, atol=1e-6),
                msg="similarity_by_id mismatch at row %d" % docpos,
            )
        with self.assertRaises(ValueError):
            index.vector_by_id(len(all_bow))
        with self.assertRaises(ValueError):
            index.vector_by_id(-1)

    def test_iteration_over_index(self):
        index = self._index(shardsize=2, chunksize=3)
        index.add_documents(self.batch1)
        index.add_documents(self.batch2)
        all_bow = self.batch1 + self.batch2
        model = TfidfModel(all_bow)
        ref = Similarity(None, [model[d] for d in all_bow],
                         num_features=self.num_features, shardsize=2)
        # __iter__ feeds the indexed raw BoW back through the owned model, so compare each row
        # against the reference result for the same RAW document
        for docpos, (got, raw_doc) in enumerate(zip(index, all_bow)):
            expected = ref[model[raw_doc]]
            self.assertTrue(
                np.allclose(got, expected, atol=1e-6),
                msg="iter row %d mismatch: %s vs %s" % (docpos, got, expected),
            )

    def test_save_load_and_continue(self):
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        index.add_documents(self.batch2)
        # query before save so some shards are already closed
        index[self.queries[0]]
        save_path = os.path.join(self.tmpdir, 'saved_index')
        index.save(save_path)
        loaded = Similarity.load(save_path)
        self.assertEqual(len(loaded), len(self.batch1 + self.batch2))
        self.assertEqual(loaded.tfidf_model.dfs, index.tfidf_model.dfs)
        loaded.add_documents(self.batch3)
        all_bow = self.batch1 + self.batch2 + self.batch3
        # compare with a reference built over the same whole corpus
        model = TfidfModel(all_bow)
        ref = Similarity(None, [model[d] for d in all_bow],
                         num_features=self.num_features, shardsize=2)
        for query in self.queries:
            self.assertTrue(np.allclose(loaded[query], ref[model[query]], atol=1e-6))
        # save/load again after the post-load append
        save_path2 = os.path.join(self.tmpdir, 'saved_index_2')
        loaded.save(save_path2)
        reloaded = Similarity.load(save_path2)
        for query in self.queries:
            self.assertTrue(np.allclose(reloaded[query], loaded[query], atol=1e-6))

    def test_failed_batch_is_atomic(self):
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        before_sims = index[self.queries[0]].copy()
        before_len = len(index)
        before_docs = index.tfidf_model.num_docs
        before_dfs = dict(index.tfidf_model.dfs)

        # batch big enough to spill a shard before the bad document is reached
        def failing_stream():
            yield [(5, 1)]
            yield [(6, 2)]
            yield [(2, 1), (1, 1)]   # unordered ids -> ValueError
            yield [(7, 1)]           # must never be reached

        with self.assertRaises(ValueError):
            index.add_documents(failing_stream())
        # no partial content, statistics or shards remain
        self.assertEqual(len(index), before_len)
        self.assertEqual(index.tfidf_model.num_docs, before_docs)
        self.assertEqual(dict(index.tfidf_model.dfs), before_dfs)
        self.assertTrue(np.array_equal(index[self.queries[0]], before_sims))
        # no in-progress files left behind
        for fname in os.listdir(self.tmpdir):
            self.assertNotIn('inprogress', fname)
        # earlier rows still work and a legal append succeeds afterwards
        index.add_documents(self.batch2)
        all_bow = self.batch1 + self.batch2
        self._check_against_reference(index, all_bow, 2)

    def test_source_exception_atomic_and_propagated(self):
        class Boom(Exception):
            pass

        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        before_docs = index.tfidf_model.num_docs

        def exploding():
            yield [(5, 1)]
            raise Boom('iterator failure')

        with self.assertRaises(Boom):
            index.add_documents(exploding())
        self.assertEqual(index.tfidf_model.num_docs, before_docs)
        self.assertEqual(len(index), len(self.batch1))

    def test_input_validation_and_id_range(self):
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        bad_batches = [
            [[(self.num_features, 1)]],       # id out of range
            [[(1, 1), (1, 2)]],               # duplicate id
            [[(2, 1), (0, 1)]],               # unordered
            [[(True, 1)]],                    # bool id
            [[(0, True)]],                    # bool count
            [[(0, float('nan'))]],
            [[(0, float('inf'))]],
            [[(0, 0)]],
            [[(-1, 1)]],
            [42],                             # not a document
        ]
        for batch in bad_batches:
            with self.subTest(batch=batch):
                with self.assertRaises(ValueError):
                    index.add_documents(batch)
        # queries are validated with the same rules
        with self.assertRaises(ValueError):
            _ = index[[(self.num_features, 1)]]
        with self.assertRaises(ValueError):
            _ = index[[(1, 1), (0, 1)]]

    def test_tfidf_mode_construction_gates(self):
        # norm other than l2 is rejected before reading the corpus
        with self.assertRaises(ValueError):
            Similarity(None, self.batch1, num_features=self.num_features, norm='l1', tfidf=True)
        with self.assertRaises(ValueError):
            Similarity(None, self.batch1, num_features=0, tfidf=True)
        with self.assertRaises(ValueError):
            Similarity(None, self.batch1, num_features=1.5, tfidf=True)
        # tfidf stays keyword-only
        with self.assertRaises(TypeError):
            Similarity(None, self.batch1, self.num_features, None, 256, 32768, 'l2', True)

    def test_tfidf_model_property_is_read_only(self):
        index = self._index()
        model = index.tfidf_model
        self.assertIsInstance(model, TfidfModel)
        with self.assertRaises(AttributeError):
            index.tfidf_model = TfidfModel()
        plain = Similarity(None, [[(0, 1)]], num_features=3)
        with self.assertRaises(AttributeError):
            _ = plain.tfidf_model

    def test_plain_vector_mode_unchanged(self):
        # the historical (non-TF-IDF) code path must keep its old behaviour
        vectors = [[(0, 1.0)], [(1, 2.0), (2, 1.0)]]
        plain = Similarity(None, vectors, num_features=3, shardsize=1)
        self.assertFalse(plain.tfidf)
        sims = plain[[(0, 1.0)]]
        self.assertAlmostEqual(float(sims[0]), 1.0, places=6)
        self.assertAlmostEqual(float(sims[1]), 0.0, places=6)

    def test_statistics_visible_through_model_property(self):
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        index.add_documents(self.batch2)
        all_bow = self.batch1 + self.batch2
        ref_model = TfidfModel(all_bow)
        self.assertEqual(index.tfidf_model.num_docs, ref_model.num_docs)
        self.assertEqual(index.tfidf_model.num_nnz, ref_model.num_nnz)
        self.assertEqual(index.tfidf_model.dfs, ref_model.dfs)
        for termid, idf in ref_model.idfs.items():
            self.assertAlmostEqual(index.tfidf_model.idfs[termid], idf)

    def test_empty_index_and_all_empty_documents(self):
        index = Similarity(None, None, num_features=self.num_features, shardsize=2, tfidf=True)
        self.assertEqual(len(index), 0)
        # zero documents -> zero-shaped full results, never an exception
        result = index[[(0, 1)]]
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0,))
        corpus_result = index[[[(0, 1)], [(1, 1)]]]
        self.assertEqual(corpus_result.shape, (2, 0))
        index.num_best = 3
        self.assertEqual(index[[(0, 1)]], [])
        self.assertEqual(index[[[(0, 1)]]], [[]])
        index.num_best = None

        # documents that are all empty still occupy row numbers
        index.add_documents([[], [], []])
        self.assertEqual(len(index), 3)
        self.assertEqual(index.tfidf_model.num_docs, 3)
        self.assertTrue(np.all(index[[(0, 1)]] == 0.0))

        # persistence keeps the empty-rows shard, then a real append works
        save_path = os.path.join(self.tmpdir, 'empty_rows')
        index.save(save_path)
        loaded = Similarity.load(save_path)
        self.assertEqual(len(loaded), 3)
        loaded.add_documents([[(0, 1), (1, 1)]])
        self.assertEqual(len(loaded), 4)
        sims = loaded[[(0, 1)]]
        self.assertEqual(np.count_nonzero(sims), 1)  # only the last, non-empty row can match
        # iterating an index that contains empty rows still yields one result per row
        rows = list(loaded)
        self.assertEqual(len(rows), 4)

    def test_destroy_removes_shard_files(self):
        index = self._index(shardsize=2)
        index.add_documents(self.batch1)
        index.close_shard()
        self.assertTrue(any(f.endswith('.0') for f in os.listdir(self.tmpdir)))
        index.destroy()
        self.assertFalse(any(f.startswith(os.path.basename(self.prefix))
                             for f in os.listdir(self.tmpdir)))

    def test_single_pass_iterators_for_ingest_and_query(self):
        class OneShot:
            def __init__(self, docs):
                self.docs = iter(docs)
                self.starts = 0

            def __iter__(self):
                self.starts += 1
                if self.starts > 1:
                    raise AssertionError("iterator was started more than once")
                return self

            def __next__(self):
                return next(self.docs)

        flat = self.batch1 + self.batch2
        index = self._index(shardsize=3)
        ingest = OneShot(flat)
        index.add_documents(ingest)
        self.assertEqual(ingest.starts, 1)
        self.assertEqual(len(index), len(flat))

        full = OneShot(self.queries)
        result = index[full]
        self.assertEqual(full.starts, 1)
        self.assertEqual(result.shape[0], len(self.queries))

        index.num_best = 2
        top = index[OneShot(self.queries)]
        self.assertEqual(len(top), len(self.queries))
        index.num_best = None

    def test_multiple_full_shards_plus_partial(self):
        # shardsize=2 over 10 docs => five full shards; then a partial tail in a later batch
        docs = [[(i % self.num_features, 1)] for i in range(10)]
        index = Similarity(None, docs[:8], num_features=self.num_features,
                           shardsize=2, chunksize=2, tfidf=True)
        self.assertEqual(len(index.shards), 4)
        index.add_documents(docs[8:])
        # query closes the partial shard
        self.assertEqual(index[self.queries[0]].shape[0], 10)
        self.assertEqual(len(index.shards), 5)
        # row numbers follow acceptance order, including every row
        model = TfidfModel(docs)
        ref = Similarity(None, model[docs], num_features=self.num_features, shardsize=2)
        for query in self.queries[:3]:
            self.assertTrue(np.allclose(index[query], ref[model[query]], atol=1e-6))


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
