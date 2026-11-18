#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Tests for incremental collection statistics of :class:`~gensim.models.tfidfmodel.TfidfModel`.

All expected values below are computed independently with plain Python / NumPy from the raw
bag-of-words; the incremental :meth:`TfidfModel.add_documents` entry point is never used to
produce an expectation.
"""

import logging
import math
import unittest

import numpy as np

from gensim.models import TfidfModel
from gensim.models.tfidfmodel import validate_bow_document, df2idf


def reference_stats(batches):
    """Independent (non-incremental) collection statistics straight from raw BoW batches."""
    num_docs = 0
    num_nnz = 0
    dfs = {}
    for batch in batches:
        for doc in batch:
            num_docs += 1
            num_nnz += len(doc)
            for termid, _ in doc:
                dfs[termid] = dfs.get(termid, 0) + 1
    idfs = {termid: df2idf(df, num_docs) for termid, df in dfs.items()}
    return num_docs, num_nnz, dfs, idfs


def reference_transform(bow, stats, normalize=True):
    """Independent TF-IDF transform of one document from hand-computed statistics."""
    num_docs, _, _, idfs = stats
    weighted = [(termid, float(count) * idfs.get(termid, 0.0)) for termid, count in bow]
    weighted = [(termid, w) for termid, w in weighted if abs(w) > 1e-12]
    if normalize:
        norm = math.sqrt(sum(w * w for _, w in weighted))
        if norm > 0.0:
            weighted = [(termid, w / norm) for termid, w in weighted]
    return weighted


class TestTfidfModelAddDocuments(unittest.TestCase):
    def setUp(self):
        # deliberately irregular: empty docs, repeated term frequencies, gaps in ids
        self.batch1 = [
            [(0, 1), (1, 2), (2, 1)],
            [(0, 3), (2, 1), (3, 2)],
            [],  # empty document still counts
        ]
        self.batch2 = [
            [(1, 1), (4, 5)],
            [(0, 2), (2, 2), (3, 1), (4, 1)],
            [(2, 7)],
        ]
        self.batch3 = [[(5, 1)], [(0, 1), (5, 2)]]

    def test_incremental_matches_one_shot_stats(self):
        for batches in ([self.batch1], [self.batch1, self.batch2],
                        [self.batch1, self.batch2, self.batch3]):
            model = TfidfModel(batches[0])
            for extra in batches[1:]:
                result = model.add_documents(extra)
                self.assertIs(result, model)  # returns self
            expected_docs, expected_nnz, expected_dfs, expected_idfs = reference_stats(batches)
            self.assertEqual(model.num_docs, expected_docs)
            self.assertEqual(model.num_nnz, expected_nnz)
            self.assertEqual(model.dfs, expected_dfs)
            self.assertEqual(set(model.idfs), set(expected_idfs))
            for termid in expected_idfs:
                self.assertAlmostEqual(model.idfs[termid], expected_idfs[termid])

    def test_different_batch_cuts_are_equivalent(self):
        flat = [doc for batch in (self.batch1, self.batch2, self.batch3) for doc in batch]
        one_shot = TfidfModel(flat)
        # one document at a time, including empty documents interleaved
        streamed = TfidfModel()
        for doc in flat:
            streamed.add_documents([doc])
        self.assertEqual(streamed.num_docs, one_shot.num_docs)
        self.assertEqual(streamed.dfs, one_shot.dfs)
        for termid in one_shot.idfs:
            self.assertAlmostEqual(streamed.idfs[termid], one_shot.idfs[termid])

    def test_empty_batch_changes_nothing(self):
        model = TfidfModel(self.batch1)
        snapshot = (model.num_docs, model.num_nnz, dict(model.dfs), dict(model.idfs))
        self.assertIs(model.add_documents([]), model)
        self.assertEqual((model.num_docs, model.num_nnz, dict(model.dfs), dict(model.idfs)), snapshot)
        # an empty batch must not even initialize a fresh model
        fresh = TfidfModel()
        fresh.add_documents([])
        self.assertIsNone(fresh.num_docs)
        self.assertIsNone(fresh.idfs)

    def test_empty_document_counts(self):
        model = TfidfModel()
        model.add_documents([[], [(0, 1)], []])
        self.assertEqual(model.num_docs, 3)
        self.assertEqual(model.num_nnz, 1)
        self.assertEqual(model.dfs, {0: 1})

    def test_weights_match_independent_reference(self):
        model = TfidfModel(self.batch1)
        model.add_documents(self.batch2)
        stats = reference_stats([self.batch1, self.batch2])
        for normalize in (True, False):
            model.normalize = normalize
            for doc in self.batch1 + self.batch2 + [[(0, 1), (4, 2)], []]:
                got = model[doc]
                expected = reference_transform(doc, stats, normalize=normalize)
                self.assertEqual([i for i, _ in got], [i for i, _ in expected])
                for (_, g), (_, e) in zip(got, expected):
                    self.assertAlmostEqual(g, e, places=12)

    def test_zero_idf_term_recovers(self):
        # term 0 is in every document of the first batch -> zero IDF
        batch = [[(0, 1), (1, 1)], [(0, 2)], [(0, 1), (1, 3)]]
        model = TfidfModel(batch)
        self.assertEqual(model.idfs[0], 0.0)
        self.assertNotIn(0, dict(model[[(0, 5)]]))
        # later batch contains a document without term 0 -> it becomes a useful term again
        model.add_documents([[(1, 1)], [(2, 2)]])
        self.assertGreater(model.idfs[0], 0.0)
        transformed = dict(model[[(0, 5)]])
        self.assertIn(0, transformed)
        self.assertGreater(transformed[0], 0.0)
        stats = reference_stats([batch, [[(1, 1)], [(2, 2)]]])
        self.assertAlmostEqual(transformed[0], dict(reference_transform([(0, 5)], stats))[0])

    def test_persistence_keeps_incremental_state(self):
        from gensim.test.utils import get_tmpfile
        model = TfidfModel(self.batch1)
        model.add_documents(self.batch2)
        fname = get_tmpfile('tfidf_incremental.tst')
        model.save(fname)
        loaded = TfidfModel.load(fname)
        self.assertEqual(loaded.num_docs, model.num_docs)
        self.assertEqual(loaded.num_nnz, model.num_nnz)
        self.assertEqual(loaded.dfs, model.dfs)
        self.assertEqual(loaded.idfs, model.idfs)
        # and the loaded model can keep growing
        loaded.add_documents(self.batch3)
        stats = reference_stats([self.batch1, self.batch2, self.batch3])
        self.assertEqual(loaded.num_docs, stats[0])
        self.assertEqual(loaded.dfs, stats[2])

    def test_generator_consumed_once_and_single_pass(self):
        class OneShotGenerator:
            def __init__(self, docs):
                self.docs = iter(docs)
                self.iterations = 0
                self.yields = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError("the corpus iterator was started more than once")
                return self

            def __next__(self):
                doc = next(self.docs)
                self.yields += 1
                return doc

        docs = self.batch1 + self.batch2
        source = OneShotGenerator(docs)
        model = TfidfModel()
        model.add_documents(source)
        self.assertEqual(source.iterations, 1)
        self.assertEqual(source.yields, len(docs))
        self.assertEqual(model.num_docs, len(docs))

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def test_validator_accepts_valid_documents(self):
        valid = [[], [(0, 1)], [(0, 1.5), (2, 2), (100, 0.001)]]
        for doc in valid:
            self.assertEqual(validate_bow_document(iter(doc)), list(doc))

    def test_validator_rejects_bad_documents(self):
        bad_documents = [
            [(1, 1), (1, 2)],          # duplicate id
            [(2, 1), (1, 2)],          # unordered id
            [(-1, 1)],                 # negative id
            [(True, 1)],               # bool id
            [(1, True)],               # bool count
            [(1, 0)],                  # zero count
            [(1, -2.0)],               # negative count
            [(1, float('nan'))],       # NaN
            [(1, float('inf'))],       # +inf
            [(1, float('-inf'))],      # -inf
            [(1, 1 + 2j)],             # complex count
            [(1, 'x')],                # non-numeric count
            [('x', 1)],                # non-int id
            [(1.5, 1)],                # float id
            42,                        # not an iterable of pairs
            None,
            [5],                       # entry not a pair
            [(1, 2, 3)],               # entry too long
        ]
        for doc in bad_documents:
            with self.subTest(doc=doc):
                with self.assertRaises(ValueError):
                    validate_bow_document(doc)

    def test_add_documents_validates_and_rolls_back(self):
        model = TfidfModel(self.batch1)
        snapshot = (model.num_docs, model.num_nnz, dict(model.dfs), dict(model.idfs))

        def malformed_batch():
            yield [(4, 1)]
            yield [(2, 1), (1, 1)]  # unordered -> ValueError halfway through

        with self.assertRaises(ValueError):
            model.add_documents(malformed_batch())
        self.assertEqual(
            (model.num_docs, model.num_nnz, dict(model.dfs), dict(model.idfs)), snapshot
        )
        # a later, valid batch still works and builds on the pre-failure state
        model.add_documents(self.batch2)
        stats = reference_stats([self.batch1, self.batch2])
        self.assertEqual((model.num_docs, model.num_nnz, model.dfs), stats[:3])

    def test_source_exception_propagates_unchanged(self):
        class CustomError(Exception):
            pass

        model = TfidfModel(self.batch1)
        snapshot = (model.num_docs, model.num_nnz, dict(model.dfs))

        def exploding():
            yield [(4, 1)]
            raise CustomError("boom from the source")

        with self.assertRaises(CustomError):
            model.add_documents(exploding())
        self.assertEqual((model.num_docs, model.num_nnz, dict(model.dfs)), snapshot)

    def test_unsupported_configuration_rejected_before_input(self):
        configured = [
            TfidfModel(self.batch1, smartirs='nfc'),
            TfidfModel(self.batch1, pivot=1.0),
            TfidfModel(self.batch1, wlocal=lambda x: x),
            TfidfModel(self.batch1, wglobal=lambda df, d: 1.0),
            TfidfModel(self.batch1, normalize=lambda v: v),
        ]
        for model in configured:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    model.add_documents(self.batch2)

        # the rejection happens before the iterator is even started
        def must_not_be_touched():
            raise AssertionError("input was read despite an unsupported configuration")
            yield  # pragma: no cover

        with self.assertRaises(ValueError):
            configured[0].add_documents(must_not_be_touched())

    def test_constructor_interface_unchanged(self):
        # fitting through the constructor and transforming still behaves exactly as before
        model = TfidfModel(self.batch1, normalize=True)
        one_shot = TfidfModel(self.batch1)
        for doc in self.batch1:
            self.assertTrue(np.allclose(model[doc], one_shot[doc]))
        # normalize=False path stays available
        raw = TfidfModel(self.batch1, normalize=False)
        vec = raw[self.batch1[0]]
        expected = reference_transform(self.batch1[0], reference_stats([self.batch1]), normalize=False)
        for (_, g), (_, e) in zip(vec, expected):
            self.assertAlmostEqual(g, e)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
