#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Compute similarities across a collection of documents in the Vector Space Model.

The main class is :class:`~gensim.similarities.docsim.Similarity`, which builds an index for a given set of documents.

Once the index is built, you can perform efficient queries like "Tell me how similar is this query document to each
document in the index?". The result is a vector of numbers as large as the size of the initial set of documents,
that is, one float for each index document. Alternatively, you can also request only the top-N most
similar index documents to the query.


How It Works
------------
The :class:`~gensim.similarities.docsim.Similarity` class splits the index into several smaller sub-indexes ("shards"),
which are disk-based. If your entire index fits in memory (~one million documents per 1GB of RAM),
you can also use the :class:`~gensim.similarities.docsim.MatrixSimilarity`
or :class:`~gensim.similarities.docsim.SparseMatrixSimilarity` classes directly.
These are more simple but do not scale as well: they keep the entire index in RAM, no sharding. They also do not
support adding new document to the index dynamically.

Once the index has been initialized, you can query for document similarity simply by

.. sourcecode:: pycon

    >>> from gensim.similarities import Similarity
    >>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
    >>>
    >>> index_tmpfile = get_tmpfile("index")
    >>> query = [(1, 2), (6, 1), (7, 2)]
    >>>
    >>> index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary))  # build the index
    >>> similarities = index[query]  # get similarities between the query and all index documents

If you have more query documents, you can submit them all at once, in a batch

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
    >>>
    >>> index_tmpfile = get_tmpfile("index")
    >>> batch_of_documents = common_corpus[:]  # only as example
    >>> index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary))  # build the index
    >>>
    >>> # the batch is simply an iterable of documents, aka gensim corpus:
    >>> for similarities in index[batch_of_documents]:
    ...     pass

The benefit of this batch (aka "chunked") querying is a much better performance.
To see the speed-up on your machine, run ``python -m gensim.test.simspeed``
(compare to my results `here <https://groups.google.com/g/gensim/c/9rg5zqoWyDQ/m/yk-ehhoXb08J>`_).

There is also a special syntax for when you need similarity of documents in the index
to the index itself (i.e. queries = the indexed documents themselves). This special syntax
uses the faster, batch queries internally and **is ideal for all-vs-all pairwise similarities**:

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
    >>>
    >>> index_tmpfile = get_tmpfile("index")
    >>> index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary))  # build the index
    >>>
    >>> for similarities in index:  # yield similarities of the 1st indexed document, then 2nd...
    ...     pass

Streaming TF-IDF over a continuously arriving corpus
-----------------------------------------------------
Passing ``tfidf=True`` to :class:`Similarity` switches the index to *streaming TF-IDF mode*:
the constructor, :meth:`~Similarity.add_documents` and queries all take raw bag-of-words
documents instead of pre-weighted vectors. The index stores the raw counts in disk shards and
owns a default :class:`~gensim.models.tfidfmodel.TfidfModel` whose collection statistics grow
on every accepted batch. Every query is weighted with the statistics of all documents
received so far, so old and new shards always share one collection-wide TF-IDF statistic:

.. sourcecode:: pycon

    >>> from gensim.similarities import Similarity
    >>>
    >>> index = Similarity(get_tmpfile("streaming"), first_batch, num_features=num_terms, tfidf=True)
    >>> index.add_documents(second_batch)  # raw BoW; the owned TfidfModel is updated atomically
    >>> similarities = index[raw_bow_query]  # weighted against the statistics of BOTH batches
    >>> stats = index.tfidf_model.num_docs, index.tfidf_model.dfs  # read-only inspection

A term whose IDF used to be zero (it occurred in every earlier document) is not lost: its raw
counts remain on disk and it becomes an effective retrieval term again once later batches
lower its document frequency. See :class:`Similarity` for the input format, failure atomicity
and persistence guarantees.

"""
import logging
import itertools
import os
import glob
import heapq
import warnings

import numpy
import scipy.sparse

from gensim import interfaces, utils, matutils
from gensim.models.tfidfmodel import TfidfModel, validate_bow_document


logger = logging.getLogger(__name__)

PARALLEL_SHARDS = False
try:
    import multiprocessing
    # by default, don't parallelize queries. uncomment the following line if you want that.
#    PARALLEL_SHARDS = multiprocessing.cpu_count() # use #parallel processes = #CPus
except ImportError:
    pass


class Shard(utils.SaveLoad):
    """A proxy that represents a single shard instance within :class:`~gensim.similarity.docsim.Similarity` index.

    Basically just wraps :class:`~gensim.similarities.docsim.MatrixSimilarity`,
    :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`, etc, so that it mmaps from disk on request (query).

    """
    def __init__(self, fname, index):
        """

        Parameters
        ----------
        fname : str
            Path to top-level directory (file) to traverse for corpus documents.
        index : :class:`~gensim.interfaces.SimilarityABC`
            Index object.

        """
        self.dirname, self.fname = os.path.split(fname)
        self.length = len(index)
        self.cls = index.__class__
        logger.info("saving index shard to %s", self.fullname())
        index.save(self.fullname())
        self.index = self.get_index()

    def fullname(self):
        """Get full path to shard file.

        Return
        ------
        str
            Path to shard instance.

        """
        return os.path.join(self.dirname, self.fname)

    def __len__(self):
        """Get length."""
        return self.length

    def __getstate__(self):
        """Special handler for pickle.

        Returns
        -------
        dict
            Object that contains state of current instance without `index`.

        """
        result = self.__dict__.copy()
        # (S)MS objects must be loaded via load() because of mmap (simple pickle.load won't do)
        if 'index' in result:
            del result['index']
        return result

    def __str__(self):
        return "%s<%i documents in %s>" % (self.cls.__name__, len(self), self.fullname())

    def get_index(self):
        """Load & get index.

        Returns
        -------
        :class:`~gensim.interfaces.SimilarityABC`
            Index instance.

        """
        if not hasattr(self, 'index'):
            logger.debug("mmaping index from %s", self.fullname())
            self.index = self.cls.load(self.fullname(), mmap='r')
        return self.index

    def get_document_id(self, pos):
        """Get index vector at position `pos`.

        Parameters
        ----------
        pos : int
            Vector position.

        Return
        ------
        {:class:`scipy.sparse.csr_matrix`, :class:`numpy.ndarray`}
            Index vector. Type depends on underlying index.

        Notes
        -----
        The vector is of the same type as the underlying index (ie., dense for
        :class:`~gensim.similarities.docsim.MatrixSimilarity`
        and scipy.sparse for :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.

        """
        assert 0 <= pos < len(self), "requested position out of range"
        return self.get_index().index[pos]

    def __getitem__(self, query):
        """Get similarities of document (or corpus) `query` to all documents in the corpus.

        Parameters
        ----------
        query : {iterable of list of (int, number) , list of (int, number))}
            Document or corpus.

        Returns
        -------
        :class:`numpy.ndarray`
            Similarities of document/corpus if index is :class:`~gensim.similarities.docsim.MatrixSimilarity` **or**
        :class:`scipy.sparse.csr_matrix`
            for case if index is :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.

        """
        index = self.get_index()
        try:
            index.num_best = self.num_best
            index.normalize = self.normalize
        except Exception:
            raise ValueError("num_best and normalize have to be set before querying a proxy Shard object")
        return index[query]


class TfidfShard(utils.SaveLoad):
    """A disk-backed shard of *raw bag-of-words* counts for TF-IDF mode.

    Unlike :class:`Shard`, which stores already weighted document vectors, a
    :class:`TfidfShard` stores the original term counts as a
    :class:`scipy.sparse.csr_matrix`. The term counts never change when more documents are
    appended: only the collection statistics owned by the
    :class:`~gensim.models.tfidfmodel.TfidfModel` change. At query time each shard is re-weighted
    with the *current* IDF vector, so old and new shards always share one collection-wide
    statistic. A term whose IDF used to be zero (it appeared in every document of an earlier
    batch) is kept here with its raw count and becomes a valid retrieval term again once later
    batches lower its document frequency.

    The count matrix is stored as separate ``.npy`` files and can be memory-mapped read-only,
    so opening a shard does not copy it into Python heap memory.

    """

    def __init__(self, fname, docs, num_features):
        """

        Parameters
        ----------
        fname : str
            Path of the shard file (without the numpy suffixes added by SaveLoad).
        docs : list of list of (int, float)
            Raw bag-of-words documents in this shard (already validated by the caller).
        num_features : int
            Fixed number of features; every term id is known to be smaller than it.

        """
        self.dirname, self.fname = os.path.split(fname)
        self.num_features = num_features
        self.length = len(docs)
        num_nnz = sum(len(doc) for doc in docs)
        # matutils.corpus2csc gives documents as columns; transpose for documents-as-rows CSR.
        # Raw counts are stored in float32: this is the same storage precision the ordinary
        # (non-TF-IDF) Similarity path uses for its shard matrices.
        self._data = matutils.corpus2csc(
            docs, num_terms=num_features, num_docs=self.length, num_nnz=num_nnz, dtype=numpy.float32
        ).T.tocsr()
        logger.info("saving TF-IDF index shard to %s", self.fullname())
        # always store the count matrix in side files (never inside the pickle), so it can be
        # memory-mapped read-only regardless of the shard's size
        self.save(self.fullname(), separately=['_data'])
        # drop the in-memory build matrix; shards are re-mapped on demand at query time
        del self._data

    def fullname(self):
        """Get full path to the shard file.

        Returns
        -------
        str
            Path to the shard.

        """
        return os.path.join(self.dirname, self.fname)

    def __len__(self):
        """Number of documents stored in this shard."""
        return self.length

    def __getstate__(self):
        """Pickle everything except the (possibly memory-mapped) matrix, which is saved separately."""
        result = self.__dict__.copy()
        if 'index' in result:
            del result['index']
        if '_data' in result:
            del result['_data']
        return result

    def __str__(self):
        return "%s<%i documents in %s>" % (self.__class__.__name__, len(self), self.fullname())

    def get_index(self):
        """Load (or memory-map) the raw count matrix.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            Raw document-term counts, documents as rows. Large arrays are memory-mapped
            read-only through the standard SaveLoad mechanism; small ones are unpickled
            together with the proxy.

        Note
        ----
        Callers that scan many shards (the query path) go through the owner's bounded LRU in
        :meth:`Similarity._all_weighted_shards` instead; this direct accessor is retained for
        single-shard operations such as :meth:`get_document_id` and tests.

        """
        if not hasattr(self, '_data'):
            logger.debug("mmaping TF-IDF shard from %s", self.fullname())
            loaded = utils.SaveLoad.load(self.fullname(), mmap='r')
            self._data = loaded._data
        return self._data

    def load_raw(self):
        """Memory-map the raw matrix without pinning it onto this proxy.

        Returns a matrix the caller owns; nothing is stored on the shard, so the query loop in
        :meth:`Similarity._all_weighted_shards` can release the mapping again after building the
        weighted matrix for one shard.
        """
        loaded = utils.SaveLoad.load(self.fullname(), mmap='r')
        return loaded._data

    def weighted_index(self, idf_values):
        """Build the current L2-normalized TF-IDF matrix for this shard.

        Parameters
        ----------
        idf_values : :class:`numpy.ndarray`
            Current collection-wide IDF weight for every feature, length ``num_features``.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            Row-L2-normalized TF-IDF vectors, one row per stored document. Entries whose weight
            is zero (unknown term or zero IDF) are dropped, matching
            :class:`~gensim.models.tfidfmodel.TfidfModel` semantics. The raw count matrix on
            disk is never modified.

        """
        raw = self.get_index()
        return self.weighted_index_from_raw(raw, idf_values)

    @staticmethod
    def weighted_index_from_raw(raw, idf_values):
        """Apply current IDF weights and row L2 normalization to a raw count matrix.

        Pure function of its two arguments: the raw matrix on disk is never modified and the
        same raw matrix can be re-weighted with successive IDF vectors as the collection grows.
        """
        idf_diag = scipy.sparse.diags(idf_values.astype(numpy.float32, copy=False))
        weighted = (raw @ idf_diag).tocsr()
        weighted.eliminate_zeros()
        # normalize each row to unit L2 length; all-zero rows are returned unchanged
        norms = numpy.sqrt(numpy.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
        inv = numpy.zeros_like(norms)
        nonzero = norms > 0.0
        inv[nonzero] = 1.0 / norms[nonzero]
        weighted = scipy.sparse.diags(inv.astype(numpy.float32)) @ weighted
        weighted.eliminate_zeros()
        return weighted.tocsr()

    def get_document_id(self, pos):
        """Raw count vector of the document at shard-local position `pos`.

        Parameters
        ----------
        pos : int
            Shard-local row position.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            One-row CSR matrix of raw counts.

        """
        assert 0 <= pos < len(self), "requested position out of range"
        return self.get_index()[pos]


def query_shard(args):
    """Helper for request query from shard, same as shard[query].

    Parameters
    ---------
    args : (list of (int, number), :class:`~gensim.interfaces.SimilarityABC`)
        Query and Shard instances

    Returns
    -------
    :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        Similarities of the query against documents indexed in this shard.

    """
    query, shard = args  # simulate starmap (not part of multiprocessing in older Pythons)
    logger.debug("querying shard %s num_best=%s in process %s", shard, shard.num_best, os.getpid())
    result = shard[query]
    logger.debug("finished querying shard %s in process %s", shard, os.getpid())
    return result


def _nlargest(n, iterable):
    """Helper for extracting n documents with maximum similarity.

    Parameters
    ----------
    n : int
        Number of elements to be extracted
    iterable : iterable of list of (int, float)
        Iterable containing documents with computed similarities

    Returns
    -------
    :class:`list`
        List with the n largest elements from the dataset defined by iterable.

    Notes
    -----
    Elements are compared by the absolute value of similarity, because negative value of similarity
    does not mean some form of dissimilarity.

    """
    return heapq.nlargest(n, itertools.chain(*iterable), key=lambda item: abs(item[1]))


class Similarity(interfaces.SimilarityABC):
    """Compute cosine similarity of a dynamic query against a corpus of documents ('the index').

    The index supports adding new documents dynamically.

    Notes
    -----
    Scalability is achieved by sharding the index into smaller pieces, each of which fits into core memory
    The shards themselves are simply stored as files to disk and mmap'ed back as needed.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora.textcorpus import TextCorpus
        >>> from gensim.test.utils import datapath, get_tmpfile
        >>> from gensim.similarities import Similarity
        >>>
        >>> corpus = TextCorpus(datapath('testcorpus.mm'))
        >>> index_temp = get_tmpfile("index")
        >>> index = Similarity(index_temp, corpus, num_features=400)  # create index
        >>>
        >>> query = next(iter(corpus))
        >>> result = index[query]  # search similar to `query` in index
        >>>
        >>> for sims in index[corpus]:  # if you have more query documents, you can submit them all at once, in a batch
        ...     pass
        >>>
        >>> # There is also a special syntax for when you need similarity of documents in the index
        >>> # to the index itself (i.e. queries=indexed documents themselves). This special syntax
        >>> # uses the faster, batch queries internally and **is ideal for all-vs-all pairwise similarities**:
        >>> for similarities in index:  # yield similarities of the 1st indexed document, then 2nd...
        ...     pass

    See Also
    --------
    :class:`~gensim.similarities.docsim.MatrixSimilarity`
        Index similarity (dense with cosine distance).
    :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`
        Index similarity (sparse with cosine distance).
    :class:`~gensim.similarities.docsim.WmdSimilarity`
        Index similarity (with word-mover distance).

    """

    def __init__(self, output_prefix, corpus, num_features, num_best=None, chunksize=256, shardsize=32768,
                 norm='l2', *, tfidf=False):
        """

        Parameters
        ----------
        output_prefix : str
            Prefix for shard filename. If None, a random filename in temp will be used.
        corpus : iterable of list of (int, number)
            Corpus in streamed Gensim bag-of-words format. With ``tfidf=True`` this must be
            raw bag-of-words counts (see the `tfidf` parameter).
        num_features : int
            Size of the dictionary (number of features).
        num_best : int, optional
            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.
            Otherwise, return a full vector with one float for every document in the index.
        chunksize : int, optional
            Size of query chunks. Used internally when the query is an entire corpus.
        shardsize : int, optional
            Maximum shard size, in documents. Choose a value so that a `shardsize x chunksize` matrix of floats fits
            comfortably into your RAM.
        norm : {'l1', 'l2'}, optional
            Normalization to use. When `tfidf` is enabled only ``'l2'`` is allowed.
        tfidf : bool, optional, keyword-only
            If ``False`` (the default), keep the historical behaviour: the index stores the
            vectors passed in by the caller and :meth:`add_documents` appends those vectors
            unchanged.

            If ``True``, switch to *streaming TF-IDF mode*: the constructor,
            :meth:`add_documents` and queries all accept **raw bag-of-words** documents
            (iterables of strictly increasing ``(term_id, count)`` pairs with finite positive
            real counts; empty documents are accepted and still occupy a row). The index keeps
            the raw counts on disk and owns a default
            :class:`~gensim.models.tfidfmodel.TfidfModel` (identity local weight,
            :func:`~gensim.models.tfidfmodel.df2idf` global weight, L2-normalized) whose
            collection statistics are extended on every batch. Every query -- against old and
            new documents alike -- is weighted with the TF-IDF statistics computed from *all
            documents received so far*, which is equivalent to rebuilding a default model over
            the whole accumulated corpus. The model is exposed read-only through the
            :attr:`tfidf_model` property. Term ids must be smaller than `num_features`.
            Only ``norm='l2'`` is supported in this mode; an invalid combination is rejected
            before any input is read.

        Notes
        -----
        Documents are split (internally, transparently) into shards of `shardsize` documents each, and each shard
        converted to a matrix, for faster BLAS calls. Each shard is stored to disk under `output_prefix.shard_number`.

        If you don't specify an output prefix, a random filename in temp will be used.

        If your entire index fits in memory (~1 million documents per 1GB of RAM), you can also use the
        :class:`~gensim.similarities.docsim.MatrixSimilarity` or
        :class:`~gensim.similarities.docsim.SparseMatrixSimilarity` classes directly.
        These are more simple but do not scale as well (they keep the entire index in RAM, no sharding).
        They also do not support adding new document dynamically.

        Streaming TF-IDF mode (``tfidf=True``)
        --------------------------------------
        * Input documents are validated as they are streamed; a malformed document or an exception
          raised by the input iterator aborts the whole batch atomically: no document, no model
          statistic and no queryable content from the failed batch is kept, earlier batches remain
          queryable and a later valid batch can be appended normally.
        * Raw counts are spilled to disk in shards of at most `shardsize` documents; no full
          historical copy of the corpus or of its weighted vectors is held in working memory.
        * A term that had zero IDF in earlier batches (it occurred in every document) keeps its raw
          counts on disk and starts contributing to queries as soon as later batches lower its
          document frequency.
        * :meth:`save` / :meth:`load` persist the raw shards together with the model statistics, so
          an index can be reloaded and appended to with identical results to an uninterrupted run.

        """
        if output_prefix is None:
            # undocumented feature: set output_prefix=None to create the server in temp
            self.output_prefix = utils.randfname(prefix='simserver')
        else:
            self.output_prefix = output_prefix
        logger.info("starting similarity index under %s", self.output_prefix)
        self.tfidf = bool(tfidf)
        if self.tfidf and norm != 'l2':
            raise ValueError(
                "streaming TF-IDF mode (tfidf=True) only supports norm='l2', got norm=%r" % (norm,)
            )
        if self.tfidf and (not isinstance(num_features, int) or isinstance(num_features, bool)
                           or num_features <= 0):
            raise ValueError(
                "tfidf=True requires a positive integer num_features, got %r" % (num_features,)
            )
        self.num_features = num_features
        self.num_best = num_best
        self.norm = norm
        self.chunksize = int(chunksize)
        self.shardsize = shardsize
        self.shards = []
        self.fresh_docs, self.fresh_nnz = [], 0
        if self.tfidf:
            # the index owns this model; callers inspect it through the read-only `tfidf_model`
            # property and are not allowed to replace or reconfigure it
            self._tfidf_model = TfidfModel()

        if corpus is not None:
            self.add_documents(corpus)

    @property
    def tfidf_model(self):
        """The :class:`~gensim.models.tfidfmodel.TfidfModel` owned by this index (TF-IDF mode only).

        Its collection statistics are extended automatically by every successful
        :meth:`add_documents` call. The property is read-only: assigning to ``tfidf_model``
        raises :class:`AttributeError`. The model object itself must not be mutated by the
        caller, because the on-disk raw-count shards and the model statistics must always
        describe the same collection.

        Raises
        ------
        AttributeError
            Outside of streaming TF-IDF mode (``tfidf=False``), and on any assignment attempt.

        """
        if not getattr(self, 'tfidf', False):
            raise AttributeError(
                "tfidf_model is only available in streaming TF-IDF mode (construct with tfidf=True)"
            )
        return self._tfidf_model

    def __len__(self):
        """Get length of index."""
        return len(self.fresh_docs) + sum(len(shard) for shard in self.shards)

    def __str__(self):
        return "%s<%i documents in %i shards stored under %s>" % (
            self.__class__.__name__, len(self), len(self.shards), self.output_prefix
        )

    def add_documents(self, corpus):
        """Extend the index with new documents.

        Parameters
        ----------
        corpus : iterable of list of (int, number)
            Corpus in BoW format. In streaming TF-IDF mode (``tfidf=True``) this must be raw
            bag-of-words counts following the format documented on the constructor; every
            term id must be smaller than ``num_features``.

        Notes
        -----
        Internally, documents are buffered and then spilled to disk when there's `self.shardsize` of them
        (or when a query is issued).

        In streaming TF-IDF mode the batch is applied atomically: documents are validated while
        streaming from `corpus` (which is iterated exactly once), and any malformed document or
        exception raised by the iterator rolls back every document and every model statistic of
        this batch, leaving earlier batches queryable.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath, get_tmpfile
            >>> from gensim.similarities import Similarity
            >>>
            >>> corpus = TextCorpus(datapath('testcorpus.mm'))
            >>> index_temp = get_tmpfile("index")
            >>> index = Similarity(index_temp, corpus, num_features=400)  # create index
            >>>
            >>> one_more_corpus = TextCorpus(datapath('testcorpus.txt'))
            >>> index.add_documents(one_more_corpus)  # add more documents in corpus

        """
        if getattr(self, 'tfidf', False):
            self._add_documents_tfidf(corpus)
            return

        min_ratio = 1.0  # 0.5 to only reopen shards that are <50% complete
        if self.shards and len(self.shards[-1]) < min_ratio * self.shardsize:
            # The last shard was incomplete (<; load it back and add the documents there, don't start a new shard
            self.reopen_shard()
        for doc in corpus:
            if isinstance(doc, numpy.ndarray):
                doclen = len(doc)
            elif scipy.sparse.issparse(doc):
                doclen = doc.nnz
            else:
                doclen = len(doc)
                if doclen < 0.3 * self.num_features:
                    doc = matutils.unitvec(matutils.corpus2csc([doc], self.num_features).T, self.norm)
                else:
                    doc = matutils.unitvec(matutils.sparse2full(doc, self.num_features), self.norm)
            self.fresh_docs.append(doc)
            self.fresh_nnz += doclen
            if len(self.fresh_docs) >= self.shardsize:
                self.close_shard()
            if len(self.fresh_docs) % 10000 == 0:
                logger.info("PROGRESS: fresh_shard size=%i", len(self.fresh_docs))

    # ------------------------------------------------------------------ #
    # streaming TF-IDF mode internals
    # ------------------------------------------------------------------ #
    def _tfidf_shard_filename(self, shardid, in_progress=False):
        """Path used for a TF-IDF shard; in-progress shards carry a temporary suffix."""
        base = self.shardid2filename(shardid)
        return base + '.inprogress' if in_progress else base

    def _spill_tfidf_shard(self, docs, shardid, in_progress):
        """Write one raw-count shard to disk and return its :class:`TfidfShard` proxy."""
        fname = self._tfidf_shard_filename(shardid, in_progress=in_progress)
        return TfidfShard(fname, docs, self.num_features)

    def _add_documents_tfidf(self, corpus):
        """Atomic raw-BoW ingestion for streaming TF-IDF mode.

        The input iterator is consumed exactly once. Nothing visible to queries changes until
        the whole batch has been validated: candidate shards are written under temporary names
        and the model statistics are merged only at commit time. On any exception the temporary
        shards are deleted and every piece of index state is restored to its pre-batch value.

        Working memory is bounded by `shardsize` documents (the current spill buffer) plus the
        model's document-frequency dictionary whose size is bounded by the vocabulary.
        """
        model = self._tfidf_model

        # snapshot of every piece of state a failure must restore (an uninitialized
        # TfidfModel has no dfs/idfs yet, which is represented by None)
        snapshot = (
            list(self.shards), len(self.fresh_docs), self.fresh_nnz,
            getattr(model, 'num_docs', None), getattr(model, 'num_nnz', None),
            getattr(model, 'dfs', None), getattr(model, 'idfs', None),
        )
        fresh_snapshot = [list(doc) for doc in self.fresh_docs]
        candidate_shards = []
        # batch-local statistics, merged into the model only at commit time
        batch_docs, batch_nnz = 0, 0
        batch_dfs = {}

        def spill(buffer_docs):
            shardid = len(self.shards) + len(candidate_shards)
            shard = self._spill_tfidf_shard(buffer_docs, shardid, in_progress=True)
            candidate_shards.append(shard)

        try:
            # an open partial shard from an earlier batch is kept as the query-time buffer; the
            # new batch fills it first, then every shardsize documents form a new shard.
            # buffer_docs always holds at most shardsize documents.
            buffer_docs = [list(doc) for doc in fresh_snapshot]
            for raw_doc in corpus:
                # the same validation as TfidfModel.add_documents, plus the fixed id range
                doc = validate_bow_document(raw_doc, max_id=self.num_features)
                batch_docs += 1
                batch_nnz += len(doc)
                for termid, _ in doc:
                    batch_dfs[termid] = batch_dfs.get(termid, 0) + 1
                buffer_docs.append(doc)
                if len(buffer_docs) >= self.shardsize:
                    spill(buffer_docs)
                    buffer_docs = []
                    if batch_docs % 10000 == 0:
                        logger.info("PROGRESS: TF-IDF batch processed %i documents", batch_docs)

            if batch_docs == 0:
                # an empty batch changes nothing; leave the pre-existing buffer untouched
                return

            # commit phase: first make the model statistics describe the whole collection ...
            model._merge_batch_statistics(batch_docs, batch_nnz, batch_dfs)

            # ... then publish the on-disk shards under their final names. shards that already
            # existed before this batch are never touched (no rewrite of historical data).
            published = []
            for position, shard in enumerate(candidate_shards):
                final_id = len(self.shards) + position
                final_name = self._tfidf_shard_filename(final_id, in_progress=False)
                tmp_name = shard.fullname()
                # the proxy plus every numpy/scipy side file SaveLoad created for this shard
                for tmp_path in sorted(glob.glob(tmp_name + '*')):
                    suffix = tmp_path[len(tmp_name):]
                    os.rename(tmp_path, final_name + suffix)
                shard.dirname, shard.fname = os.path.split(final_name)
                if hasattr(shard, 'index'):
                    del shard.index
                if hasattr(shard, '_data'):
                    del shard._data
                published.append(shard)
            self.shards.extend(published)
            # documents left in the spill buffer stay queryable via close_shard() at query/save
            # time; buffer_docs already contains the pre-existing partial shard plus new tail docs
            self.fresh_docs = buffer_docs
            self.fresh_nnz = sum(len(doc) for doc in buffer_docs)
        except BaseException:
            # roll back: drop every file the failed batch wrote and restore the snapshot
            for shard in candidate_shards:
                self._delete_shard_files(shard.fullname())
            self.shards = snapshot[0]
            self.fresh_docs = fresh_snapshot
            self.fresh_nnz = snapshot[2]
            model.num_docs, model.num_nnz = snapshot[3], snapshot[4]
            if snapshot[5] is None:
                # the model was uninitialized before this batch: remove the attributes a merge
                # would have created, instead of leaving empty dictionaries behind
                for attr in ('dfs', 'idfs', 'cfs', 'term_lengths'):
                    if hasattr(model, attr):
                        delattr(model, attr)
            else:
                model.dfs, model.idfs = snapshot[5], snapshot[6]
            raise

    @staticmethod
    def _delete_shard_files(prefix):
        """Best-effort removal of every file belonging to one shard (proxy + numpy arrays)."""
        for fname in glob.glob(prefix + '*'):
            try:
                os.remove(fname)
            except OSError:
                logger.warning("could not remove temporary shard file %s", fname)

    def _current_idf_vector(self):
        """Current IDF weight of every feature as a length-``num_features`` numpy array."""
        idfs = self._tfidf_model.idfs
        idf = numpy.zeros(self.num_features, dtype=numpy.float32)
        if idfs:
            ids = numpy.fromiter(idfs.keys(), dtype=numpy.int64, count=len(idfs))
            vals = numpy.fromiter(idfs.values(), dtype=numpy.float64, count=len(idfs))
            idf[ids.astype(numpy.intp)] = vals.astype(numpy.float32)
        return idf

    def _all_weighted_shards(self, idf):
        """Yield ``(row_offset, weighted_csr)`` for every closed shard and the current buffer.

        The buffer (documents accepted since the last query but not yet a full shard) is
        materialized through the same raw-count path, so buffered and closed documents share
        exactly the same weighting code.

        Each shard's raw matrix is memory-mapped, used to build one weighted matrix and
        released again within the loop: resident memory stays bounded by one shard plus the
        current query chunk instead of growing with the number of historical shards. The
        operating system page cache still makes repeated accesses fast.
        """
        offset = 0
        for shard in self.shards:
            raw = shard.load_raw()
            try:
                weighted = shard.weighted_index_from_raw(raw, idf)
            finally:
                del raw  # close the mapping as soon as the weighted copy exists
            yield offset, weighted
            offset += len(shard)
        if self.fresh_docs:
            tmp = TfidfShard.__new__(TfidfShard)
            tmp.dirname, tmp.fname = '', ''
            tmp.num_features = self.num_features
            tmp.length = len(self.fresh_docs)
            tmp._data = matutils.corpus2csc(
                self.fresh_docs, num_terms=self.num_features, num_docs=len(self.fresh_docs),
                num_nnz=sum(len(doc) for doc in self.fresh_docs), dtype=numpy.float32
            ).T.tocsr()
            yield offset, tmp.weighted_index(idf)

    def _build_query_csr(self, query_docs):
        """Transform one validated raw-BoW query chunk into an L2-normalized CSR query matrix."""
        model = self._tfidf_model
        transformed = [model[doc] for doc in query_docs]
        return matutils.corpus2csc(
            transformed, num_terms=self.num_features, num_docs=len(transformed),
            num_nnz=sum(len(doc) for doc in transformed), dtype=numpy.float32
        ).T.tocsr()

    def _query_tfidf(self, query):
        """Query entry point for streaming TF-IDF mode."""
        self.close_shard()  # flush buffered documents so row numbers cover every accepted doc

        is_corpus, query_obj = utils.is_corpus(query)
        is_corpus = is_corpus or (
            hasattr(query_obj, 'ndim') and query_obj.ndim > 1 and query_obj.shape[0] > 1
        )

        # an index that never received documents has no statistics at all: every result is zero
        if len(self) == 0:
            if not is_corpus:
                result = numpy.zeros(0, dtype=numpy.float32)
            else:
                n_queries = query_obj.shape[0] if hasattr(query_obj, 'shape') else len(query_obj)
                result = numpy.zeros((n_queries, 0), dtype=numpy.float32)
            if self.num_best is not None:
                # no documents means no top-k hits; one empty list per query document
                if not is_corpus:
                    return []
                return [[] for _ in range(result.shape[0])]
            return result

        idf = self._current_idf_vector()

        if not is_corpus:
            # a single raw-BoW document; validate before touching any shard
            bow = validate_bow_document(query_obj, max_id=self.num_features)
            transformed = self._tfidf_model[bow]
            qvec = matutils.corpus2csc(
                [transformed], num_terms=self.num_features, num_docs=1,
                num_nnz=len(transformed), dtype=numpy.float32
            ).T.tocsr()
            return self._score_tfidf(qvec, idf, single=True)

        # corpus query: stream the input in chunksize-bounded pieces. Only one chunk of query
        # documents and one chunk of score rows is held at a time; the final stacked score array
        # is the documented return value and is not counted as working area (same convention as
        # the ordinary Similarity path).
        result_chunks = []
        chunk = []
        saw_any = False
        for raw_query_doc in query_obj:
            chunk.append(validate_bow_document(raw_query_doc, max_id=self.num_features))
            saw_any = True
            if len(chunk) >= self.chunksize:
                qchunk = self._build_query_csr(chunk)
                result_chunks.append(self._score_tfidf(qchunk, idf, single=False))
                chunk = []
        if not saw_any:
            # an empty corpus query returns an empty result, shaped like the ordinary path
            return numpy.empty((0, len(self)), dtype=numpy.float32)
        if chunk:
            qchunk = self._build_query_csr(chunk)
            result_chunks.append(self._score_tfidf(qchunk, idf, single=False))
        result = numpy.vstack(result_chunks)

        if self.num_best is not None:
            return [self._topk_rows(row) for row in result]
        return result

    def _score_tfidf(self, qcsr, idf, single):
        """Compute similarity of one CSR query matrix against every weighted shard."""
        pieces = []
        for offset, weighted in self._all_weighted_shards(idf):
            # (N x T) @ (T x Q) -> N x Q; both matrices' rows are L2-normalized already.
            # shards are emitted in row order with contiguous offsets, so hstacking their
            # results reproduces the global document row numbers
            sims = (weighted @ qcsr.T).toarray().T.astype(numpy.float32)
            pieces.append(sims)
        if not pieces:
            sims = numpy.empty((qcsr.shape[0], 0), dtype=numpy.float32)
        else:
            sims = numpy.hstack(pieces)

        if self.num_best is not None:
            if single:
                return self._topk_rows(sims[0])
            return sims
        if single:
            return sims[0]
        return sims

    def _topk_rows(self, row):
        """Top-`num_best` ``(row_number, score)`` pairs of one score row.

        Zero scores are excluded; equal positive scores break ties by ascending row number,
        which makes the ordering deterministic and independent of shard boundaries.
        """
        if self.num_best <= 0:
            return []
        nnz = numpy.nonzero(row > 1e-12)[0]
        if nnz.size == 0:
            return []
        # stable sort on (-score, row): exact zeros are already removed and ties keep row order
        order = numpy.lexsort((nnz, -row[nnz]))
        picked = nnz[order[:self.num_best]]
        return [(int(pos), float(row[pos])) for pos in picked]

    def shardid2filename(self, shardid):
        """Get shard file by `shardid`.

        Parameters
        ----------
        shardid : int
            Shard index.

        Return
        ------
        str
            Path to shard file.

        """
        if self.output_prefix.endswith('.'):
            return "%s%s" % (self.output_prefix, shardid)
        else:
            return "%s.%s" % (self.output_prefix, shardid)

    def close_shard(self):
        """Force the latest shard to close (be converted to a matrix and stored to disk).
         Do nothing if no new documents added since last call.

        Notes
        -----
        The shard is closed even if it is not full yet (its size is smaller than `self.shardsize`).
        If documents are added later via :meth:`~gensim.similarities.docsim.MatrixSimilarity.add_documents`
        this incomplete shard will be loaded again and completed.

        """
        if not self.fresh_docs:
            return
        shardid = len(self.shards)
        if getattr(self, 'tfidf', False):
            # raw counts are spilled as-is; TF-IDF weighting is deferred to query time so that
            # the shard always uses the collection statistics current at query time
            shard = self._spill_tfidf_shard(self.fresh_docs, shardid, in_progress=False)
            self.shards.append(shard)
            self.fresh_docs, self.fresh_nnz = [], 0
            return
        # consider the shard sparse if its density is < 30%
        issparse = 0.3 > 1.0 * self.fresh_nnz / (len(self.fresh_docs) * self.num_features)
        if issparse:
            index = SparseMatrixSimilarity(
                self.fresh_docs, num_terms=self.num_features, num_docs=len(self.fresh_docs), num_nnz=self.fresh_nnz
            )
        else:
            index = MatrixSimilarity(self.fresh_docs, num_features=self.num_features)
        logger.info("creating %s shard #%s", 'sparse' if issparse else 'dense', shardid)
        shard = Shard(self.shardid2filename(shardid), index)
        shard.num_best = self.num_best
        shard.num_nnz = self.fresh_nnz
        self.shards.append(shard)
        self.fresh_docs, self.fresh_nnz = [], 0

    def reopen_shard(self):
        """Reopen an incomplete shard."""
        assert self.shards
        if self.fresh_docs:
            raise ValueError("cannot reopen a shard with fresh documents in index")
        last_shard = self.shards[-1]
        last_index = last_shard.get_index()
        logger.info("reopening an incomplete shard of %i documents", len(last_shard))

        self.fresh_docs = list(last_index.index)
        self.fresh_nnz = last_shard.num_nnz
        del self.shards[-1]  # remove the shard from index, *but its file on disk is not deleted*
        logger.debug("reopen complete")

    def query_shards(self, query):
        """Apply shard[query] to each shard in `self.shards`. Used internally.

        Parameters
        ----------
        query : {iterable of list of (int, number) , list of (int, number))}
            Document in BoW format or corpus of documents.

        Returns
        -------
        (None, list of individual shard query results)
            Query results.

        """
        args = zip([query] * len(self.shards), self.shards)
        if PARALLEL_SHARDS and PARALLEL_SHARDS > 1:
            logger.debug("spawning %i query processes", PARALLEL_SHARDS)
            pool = multiprocessing.Pool(PARALLEL_SHARDS)
            result = pool.imap(query_shard, args, chunksize=1 + len(self.shards) / PARALLEL_SHARDS)
        else:
            # serial processing, one shard after another
            pool = None
            result = map(query_shard, args)
        return pool, result

    def __getitem__(self, query):
        """Get similarities of the document (or corpus) `query` to all documents in the corpus.

        Parameters
        ----------
        query : {iterable of list of (int, number) , list of (int, number))}
            A single document in bag-of-words format, or a corpus (iterable) of such documents.

        Return
        ------
        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
            Similarities of the query against this index.

        Notes
        -----
        If `query` is a corpus (iterable of documents), return a matrix of similarities of
        all query documents vs. all corpus document. This batch query is more efficient than computing the similarities
        one document after another.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath
            >>> from gensim.similarities import Similarity
            >>>
            >>> corpus = TextCorpus(datapath('testcorpus.txt'))
            >>> index = Similarity('temp', corpus, num_features=400)
            >>> result = index[corpus]  # pairwise similarities of each document against each document

        """
        if getattr(self, 'tfidf', False):
            # streaming TF-IDF mode: queries are raw BoW and get weighted with the current
            # collection statistics inside _query_tfidf
            return self._query_tfidf(query)

        self.close_shard()  # no-op if no documents added to index since last query

        # reset num_best and normalize parameters, in case they were changed dynamically
        for shard in self.shards:
            shard.num_best = self.num_best
            shard.normalize = self.norm

        # there are 4 distinct code paths, depending on whether input `query` is
        # a corpus (or numpy/scipy matrix) or a single document, and whether the
        # similarity result should be a full array or only num_best most similar
        # documents.
        pool, shard_results = self.query_shards(query)
        if self.num_best is None:
            # user asked for all documents => just stack the sub-results into a single matrix
            # (works for both corpus / single doc query)
            result = numpy.hstack(list(shard_results))
        else:
            # the following uses a lot of lazy evaluation and (optionally) parallel
            # processing, to improve query latency and minimize memory footprint.
            offsets = numpy.cumsum([0] + [len(shard) for shard in self.shards])

            def convert(shard_no, doc):
                return [(doc_index + offsets[shard_no], sim) for doc_index, sim in doc]

            is_corpus, query = utils.is_corpus(query)
            is_corpus = is_corpus or hasattr(query, 'ndim') and query.ndim > 1 and query.shape[0] > 1
            if not is_corpus:
                # user asked for num_best most similar and query is a single doc
                results = (convert(shard_no, result) for shard_no, result in enumerate(shard_results))
                result = _nlargest(self.num_best, results)
            else:
                # the trickiest combination: returning num_best results when query was a corpus
                results = []
                for shard_no, result in enumerate(shard_results):
                    shard_result = [convert(shard_no, doc) for doc in result]
                    results.append(shard_result)
                result = []
                for parts in zip(*results):
                    merged = _nlargest(self.num_best, parts)
                    result.append(merged)
        if pool:
            # gc doesn't seem to collect the Pools, eventually leading to
            # "IOError 24: too many open files". so let's terminate it manually.
            pool.terminate()

        return result

    def vector_by_id(self, docpos):
        """Get the indexed vector corresponding to the document at position `docpos`.

        Parameters
        ----------
        docpos : int
            Document position

        Return
        ------
        :class:`scipy.sparse.csr_matrix`
            Indexed vector.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath
            >>> from gensim.similarities import Similarity
            >>>
            >>> # Create index:
            >>> corpus = TextCorpus(datapath('testcorpus.txt'))
            >>> index = Similarity('temp', corpus, num_features=400)
            >>> vector = index.vector_by_id(1)

        """
        self.close_shard()  # no-op if no documents added to index since last query
        pos = 0
        for shard in self.shards:
            pos += len(shard)
            if docpos < pos:
                break
        if not self.shards or docpos < 0 or docpos >= pos:
            raise ValueError("invalid document position: %s (must be 0 <= x < %s)" % (docpos, len(self)))
        local_pos = docpos - pos + len(shard)
        if getattr(self, 'tfidf', False):
            # return the *current* L2-normalized TF-IDF vector, not the raw counts
            idf = self._current_idf_vector()
            return shard.weighted_index(idf)[local_pos]
        result = shard.get_document_id(local_pos)
        return result

    def similarity_by_id(self, docpos):
        """Get similarity of a document specified by its index position `docpos`.

        Parameters
        ----------
        docpos : int
            Document position in the index.

        Return
        ------
        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
            Similarities of the given document against this index.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath
            >>> from gensim.similarities import Similarity
            >>>
            >>> corpus = TextCorpus(datapath('testcorpus.txt'))
            >>> index = Similarity('temp', corpus, num_features=400)
            >>> similarities = index.similarity_by_id(1)

        """
        if getattr(self, 'tfidf', False):
            # vector_by_id already returns the current normalized TF-IDF row; score it directly
            # against every weighted shard without round-tripping through the raw-BoW validator
            self.close_shard()
            idf = self._current_idf_vector()
            qvec = self.vector_by_id(docpos)
            return self._score_tfidf(qvec, idf, single=True)
        query = self.vector_by_id(docpos)
        norm, self.norm = self.norm, False
        result = self[query]
        self.norm = norm
        return result

    def __iter__(self):
        """For each index document in index, compute cosine similarity against all other documents in the index.
        Uses :meth:`~gensim.similarities.docsim.Similarity.iter_chunks` internally.

        Yields
        ------
        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
            Similarities of each document in turn against the index.

        """
        if getattr(self, 'tfidf', False):
            # stream the indexed raw BoW documents themselves as chunked queries, so each row is
            # weighted with the current collection statistics exactly like an external query
            self.close_shard()
            for chunk in self.iter_chunks():
                # chunk is a small list of raw BoW documents
                sims = self[chunk]
                if sims.shape[0] > 1:
                    for sim in sims:
                        yield sim
                else:
                    yield sims[0]
            return

        # turn off query normalization (vectors in the index are already normalized, save some CPU)
        norm, self.norm = self.norm, False

        for chunk in self.iter_chunks():
            if chunk.shape[0] > 1:
                for sim in self[chunk]:
                    yield sim
            else:
                yield self[chunk]

        self.norm = norm  # restore normalization

    def iter_chunks(self, chunksize=None):
        """Iteratively yield the index as chunks of document vectors, each of size <= chunksize.

        Parameters
        ----------
        chunksize : int, optional
            Size of chunk,, if None - `self.chunksize` will be used.

        Yields
        ------
        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
            Chunks of the index as 2D arrays. The arrays are either dense or sparse, depending on
            whether the shard was storing dense or sparse vectors.

        """
        self.close_shard()

        if chunksize is None:
            # if not explicitly specified, use the chunksize from the constructor
            chunksize = self.chunksize

        if getattr(self, 'tfidf', False):
            # yield raw BoW chunks (lists of (term_id, count) pairs) instead of matrix slices;
            # each yielded chunk holds at most `chunksize` documents
            for shard in self.shards:
                raw = shard.get_index()
                for chunk_start in range(0, len(shard), chunksize):
                    chunk_end = min(len(shard), chunk_start + chunksize)
                    yield [matutils.scipy2sparse(raw[pos], eps=0.0) for pos in range(chunk_start, chunk_end)]
            return

        for shard in self.shards:
            query = shard.get_index().index
            for chunk_start in range(0, query.shape[0], chunksize):
                # scipy.sparse doesn't allow slicing beyond real size of the matrix
                # (unlike numpy). so, clip the end of the chunk explicitly to make
                # scipy.sparse happy
                chunk_end = min(query.shape[0], chunk_start + chunksize)
                chunk = query[chunk_start: chunk_end]  # create a view
                yield chunk

    def check_moved(self):
        """Update shard locations, for case where the server prefix location changed on the filesystem."""
        dirname = os.path.dirname(self.output_prefix)
        for shard in self.shards:
            shard.dirname = dirname

    def save(self, fname=None, *args, **kwargs):
        """Save the index object via pickling under `fname`. See also :meth:`~gensim.docsim.Similarity.load()`.

        Parameters
        ----------
        fname : str, optional
            Path for save index, if not provided - will be saved to `self.output_prefix`.
        *args : object
            Arguments, see :meth:`gensim.utils.SaveLoad.save`.
        **kwargs : object
            Keyword arguments, see :meth:`gensim.utils.SaveLoad.save`.

        Notes
        -----
        Will call :meth:`~gensim.similarities.Similarity.close_shard` internally to spill
        any unfinished shards to disk first.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath, get_tmpfile
            >>> from gensim.similarities import Similarity
            >>>
            >>> temp_fname = get_tmpfile("index")
            >>> output_fname = get_tmpfile("saved_index")
            >>>
            >>> corpus = TextCorpus(datapath('testcorpus.txt'))
            >>> index = Similarity(output_fname, corpus, num_features=400)
            >>>
            >>> index.save(output_fname)
            >>> loaded_index = index.load(output_fname)

        """
        self.close_shard()
        if fname is None:
            fname = self.output_prefix
        super(Similarity, self).save(fname, *args, **kwargs)

    def destroy(self):
        """Delete all files under self.output_prefix Index is not usable anymore after calling this method."""
        import glob
        for fname in glob.glob(self.output_prefix + '*'):
            logger.info("deleting %s", fname)
            os.remove(fname)


class MatrixSimilarity(interfaces.SimilarityABC):
    """Compute cosine similarity against a corpus of documents by storing the index matrix in memory.

    Unless the entire matrix fits into main memory, use :class:`~gensim.similarities.docsim.Similarity` instead.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.similarities import MatrixSimilarity
        >>>
        >>> query = [(1, 2), (5, 4)]
        >>> index = MatrixSimilarity(common_corpus, num_features=len(common_dictionary))
        >>> sims = index[query]

    """
    def __init__(self, corpus, num_best=None, dtype=numpy.float32, num_features=None, chunksize=256, corpus_len=None):
        """

        Parameters
        ----------
        corpus : iterable of list of (int, number)
            Corpus in streamed Gensim bag-of-words format.
        num_best : int, optional
            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.
            Otherwise, return a full vector with one float for every document in the index.
        num_features : int
            Size of the dictionary (number of features).
        corpus_len : int, optional
            Number of documents in `corpus`. If not specified, will scan the corpus to determine the matrix size.
        chunksize : int, optional
            Size of query chunks. Used internally when the query is an entire corpus.
        dtype : numpy.dtype, optional
            Datatype to store the internal matrix in.

        """
        if num_features is None:
            logger.warning(
                "scanning corpus to determine the number of features (consider setting `num_features` explicitly)"
            )
            num_features = 1 + utils.get_max_id(corpus)

        self.num_features = num_features
        self.num_best = num_best
        self.normalize = True
        self.chunksize = chunksize
        if corpus_len is None:
            corpus_len = len(corpus)

        if corpus is not None:
            if self.num_features <= 0:
                raise ValueError(
                    "cannot index a corpus with zero features (you must specify either `num_features` "
                    "or a non-empty corpus in the constructor)"
                )
            logger.info("creating matrix with %i documents and %i features", corpus_len, num_features)
            self.index = numpy.empty(shape=(corpus_len, num_features), dtype=dtype)
            # iterate over corpus, populating the numpy index matrix with (normalized)
            # document vectors
            for docno, vector in enumerate(corpus):
                if docno % 1000 == 0:
                    logger.debug("PROGRESS: at document #%i/%i", docno, corpus_len)
                # individual documents in fact may be in numpy.scipy.sparse format as well.
                # it's not documented because other it's not fully supported throughout.
                # the user better know what he's doing (no normalization, must
                # explicitly supply num_features etc).
                if isinstance(vector, numpy.ndarray):
                    pass
                elif scipy.sparse.issparse(vector):
                    vector = vector.toarray().flatten()
                else:
                    vector = matutils.unitvec(matutils.sparse2full(vector, num_features))
                self.index[docno] = vector

    def __len__(self):
        return self.index.shape[0]

    def get_similarities(self, query):
        """Get similarity between `query` and this index.

        Warnings
        --------
        Do not use this function directly, use the :class:`~gensim.similarities.docsim.MatrixSimilarity.__getitem__`
        instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`}
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix.

        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = numpy.asarray(
                [matutils.sparse2full(vec, self.num_features) for vec in query],
                dtype=self.index.dtype
            )
        else:
            if scipy.sparse.issparse(query):
                query = query.toarray()  # convert sparse to dense
            elif isinstance(query, numpy.ndarray):
                pass
            else:
                # default case: query is a single vector in sparse gensim format
                query = matutils.sparse2full(query, self.num_features)
            query = numpy.asarray(query, dtype=self.index.dtype)

        # do a little transposition dance to stop numpy from making a copy of
        # self.index internally in numpy.dot (very slow).
        result = numpy.dot(self.index, query.T).T  # return #queries x #index
        return result  # XXX: removed casting the result from array to list; does anyone care?

    def __str__(self):
        return "%s<%i docs, %i features>" % (self.__class__.__name__, len(self), self.index.shape[1])


class SoftCosineSimilarity(interfaces.SimilarityABC):
    """Compute soft cosine similarity against a corpus of documents by storing the index matrix in memory.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_texts
        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import Word2Vec
        >>> from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
        >>> from gensim.similarities import WordEmbeddingSimilarityIndex
        >>>
        >>> model = Word2Vec(common_texts, vector_size=20, min_count=1)  # train word-vectors
        >>> termsim_index = WordEmbeddingSimilarityIndex(model.wv)
        >>> dictionary = Dictionary(common_texts)
        >>> bow_corpus = [dictionary.doc2bow(document) for document in common_texts]
        >>> similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix
        >>> docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
        >>>
        >>> query = 'graph trees computer'.split()  # make a query
        >>> sims = docsim_index[dictionary.doc2bow(query)]  # calculate similarity of query to each doc from bow_corpus

    Check out `the Gallery <https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html>`__
    for more examples.

    """
    def __init__(self, corpus, similarity_matrix, num_best=None, chunksize=256, normalized=None,
                 normalize_queries=True, normalize_documents=True):
        """

        Parameters
        ----------
        corpus: iterable of list of (int, float)
            A list of documents in the BoW format.
        similarity_matrix : :class:`gensim.similarities.SparseTermSimilarityMatrix`
            A term similarity matrix.
        num_best : int, optional
            The number of results to retrieve for a query, if None - return similarities with all elements from corpus.
        chunksize: int, optional
            Size of one corpus chunk.
        normalized : tuple of {True, False, 'maintain', None}, optional
            A deprecated alias for `(normalize_queries, normalize_documents)`. If None, use
            `normalize_queries` and `normalize_documents`. Default is None.
        normalize_queries : {True, False, 'maintain'}, optional
            Whether the query vector in the inner product will be L2-normalized (True; corresponds
            to the soft cosine similarity measure; default), maintain their L2-norm during change of
            basis ('maintain'; corresponds to queryexpansion with partial membership), or kept as-is
            (False;  corresponds to query expansion).
        normalize_documents : {True, False, 'maintain'}, optional
            Whether the document vector in the inner product will be L2-normalized (True; corresponds
            to the soft cosine similarity measure; default), maintain their L2-norm during change of
            basis ('maintain'; corresponds to queryexpansion with partial membership), or kept as-is
            (False;  corresponds to query expansion).

        See Also
        --------
        :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
            A sparse term similarity matrix built using a term similarity index.
        :class:`~gensim.similarities.termsim.LevenshteinSimilarityIndex`
            A term similarity index that computes Levenshtein similarities between terms.
        :class:`~gensim.similarities.termsim.WordEmbeddingSimilarityIndex`
            A term similarity index that computes cosine similarities between word embeddings.

        """
        self.similarity_matrix = similarity_matrix

        self.corpus = list(corpus)
        self.num_best = num_best
        self.chunksize = chunksize
        if normalized is not None:
            warnings.warn(
                'Parameter normalized will be removed in 5.0.0, use normalize_queries and normalize_documents instead',
                category=DeprecationWarning,
            )
            self.normalized = normalized
        else:
            self.normalized = (normalize_queries, normalize_documents)

        # Normalization of features is undesirable, since soft cosine similarity requires special
        # normalization using the similarity matrix. Therefore, we would just be normalizing twice,
        # increasing the numerical error.
        self.normalize = False

        # index is simply an array from 0 to size of corpus.
        self.index = numpy.arange(len(corpus))

    def __len__(self):
        return len(self.corpus)

    def get_similarities(self, query):
        """Get similarity between `query` and this index.

        Warnings
        --------
        Do not use this function directly; use the `self[query]` syntax instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number)}
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix.

        """
        if not self.corpus:
            return numpy.array()

        is_corpus, query = utils.is_corpus(query)
        if not is_corpus and isinstance(query, numpy.ndarray):
            query = [self.corpus[i] for i in query]  # convert document indexes to actual documents
        result = self.similarity_matrix.inner_product(query, self.corpus, normalized=self.normalized)

        if scipy.sparse.issparse(result):
            return numpy.asarray(result.todense())
        if numpy.isscalar(result):
            return numpy.array(result)
        return numpy.asarray(result)[0]

    def __str__(self):
        return "%s<%i docs, %i features>" % (self.__class__.__name__, len(self), self.similarity_matrix.shape[0])


class WmdSimilarity(interfaces.SimilarityABC):
    """Compute negative WMD similarity against a corpus of documents.

    Check out `the Gallery <https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html>`__
    for more examples.

    When using this code, please consider citing the following papers:

    * `Rémi Flamary et al. "POT: Python Optimal Transport"
      <https://jmlr.org/papers/v22/20-451.html>`_
    * `Matt Kusner et al. "From Word Embeddings To Document Distances"
      <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_

    Example
    -------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_texts
        >>> from gensim.models import Word2Vec
        >>> from gensim.similarities import WmdSimilarity
        >>>
        >>> model = Word2Vec(common_texts, vector_size=20, min_count=1)  # train word-vectors
        >>>
        >>> index = WmdSimilarity(common_texts, model.wv)
        >>> # Make query.
        >>> query = ['trees']
        >>> sims = index[query]

    """
    def __init__(self, corpus, kv_model, num_best=None, chunksize=256):
        """

        Parameters
        ----------
        corpus: iterable of list of str
            A list of documents, each of which is a list of tokens.
        kv_model: :class:`~gensim.models.keyedvectors.KeyedVectors`
            A set of KeyedVectors
        num_best: int, optional
            Number of results to retrieve.
        chunksize : int, optional
            Size of chunk.

        """
        self.corpus = corpus
        self.wv = kv_model
        self.num_best = num_best
        self.chunksize = chunksize

        # Normalization of features is not possible, as corpus is a list (of lists) of strings.
        self.normalize = False

        # index is simply an array from 0 to size of corpus.
        self.index = numpy.arange(len(corpus))

    def __len__(self):
        """Get size of corpus."""
        return len(self.corpus)

    def get_similarities(self, query):
        """Get similarity between `query` and this index.

        Warnings
        --------
        Do not use this function directly; use the `self[query]` syntax instead.

        Parameters
        ----------
        query : {list of str, iterable of list of str}
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix.

        """
        if isinstance(query, numpy.ndarray):
            # Convert document indexes to actual documents.
            query = [self.corpus[i] for i in query]

        if not query or not isinstance(query[0], list):
            query = [query]

        n_queries = len(query)
        result = []
        for qidx in range(n_queries):
            # Compute similarity for each query.
            qresult = [self.wv.wmdistance(document, query[qidx]) for document in self.corpus]
            qresult = numpy.array(qresult)
            qresult = 1. / (1. + qresult)  # Similarity is the negative of the distance.

            # Append single query result to list of all results.
            result.append(qresult)

        if len(result) == 1:
            # Only one query.
            result = result[0]
        else:
            result = numpy.array(result)

        return result

    def __str__(self):
        return "%s<%i docs, %i features>" % (self.__class__.__name__, len(self), self.wv.vectors.shape[1])


class SparseMatrixSimilarity(interfaces.SimilarityABC):
    """Compute cosine similarity against a corpus of documents by storing the index matrix in memory.

    Examples
    --------
    Here is how you would index and query a corpus of documents in the bag-of-words format using the
    cosine similarity:

    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.similarities import SparseMatrixSimilarity
        >>> from gensim.test.utils import common_texts as corpus
        >>>
        >>> dictionary = Dictionary(corpus)  # fit dictionary
        >>> bow_corpus = [dictionary.doc2bow(line) for line in corpus]  # convert corpus to BoW format
        >>> index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary))
        >>>
        >>> query = 'graph trees computer'.split()  # make a query
        >>> bow_query = dictionary.doc2bow(query)
        >>> similarities = index[bow_query]  # calculate similarity of query to each doc from bow_corpus

    Here is how you would index and query a corpus of documents using the Okapi BM25 scoring
    function:

    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import TfidfModel, OkapiBM25Model
        >>> from gensim.similarities import SparseMatrixSimilarity
        >>> from gensim.test.utils import common_texts as corpus
        >>>
        >>> dictionary = Dictionary(corpus)  # fit dictionary
        >>> query_model = TfidfModel(dictionary=dictionary, smartirs='bnn')  # enforce binary weights
        >>> document_model = OkapiBM25Model(dictionary=dictionary)  # fit bm25 model
        >>>
        >>> bow_corpus = [dictionary.doc2bow(line) for line in corpus]  # convert corpus to BoW format
        >>> bm25_corpus = document_model[bow_corpus]
        >>> index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
        ...                                normalize_queries=False, normalize_documents=False)
        >>>
        >>>
        >>> query = 'graph trees computer'.split()  # make a query
        >>> bow_query = dictionary.doc2bow(query)
        >>> bm25_query = query_model[bow_query]
        >>> similarities = index[bm25_query]  # calculate similarity of query to each doc from bow_corpus

    Notes
    -----
    Use this if your input corpus contains sparse vectors (such as TF-IDF documents) and fits into RAM.

    The matrix is internally stored as a :class:`scipy.sparse.csr_matrix` matrix. Unless the entire
    matrix fits into main memory, use :class:`~gensim.similarities.docsim.Similarity` instead.

    Takes an optional `maintain_sparsity` argument, setting this to True
    causes `get_similarities` to return a sparse matrix instead of a
    dense representation if possible.

    See also
    --------
    :class:`~gensim.similarities.docsim.Similarity`
        Index similarity (wrapper for other inheritors of :class:`~gensim.interfaces.SimilarityABC`).
    :class:`~gensim.similarities.docsim.MatrixSimilarity`
        Index similarity (dense with cosine distance).

    """
    def __init__(self, corpus, num_features=None, num_terms=None, num_docs=None, num_nnz=None,
                 num_best=None, chunksize=500, dtype=numpy.float32, maintain_sparsity=False,
                 normalize_queries=True, normalize_documents=True):
        """

        Parameters
        ----------
        corpus: iterable of list of (int, float)
            A list of documents in the BoW format.
        num_features : int, optional
            Size of the dictionary. Must be either specified, or present in `corpus.num_terms`.
        num_terms : int, optional
            Alias for `num_features`, you can use either.
        num_docs : int, optional
            Number of documents in `corpus`. Will be calculated if not provided.
        num_nnz : int, optional
            Number of non-zero elements in `corpus`. Will be calculated if not provided.
        num_best : int, optional
            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.
            Otherwise, return a full vector with one float for every document in the index.
        chunksize : int, optional
            Size of query chunks. Used internally when the query is an entire corpus.
        dtype : numpy.dtype, optional
            Data type of the internal matrix.
        maintain_sparsity : bool, optional
            Return sparse arrays from :meth:`~gensim.similarities.docsim.SparseMatrixSimilarity.get_similarities`?
        normalize_queries : bool, optional
            If queries are in bag-of-words (int, float) format, as opposed to a sparse or dense
            2D arrays, they will be L2-normalized. Default is True.
        normalize_documents : bool, optional
            If `corpus` is in bag-of-words (int, float) format, as opposed to a sparse or dense
            2D arrays, it will be L2-normalized. Default is True.
        """
        self.num_best = num_best
        self.normalize = normalize_queries
        self.chunksize = chunksize
        self.maintain_sparsity = maintain_sparsity

        if corpus is not None:
            logger.info("creating sparse index")

            # iterate over input corpus, populating the sparse index matrix
            try:
                # use the more efficient corpus generation version, if the input
                # `corpus` is MmCorpus-like (knows its shape and number of non-zeroes).
                num_terms, num_docs, num_nnz = corpus.num_terms, corpus.num_docs, corpus.num_nnz
                logger.debug("using efficient sparse index creation")
            except AttributeError:
                # no MmCorpus, use the slower version (or maybe user supplied the
                # num_* params in constructor)
                pass
            if num_features is not None:
                # num_terms is just an alias for num_features, for compatibility with MatrixSimilarity
                num_terms = num_features
            if num_terms is None:
                raise ValueError("refusing to guess the number of sparse features: specify num_features explicitly")
            corpus = (matutils.scipy2sparse(v) if scipy.sparse.issparse(v) else
                      (matutils.full2sparse(v) if isinstance(v, numpy.ndarray) else
                       matutils.unitvec(v) if normalize_documents else v) for v in corpus)
            self.index = matutils.corpus2csc(
                corpus, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                dtype=dtype, printprogress=10000,
            ).T

            # convert to Compressed Sparse Row for efficient row slicing and multiplications
            self.index = self.index.tocsr()  # currently no-op, CSC.T is already CSR
            logger.info("created %r", self.index)

    def __len__(self):
        """Get size of index."""
        return self.index.shape[0]

    def get_similarities(self, query):
        """Get similarity between `query` and this index.

        Warnings
        --------
        Do not use this function directly; use the `self[query]` syntax instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`}
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix (if maintain_sparsity=False) **OR**
        :class:`scipy.sparse.csc`
            otherwise

        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.T  # convert documents=rows to documents=columns
            elif isinstance(query, numpy.ndarray):
                if query.ndim == 1:
                    query.shape = (1, len(query))
                query = scipy.sparse.csr_matrix(query, dtype=self.index.dtype).T
            else:
                # default case: query is a single vector, in sparse gensim format
                query = matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)

        # compute cosine similarity against every other document in the collection
        result = self.index * query.tocsc()  # N x T * T x C = N x C
        if result.shape[1] == 1 and not is_corpus:
            # for queries of one document, return a 1d array
            result = result.toarray().flatten()
        elif self.maintain_sparsity:
            # avoid converting to dense array if maintaining sparsity
            result = result.T
        else:
            # otherwise, return a 2d matrix (#queries x #index)
            result = result.toarray().T
        return result
