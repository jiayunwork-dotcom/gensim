# Streaming TF-IDF over a continuously appended corpus — design notes

## Problem

`TfidfModel` (gensim 4.3.3) collects document frequencies only at construction time, and
`Similarity.add_documents` appends already-weighted vectors. When a new text batch arrives,
old shards therefore keep TF-IDF weights computed against the *old* collection statistics,
while new documents use the new statistics; the two are not comparable. The goal is to make
every query — against old and new documents alike — use the TF-IDF statistics of **all
documents received so far**, while keeping the sharded, disk-backed, bounded-memory design of
`Similarity` and supporting one-shot generator batches, atomic failures and save/load.

## Two viable disk organizations

### Option A (implemented): immutable raw-count shards + statistics owned by the model

Each shard stores the **raw bag-of-words counts** as a memory-mappable CSR matrix
(`TfidfShard`). The `Similarity` owns a default `TfidfModel` whose `num_docs`, `num_nnz`,
`dfs` and `idfs` are extended cumulatively on every successful batch. At query time every
shard is re-weighted with the *current* IDF vector (`raw @ diag(idf)`, row L2-normalized);
the raw counts on disk are never modified.

### Option B: rewrite every existing shard on each batch

Keep weighted vectors on disk as today, but after each batch recompute every shard's
TF-IDF matrix from stored raw counts (which would still have to be retained) and rewrite all
shard files with the new weights; new documents are appended as today.

### Comparison

| Dimension | Option A — raw counts, re-weight at query | Option B — rewrite shards at append |
| --- | --- | ---|
| Append write amplification | **~1×**: only the new shard's bytes are written; historical shard files are never opened for writing | **O(number of batches)**: every old shard is rewritten on every batch, so total bytes written grow quadratically with the number of batches |
| Query work | One sparse diagonal scaling + row normalization per shard per query, plus the same sparse matvec the ordinary index already does. The scaling is proportional to shard non-zeros and is vectorized SciPy; measured query time stays in the same order of magnitude as the ordinary path | Ordinary weighted matvec only (no re-weighting at query time), so queries are slightly cheaper in pure arithmetic — but only after paying the rewrite cost up front |
| Zero-weight recovery | **Native.** A term with IDF zero in an early collection is still present in the raw CSR data; when later batches lower its document frequency it immediately contributes again | Only works if raw counts are *also* retained (otherwise the zero component was eliminated from the weighted matrix and can never return), in which case Option B strictly needs the same raw storage plus the rewrites |
| Persistence cost | One small proxy pickle + three `.npy` arrays per shard (data/indices/indptr, mmap-able); the model statistics are one small pickle inside the index object. Save only flushes the open partial shard | Same raw/weighted files plus full-shard rewrites on every batch; save/load themselves are similar but total disk churn is much higher |
| Working memory | One shard's CSR (≤ `shardsize` rows), one query chunk (≤ `chunksize × shardsize` floats) and the vocabulary-sized DF dict | Same bound, plus a rewrite working set that touches every shard per batch |
| Failure atomicity | Candidate shards are written under `.inprogress` names and renamed only after the whole batch validates and the model statistics merge; rollback deletes the temp files and restores a state snapshot | A failed rewrite can leave partially rewritten historical shards; restoring them needs either backups (more writes) or a transaction log |
| Save/load + continue | Shards are content-addressed by immutable row ranges; loading just re-mmaps them and appends new shard ids | The reload must re-run any pending rewrites before the index is consistent |

### Why Option A

The workload is *many small append batches over a long-lived index*, where the query rate is
moderate and the collection statistics keep changing. Option A pays a small, linear,
per-query scaling cost instead of quadratic append write amplification, it makes zero-IDF
recovery automatic (the requirement that a term missing from old weighted vectors must become
retrievable again), and it gives clean batch atomicity because historical files are simply
never touched. Option B only wins when writes are extremely rare and query throughput is the
dominant cost, which does not match this design brief.

A third alternative — keeping an in-memory copy of all historical BoW or weighted vectors —
was rejected outright: it violates the bounded-memory requirement and removes the point of
the disk sharding.

## Where the statistics and the shards meet

* `TfidfModel.add_documents(corpus)` (and the internal `_merge_batch_statistics`) update
  `num_docs`, `num_nnz`, `dfs`, `idfs` cumulatively; an empty batch is a no-op and a failed
  batch leaves no partial statistics.
* In TF-IDF mode `Similarity.add_documents` streams each batch once, validates every document
  and accumulates the same batch-local statistics in the same pass, spilling at most
  `shardsize` documents into a temporary `TfidfShard`. Commit order is: merge model
  statistics → rename temp shards to their final ids → publish them in `self.shards`.
* At query time `Similarity` builds one current IDF vector and asks **every** shard (old and
  new, closed and buffered) to materialize its current weighted matrix from the same raw
  counts, so the collection statistics truly govern both sides of every cosine.
* `vector_by_id` returns the current weighted row through the same code path; `save/load`
  persist the raw shards (mmap-ed back) and the model statistics, so a reloaded index
  appends and scores identically to an uninterrupted run.

## Working-memory upper bound

With `shardsize = S`, `chunksize = C`, vocabulary `V` and at most `k` non-zeros per document:

* spill buffer: at most `S` validated BoW documents,
* one shard weighted matrix: sparse CSR of at most `S·k` entries (worst dense case
  `S·V` float32 = 4 MiB at `S = 256, V = 4096`) — only one is resident at a time, because
  each raw shard is memory-mapped, converted and the mapping released inside the query loop,
* one query intermediate: `C·S` float32 values (32 KiB at `C = 32, S = 256`),
* the DF dictionary: one Python `int` key + float value per term, ~0.2 MiB at `V = 4096`.

The final per-query score array (one float per indexed document) is the documented return
value and, as in the ordinary `Similarity`, is not counted as working area. Adversarial
inputs can only raise the per-shard CSR toward the dense bound `S·V`; they cannot make the
index retain history beyond the current shard and the vocabulary-sized statistics. A bounded
LRU of open shard matrices was prototyped but measured to give no speed-up over the kernel's
page cache (0.92× at this data density) while complicating eviction, so the implementation
keeps exactly one shard mapping resident at a time; `/proc/<pid>/maps` confirms no `.npy`
mapping remains after a query.

## Measured cost (vocabulary 4096, 16 non-zeros/doc, 2048 docs/batch, shardsize 256)

| batches (docs) | append total (s) | 16 full queries (s) | Python peak (MiB) | RSS total (MiB) | disk (MiB) | rewrite (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 → +1 (10 240) | 0.87 | 0.73 | 3.06 | 107.2 | 1.47 | 1.47 |
| 16 → +1 (34 816) | 3.54 | 3.00 | 4.95 | 124.2 | 4.68 | 4.68 |
| 64 → +1 (133 120) | 15.41 | 12.31 | 12.97 | 124.2 | 17.54 | 17.54 |

The extra Python allocation peak stays an order of magnitude under the 96 MiB budget and RSS
is flat between the 16- and 64-batch scales. Rewrite volume equals the final disk volume
(write amplification ≈ 1×): historical shard files are never rewritten. All independent
reference checks (full rebuild at the small scale, per-document streaming checks at every
scale) report a maximum absolute cosine error of 0.0 in float32, well within the 1e-6 target.

## Known limits of this first version

* Single process, serial calls only (no multiprocess shard querying in TF-IDF mode).
* Only the default model configuration and `norm='l2'`; other TF-IDF schemes raise
  `ValueError` before the input is read.
* Durability covers successful operations; a process crash mid-batch relies on the
  `.inprogress` temp files being discarded on the next run (crash-recovery automation is
  explicitly out of scope).
