## KV cache structures
The conventional way of organizing KV caches adopts from the PagedAttention paper, and in practice paged KV caches are
stored as per layer and have this shape:

`[page_size, 2, num_kvheads, hidden_size]`

Different from this structure that contains all KV heads, we propose a BSA KV cache that is also per layer, but only
contains a single KV head. It has this shape:

`[block_size, 2, hidden_size]`

We'll discuss its benefits compared to paged KV cache below.

## Back-of-the-envelope
Suppose we have these parameters:
```
num_layers=28
num_heads=16
num_kvheads=8 (due to GQA)
hidden_size=1024
page_size=32
block_size=512
context_size=4096
dtype=bf16
```

First we compute the per page and per block size.

For paged KV cache, it's `32(num of tokens) * 2(K and V) * 8(KV heads) * 1024(hidden_size) * 2(dtype bytes) = 1MB`.

For BSA KV cache, it's `512(num of tokens) * 2(K and V) * 1024(hidden_size) * 2(dtype bytes) = 2MB`.

Then we compute the total size of the KV caches for the whole user query.

For paged KV cache, in each layer we retrieve the set of BSA blocks (here it means MOBA blocks) for all heads. Note that since
paged KV caches contain all KV heads, we need to read those blocks no matter which query head we're currently in, despite that
many KV pairs belonging to other heads may not get used. For example, Q head 0 reads block (1, 3, 7) and Q head 1 reads block (1, 4, 7) (Note these two Q heads both correspond to KV head 0). This will result in reading (1, 3, 4, 7), and for all pages in each block only KV head 0 is used. (More precisely, if other query heads in this layer read them again then the other KV heads in those pages may be used too, but there must be some KV heads unused)

Therefore, in the worst case all the 8 blocks are read in each layer, hence the total KV cache size is
`8(blocks) * 16(num of pages per block) * 28(layers) * 1MB = 3.5GB`.

For BSA KV cache, since it's per layer per KV head, in each layer we retrieve only the BSA KV cache for the accessed blocks, **just
for that KV head**. For example, Q head reads block (1, 3, 7) and Q head 1 reads block (1, 4, 7), then we just read BSA KV cache
(1, 3, 4, 7) for KV head 0. No unused data at all.

So for BSA KV cache, in the worst case we read 5 blocks (2 + 2 + 1) per query head ,hence total size is
`5(BSA KV cache per query KV head) * 8(KV heads) * 28(layers) * 2MB = 2.24GB`.

## Implementation notes
1. As we discussed above, because conventional paged KV cache contains all KV heads, it can cause non-trivial waste of memory. In the example above BSA KV cache saves ~40% memory in the worst case, and the reality is paged KV cache tends to behave under the worst case while BSA KV cache tends to behave under the average case due to higher probability of KV cache sharing in each KV head.
2. We don't consider cross-requests KV cache sharing for now, which means KV caches of a sequence can't be reused in other sequences, effectively meaning batch size=1.
3. Due to 2, we use the averaged cache hit rate for all sequences as the performance metric, and we reset the cache's state for each sequence.
4. In reality people batch multiple sequences during decoding to fully leverage GPU's compute power and typical capacity for KV caches is tens of GB. Since we set batch size=1, we should limit the available KV cache size to be single-digit GBs and compare how different eviction algorithms perform under the same cache size.
5. Preliminary experiments show that with LRU<3.5GB effectively leads to miss rato=1 since when cache is full the previous layers will always get kicked out, so in the next iteration it has to load them back again. This aligns with the common sense that sequential accessing + recency-based eviction results in poor cache hit rate. Comparing with S3FIFO also shows for 3GB cache size and paged KV cache, LRU gives 100% miss ratio and S3FIFO gives 35%.