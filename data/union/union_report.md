# IVF batch union and worker load

Device coarse search supplied uint32 list IDs and BF16 scores. The host selected up to nprobe distinct IDs per query. Each configuration used 10,000 valid queries in 313 batches. List tasks were not chunked. The existing scheduler orders lists by pages times active batches and places them to minimize the peak batch load. The Faiss-style count is a counterfactual using these same device-selected lists, not a separate Faiss coarse search.

The requested union columns use valid queries. The scheduled union and worker load columns reflect the current host scheduler, including any padded query rows in the last batch. Empty lists have zero measured pages and receive no worker task.

The tables show median/p95/max across batches. `union_summary.csv` also contains each metric's mean.

| Nlist | Nprobe | Union lists | Union pages | TT scanned fraction | Faiss mean scanned fraction | Extra scan factor |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 31.00/32.00/32.00 | 2655.00/2959.60/3250.00 | 0.0713/0.0796/0.0874 | 0.0023 | 30.79/32.00/32.00 |
| 512 | 2 | 60.00/63.00/64.00 | 5030.00/5439.40/5697.00 | 0.1352/0.1462/0.1533 | 0.0046 | 29.64/31.43/32.00 |
| 512 | 4 | 111.00/117.40/123.00 | 9107.00/9766.80/10265.00 | 0.2448/0.2626/0.2760 | 0.0089 | 27.37/29.20/30.65 |
| 512 | 8 | 194.00/205.40/217.00 | 15318.00/16490.80/17262.00 | 0.4114/0.4432/0.4641 | 0.0174 | 23.72/25.47/26.77 |
| 512 | 16 | 304.00/325.40/341.00 | 23214.00/24807.80/26124.00 | 0.6234/0.6665/0.7019 | 0.0340 | 18.37/19.84/20.84 |
| 512 | 32 | 412.00/434.40/448.00 | 30545.00/32134.60/33078.00 | 0.8204/0.8631/0.8884 | 0.0660 | 12.40/13.17/13.65 |
| 1024 | 1 | 32.00/32.00/32.00 | 1370.00/1537.40/1667.00 | 0.0366/0.0412/0.0446 | 0.0012 | 32.00/32.00/32.00 |
| 1024 | 2 | 62.00/64.00/64.00 | 2653.00/2930.80/3129.00 | 0.0709/0.0785/0.0838 | 0.0023 | 30.87/32.00/32.00 |
| 1024 | 4 | 119.00/124.00/127.00 | 5024.00/5408.60/5828.00 | 0.1343/0.1446/0.1561 | 0.0046 | 29.46/30.89/31.73 |
| 1024 | 8 | 221.00/232.00/241.00 | 9094.00/9799.00/10239.00 | 0.2429/0.2617/0.2740 | 0.0090 | 27.19/28.79/29.83 |
| 1024 | 16 | 385.00/407.00/421.00 | 15404.00/16297.20/17048.00 | 0.4113/0.4353/0.4555 | 0.0176 | 23.39/25.04/26.22 |
| 1024 | 32 | 605.00/641.40/665.00 | 23302.00/24837.60/25647.00 | 0.6220/0.6630/0.6846 | 0.0344 | 18.08/19.52/20.35 |
| 2048 | 1 | 32.00/32.00/32.00 | 673.00/746.60/797.00 | 0.0178/0.0198/0.0211 | 0.0006 | 32.00/32.00/32.00 |
| 2048 | 2 | 63.00/64.00/64.00 | 1326.00/1429.40/1527.00 | 0.0350/0.0379/0.0405 | 0.0011 | 31.41/32.00/32.00 |
| 2048 | 4 | 124.00/127.00/128.00 | 2558.00/2728.40/2893.00 | 0.0675/0.0722/0.0767 | 0.0022 | 30.83/31.75/32.00 |
| 2048 | 8 | 238.00/246.00/251.00 | 4856.00/5108.40/5408.00 | 0.1281/0.1350/0.1431 | 0.0043 | 29.65/30.73/31.36 |
| 2048 | 12 | 345.00/358.00/365.00 | 6930.00/7305.60/7696.00 | 0.1827/0.1928/0.2035 | 0.0064 | 28.58/29.76/30.58 |
| 2048 | 16 | 444.00/463.00/473.00 | 8843.00/9316.60/9653.00 | 0.2332/0.2457/0.2547 | 0.0085 | 27.50/28.70/29.84 |
| 2048 | 32 | 768.00/814.20/840.00 | 15121.00/15956.80/16555.00 | 0.3985/0.4207/0.4363 | 0.0168 | 23.68/25.27/26.23 |

| Nlist | Nprobe | Active workers | Max worker pages | Imbalance | Critical pages | 7 µs estimate (ms) | Measured fine median (ms) | Estimate / measured |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 26.00/29.00/31.00 | 175.00/212.00/212.00 | 4.21/5.07/6.57 | 55988 | 391.92 | 232.37 | 1.69× |
| 512 | 2 | 41.00/45.00/48.00 | 227.00/249.00/271.00 | 2.87/3.23/4.84 | 71619 | 501.33 | — | — |
| 512 | 4 | 55.00/58.00/61.00 | 333.00/343.00/351.00 | 2.30/2.51/3.34 | 103435 | 724.04 | — | — |
| 512 | 8 | 62.00/63.00/63.00 | 448.00/460.00/467.00 | 1.83/1.99/2.31 | 139366 | 975.56 | 899.90 | 1.08× |
| 512 | 16 | 63.00/63.00/63.00 | 561.00/573.00/581.00 | 1.52/1.62/2.16 | 174550 | 1221.85 | 1354.51 | 0.90× |
| 512 | 32 | 63.00/63.00/63.00 | 605.00/622.00/622.00 | 1.25/1.32/1.46 | 189760 | 1328.32 | 1768.92 | 0.75× |
| 1024 | 1 | 27.00/30.00/31.00 | 87.00/106.00/106.00 | 4.04/4.97/7.04 | 27923 | 195.46 | — | — |
| 1024 | 2 | 41.00/44.00/47.00 | 115.00/122.00/149.00 | 2.75/3.04/4.85 | 36313 | 254.19 | — | — |
| 1024 | 4 | 55.00/58.00/61.00 | 170.00/173.00/178.00 | 2.12/2.29/3.34 | 52872 | 370.10 | — | — |
| 1024 | 8 | 62.00/63.00/63.00 | 254.00/259.00/262.00 | 1.75/1.88/2.30 | 79215 | 554.50 | — | — |
| 1024 | 16 | 63.00/63.00/63.00 | 370.00/376.00/379.00 | 1.51/1.62/2.23 | 115459 | 808.21 | — | — |
| 1024 | 32 | 63.00/63.00/63.00 | 496.00/504.00/508.00 | 1.34/1.41/1.52 | 154478 | 1081.35 | — | — |
| 2048 | 1 | 27.00/30.00/31.00 | 38.00/58.00/60.00 | 3.67/5.27/8.59 | 12777 | 89.44 | — | — |
| 2048 | 2 | 42.00/46.00/49.00 | 48.00/60.00/60.00 | 2.34/2.81/4.73 | 15623 | 109.36 | — | — |
| 2048 | 4 | 56.00/59.00/61.00 | 74.00/76.00/78.00 | 1.83/1.96/2.87 | 23256 | 162.79 | — | — |
| 2048 | 8 | 62.00/63.00/63.00 | 123.00/125.00/128.00 | 1.60/1.72/2.45 | 38481 | 269.37 | — | — |
| 2048 | 12 | 63.00/63.00/63.00 | 166.00/168.00/172.00 | 1.51/1.61/2.51 | 51919 | 363.43 | — | — |
| 2048 | 16 | 63.00/63.00/63.00 | 202.00/205.00/208.00 | 1.43/1.52/1.93 | 63022 | 441.15 | — | — |
| 2048 | 32 | 63.00/63.00/63.00 | 317.00/320.00/323.00 | 1.32/1.40/1.80 | 98887 | 692.21 | — | — |

![TT and Faiss scanned fraction](scanned_fraction.png)

![Worker imbalance distribution](worker_imbalance.png)

## Sanity checks

- nlist=512: list sizes sum to 1,183,514 vectors; 512 lists recorded.
- nlist=1024: list sizes sum to 1,183,514 vectors; 1024 lists recorded.
- nlist=2048: list sizes sum to 1,183,514 vectors; 2048 lists recorded.
- 512/1: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 512/2: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 512/4: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 512/8: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 512/16: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 512/32: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 1024/1: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 1024/2: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 1024/4: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 1024/8: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 1024/16: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 1024/32: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 2048/1: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 2048/2: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 2048/4: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 2048/8: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 2048/12: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 2048/16: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 2048/32: 313 batches passed union bounds, selections, 63-worker page sums and nprobe=1 active-worker limit; 1 batches had additional scheduled pages from padded rows.
- 512/8: two device-coarse repetitions produced identical per-batch union ID sets.

The 7 µs/page estimate uses a single-list trace. It does not model concurrent DRAM traffic, startup, overlap, or final aggregation. Reference fine times at nlist=512 are 232.37, 899.90, 1354.51, and 1768.92 ms for nprobe=1, 8, 16, and 32. Those timings may use a different cluster chunk limit; the ratio is descriptive only.

The logging implementation is in the parent folder. [logging_changes.patch](../logging_changes.patch) records the source changes relative to the energy executable; [source_sha256.csv](../source_sha256.csv) records the captured source files.
