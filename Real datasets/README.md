# Real Datasets Documentation

This repository contains two curated real biological datasets for circadian rhythm research:  
- **CGDB-20K** – for circadian oscillation tasks (20,886 genes)  
- **DIFF-13K** – for differential rhythmicity tasks (13,619 genes)  

The datasets provide experimentally validated time-series measurements with detailed annotations.

---

## Datasets Overview

### 1. CGDB-20K (Circadian Oscillation)

This dataset contains **18 sub-datasets** collected from public sources, covering a total of **20,886 circadian genes** across multiple organisms and tissues.

| Sub-dataset  | Dataset                 | Organism            | Tissue               | Duration | Interval     | Replicates | No. genes | Max (Amp) |
|:------------:|:-----------------------:|:-------------------:|:--------------------:|:--------:|:------------:|:----------:|:---------:|:---------:|
| CGDB20K-1    | GSE67305                | *M. musculus*       | Liver                | 24h      | 2h           | 2          | 501       | 11.44     |
| CGDB20K-2    | Vollmers, et al.        | *M. musculus*       | Liver                | 24h      | 3h           | 1          | 692       | 26.33     |
| CGDB20K-3    | GSE52333                | *M. musculus*       | Liver                | 24h      | 4h           | 3          | 2097      | 12.20     |
| CGDB20K-4    | GSE33381                | *M. musculus*       | Liver                | 24h      | 6h           | 3          | 2440      | 11.70     |
| CGDB20K-5    | GSE11923                | *M. musculus*       | Liver                | 48h      | 1h           | 1          | 2080      | 17.01     |
| CGDB20K-6    | GSE3751                 | *M. musculus*       | Liver                | 48h      | 4h           | 2          | 327       | 16.88     |
| CGDB20K-7    | GSE72095                | *M. musculus*       | SCN                  | 24h      | 4h           | 3          | 3019      | 16.66     |
| CGDB20K-8    | GSE11922                | *M. musculus*       | NIH3T3               | 48h      | 1h           | 1          | 7         | 14.27     |
| CGDB20K-9    | GSE38623                | *M. musculus*       | Anagen epidermis     | 48h      | 4h           | 1          | 252       | 10.14     |
| CGDB20K-10   | GSE38622                | *M. musculus*       | Telogen epidermis    | 48h      | 4h           | 1          | 679       | 12.50     |
| CGDB20K-11   | GSE3746                 | *M. musculus*       | Skeletal muscle      | 48h      | 4h           | 2          | 147       | 15.64     |
| CGDB20K-12   | Gill, et al.            | *D. melanogaster*   | Periphery            | 24h      | 3h           | 1          | 626       | 9.95      |
| CGDB20K-13   | Gill, et al.            | *D. melanogaster*   | Head                 | 24h      | 3h           | 1          | 438       | 9.33      |
| CGDB20K-14   | GSE25612                | *R. norvegicus*     | Lung                 | 24h      | Non-uniform  | -          | 407       | 11.36     |
| CGDB20K-15   | SRA054264               | *D. rerio*          | Pineal gland         | 24h      | 4h           | 2          | 11        | 11.04     |
| CGDB20K-16   | GSE51277                | *D. rerio*          | Brain                | 48h      | 4h           | 1          | 97        | 1.721     |
| CGDB20K-17   | GSE37278                | *A. thaliana*       | Seedlings            | 48h      | 4h           | 1          | 6729      | 12.20     |
| CGDB20K-18   | Covington, et al.       | *A. thaliana*       | Seedlings            | 48h      | 4h           | 2          | 337       | 3.25      |

---

### 2. DIFF-13K (Differential Rhythmicity)

This dataset contains **6 sub-datasets** collected from public sources, covering a total of **13,619 differential rhythmicity genes**.

| Sub-dataset  | Dataset      | Organism        | Tissue     | Duration | Interval     | Replicates | No. genes | Max (Amp) |
|:------------:|:------------:|:---------------:|:----------:|:--------:|:------------:|:----------:|:---------:|:---------:|
| DIFF13K-1    | GSE67305     | *M. musculus*   | Liver      | 24h      | 2h           | 2          | 4581      | 13.48     |
| DIFF13K-2    | GSE93239     | *M. musculus*   | Liver      | 24h      | 4h           | 2          | 251       | 5.58      |
| DIFF13K-3    | GSE115264    | *M. musculus*   | Liver      | 24h      | 4h           | 4          | 1051      | 6.10      |
| DIFF13K-4    | GSE126851    | *M. musculus*   | SCN        | 24h      | 6h           | 4          | 4883      | 13.58     |
| DIFF13K-5    | GSE165198    | *M. musculus*   | Pancreas   | 48h      | 4h           | 2          | 2369      | 3.46      |
| DIFF13K-6    | GSE122541    | *H. sapiens*    | Blood      | 24h      | 3h           | 3          | 484       | 12.74     |

---

## File Structure

- **CGDB-20K sub-datasets** each contain training, validation, and test splits (`TRAIN.ts`, `VALIDATION.ts`, `TEST.ts`).
- **DIFF-13K sub-datasets** each contain only test sets (`TEST.ts`) for evaluation, consistent with the differential rhythmicity task format.

```text
├── CGDB20K-1/ 
│   └── TEST.ts
│
├── CGDB20K-2/   
│   └── TEST.ts
│
└── ... # (18 sub-datasets total for CGDB-20K)

├── DIFF13K-1/   
│   └── TEST.ts
│
├── DIFF13K-2/   
│   └── TEST.ts
│
└── ... # (6 sub-datasets total for DIFF-13K)
```

## Data Availability:
Due to file size limitations, the datasets are hosted on figshare and can be accessed via the following link:
[figshare](https://doi.org/10.6084/m9.figshare.29322500)
