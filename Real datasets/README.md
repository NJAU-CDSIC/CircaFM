# Real Datasets Documentation

This repository contains two curated real biological datasets for circadian rhythm research:  
- **CGDB-20K** – for circadian oscillation tasks (20,886 genes)  
- **DIFF-13K** – for differential rhythmicity tasks (13,619 genes)  

The datasets provide experimentally validated time-series measurements with detailed annotations.


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
