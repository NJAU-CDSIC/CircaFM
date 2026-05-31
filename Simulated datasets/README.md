# Synthetic Datasets Documentation

This repository contains artificially generated datasets for **Circadian Oscillation** and **Differential Rhythmicity** tasks. All sample counts are measured in **thousands (K)** (1K = 1,000 samples).

---

## Dataset Overview

### 1. Circadian Oscillation Task

| Dataset     | Duration | Duplicates | Interval     | Training (K) | Validation (K) | Test (K) |
|:-----------:|:--------:|:----------:|:------------:|:------------:|:--------------:|:--------:|
| SynthDST-1  | 24h      | 1          | 2h           | 480          | 80             | 240      |
| SynthDST-2  | 24h      | 2          | 2h           | 480          | 80             | 240      |
| SynthDST-3  | 24h      | 1          | 3h           | 480          | 80             | 240      |
| SynthDST-4  | 24h      | 2          | 3h           | 480          | 80             | 240      |
| SynthDST-5  | 24h      | 3          | 3h           | 480          | 80             | 240      |
| SynthDST-6  | 24h      | 1          | 4h           | 480          | 80             | 240      |
| SynthDST-7  | 24h      | 1          | Non-uniform  | 960          | 160            | 480      |
| SynthDST-8  | 48h      | 1          | 1h           | 480          | 80             | 240      |
| SynthDST-9  | 48h      | 1          | 2h           | 480          | 80             | 240      |
| SynthDST-10 | 48h      | 1          | 3h           | 480          | 80             | 240      |
| SynthDST-11 | 48h      | 1          | 4h           | 480          | 80             | 240      |
| SynthDST-12 | 48h      | 1          | 6h           | 480          | 80             | 240      |

---

### 2. Differential Rhythmicity Task

| Dataset     | Duration | Duplicates | Interval     | Training (K) | Validation (K) | Test (K) |
|:-----------:|:--------:|:----------:|:------------:|:------------:|:--------------:|:--------:|
| SynthDST-13 | 24h      | 1          | 2h           | 192          | 32             | 96       |
| SynthDST-14 | 24h      | 1          | 3h           | 192          | 32             | 96       |
| SynthDST-15 | 24h      | 3          | 3h           | 192          | 32             | 96       |
| SynthDST-16 | 24h      | 1          | 4h           | 192          | 32             | 96       |
| SynthDST-17 | 24h      | 1          | Non-uniform  | 384          | 64             | 192      |
| SynthDST-18 | 48h      | 1          | 2h           | 192          | 32             | 96       |
| SynthDST-19 | 48h      | 1          | 3h           | 192          | 32             | 96       |
| SynthDST-20 | 48h      | 1          | 4h           | 192          | 32             | 96       |
| SynthDST-21 | 48h      | 1          | 6h           | 192          | 32             | 96       |

---

## File Structure

Each dataset consists of three time-series files in the following structure:

**Examples**:

```text
├── SynthDST-1/   # Dataset folder
│   ├── TRAIN.ts
│   ├── VALIDATION.ts
│   └── TEST.ts
│
├── SynthDST-2/   # Dataset folder
│   ├── TRAIN.ts
│   ├── VALIDATION.ts
│   └── TEST.ts
│
└── ... # (21 Datasets total)

## Data Availability
Due to file size limitations, the datasets are hosted on figshare and can be accessed via the following link:
[figshare](https://doi.org/10.6084/m9.figshare.29322500)
