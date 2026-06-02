# CircaFM: a foundation model for circadian pattern analysis of gene expression

This repository contains the complete implementation of CircaFM, a novel framework for circadian rhythm analysis using large language models, along with both synthetic and real biological datasets.

---

The folders in the CircaFM repository:

- **CircaFM_codes**: Source code for CircaFM model.
  
  - models: Core model implementations
  
  - data_provider: Data loading and preprocessing modules
  
  - layers: Custom neural network components
  
  - utils: Support utilities and tools

  - pretrained: Contains pre-trained model checkpoints (Empty in GitHub - download from [FigShare](https://doi.org/10.6084/m9.figshare.29322500))

  - assets: Contains experimental results (large files available on [FigShare](https://doi.org/10.6084/m9.figshare.29322500))
 
  - Oscillation detection demo.ipynb: This Jupyter Notebook demonstrates oscillation detection in time-series data using a pre-trained model.
 
  - Circadian variation detection demo.ipynb: This Jupyter Notebook demonstrates Circadian variation detection in time-series data using a pre-trained model.

- **Simulated datasets**: Artificially generated time-series data
  - 21 specialized datasets (SynthDST-1 to SynthDST-21) for two primary tasks:
    - **Circadian Oscillation**: 12 datasets (SynthDST-1 to SynthDST-12)
    - **Differential Rhythmicity**: 9 datasets (SynthDST-13 to SynthDST-21)
  - Readme.md: Dataset documentation

- **Real datasets**: Curated circadian rhythm transcriptome datasets
  - Two comprehensive biological datasets:
    - **CGDB-20K**: 18 sub-datasets covering 20,886 circadian genes
    - **DIFF-13K**: 6 sub-datasets covering 13,619 differential rhythmicity genes
  - Readme.md: Dataset documentation
 
- **Datasets for Case study**: Curated real-world time-course datasets for in-depth biological analysis  
  - Contains 15 specialized case studies across diverse biological contexts
 
- **SOTA**: 10 models used in the comparative study:

  - JTK_CYCLE: <https://CRAN.R-project.org/package=MetaCycle>

  - Cosiner: <https://CRAN.R-project.org/package=cosinor>

  - ARSER: <https://CRAN.R-project.org/package=MetaCycle>

  - MetaCycle: <https://CRAN.R-project.org/package=MetaCycle>

  - DODR: <https://CRAN.R-project.org/package=DODR>
 
  - LimoRhyde: <https://CRAN.R-project.org/package=limorhyde>
 
  - dryR: <https://github.com/naef-lab/dryR>
 
  - BayesDCirc: <https://github.com/lichen-lab/BayesDCirc>

  - diffCircadian: <https://github.com/diffCircadian/diffCircadian>
 
  - CircaCompare: <https://CRAN.R-project.org/package=circacompare>

---

## **Step-by-step Running:**

## 1. Installation

```bash
git clone https://github.com/NJAU-CDSIC/CircaFM.git
cd CircaFM
```

## 2. Environment Installation

It is recommended to use the conda environment (python 3.11), mainly installing the following dependencies:

```bash
conda env create -n circafm python==3.11
conda activate circafm
pip install -r requirements.txt
```

See requirements.txt for details.

## 3. Datasets and model parameters

To ensure repository stability and comply with platform size limitations, critical components of CircaFM are hosted externally on figshare. **Full functionality requires downloading these resources separately**:

### 🔗 Essential Figshare Resources

| Resource Category | Figshare Access | Required For |
|:-----------------:|:---------------:|:------------:|
| **Pre-trained Models** | [Download Link](https://doi.org/10.6084/m9.figshare.29322500) | Model inference, Transfer learning |
| **Visual Assets** | [Download Link](https://doi.org/10.6084/m9.figshare.29322500) | Training curves, Result visualizations |
| **Simulated Datasets** | [Download Link](https://doi.org/10.6084/m9.figshare.29322500) | Model training, Method validation |
| **Real Datasets** | [Download Link](https://doi.org/10.6084/m9.figshare.29322500) |  Method validation |

## 4.  Execute the code cell by cell in the Python interactive window
  
Run the Jupyter notebook at the following path:  
**`CircaFM_code/Oscillation detection demo.ipynb`** for Oscillation detection

**`CircaFM_code/Circadian variation detection demo.ipynb`** for Circadian variation detection
