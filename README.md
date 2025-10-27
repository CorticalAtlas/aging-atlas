# Cortical-Aging-Atlas
A surface area aging atlas for healthy middle-aged and elderly populations


## Atlas
![Git_atlas](https://github.com/user-attachments/assets/8b015ec0-dc41-448d-a47f-2a6aea4a5537)


## The original NMF results
![NMF](https://github.com/user-attachments/assets/2969ff90-f7ae-4c66-8bf4-459d005094b6)

## Different λ results
![λ](https://github.com/user-attachments/assets/a36b8b8a-0103-43d8-9808-f12374543104)


## Atrophy subtypes
![Git_atrophy](https://github.com/user-attachments/assets/0b78838b-f7d9-4767-8b06-67d45699a89a)



We constructed a healthy aging atlas using only healthy control group data from the Alzheimer's Disease Neuroimaging Initiative ADNI（https://ida.loni.usc.edu/pages/access/search.jsp?tab=collection&project=ADNI&page=DOWNLOADS&subPage=IMAGE_COLLECTIONS）, comprising 1,069 longitudinal scans (male: 76.94 ± 6.59 years; female: 76.73 ± 6.39 years) to establish a baseline for healthy aging. T1-weighted MR scans were preprocessed using FreeSurfer's standard pipeline for cortical surface reconstruction, and the resulting data were concatenated to generate a surface area matrix V. Under the Riemannian manifold framework, the cortical surface was modeled as a triangular mesh manifold M. The topological connections between cortical vertices were encoded using a graph Laplacian operator. A two-stage Non-negative Matrix Factorization (NMF) architecture was employed:
```plaintext
Stage 1: Standard NMF was applied to obtain an initial basis matrix W⁽⁰⁾ and coefficient matrix H⁽⁰⁾, ensuring sparsity without topological constraints.
Stage 2: With H⁽⁰⁾ fixed, the basis matrix W⁽⁰⁾ was fine-tuned to promote similarity in factor expressions between adjacent cortical vertices.
```
This study aims to build a normative atlas of healthy brain aging, providing a reference baseline for evaluating aging rates in healthy individuals and supporting clinical diagnosis.

Hierarchical
```plaintext
Aging Atlas/
├── SpatiallyRegularizedNMF.py
├── requirements.txt
├── CN/
│   ├── lh_area.mgh # input LH area matrix
│   ├── rh_area.mgh # input RH area matrix
│   ├── lh_V.npy
│   ├── rh_V.npy
│   ├── lh_filtered.npy
│   ├── lh_mask.npy
│   ├── rh_filtered.npy
│   └── rh_mask.npy
└── atlas/
│   ├── nmf_clusters_lh_17.annot
│   └── nmf_clusters_rh_17.annot
└── results/
    ├── nmf_clusters_lh_2.annot
    ├── nmf_clusters_lh_3.annot
    ├── analysis_lh.png
    └── . . .
```

## Quick start
# Install dependencies
```bash
   pip install -r requirements.txt
```
# Run codes

1、Loading the cortical surface area matrices (lh_area.mgh for left hemisphere and rh_area.mgh for right hemisphere) from the FreeSurfer mgh format.
```Python
python read_modalities.py
```
2、Process data matrix.
```Python
python mask_modalities.py
```
3、Run Spatially Regularized NMF.
(1) Process single hemisphere
```Python
python SpatiallyRegularizedNMF.py --hemi lh
```
(2) Process bilateral hemispheres (default)
```Python
python SpatiallyRegularizedNMF.py
```
