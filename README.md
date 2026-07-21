# Cortical-Aging-Atlas
A surface area aging atlas for healthy middle-aged and elderly populations


## Atlas
<img width="2458" height="950" alt="atlas" src="https://github.com/user-attachments/assets/7af89730-acf7-42a5-bf82-2d2387fb5d56" />


## Different methods results
<img width="2778" height="1824" alt="different methods" src="https://github.com/user-attachments/assets/1c862342-687a-438c-a097-5420214c583c" />


## Atrophy subtypes
<img width="1818" height="520" alt="subtypes" src="https://github.com/user-attachments/assets/85bfc984-c153-4476-8f0d-994b9ef4de6f" />



We constructed a healthy aging atlas using only healthy control group data from the Alzheimer's Disease Neuroimaging Initiative ADNI（https://ida.loni.usc.edu/pages/access/search.jsp?tab=collection&project=ADNI&page=DOWNLOADS&subPage=IMAGE_COLLECTIONS）, comprising 1,066 longitudinal scans (male: 76.36 ± 6.78 years; female: 75.36 ± 6.98 years) to establish a baseline for healthy aging. T1-weighted MR scans were preprocessed using FreeSurfer's standard pipeline for cortical surface reconstruction, and the resulting data were concatenated to generate a surface area matrix V. Under the Riemannian manifold framework, the cortical surface was modeled as a triangular mesh manifold M. The topological connections between cortical vertices were encoded using a graph Laplacian operator. A two-stage Non-negative Matrix Factorization (NMF) architecture was employed:
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
│   ├── nmf_clusters_lh_16.annot
│   └── nmf_clusters_rh_16.annot
└── results/
    ├── nmf_clusters_lh_2.annot
    ├── nmf_clusters_lh_3.annot
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
