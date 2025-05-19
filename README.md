# Cortical-Aging-Atlas
A surface area aging atlas for healthy middle-aged and elderly populations

We constructed a surface area-based brain aging atlas using healthy middle-aged and elderly control data from the ADNI（https://ida.loni.usc.edu/pages/access/search.jsp?tab=collection&project=ADNI&page=DOWNLOADS&subPage=IMAGE_COLLECTIONS） dataset, employing non-negative matrix factorization constrained by cortical surface topology. This methodology enabled the identification of three distinct cortical atrophy subtypes through advanced pattern decomposition.

Hierarchical
```plaintext
Aging Atlas/
├── SpatiallyRegularizedNMF.py
├── requirements.txt
├── CN/
│   ├── lh_area.mgh # LH area matrix
│   ├── rh_area.mgh # RH area matrix
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

## Quick star
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
3、Run NMF.
(1) Process single hemisphere
```Python
python spatial_nmf.py --hemi lh
```
(2) Process bilateral hemispheres (default)
```Python
python SpatiallyRegularizedNMF.py
```

## Atlas
![atlas](https://github.com/user-attachments/assets/a0ca5201-914d-4f55-ba82-11629476b1d8)

## Atrophy subtypes
![atrophy subtypes](https://github.com/user-attachments/assets/0a951efa-3be1-4bf5-9667-ec57aaad2283)


