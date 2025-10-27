"""
Spatially Regularized NMF Implementation for Neuroimaging Data Analysis
Authors: anonymous
License: MIT License
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nibabel as nib
from nibabel.freesurfer.io import write_annot
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csgraph

class Config:
    FREESURFER_HOME = os.environ.get('FREESURFER_HOME', '/opt/freesurfer')
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'CN')  
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')  

def validate_paths():
    required_paths = [
        os.path.join(Config.FREESURFER_HOME, 'subjects/fsaverage6/surf'),
        Config.DATA_DIR,
        Config.OUTPUT_DIR
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

def spatially_regularized_nmf(V, components, adjacency_matrix, L, initial_lambda_reg, iternum, random_state):
    scaler = MinMaxScaler()
    V_normalized = scaler.fit_transform(V)

    model = NMF(n_components=components, init='nndsvdar', solver='mu', max_iter=iternum, random_state=random_state)
    W = model.fit_transform(V_normalized)
    H = model.components_

    reconstruction_error = model.reconstruction_err_
        
    lambda_reg = initial_lambda_reg

    for iteration in range(iternum):
        W -= lambda_reg * np.dot(L, W)  
        
        negative_count = np.sum(W < 0)
        if negative_count > 0:
            negative_percentage = negative_count / W.size * 100
            W[W < 0] = 0 
        V_approx = np.dot(W, H)

        new_error = np.linalg.norm(V_normalized - V_approx)
        
        if abs(reconstruction_error - new_error) < 1e-6:
            break
        reconstruction_error = new_error
    return W, H, reconstruction_error

def build_adjacency_matrix(surf_file, mask):
    _, faces = nib.freesurfer.read_geometry(surf_file)
    included_indices = np.where(mask)[0]
    index_map = {orig: new for new, orig in enumerate(included_indices)}
    
    adjacency = np.zeros((len(included_indices), len(included_indices)), dtype=int)
    
    for face in faces:
        for i, j in [(0,1), (0,2), (1,2)]:
            if face[i] in index_map and face[j] in index_map:
                idx_i = index_map[face[i]]
                idx_j = index_map[face[j]]
                adjacency[idx_i, idx_j] = 1
                adjacency[idx_j, idx_i] = 1
                
    return adjacency

def process_hemisphere(hemi,seeds):
    print(f'Processing {hemi.upper()} hemisphere')
    
    input_dir = Config.DATA_DIR
    V = np.load(os.path.join(input_dir, f'{hemi}_filtered.npy'))
    mask = np.load(os.path.join(input_dir, f'{hemi}_mask.npy'))
    
    surf_path = os.path.join(Config.FREESURFER_HOME, 'subjects/fsaverage6/surf', f'{hemi}.pial')
    adjacency_matrix = build_adjacency_matrix(surf_path, mask)
    L = csgraph.laplacian(adjacency_matrix, normed=True)
    
    results = []
    output_dir = Config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for n_components in range(2, 21):
        all_labels = []  
        all_W = []      
        silhouette_scores = [] 
        reconstruction_errors = [] 

        for seed in seeds:                
            W, H, reconstruction_error = spatially_regularized_nmf(
                    V, n_components, adjacency_matrix, L, 
                    initial_lambda_reg=0.05, iternum=10000, random_state=seed) 
            kmeans = KMeans(n_clusters=n_components, random_state=seed, init='k-means++').fit(W)
            labels = kmeans.labels_

            all_labels.append(labels)
            all_W.append(W)
            reconstruction_errors.append(reconstruction_error)
            if n_components > 1:
                silhouette_avg = silhouette_score(W, labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(np.nan)
                
        if n_components > 1:
            best_idx = np.argmax(silhouette_scores)
            best_silhouette = silhouette_scores[best_idx]
            best_reconstruction_error = reconstruction_errors[best_idx]
            final_labels = all_labels[best_idx]
            best_seed = seeds[best_idx]
        else:
            best_silhouette = np.nan
            best_reconstruction_error = reconstruction_errors[0]
            final_labels = all_labels[0]
            best_seed = seeds[0]
            
        avg_silhouette = np.mean([s for s in silhouette_scores if not np.isnan(s)]) if n_components > 1 else np.nan
        avg_reconstruction_error = np.mean(reconstruction_errors)
        
        save_cluster_annot(final_labels, mask, output_dir, hemi, n_components)
        results.append({
                'n_components': n_components,
                'reconstruction_error': avg_reconstruction_error,
                'silhouette_score': avg_silhouette,
                'best_silhouette': best_silhouette,
                'best_reconstruction_error': best_reconstruction_error,
                'best_seed': best_seed
            })
    
    save_metrics(results, output_dir, hemi)
    generate_plots(results, output_dir, hemi)

def save_cluster_annot(labels, mask, output_dir, hemi, n_components):
    vertex_labels = np.zeros(40962, dtype=int)
    vertex_labels[mask] = labels + 1  
    
    colormap = np.zeros((n_components+1, 5), dtype=int)
    cmap = plt.cm.get_cmap('jet', n_components)
    
    for i in range(n_components):
        colormap[i+1, :4] = (cmap(i)[:3] * 255).astype(int)  
    
    names = [f'Cluster{i+1}' for i in range(n_components)]
    names.insert(0, 'unknown')  
    
    annot_path = os.path.join(output_dir, f'nmf_clusters_{hemi}_{n_components}.annot')
    write_annot(annot_path, vertex_labels, colormap, names)

def save_metrics(results, output_dir, hemi):
    metrics_path = os.path.join(output_dir, f'nmf_kmeans_results_{hemi}.txt')
    with open(metrics_path, 'w') as f:
        f.write('n_components\treconstruction_error\tsilhouette_score\n')
        for res in results:
            f.write(f"{res['n_components']}\t{res['reconstruction_error']:.6f}\t{res['silhouette_score']:.6f}\n")

def generate_plots(results, output_dir, hemi):
    x = [r['n_components'] for r in results]
    errors = [r['reconstruction_error'] for r in results]
    scores = [r['silhouette_score'] for r in results]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, errors, 'b-o', markersize=5)
    plt.title(f'{hemi} Reconstruction Error')
    plt.xlabel('Number of Components')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x[1:], scores[1:], 'r-o', markersize=5)
    plt.title(f'{hemi} Silhouette Score')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f'analysis_{hemi}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spatially Regularized NMF for Cortical Parcellation')
    parser.add_argument('--hemi', choices=['both', 'lh', 'rh'], default='both',
                       help='Hemisphere(s) to process (default: both)')
    
    args = parser.parse_args()
    
    try:
        validate_paths()
        start_time = time.time()
        
        hemispheres = ['lh', 'rh'] if args.hemi == 'both' else [args.hemi]

        seeds = [2, 17, 41, 67, 97]
        
        for hemi in hemispheres:
            process_hemisphere(hemi,seeds)
        
        print(f'Total processing time: {time.time()-start_time:.2f} seconds')
    
    except Exception as e:
        print(f'Error occurred: {str(e)}')
