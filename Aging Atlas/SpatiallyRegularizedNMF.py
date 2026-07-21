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
import nibabel as nib
from nibabel.freesurfer.io import write_annot
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (silhouette_score,confusion_matrix)
from scipy.sparse import csgraph
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
class Config:
    FREESURFER_HOME = os.environ.get(
        "FREESURFER_HOME",
        "/opt/freesurfer"
    )
    DATA_DIR = os.path.join(
        os.path.dirname(__file__),
        "CN"
    )

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__),
        "results"
    )

    FSAVERAGE_VERTEX_NUM = 40962

# ---------------------------------------------------------------------
# Validate Required Paths
# ---------------------------------------------------------------------

def validate_paths():
    required_paths = [
        os.path.join(
            Config.FREESURFER_HOME,
            "subjects/fsaverage6/surf"
        ),
        Config.DATA_DIR,
        Config.OUTPUT_DIR
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

# ---------------------------------------------------------------------
# Stability Analysis Functions
# ---------------------------------------------------------------------

def compute_dice_matrix(
        labels1,
        labels2,
        n_clusters):

    """
    Compute Dice similarity matrix between two clustering solutions.
    Parameters
    ----------
    labels1 : ndarray
    labels2 : ndarray
    n_clusters : int
    Returns
    -------
    dice_matrix : ndarray
    """

    cm = confusion_matrix(
        labels1,
        labels2,
        labels=np.arange(n_clusters)
    )

    sum1 = cm.sum(axis=1)[:, np.newaxis]
    sum2 = cm.sum(axis=0)[np.newaxis, :]
    denominator = sum1 + sum2

    with np.errstate(
            divide="ignore",
            invalid="ignore"):
        dice_matrix = np.where(
            denominator == 0,
            0.0,
            2.0 * cm / denominator
        )

    return dice_matrix

def calculate_pairwise_similarity(
        solutions,
        n_clusters):

    """
    Calculate pairwise atlas similarity.
    Parameters
    ----------
    solutions : list
        List of label arrays.
    n_clusters : int
    Returns
    -------
    average_similarity : ndarray
        Mean Dice similarity for each run.
    overall_stability : float
        Mean pairwise Dice similarity across runs.
    """

    n_runs = len(solutions)

    if n_runs < 2:
        return np.array([1.0]), 1.0

    similarity_matrix = np.zeros(
        (n_runs, n_runs)
    )

    for i in range(n_runs):
        for j in range(i, n_runs):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue
                
            dice_matrix = compute_dice_matrix(
                solutions[i],
                solutions[j],
                n_clusters
            )

            row_ind, col_ind = linear_sum_assignment(-dice_matrix)

            mean_dice = dice_matrix[row_ind,col_ind].mean()

            similarity_matrix[i, j] = mean_dice
            similarity_matrix[j, i] = mean_dice

    average_similarity = (
        np.sum(
            similarity_matrix,
            axis=1
        ) - 1.0
    ) / (n_runs - 1)

    upper_triangle = np.triu_indices(n_runs,k=1)

    overall_stability = np.mean(similarity_matrix[upper_triangle])

    return (average_similarity,overall_stability)


# ---------------------------------------------------------------------
# Spatially Regularized NMF
# ---------------------------------------------------------------------

def spatially_regularized_nmf(
        V_normalized,
        n_components,
        L,
        lambda_reg,
        iternum,
        random_state):
    """
    Spatially Regularized NMF.
    Parameters
    ----------
    V_normalized : ndarray
        Min-Max normalized data matrix.
    n_components : int
        Number of components.
    L : ndarray
        Graph Laplacian matrix.
    lambda_reg : float
        Spatial regularization parameter.
    iternum : int
        Maximum iteration number.
    random_state : int
        Random seed.
    Returns
    -------
    W : ndarray
        Basis matrix.
    H : ndarray
        Coefficient matrix.
    reconstruction_error : float
        Final reconstruction error.
    """

    model = NMF(
        n_components=n_components,
        init="nndsvdar",
        solver="mu",
        max_iter=iternum,
        random_state=random_state
    )

    W = model.fit_transform(V_normalized)
    H = model.components_

    reconstruction_error = model.reconstruction_err_

    # Pre-allocate memory
    V_approx = np.empty_like(V_normalized)

    for iteration in range(iternum):
        # Avoid denormal floating-point numbers
        W = np.maximum(W, 1e-12)
        # Laplacian regularization
        W -= lambda_reg * L.dot(W)
        # Keep non-negative
        W[W < 0] = 0
        # Check convergence every 200 iterations
        if iteration % 200 == 0:
            np.dot(W, H, out=V_approx)
            new_error = np.linalg.norm(
                V_normalized - V_approx
            )

            if abs(reconstruction_error- new_error) < 1e-6:
                print(f"Converged at iteration " f"{iteration}")
                break

            reconstruction_error = new_error

    return W, H, reconstruction_error

# ---------------------------------------------------------------------
# Build Adjacency Matrix
# ---------------------------------------------------------------------

def build_adjacency_matrix(surf_file,mask):
    """
    Build adjacency matrix from fsaverage6 surface.
    """

    _, faces = nib.freesurfer.read_geometry(surf_file)

    included_indices = np.where(mask)[0]
    n_vertices = len(included_indices)
    index_map = {
        original_idx: new_idx
        for new_idx,
        original_idx
        in enumerate(included_indices)
    }

    adjacency_matrix = np.zeros((n_vertices, n_vertices),dtype=int)

    for face in faces:
        i, j, k = face
        if (i in index_map and j in index_map):
            u = index_map[i]
            v = index_map[j]
            adjacency_matrix[u, v] = 1
            adjacency_matrix[v, u] = 1

        if (i in index_map and k in index_map):
            u = index_map[i]
            v = index_map[k]
            adjacency_matrix[u, v] = 1
            adjacency_matrix[v, u] = 1

        if (j in index_map and k in index_map):
            u = index_map[j]
            v = index_map[k]
            adjacency_matrix[u, v] = 1
            adjacency_matrix[v, u] = 1

    return adjacency_matrix

# ---------------------------------------------------------------------
# Save FreeSurfer Annotation File
# ---------------------------------------------------------------------

def save_cluster_annot(
        labels,
        mask,
        output_dir,
        hemi,
        n_components):
    """
    Save Medoid atlas as FreeSurfer .annot file.
    """

    vertex_labels = np.zeros(Config.FSAVERAGE_VERTEX_NUM,dtype=int)

    # 0 = unknown
    # 1~K = cluster labels

    vertex_labels[mask] = labels + 1
    colormap = np.zeros((n_components + 1, 5),dtype=np.int32)

    cmap = plt.cm.get_cmap("jet",n_components)

    for i in range(n_components):
        rgba = cmap(i)
        colormap[i + 1, :4] = [
            int(x * 255)
            for x in rgba
        ]

        colormap[i + 1, 4] = i

    names = [f"Cluster{i + 1}"for i in range(n_components)]

    names.insert(0, "unknown")

    annot_path = os.path.join(
        output_dir,
        f"stable_nmf_"
        f"{hemi}_"
        f"{n_components}.annot"
    )

    write_annot(
        annot_path,
        vertex_labels,
        colormap,
        names
    )

# ---------------------------------------------------------------------
# Save Metrics
# ---------------------------------------------------------------------

def save_metrics(
        results,
        output_dir,
        hemi):
    """
    Save quantitative metrics.
    """

    metrics_path = os.path.join(
        output_dir,
        f"stable_nmf_results_"
        f"{hemi}.txt"
    )

    with open(metrics_path, "w") as f:
        f.write(
            "n_components\t"
            "reconstruction_error\t"
            "silhouette_score\t"
            "stability\n"
        )

        for res in results:
            f.write(
                f"{res['n_components']}\t"
                f"{res['reconstruction_error']:.6f}\t"
                f"{res['silhouette_score']:.6f}\t"
                f"{res['stability']:.6f}\n"
            )

# ---------------------------------------------------------------------
# Generate Figures
# ---------------------------------------------------------------------

def generate_plots(
        results,
        output_dir,
        hemi):
    """
    Generate quantitative evaluation plots.
    """

    n_components = [
        r["n_components"]
        for r in results
    ]

    reconstruction_errors = [
        r["reconstruction_error"]
        for r in results
    ]

    silhouette_scores = [
        r["silhouette_score"]
        for r in results
    ]

    stability_scores = [
        r["stability"]
        for r in results
    ]

    error_gradient = np.diff(reconstruction_errors)

    # --------------------------------------------------------------
    # Reconstruction Error
    # --------------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        n_components,
        reconstruction_errors,
        marker="o"
    )

    plt.title(
        f"{hemi.upper()} "
        f"Reconstruction Error (Medoid)"
    )

    plt.xlabel(
        "Number of Components (K)"
    )

    plt.ylabel(
        "Reconstruction Error"
    )

    plt.grid(False)

    plt.savefig(
        os.path.join(
            output_dir,
            f"Error_{hemi}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------------
    # Silhouette Score
    # --------------------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(
        n_components,
        silhouette_scores,
        marker="o"
    )

    plt.title(
        f"{hemi.upper()} "
        f"Silhouette Score (Medoid)"
    )

    plt.xlabel(
        "Number of Components (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.grid(False)

    plt.savefig(

        os.path.join(
            output_dir,
            f"Silhouette_{hemi}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------------
    # Stability Index
    # --------------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        n_components,
        stability_scores,
        marker="o"
    )

    plt.title(
        f"{hemi.upper()} "
        f"Atlas Stability"
    )

    plt.xlabel(
        "Number of Components (K)"
    )

    plt.ylabel(
        "Mean Pairwise Dice"
    )

    plt.grid(False)
    plt.savefig(

        os.path.join(
            output_dir,
            f"Stability_{hemi}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------------
    # Error Gradient
    # --------------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        n_components[1:],
        error_gradient,
        marker="o"
    )

    plt.title(
        f"{hemi.upper()} "
        f"Error Gradient"
    )

    plt.xlabel(
        "Number of Components (K)"
    )

    plt.ylabel(
        "Change in Error"
    )

    plt.grid(False)

    plt.savefig(

        os.path.join(
            output_dir,
            f"ErrorGradient_{hemi}.png"
        ),
        dpi=300
    )

    plt.close()

# ---------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------

def process_hemisphere(hemi,seeds):
    """
    Process one hemisphere.
    Workflow
    --------
    1. Load data.
    2. Build graph Laplacian.
    3. Run NMF multiple times.
    4. Compute atlas stability.
    5. Select Medoid solution.
    6. Save annotation files.
    7. Save metrics.
    8. Generate plots.
    """

    print(f"\nProcessing " f"{hemi.upper()} hemisphere ...")

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    input_dir = Config.DATA_DIR

    V = np.load(
        os.path.join(
            input_dir,
            f"{hemi}_filtered.npy"
        )
    )

    mask = np.load(
        os.path.join(
            input_dir,
            f"{hemi}_mask.npy"
        )
    )

    scaler = MinMaxScaler()

    V_normalized = scaler.fit_transform(V)

    # --------------------------------------------------------------
    # Surface adjacency matrix
    # --------------------------------------------------------------

    surf_file = os.path.join(
        Config.FREESURFER_HOME,
        "subjects",
        "fsaverage6",
        "surf",
        f"{hemi}.pial"
    )

    adjacency_matrix = build_adjacency_matrix(
        surf_file,
        mask
    )

    L = csgraph.laplacian(
        adjacency_matrix,
        normed=True
    )

    # --------------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------------

    output_dir = Config.OUTPUT_DIR

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    results = []

    # --------------------------------------------------------------
    # Iterate over K
    # --------------------------------------------------------------

    for n_components in range(2, 21):
        print(
            f"\nK = {n_components}"
        )

        run_solutions = []
        run_Ws = []
        run_errors = []
        run_silhouettes = []

        # ----------------------------------------------------------
        # Multiple repetitions
        # ----------------------------------------------------------

        for i, seed in enumerate(seeds):

            print(f"  Run "
                f"{i + 1}/"
                f"{len(seeds)} "
                f"(seed={seed})"
            )

            W, H, reconstruction_error = spatially_regularized_nmf(V_normalized,
                    n_components,
                    L,
                    lambda_reg=0.05,
                    iternum=10000,
                    random_state=seed
                )

            kmeans = KMeans(
                n_clusters=n_components,
                random_state=0,
                init="k-means++"
            ).fit(W)

            labels = kmeans.labels_

            silhouette_avg = silhouette_score(W,labels)

            run_solutions.append(labels)

            run_Ws.append(W)

            run_errors.append(reconstruction_error)

            run_silhouettes.append(silhouette_avg)

        # ----------------------------------------------------------
        # Stability analysis
        # ----------------------------------------------------------

        similarity_scores, stability_index = calculate_pairwise_similarity(run_solutions,n_components)

        medoid_index = np.argmax(similarity_scores)

        print(
            f"  Stability = "
            f"{stability_index:.4f}"
        )

        print(
            f"  Medoid run = "
            f"{medoid_index + 1}"
        )

        medoid_labels = run_solutions[medoid_index]

        medoid_error = run_errors[medoid_index]

        medoid_silhouette = run_silhouettes[medoid_index]


        # ----------------------------------------------------------
        # Save Medoid atlas
        # ----------------------------------------------------------

        save_cluster_annot(
            labels=medoid_labels,
            mask=mask,
            output_dir=output_dir,
            hemi=hemi,
            n_components=n_components
        )

        # ----------------------------------------------------------
        # Record results
        # ----------------------------------------------------------

        results.append({
            "n_components":n_components,
            "reconstruction_error":medoid_error,
            "silhouette_score":medoid_silhouette,
            "stability":stability_index
        })

    # --------------------------------------------------------------
    # Save metrics and figures
    # --------------------------------------------------------------

    save_metrics(
        results,
        output_dir,
        hemi
    )

    generate_plots(
        results,
        output_dir,
        hemi
    )

    print(
        f"\nFinished processing "
        f"{hemi.upper()} hemisphere."
    )

# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Spatially Regularized NMF with "
            "Atlas Stability Analysis"
        )
    )

    parser.add_argument(
        "--hemi",
        choices=["both", "lh", "rh"],
        default="both",
        help=(
            "Hemisphere(s) to process "
            "(default: both)"
        )
    )

    parser.add_argument(
        "--k_min",
        type=int,
        default=2,
        help=(
            "Minimum number of components "
            "(default: 2)"
        )
    )

    parser.add_argument(
        "--k_max",
        type=int,
        default=20,
        help=(
            "Maximum number of components "
            "(default: 20)"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Validate directories
    # --------------------------------------------------------------

    validate_paths()

    # --------------------------------------------------------------
    # Random seeds
    # --------------------------------------------------------------

    seeds = [2,17,41,67,97]

    # --------------------------------------------------------------
    # Hemisphere selection
    # --------------------------------------------------------------

    if args.hemi == "both":
        hemispheres = ["lh","rh"]

    else:
        hemispheres = [args.hemi]

    # --------------------------------------------------------------
    # Processing
    # --------------------------------------------------------------

    start_time = time.time()

    for hemi in hemispheres:
        process_hemisphere(
            hemi=hemi,
            seeds=seeds
        )

    elapsed_time = (time.time() - start_time)

    print("\n===================================")
    print("All processing completed.")
    print(
        f"Total time: "
        f"{elapsed_time:.2f} seconds"
    )
    print("===================================\n")

# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print("\nAn error occurred:")
        print(str(e))
        raise
