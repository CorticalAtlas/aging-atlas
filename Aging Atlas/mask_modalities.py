import numpy as np
from nibabel.freesurfer.io import write_annot, read_annot
import os

class Config:
    FREESURFER_HOME = os.environ.get('FREESURFER_HOME', '/opt/freesurfer')
    DATA_ROOT = os.path.join(os.path.dirname(__file__), 'CN')

hemispheres = ['lh', 'rh']

def create_mask(hemi: str) -> None:
    try:
        annot_path = os.path.join(
            Config.FREESURFER_HOME,
            'subjects/fsaverage6/label',
            f'{hemi}.aparc.a2009s.annot'
        )
        input_npy = os.path.join(Config.DATA_ROOT, f'{hemi}_V.npy')
        output_dir = Config.DATA_ROOT
        
        os.makedirs(output_dir, exist_ok=True)

        labels, ctab, names = read_annot(annot_path)
        
        exclude_label = np.where(np.array(names) == b'Medial_wall')[0][0]
        mask = labels != exclude_label
        
        mask_path = os.path.join(output_dir, f'{hemi}_mask.npy')
        np.save(mask_path, mask)
        print(f"[{hemi.upper()}] Mask saved: {mask_path}")

        matrix = np.load(input_npy)
        filtered_matrix = matrix[mask]
        filtered_path = os.path.join(output_dir, f'{hemi}_filtered.npy')
        np.save(filtered_path, filtered_matrix)
        print(f"[{hemi.upper()}] Filtered matrix saved: {filtered_path}")
        print(f"Original shape: {matrix.shape} -> Filtered shape: {filtered_matrix.shape}")

        new_labels = np.where(mask, 0, 1)  
        annot_output = os.path.join(output_dir, f'{hemi}_mask.annot')
        write_annot(
            annot_output,
            new_labels,
            np.array([[0]*5, [255]*4+[0]], dtype=int),  
            [b'Included', b'Background']
        )
        print(f"[{hemi.upper()}] Mask annot saved: {annot_output}")

    except Exception as e:
        print(f"Error processing {hemi.upper()}: {str(e)}")
        raise

if __name__ == '__main__':
    required_paths = [
        Config.FREESURFER_HOME,
        Config.DATA_ROOT
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

    for hemi in hemispheres:
        print(f"\n{'='*30} Processing {hemi.upper()} {'='*30}")
        create_mask(hemi)