import nibabel as nib
import numpy as np
import os

hemispheres = ['lh', 'rh'] 

input_template = os.path.join('CN', '{hemi}_area.mgh')  
output_dir = os.path('CN') 

def z_score_standardization(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

os.makedirs(output_dir, exist_ok=True)

for hemi in hemispheres:
    input_path = input_template.format(hemi=hemi)
    output_path = os.path.join(output_dir, f"{hemi}_V.npy")  
    
    try:
        mgh = nib.load(input_path)
        original_data = mgh.get_fdata()
        processed_data = z_score_standardization(np.squeeze(original_data))
        
        np.save(output_path, processed_data)
        
        print(f"[{hemi.upper()}] Successfully processed:")
        print(f"Original shape: {original_data.shape}")
        print(f"Processed shape: {processed_data.shape}")
        print(f"Saved to: {output_path}\n")

    except FileNotFoundError:
        print(f"Error: Input file not found - {input_path}")
    except Exception as e:
        print(f"Error processing {hemi.upper()}: {str(e)}")