#!/usr/bin/env python

import argparse
import numpy as np
import nibabel as nib
import os

# code for segmentation in nifti image
class NIFTI_CODE:
    necrotic = 4
    enhancing = 1
    edema = 3

class SEG_CODE:
    pass

def find_data_paths(patient_path):
    seg_path = os.path.join(patient_path, "segm.nii.gz")
    gm_path = os.path.join(patient_path, "t1_gm.nii.gz")
    wm_path = os.path.join(patient_path, "t1_wm.nii.gz")

    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Could not find segmentation file {seg_path}")
    if not os.path.isfile(gm_path):
        raise FileNotFoundError(f"Could not find grey matter file {gm_path}")
    if not os.path.isfile(wm_path):
        raise FileNotFoundError(f"Could not find white matter file {wm_path}")

    return seg_path, gm_path, wm_path

def read_nifti(path):
    nifti_img = nib.load(path)
    volume_array = nifti_img.get_fdata()
    return volume_array, nifti_img.affine, nifti_img.header

def crop_scale_data(seg, gm, wm, threshold, scale):
    """
    Crop data to keep only grid sizes that are above a given threshold.
    """
    idx = seg > threshold
    # get x range
    x_min = np.argmax(idx.any(2).any(1).astype(int))
    x_max = seg.shape[0] - np.argmax(idx.any(2).any(1).astype(int)[::-1])
    # get y range
    y_min = np.argmax(idx.any(2).any(0).astype(int))
    y_max = seg.shape[1] - np.argmax(idx.any(2).any(0).astype(int)[::-1])
    # get z range
    z_min = np.argmax(idx.any(1).any(0).astype(int))
    z_max = seg.shape[2] - np.argmax(idx.any(1).any(0).astype(int)[::-1])

    if scale == 1:
        return seg[x_min:x_max,y_min:y_max,z_min:z_max], \
            gm[x_min:x_max,y_min:y_max,z_min:z_max], \
            wm[x_min:x_max,y_min:y_max,z_min:z_max]
    else:
        x_min_new = np.ceil(x_min - ((x_min + x_max)/2 - x_min)*(scale-1)).astype(int)
        x_max_new = np.ceil(x_max + ((x_min + x_max)/2 - x_min)*(scale-1)).astype(int)

        y_min_new = np.ceil(y_min - ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
        y_max_new = np.ceil(y_max + ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)

        y_min_new = np.ceil(y_min - ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
        y_max_new = np.ceil(y_max + ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)

        z_min_new = np.ceil(z_min - ((z_min + z_max)/2 - z_min)*(scale-1)).astype(int)
        z_max_new = np.ceil(z_max + ((z_min + z_max)/2 - z_min)*(scale-1)).astype(int)

        return seg[x_min_new:x_max_new,y_min_new:y_max_new,z_min_new:z_max_new], \
            gm[x_min_new:x_max_new,y_min_new:y_max_new,z_min_new:z_max_new], \
            wm[x_min_new:x_max_new,y_min_new:y_max_new,z_min_new:z_max_new]


def load_data(data_path, trim_threshold=0.1, trim_scale=1.5):
    seg_path, gm_path, wm_path = find_data_paths(data_path)

    seg, seg_affine, seg_header = read_nifti(seg_path)
    gm, _, _ = read_nifti(gm_path)
    wm, _, _ = read_nifti(wm_path)

    seg = np.where(seg == NIFTI_CODE.edema, 1,
                   np.where(np.isin(seg, [NIFTI_CODE.enhancing, NIFTI_CODE.necrotic]), 2,
                            0))

    trimmed_seg, trimmed_gm, trimmed_wm = crop_scale_data(seg, gm, wm, trim_threshold, trim_scale)

    meta_data = {
        'nifti_affine': seg_affine,
        'nifti_header': seg_header
    }

    raw_data = {
        'seg': seg,
        'gm': gm,
        'wm': wm
    }

    trimmed_data = {
        'seg': trimmed_seg,
        'gm': trimmed_gm,
        'wm': trimmed_wm
    }

    return meta_data, raw_data, trimmed_data




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to .nii files')
    args = parser.parse_args()

    meta_data, raw_data, trimmed_data = load_data(args.data_path)





if __name__ == '__main__':
    main()
