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
    healthy = 0
    edema = 1
    core = 2

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

def _get_smallest_rectangle_with_all_nonhealthy_cells(seg):
    nx, ny, nz = seg.shape
    # select "interesting" region
    idx = np.isin(seg, [SEG_CODE.edema, SEG_CODE.core])

    idxx = idx.any((1,2))
    xmin = np.argmax(idxx)
    xmax = nx - np.argmax(idxx[::-1])

    idxy = idx.any((0,2))
    ymin = np.argmax(idxy)
    ymax = ny - np.argmax(idxy[::-1])

    idxz = idx.any((0,1))
    zmin = np.argmax(idxz)
    zmax = nz - np.argmax(idxz[::-1])

    return xmin, xmax, ymin, ymax, zmin, zmax

def _get_scaled_range(lo, hi, scale):
    lo_new = np.ceil(lo - ((lo + hi)/2 - lo) * (scale - 1)).astype(int)
    hi_new = np.ceil(hi + ((lo + hi)/2 - lo) * (scale - 1)).astype(int)
    return lo_new, hi_new

def crop_scale_data(seg, gm, wm, scale):
    """
    Crop data to only keep region of the grid that contains tumor or necrotic cells.

    scale: used to select a region that is larger/smaller than the smallest rectangle that contains the tumor cells.
    if Lx, Ly, Lz is the size of this rectangle, the selected cropped data will be the rectangle with the same center,
    and with size scale * Lx, scale * Ly, scale * Lz.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = _get_smallest_rectangle_with_all_nonhealthy_cells(seg)

    xmin, xmax = _get_scaled_range(xmin, xmax, scale)
    ymin, ymax = _get_scaled_range(ymin, ymax, scale)
    zmin, zmax = _get_scaled_range(zmin, zmax, scale)

    return seg[xmin:xmax,ymin:ymax,zmin:zmax], \
        gm[xmin:xmax,ymin:ymax,zmin:zmax], \
        wm[xmin:xmax,ymin:ymax,zmin:zmax]

def restore_cropped_data(seg_original, scale, trimmed_seg):
    xmin, xmax, ymin, ymax, zmin, zmax = _get_smallest_rectangle_with_all_nonhealthy_cells(seg_original)

    xmin, xmax = _get_scaled_range(xmin, xmax, scale)
    ymin, ymax = _get_scaled_range(ymin, ymax, scale)
    zmin, zmax = _get_scaled_range(zmin, zmax, scale)

    seg = np.zeros_like(seg_original)
    seg[xmin:xmax,ymin:ymax,zmin:zmax] = trimmed_seg
    return seg

def load_data(data_path, trim_scale=1.5):
    seg_path, gm_path, wm_path = find_data_paths(data_path)

    seg, seg_affine, seg_header = read_nifti(seg_path)
    gm, _, _ = read_nifti(gm_path)
    wm, _, _ = read_nifti(wm_path)

    seg = np.where(seg == NIFTI_CODE.edema, SEG_CODE.edema,
                   np.where(np.isin(seg, [NIFTI_CODE.enhancing, NIFTI_CODE.necrotic]), SEG_CODE.core,
                            SEG_CODE.healthy))

    trimmed_seg, trimmed_gm, trimmed_wm = crop_scale_data(seg, gm, wm, trim_scale)

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

    trim_scale = 1.5
    meta_data, raw_data, trimmed_data = load_data(args.data_path, trim_scale)

    # test to reconstruct seg from trimmed data.
    seg = restore_cropped_data(raw_data['seg'], trim_scale, trimmed_data['seg'])

    nifti_file = nib.Nifti1Image(seg, meta_data['nifti_affine'], header=meta_data['nifti_header'])
    nib.save(nifti_file, 'test.nii')

if __name__ == '__main__':
    main()
