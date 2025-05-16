#!/usr/bin/env  python3

import argparse
import glob
import numpy as np
import os
import scipy
import skimage
import trimesh

from utils import read_vtk
from compute_standard_plan import compute_PTV_volume

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_samples', type=str, help='path to directory that contains samples')
    parser.add_argument('--patient-data-dir', required=True, type=str, help='path to patient data directory')
    parser.add_argument('--out-dir', required=True, type=str, help='path to output directory')
    args = parser.parse_args()

    path_samples = args.path_samples
    path_patient_data_dir = args.patient_data_dir
    out_dir = args.out_dir
    dist_standard_plan = 15 # mm

    _, target_volume = compute_PTV_volume(path_patient_data_dir, dist_standard_plan)

    os.makedirs(out_dir, exist_ok=True)
    vtk_paths = glob.glob(os.path.join(path_samples, 'x0_*_y0_*_z0_*', 'seg_final.vtk'))

    for i, path in enumerate(vtk_paths):
        u, spacing, origin, varname = read_vtk(path)
        dx, dy, dz = spacing

        def get_surface(u, level):
            vertices, faces, normals, values = skimage.measure.marching_cubes(u, level, spacing=spacing)
            vertices = vertices[:,::-1]
            mesh = trimesh.Trimesh(faces=faces, vertices=vertices)
            return mesh

        def f(level):
            mesh = get_surface(u, level)
            volume = abs(float(mesh.volume))
            return volume - target_volume

        ulo, uhi = np.min(u) + 1e-2,  np.max(u) - 1e-2
        level = scipy.optimize.bisect(f, ulo, uhi, xtol=1e-4)

        mesh = get_surface(u, level)
        out_path = os.path.join(out_dir, f'sample-{i:06d}.ply')
        mesh.export(out_path)

if __name__ == '__main__':
    main()
