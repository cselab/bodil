#!/usr/bin/env  python3

import argparse
import numpy as np
import scipy
import skimage
import trimesh

from prepare_data import SEG_CODE, load_data, get_grid_spacing

def compute_PTV_volume(data_dir, dist_standard_plan):
    """
    Compute the planning target volume (PTV) and the corresponding contour surface obtained with the standard plan.

    Arguments:
        data_dir: path to patient data
        dist_standard_plan: distance of the standard plan, in mm
    Return:
        mesh: triangle mesh
        volume: volume of the mesh, in mm^3
    """
    meta_data, raw_data, trimmed_data = load_data(data_dir)

    dx, dy, dz = get_grid_spacing(meta_data['nifti_header'])
    seg = trimmed_data['seg']

    mask = np.where(seg == SEG_CODE.core, 0, 1)
    distances = scipy.ndimage.distance_transform_edt(mask, sampling=[dx, dy, dz])

    vertices, faces, normals, values = skimage.measure.marching_cubes(distances,
                                                                      dist_standard_plan,
                                                                      spacing=(dx, dy, dz))
    #vertices += np.array(meta_data['crop_offset'])[None,:]
    vertices = vertices[:,::-1]

    mesh = trimesh.Trimesh(faces=faces, vertices=vertices)
    volume = abs(float(mesh.volume))
    return mesh, volume

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_data', type=str, help='path to patient directory')
    parser.add_argument('--out-mesh', type=str, default=None, help='path to output mesh')
    args = parser.parse_args()

    path_datadir = args.path_data
    path_out_mesh = args.out_mesh
    dist_standard_plan = 15 # mm

    mesh, volume = compute_PTV_volume(path_datadir, dist_standard_plan)

    print(f"Volume: {volume*1e-3} cm**3")
    if path_out_mesh:
        mesh.export(path_out_mesh)



if __name__ == '__main__':
    main()
