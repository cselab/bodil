#!/usr/bin/env  python3

import argparse
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import skimage
import trimesh

from utils import read_vtk
from prepare_data import SEG_CODE, load_data, get_grid_spacing

def compute_standard_PTV_volume(trimmed_data, meta_data, standard_plan_margin, threshold_matter=0.05):
    """
    Compute the planning target volume (PTV) and the corresponding contour surface obtained with the standard plan.

    Arguments:
        trimmed_data: patient data, trimmed; see prepare_data.py
        meta_data: patient metadata
        standard_plan_margin: distance of the standard plan, in mm
    Return:
        mesh: triangle mesh
        volume: volume of the mesh, in mm^3
    """
    dx, dy, dz = get_grid_spacing(meta_data['nifti_header'])
    seg = trimmed_data['seg']
    matter = trimmed_data['wm'] + trimmed_data['gm']

    mask = np.where(seg == SEG_CODE.core, 0, 1)
    distances = scipy.ndimage.distance_transform_edt(mask, sampling=[dx, dy, dz])

    # set large distance outside brain matter
    distances = np.where(matter > threshold_matter, distances, 2 * standard_plan_margin)

    vertices, faces, normals, values = skimage.measure.marching_cubes(distances,
                                                                      standard_plan_margin,
                                                                      spacing=(dx, dy, dz))
    offset = np.array(meta_data['crop_offset'])
    vertices += offset[None,:]
    vertices = vertices[:,::-1]

    mesh = trimesh.Trimesh(faces=faces, vertices=vertices)
    volume = abs(float(mesh.volume))
    return mesh, volume

def get_slice(meta_data, field, plane_origin, plane_normal):
    d = np.argmax(plane_normal)
    if np.linalg.norm(plane_normal) != abs(plane_normal[d]):
        raise ValueError(f"Expected a normal aligned with x, y or z axis")

    dx, dy, dz = get_grid_spacing(meta_data['nifti_header'])
    spacing = np.array([dx, dy, dz])
    r = plane_origin
    im = (r / spacing).astype(int)
    ip = im + 1
    l = (r / spacing - im)[d]

    sm = tuple(im[j] if j == d else slice(None) for j in range(3))
    sp = tuple(ip[j] if j == d else slice(None) for j in range(3))

    return (1.0 - l) * field[sm] + l * field[sp]


def main():
    parser = argparse.ArgumentParser(description="Compute the PTV of standard and uq-gliodil plans for a specific patient.")
    parser.add_argument('patient_data', type=str, help='path to patient directory')
    parser.add_argument('--standard-plan-margin', type=float, default=15, help='standard plan margin, in mm')
    parser.add_argument('--out-dir', type=str, default="out_PTV", help='output directory')
    parser.add_argument('--path-samples', type=str, default=None, help='path to directory that contains samples')
    args = parser.parse_args()

    path_patient_data = args.patient_data
    standard_plan_margin = args.standard_plan_margin
    out_dir = args.out_dir
    path_samples = args.path_samples

    os.makedirs(out_dir, exist_ok=True)

    meta_data, raw_data, trimmed_data = load_data(path_patient_data)
    offset = np.array(meta_data['crop_offset'])

    mesh_sPTV, volume_sPTV = compute_standard_PTV_volume(trimmed_data=trimmed_data,
                                                         meta_data=meta_data,
                                                         standard_plan_margin=standard_plan_margin)

    print(f"standard PTV volume: {volume_sPTV * 1e-3} cm^3")
    mesh_sPTV.export(os.path.join(out_dir, 'standard_PTV.ply'))


    mesh_gliodil = []

    if path_samples:
        vtk_paths = glob.glob(os.path.join(path_samples, 'x0_*_y0_*_z0_*', 'seg_final.vtk'))

        for i, path in enumerate(vtk_paths):
            u, spacing, origin, varname = read_vtk(path)
            dx, dy, dz = spacing

            def get_surface(u, level):
                vertices, faces, normals, values = skimage.measure.marching_cubes(u, level, spacing=spacing)
                vertices += offset[None,:]
                vertices = vertices[:,::-1]
                mesh = trimesh.Trimesh(faces=faces, vertices=vertices)
                return mesh

            def f(level):
                mesh = get_surface(u, level)
                volume = abs(float(mesh.volume))
                return volume - volume_sPTV

            ulo, uhi = np.min(u) + 1e-2,  np.max(u) - 1e-2
            level = scipy.optimize.bisect(f, ulo, uhi, xtol=1e-4)

            mesh = get_surface(u, level)
            out_path = os.path.join(out_dir, f'sample-{i:06d}.ply')
            mesh.export(out_path)
            mesh_gliodil.append(mesh)


    plane_origin = np.array(mesh_sPTV.center_mass)[::-1]
    plane_normal = np.array([0.0, 0.0, 1.0])

    slice_wm = get_slice(meta_data=meta_data, field=raw_data['wm'], plane_origin=plane_origin, plane_normal=plane_normal)
    slice_gm = get_slice(meta_data=meta_data, field=raw_data['gm'], plane_origin=plane_origin, plane_normal=plane_normal)

    lines_sPTV = trimesh.intersections.mesh_plane(mesh_sPTV, plane_normal[::-1], plane_origin[::-1])

    fig, ax = plt.subplots()
    ax.imshow(slice_gm, origin='lower', cmap='grey')
    lc = matplotlib.collections.LineCollection(lines_sPTV[:,:,1:], colors='r', linewidths=2)
    ax.add_collection(lc)

    for mesh in mesh_gliodil:
        lines = trimesh.intersections.mesh_plane(mesh, plane_normal[::-1], plane_origin[::-1])
        lc = matplotlib.collections.LineCollection(lines[:,:,1:], colors='b', linewidths=0.2)
        ax.add_collection(lc)

    plt.show()

if __name__ == '__main__':
    main()
