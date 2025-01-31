#!/usr/bin/env python3

import numpy as np
import torch
import trimesh

def main():
    mesh = trimesh.creation.icosphere(radius=1, subdivisions=2)
    print(mesh)

if __name__ == '__main__':
    main()
