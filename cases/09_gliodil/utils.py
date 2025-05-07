import numpy as np

def dump_vtk(u, dx, dy, dz, origin, path, varname='u'):
    nx, ny, nz = u.shape
    num_points = nx * ny * nz
    spacing = (dx, dy, dz)

    with open(path, "wb") as f:
        header = f"""# vtk DataFile Version 3.0
Binary uniform grid
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS {nz} {ny} {nx}
ORIGIN {origin[2]} {origin[1]} {origin[0]}
SPACING {spacing[2]} {spacing[1]} {spacing[0]}
POINT_DATA {num_points}
SCALARS {varname} float
LOOKUP_TABLE default
"""
        f.write(header.encode("utf-8"))
        f.write(u.astype('>f4').tobytes())  # '>f4' = big-endian float32


def read_vtk(path):
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            if b'SCALARS' in line:
                header_lines.append(line)
                header_lines.append(f.readline())  # also read LOOKUP_TABLE line
                break
            header_lines.append(line)

        header_text = b''.join(header_lines).decode('utf-8')

        for line in header_text.splitlines():
            if line.startswith("DIMENSIONS"):
                nz, ny, nx = map(int, line.split()[1:])
            elif line.startswith("ORIGIN"):
                loz, loy, lox = map(float, line.split()[1:])
            elif line.startswith("SPACING"):
                dz, dy, dx = map(float, line.split()[1:])
            elif line.startswith("POINT_DATA"):
                num_points = int(line.split()[1])
            elif line.startswith("SCALARS"):
                varname = line.split()[1]

        assert num_points == nx * ny * nz
        data = np.fromfile(f, dtype='>f4', count=num_points)  # big-endian float32

    data = data.reshape((nx, ny, nz))
    spacing = (dx, dy, dz)
    origin = (lox, loy, loz)
    return data, spacing, origin, varname
