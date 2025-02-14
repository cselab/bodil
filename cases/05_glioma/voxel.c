#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  const int n = 256;
  const char *input_path = "GM.raw";
  const char *output_path = "voxel.raw";
  const char *xdmf_path = "voxel.xdmf2";
  const int shift[8][3] = {
      {0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0},
      {1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0},
  };
  FILE *input, *output, *xdmf;
  int i, j, k, l, p;
  uint8_t val;
  float xyz[3 * 8];
  long ncell;

  if ((input = fopen(input_path, "r")) == NULL) {
    fprintf(stderr, "voxel: error: fail to open '%s'\n", input_path);
    exit(1);
  }

  if ((output = fopen(output_path, "w")) == NULL) {
    fprintf(stderr, "voxel: error: fail to open '%s'\n", output_path);
    exit(1);
  }
  ncell = 0;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
        if (fread(&val, sizeof val, 1, input) != 1) {
          fprintf(stderr, "voxel: error: fail to read '%s'\n", input_path);
          exit(1);
        }
        if (val == 1) {
          p = 0;
          for (l = 0; l < 8; l++) {
            xyz[p++] = i + shift[l][0];
            xyz[p++] = j + shift[l][1];
            xyz[p++] = k + shift[l][2];
          }
          if (fwrite(xyz, sizeof xyz, 1, output) != 1) {
            fprintf(stderr, "voxel: error: fail to write to '%s'\n",
                    output_path);
            exit(1);
          }
          ncell++;
        }
      }

  if ((xdmf = fopen(xdmf_path, "w")) == NULL) {
    fprintf(stderr, "voxel: error: fail to open '%s'\n", xdmf_path);
    exit(1);
  }
  fprintf(xdmf,
          "<Xdmf\n"
          "    Version=\"2\">\n"
          "  <Domain>\n"
          "    <Grid>\n"
          "      <Topology\n"
          "          TopologyType=\"Hexahedron\"\n"
          "          Dimensions=\"%ld\"/>\n"
          "      <Geometry>\n"
          "        <DataItem\n"
          "            Dimensions=\"%ld 3\"\n"
          "            Format=\"Binary\">\n"
          "          %s\n"
          "        </DataItem>\n"
          "      </Geometry>\n"
          "    </Grid>\n"
          "  </Domain>\n"
          "</Xdmf>\n",
          ncell, 8l * ncell, output_path);

  fclose(input);
  fclose(output);
}
