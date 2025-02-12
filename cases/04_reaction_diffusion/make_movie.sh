#!/bin/sh

: ${out_dir=out_forward}

ffmpeg -framerate 25 -i $out_dir/u-%06d.png -c:v libx264 -crf 20 $out_dir/rd.mp4
