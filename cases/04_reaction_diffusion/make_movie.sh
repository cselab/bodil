#!/bin/sh

ffmpeg -framerate 25 -i u-%06d.png -c:v libx264 -crf 20 rd.mp4
