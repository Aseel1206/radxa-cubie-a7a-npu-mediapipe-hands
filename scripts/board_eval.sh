#!/bin/bash
# Run one .nb over a folder of raw input files on the Radxa and fetch the outputs back.
#   scripts/board_eval.sh radxa@<board-ip> model.nb <dir with *.in.bin> <out dir>
# Needs nb_bench built on the board (see README) at ~/nb_bench/nb_bench and the awnn_inference install at ~/awnn_inference.
set -e
H=$1; NB=$2; IN=$3; OUT=$4; mkdir -p "$OUT"
R=npu_eval; ssh "$H" "mkdir -p ~/$R/in ~/$R/out"
scp -q "$NB" "$H:~/$R/model.nb"; scp -q "$IN"/*.in.bin "$H:~/$R/in/"
ssh "$H" 'cd ~/npu_eval && export LD_LIBRARY_PATH=$HOME/awnn_inference/install/lib
  for f in in/*.in.bin; do n=$(basename $f .in.bin); ~/nb_bench/nb_bench -m model.nb -n 1 -i0 $f -o out/$n.bin > /dev/null 2>&1; done
  echo "--- timing (200 runs, random input) ---"; ~/nb_bench/nb_bench -m model.nb -n 200 2>&1 | grep -E "^run|^output\["'
scp -q "$H:~/$R/out/*" "$OUT/"; echo "outputs in $OUT"
