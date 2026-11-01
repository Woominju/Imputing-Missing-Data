#!/bin/sh
# Launches the main grids on reserved GPUs 0 and 1 (PCI_BUS_ID order). Per-process CSVs are merged later.
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.pydeps OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P=${PYTHON:-$HOME/.conda/envs/reef_gpu/bin/python}
COMMON="--tau 1 --m 8 --epochs 150 --patience 15"
mkdir -p results/parts results/logs
g=0
for p in 0.1 0.3 0.5 0.7; do
  CUDA_VISIBLE_DEVICES=$g $P experiments/run_lorenz.py $COMMON --models A,B,C,D --rates $p --out parts/lorenz_p$p.csv > results/logs/run_lorenz_p$p.txt 2>&1 &
  g=$((1-g))
done
for k in 0.0 0.5 2.0; do
  CUDA_VISIBLE_DEVICES=$g $P experiments/run_coupled_lorenz.py $COMMON --models C,D --kappas $k --out parts/coupled_k$k.csv > results/logs/run_coupled_k$k.txt 2>&1 &
  g=$((1-g))
done
for d in enso eeg wind exchange transportation; do
  CUDA_VISIBLE_DEVICES=$g $P experiments/run_real_data.py $COMMON --models A,B,C --datasets $d --out parts/real_$d.csv > results/logs/run_real_$d.txt 2>&1 &
  g=$((1-g))
done
wait
