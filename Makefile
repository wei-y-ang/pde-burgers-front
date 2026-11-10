.PHONY: install test smoke data train reproduce clean

install:
	pip install -r requirements.txt

test:
	python -m unittest discover -s tests -v

smoke:
	python scripts/smoke_test.py

data:
	python scripts/download_subset.py --file-id 281363 --n-samples 200 --nu 0.01 --out data/burgers_nu0.01_200.npz

train:
	python -m pde_burgers.train --data data/burgers_nu0.01_200.npz --model fno --out results/runs/fno --epochs 40 --batch-size 32 --width 32 --modes 16 --n-layers 4

reproduce: test data train
	python scripts/plot_rollout.py --npz results/runs/fno/rollout_example.npz --out-dir results/figures/fno

clean:
	rm -rf results/runs results/figures
