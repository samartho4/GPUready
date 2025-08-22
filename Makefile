.PHONY: setup data tune train eval stats baselines dataset bench results figs verify repro test determinism generalization ablations symbolic forensics

setup:
	bin/mg setup

data:
	bin/mg data

tune:
	bin/mg tune

train:
	bin/mg train

eval:
	bin/mg eval

stats:
	bin/mg stats

baselines:
	bin/mg baselines

dataset:
	bin/mg dataset

bench:
	bin/mg bench

results:
	bin/mg results

figs:
	bin/mg figs

verify:
	bin/mg verify

repro:
	bin/mg repro

determinism:
	bin/mg determinism

generalization:
	bin/mg generalization

ablations:
	bin/mg ablations

symbolic:
	bin/mg symbolic

forensics:
	bin/mg forensics

test:
	julia --project=. -e 'using Pkg; Pkg.test()' 