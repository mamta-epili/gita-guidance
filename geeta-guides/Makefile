# Every target uses the venv's interpreter by absolute path, so nothing your
# shell does to the word `python` — aliases, pyenv/conda shims, an IDE
# activating a different interpreter — can affect these commands.
#
#   make setup           create .venv, install deps, fetch data, check hardware
#   make check M=1       run one milestone (M=0..4, default 0)
#   make check-all       run every milestone
#   make ref M=2         run a milestone against the sealed reference
#   make train           train your model, then sample
#   make sample          sample from your saved checkpoint
#   make device          report cuda / mps / cpu
#   make data            re-download and re-clean the char-GPT corpus
#   make verses          build data/verses.json (Sanskrit + Hindi + English)
#   make verses-verify   re-check verse counts and licence coverage
#   make speakers        label every verse with who is speaking
#   make speakers-check  verify the labels against traditional counts
#   make eval-validate   check the eval set is sound (incl. circularity)
#   make eval-score      score retrieval against the eval set
#   make eval            validate, then score
#   make show V="2:47 6:34"   render verses: shloka first, then translations
#   make embed           embed the 700 verses (downloads the model on first run)
#   make ask Q="..."     retrieve verses for a question, with scores
#   make eval-dense      score the dense retriever against the eval set
#   make eval-compare    BM25 vs dense, side by side

PY  := $(CURDIR)/.venv/bin/python
M   ?= 0

.PHONY: help setup check check-all ref train sample device data clean venv-ok \
        verses verses-verify eval eval-validate eval-score

help:
	@sed -n 's/^#   //p' $(MAKEFILE_LIST)

setup:
	./setup.sh

venv-ok:
	@test -x "$(PY)" || { \
		echo "No interpreter at $(PY)"; \
		echo "Run: make setup"; exit 1; }

check: venv-ok
	$(PY) checks.py --milestone $(M)

check-all: venv-ok
	$(PY) checks.py --milestone all

ref: venv-ok
	$(PY) checks.py --milestone $(M) --module reference/solution.py

train: venv-ok
	$(PY) gita_gpt.py

sample: venv-ok
	$(PY) gita_gpt.py --sample-only

device: venv-ok
	$(PY) device_check.py

data: venv-ok
	$(PY) prepare_data.py

# speakers.py must run after prepare_verses.py, which rewrites verses.json and
# would otherwise drop the speaker labels. Chained here so that cannot happen.
verses: venv-ok
	$(PY) prepare_verses.py --public-only
	$(PY) retrieval/speakers.py

speakers: venv-ok
	$(PY) retrieval/speakers.py

speakers-check: venv-ok
	$(PY) retrieval/speakers.py --check

verses-verify: venv-ok
	$(PY) prepare_verses.py --verify

eval-validate: venv-ok
	$(PY) eval/validate.py

eval-score: venv-ok
	$(PY) eval/score.py --show-misses

eval: eval-validate eval-score

V ?= 2:47
show: venv-ok
	$(PY) show.py $(V)

MODEL ?= BAAI/bge-m3
embed: venv-ok
	$(PY) retrieval/embed.py --model $(MODEL)

Q ?= i am stressed guide me?
ask: venv-ok
	@$(PY) retrieval/dense.py "$(Q)"

eval-dense: venv-ok
	$(PY) eval/score.py --retriever retrieval/dense.py --show-misses

eval-compare: venv-ok
	@echo "=========== BM25 baseline ==========="; $(PY) eval/score.py
	@echo; echo "=========== dense ==========="; $(PY) eval/score.py --retriever retrieval/dense.py

clean:
	rm -rf __pycache__ reference/__pycache__
