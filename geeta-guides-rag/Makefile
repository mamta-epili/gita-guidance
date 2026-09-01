# Python targets use .venv/bin/python by absolute path, so shell aliases,
# pyenv/conda shims and IDE interpreter selection cannot interfere.
#
# Two terminals for development:
#   make backend         FastAPI on :8000   (serves the API + no-build fallback UI)
#   make frontend        Angular on :4200   (proxies /api to :8000)
#
# Other targets:
#   make setup           backend .venv + torch/fastapi/uvicorn
#   make install         frontend dependencies (pnpm)
#   make build           production Angular build into frontend/dist
#   make serve           backend only, serving the built Angular bundle
#   make check           load the checkpoint and report, no server
#   make demo            rebuild docs/demo.html from real retrieval results
#   make safety          run the crisis-triage test suite
#   make clean           remove __pycache__ and dist

PY   := $(CURDIR)/.venv/bin/python
PORT ?= 8000
PM   ?= pnpm

.PHONY: help setup install backend frontend build serve check clean venv-ok fe-ok \
        demo safety pairs try

help:
	@sed -n 's/^#   \{0,1\}//p;s/^# \{0,1\}//p' $(MAKEFILE_LIST) | sed -n '3,20p'

setup:
	./setup.sh

venv-ok:
	@test -x "$(PY)" || { echo "No interpreter at $(PY)"; echo "Run: make setup"; exit 1; }

fe-ok:
	@test -d frontend/node_modules || { \
		echo "frontend dependencies missing"; echo "Run: make install"; exit 1; }

install:
	cd frontend && $(PM) install

backend: venv-ok
	@echo "→ API on http://127.0.0.1:$(PORT)   (fallback UI at the same address)"
	$(PY) -m uvicorn app.main:app --reload --port $(PORT)

frontend: fe-ok
	@echo "→ Angular on http://127.0.0.1:4200   (needs 'make backend' in another terminal)"
	cd frontend && $(PM) start

build: fe-ok
	cd frontend && $(PM) run build

serve: venv-ok
	@test -d frontend/dist || { echo "No build yet. Run: make build"; exit 1; }
	SERVE_ANGULAR=1 $(PY) -m uvicorn app.main:app --port $(PORT)

check: venv-ok
	$(PY) -c "from app.model import CharGPT; import json; \
	g = CharGPT(); i = g.info(); i.pop('vocab'); print(json.dumps(i, indent=2))"

# Records real answers into a standalone page. public_only=True, so the Hindi
# (copyrighted) never reaches it — the script aborts if any leaks through.
demo: venv-ok
	$(PY) scripts/build_demo.py

clean:
	rm -rf app/__pycache__ __pycache__ frontend/dist frontend/.angular

# Safety code does not ship without its tests. 38 cases: self-harm, abuse,
# acute distress, and — just as important — questions that must NOT trigger.
safety: venv-ok
	$(PY) -m app.safety --test

# Every "Arjuna asks / Krishna answers" caption the app would print, listed so
# it can be checked against the text. Needs no embedding index.
#   make pairs              the shipped threshold
#   make pairs W=2          what a wider one would claim
#   GITA_PAIR_WITHIN=0 ...  turn the pairing off entirely
pairs: venv-ok
	$(PY) scripts/pair_report.py $(if $(W),--within $(W),)

# Try a question against the real retriever without rebuilding the demo page.
#   make try Q="Why do I keep doing the thing I know is wrong?"
try: venv-ok
	@test -n "$(Q)" || { echo 'usage: make try Q="your question"'; exit 2; }
	$(PY) scripts/try_question.py --public-only "$(Q)"
