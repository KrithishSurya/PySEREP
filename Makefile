# pyserep Makefile
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: install install-dev test test-unit test-integration test-fast \
        lint lint-fix format typecheck coverage \
        benchmark-pipeline benchmark-selectors \
        docs-build docs-serve docs-clean \
        build clean smoke help

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-fast:
	pytest tests/unit/ -q --tb=line -x

test-slow:
	pytest tests/ -v --tb=short -m slow

coverage:
	pytest tests/ \
	  --cov=pyserep \
	  --cov-report=html:htmlcov \
	  --cov-report=term-missing \
	  --cov-report=xml
	@echo "Coverage report → htmlcov/index.html"

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check pyserep/

lint-fix:
	ruff check pyserep/ --fix
	black pyserep/ tests/

format:
	black pyserep/ tests/ examples/ benchmarks/

typecheck:
	mypy pyserep/ --ignore-missing-imports

# ── Benchmarks ────────────────────────────────────────────────────────────────

benchmark-pipeline:
	python benchmarks/benchmark_pipeline.py --sizes 500 2000 5000

benchmark-selectors:
	python benchmarks/benchmark_dof_selectors.py --sizes 200 500 1000 2000

# ── Documentation ─────────────────────────────────────────────────────────────

docs-build:
	sphinx-build -b html docs/source docs/build/html
	@echo "Docs → docs/build/html/index.html"

docs-serve: docs-build
	cd docs/build/html && python -m http.server 8080

docs-clean:
	rm -rf docs/build/

docs-check:
	sphinx-build -b html docs/source docs/build/html -W --keep-going

# ── Packaging ─────────────────────────────────────────────────────────────────

build:
	python -m build
	twine check dist/*
	@echo "Built packages in dist/"

# ── Quick checks ──────────────────────────────────────────────────────────────

smoke:
	@python -c "\
import pyserep; \
from pyserep.models.synthetic import spring_chain; \
from pyserep.core.eigensolver import solve_eigenproblem; \
K, M = spring_chain(100); \
freqs, phi = solve_eigenproblem(K, M, n_modes=15, verbose=False); \
print(f'pyserep {pyserep.__version__} — smoke test PASSED'); \
print(f'  {K.shape[0]} DOFs  |  {len(freqs)} modes  |  f_max={freqs[-1]:.2f} Hz')"

check: lint typecheck test-fast
	@echo "✓ lint + typecheck + fast tests all passed"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"      -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist"         -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
	@echo "Clean"

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  pyserep development commands"
	@echo ""
	@echo "  INSTALL"
	@echo "    make install          pip install -e ."
	@echo "    make install-dev      pip install -e '.[dev]'"
	@echo ""
	@echo "  TESTING"
	@echo "    make test             All tests (179 unit + integration)"
	@echo "    make test-unit        Unit tests only"
	@echo "    make test-integration Integration tests only"
	@echo "    make test-fast        Quick unit tests (-x on first failure)"
	@echo "    make coverage         Full coverage report → htmlcov/"
	@echo ""
	@echo "  CODE QUALITY"
	@echo "    make lint             ruff check"
	@echo "    make lint-fix         ruff --fix + black"
	@echo "    make format           black formatter"
	@echo "    make typecheck        mypy type checking"
	@echo "    make check            lint + typecheck + fast tests"
	@echo ""
	@echo "  BENCHMARKS"
	@echo "    make benchmark-pipeline    End-to-end pipeline timing"
	@echo "    make benchmark-selectors   DS1–DS4 condition number comparison"
	@echo ""
	@echo "  DOCUMENTATION"
	@echo "    make docs-build       Build Sphinx HTML docs"
	@echo "    make docs-serve       Build + serve at localhost:8080"
	@echo "    make docs-check       Build with -W (warnings as errors)"
	@echo ""
	@echo "  PACKAGING"
	@echo "    make build            Build wheel + sdist, run twine check"
	@echo "    make smoke            Quick import + model test"
	@echo "    make clean            Remove all build artefacts"
	@echo ""

.DEFAULT_GOAL := help
