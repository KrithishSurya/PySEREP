# Contributing to pyserep

Thank you for your interest in contributing!

## Setup

```bash
git clone https://github.com/YourOrg/pyserep.git
cd pyserep
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/unit/          # unit tests only
pytest tests/integration/   # integration tests only
```

## Code Style

```bash
ruff check pyserep/        # linting
black pyserep/ tests/      # formatting
mypy pyserep/              # type checking
```

## Adding a New DOF Selector

1. Add a function `select_dofs_<name>(phi, selected_modes, ...)` to `pyserep/selection/dof_selector.py`
2. Register it in the `_DOF_SELECTOR_MAP` in `pyserep/pipeline/serep_pipeline.py`
3. Add it to `compare_dof_selectors` in `dof_selector.py`
4. Export it from `pyserep/__init__.py`
5. Add unit tests in `tests/unit/test_dof_selector.py`

## Adding a New Damping Model

1. Add a `_<name>_ca(Ka, Ma, ...)` builder to `pyserep/frf/direct_frf.py`
2. Register it in the `if/elif` block in `compute_frf_direct`
3. Document it in the module docstring table
4. Add a test case to `tests/unit/test_direct_frf.py`

## Pull Request Checklist

- [ ] New code has docstrings (NumPy style)
- [ ] Unit tests added / updated
- [ ] `ruff` and `black` pass
- [ ] `pytest` passes on Python 3.9+
