# Contributing to Project 512D

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/euphoricdoom/project-512d.git
cd project-512d
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Code Quality Standards

- **Type hints**: All public functions/methods must have complete type hints.
- **Docstrings**: Google style for all public APIs; describe *why*, not just *what*.
- **Tests**: New features must include tests. Prefer testing invariants over values.
- **Style**: Follow PEP 8. No magic numbers — use `config/constants.py`.
- **Comments**: Only add a comment when the WHY is non-obvious.

## Architecture Constraints

- The weightless learning layer (`weightless_model.py`, `zero_forgetting.py`,
  `hf_adapter.py`, `int8_matmul.py`) depends **only on numpy + scikit-learn +
  datasets**. Do not add PyTorch or scipy imports to these files.
- The kernel research layer (`core/`, `system/`) may use scipy, networkx, matplotlib.
- PyTorch only in `system/pytorch_helix.py`.
- All seeds default to 42 for reproducibility.

## Running Tests

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_phase_encoding.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Adding New Configuration

If your change introduces new hyperparameters:
1. Add named constants to `config/constants.py` with a comment explaining their origin.
2. Add a typed field to the appropriate dataclass in `config/model_config.py`.
3. Add validation in `__post_init__` with an actionable error message.
4. Add a test in `tests/test_config_system.py`.

## Experiment Workflow

1. **Configure** — define experiment in `config/experiments/`
2. **Implement** — create runner in `experiments/`
3. **Validate** — add tests for new components
4. **Analyze** — use `analysis/` tools
5. **Document** — update `docs/RESEARCH_LEDGER.md`

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run the full test suite: `pytest tests/ -v`
5. Commit with a clear message
6. Open a pull request against `main`

## Questions?

Open an issue or email euphoricdoom@gmail.com
