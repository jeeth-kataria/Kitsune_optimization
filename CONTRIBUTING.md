# Contributing to Kitsune

Thank you for your interest in contributing to Kitsune! 🦊

This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Welcome](#welcome)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Code of Conduct](#code-of-conduct)

---

## Welcome!

We're excited to have you here! Kitsune is an open-source project, and we welcome contributions of all kinds:

- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📚 Documentation improvements
- 🧪 Test coverage additions
- 💡 Ideas and suggestions

Whether you're fixing a typo or implementing a major feature, your contribution is valued!

---

## Development Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA Toolkit 11.0+ (for GPU features)
- Git

### Step 1: Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/Kitsune_optimization.git
cd Kitsune_optimization
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n kitsune python=3.11
conda activate kitsune
```

### Step 3: Install Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install with all optional dependencies
pip install -e ".[dev,triton]"
```

### Step 4: Verify Installation

```bash
# Run tests to verify everything works
pytest tests/unit -v

# Run a quick example
python examples/basic_usage.py
```

---

## Code Style

We maintain consistent code style across the project. Please follow these guidelines:

### Formatting & Linting

- **Black**: For code formatting (line length: 88)
- **Ruff**: For linting
- **isort**: For import sorting (via Ruff)

```bash
# Format code
black kitsune/ tests/

# Lint code
ruff check kitsune/ tests/

# Fix auto-fixable issues
ruff check --fix kitsune/ tests/
```

### Type Hints

All public functions and methods **must** include type hints:

```python
# ✅ Good
def process_tensor(data: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Process tensor with scaling."""
    return data * scale

# ❌ Bad
def process_tensor(data, scale=1.0):
    return data * scale
```

### Docstrings

Use **Google-style docstrings** for all public functions, classes, and modules:

```python
def schedule_task(task: Task, priority: int = 0) -> bool:
    """Schedule a task for execution on CUDA streams.

    This function adds the task to the scheduler's queue and assigns
    it to an appropriate CUDA stream based on dependencies.

    Args:
        task: The task to schedule. Must have a valid compute graph.
        priority: Optional priority level (higher = more urgent).
            Defaults to 0.

    Returns:
        True if the task was successfully scheduled, False otherwise.

    Raises:
        SchedulerError: If the task has unresolved dependencies.
        CUDAError: If no CUDA streams are available.

    Example:
        >>> task = Task(operation=matmul_op)
        >>> success = schedule_task(task, priority=10)
        >>> assert success
    """
    ...
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `MemoryPool`, `StreamScheduler` |
| Functions/Methods | snake_case | `allocate_tensor`, `get_stream` |
| Constants | UPPER_SNAKE_CASE | `MAX_STREAMS`, `DEFAULT_POOL_SIZE` |
| Private | Leading underscore | `_internal_buffer`, `_compute_hash` |

---

## Testing

### Running Tests

```bash
# Run all unit tests
pytest tests/unit -v

# Run specific test file
pytest tests/unit/test_memory_pool.py -v

# Run with coverage
pytest tests/unit --cov=kitsune --cov-report=html

# Run benchmarks (requires GPU)
python -m tests.benchmarks.baseline
```

### Writing Tests

1. **Location**: Add tests to `tests/unit/` directory
2. **Naming**: Test files should be named `test_<module>.py`
3. **Structure**: Use pytest with descriptive test names

```python
import pytest
import torch
from kitsune.memory import MemoryPool

class TestMemoryPool:
    """Tests for the MemoryPool class."""

    def test_allocate_returns_correct_size(self):
        """Verify allocation returns tensor of requested size."""
        pool = MemoryPool(max_size="1GB")
        tensor = pool.allocate(1024)
        assert tensor.numel() == 1024

    def test_allocate_raises_on_oom(self):
        """Verify OOM error when pool exhausted."""
        pool = MemoryPool(max_size="1KB")
        with pytest.raises(MemoryError):
            pool.allocate(1_000_000)

    @pytest.mark.cuda
    def test_cuda_allocation(self):
        """Test allocation on CUDA device."""
        pytest.importorskip("torch.cuda")
        pool = MemoryPool(device="cuda")
        tensor = pool.allocate(1024)
        assert tensor.is_cuda
```

### Coverage Requirements

- Aim for **90%+** code coverage
- All new features must include tests
- Bug fixes should include regression tests

---

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Format and lint
black kitsune/ tests/
ruff check kitsune/ tests/

# Run tests
pytest tests/unit -v

# Check types (optional but recommended)
mypy kitsune/
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "feat: add async memory prefetching for overlapped execution"
git commit -m "fix: resolve race condition in stream synchronization"
git commit -m "docs: update installation guide for Windows users"
git commit -m "test: add coverage for edge cases in fusion detector"

# Commit types:
# feat: new feature
# fix: bug fix
# docs: documentation only
# test: adding/updating tests
# refactor: code change that neither fixes nor adds
# perf: performance improvement
# chore: maintenance tasks
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Reference to related issues (e.g., "Fixes #42")
- Screenshots/benchmarks if applicable

### 6. PR Review

- Address reviewer feedback promptly
- Keep discussions focused and professional
- Squash commits if requested

---

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, PyTorch version, CUDA version, GPU model
2. **Description**: Clear explanation of the issue
3. **Steps to Reproduce**: Minimal code to reproduce
4. **Expected vs Actual**: What should happen vs what happens
5. **Error Message**: Full traceback if applicable

### Feature Requests

When suggesting features:

1. **Problem**: What problem does this solve?
2. **Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Willingness**: Would you like to implement it?

---

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). 

We are committed to providing a welcoming and inclusive environment for all contributors.

---

## Questions?

- 💬 [GitHub Discussions](https://github.com/jeeth-kataria/Kitsune_optimization/discussions) - Ask questions
- 🐛 [GitHub Issues](https://github.com/jeeth-kataria/Kitsune_optimization/issues) - Report bugs
- 📧 Contact maintainers for sensitive issues

---

Thank you for contributing to Kitsune! 🦊✨
