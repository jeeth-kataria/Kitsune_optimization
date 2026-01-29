# Contributing to Kitsune

Thank you for your interest in contributing to Kitsune! This document provides guidelines and instructions for contributing.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](code-of-conduct.md).

---

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- Clear descriptive title
- Detailed steps to reproduce
- Expected vs actual behavior
- Code samples (minimal reproducible example)
- Environment details (Python version, PyTorch version, CUDA version, GPU model)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:

- Clear use case description
- Why this enhancement would be useful
- Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Update documentation
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

---

## Development Setup

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- CUDA ≥ 11.8 (for GPU features)

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/kitsune.git
cd kitsune

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_optimizer.py

# Run with coverage
pytest --cov=kitsune tests/
```

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **isort** for import sorting

Run all checks:
```bash
# Format code
black kitsune/ tests/

# Check linting
flake8 kitsune/ tests/

# Type checking
mypy kitsune/

# Sort imports
isort kitsune/ tests/
```

Or use pre-commit:
```bash
pre-commit run --all-files
```

### Building Documentation

```bash
# Install docs dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

---

## Project Structure

```
kitsune/
├── kitsune/           # Main package
│   ├── core/          # Core execution engine
│   ├── cuda/          # CUDA utilities
│   ├── amp/           # Mixed precision
│   ├── fusion/        # Kernel fusion
│   ├── memory/        # Memory management
│   └── profiler/      # Performance profiling
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   └── benchmarks/    # Performance benchmarks
├── docs/              # Documentation
└── examples/          # Usage examples
```

---

## Coding Guidelines

### Python Style

- Follow PEP 8
- Use type hints for all functions
- Write docstrings in Google style
- Keep functions focused and small

Example:
```python
def process_data(
    input_tensor: torch.Tensor,
    batch_size: int,
    device: str = "cuda"
) -> torch.Tensor:
    """Process input tensor in batches.
    
    Args:
        input_tensor: Input data to process
        batch_size: Number of samples per batch
        device: Target device for computation
        
    Returns:
        Processed tensor
        
    Raises:
        ValueError: If batch_size <= 0
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    return process_impl(input_tensor, batch_size, device)
```

### Testing

- Write tests for all new features
- Aim for >90% code coverage
- Use pytest fixtures for common setup
- Mock CUDA operations when necessary

Example:
```python
import pytest
import torch
from kitsune import KitsuneOptimizer

@pytest.fixture
def simple_model():
    return torch.nn.Linear(10, 5).cuda()

def test_optimizer_step(simple_model):
    optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
    kitsune_opt = KitsuneOptimizer(optimizer)
    
    # Test logic here
    assert kitsune_opt is not None
```

---

## Git Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Follow conventional commits format:

```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(fusion): add support for reduction fusion

Implement automatic detection and fusion of reduction operations
including mean, sum, and softmax.

Closes #123
```

---

## Review Process

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] No merge conflicts
- [ ] CI checks pass

### Review Criteria

Reviewers will check:
- Code quality and clarity
- Test coverage
- Documentation completeness
- Performance impact
- Backward compatibility

---

## Release Process

1. Update version in `kitsune/__init__.py`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Publish to PyPI

---

## Getting Help

- 💬 [GitHub Discussions](https://github.com/yourusername/kitsune/discussions)
- 📧 Email maintainers: dev@kitsune-ml.org

---

## Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Documentation

Thank you for contributing to Kitsune! 🦊
