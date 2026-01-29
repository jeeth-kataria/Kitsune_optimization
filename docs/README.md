# Kitsune Documentation

This directory contains the source files for Kitsune's documentation, built with [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install individually:

```bash
pip install mkdocs-material mkdocstrings[python] mkdocs-git-revision-date-localized-plugin
```

### Local Development

Serve the documentation locally with live reload:

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### Building Static Site

Build the static documentation site:

```bash
mkdocs build
```

The built site will be in the `site/` directory.

### Strict Mode

Build with strict mode to catch warnings:

```bash
mkdocs build --strict
```

## Documentation Structure

```
docs/
├── index.md                    # Homepage
├── getting-started/            # Getting started guides
│   ├── installation.md         # Installation instructions
│   └── quickstart.md          # 5-minute quick start
├── user-guide/                 # User guides
│   ├── overview.md            # Guide overview
│   ├── stream-parallelism.md  # Stream parallelism guide
│   ├── kernel-fusion.md       # Kernel fusion guide
│   ├── memory-management.md   # Memory management guide
│   ├── amp.md                 # Mixed precision guide
│   └── profiling.md           # Profiling guide
├── api/                        # API reference
│   ├── optimizer.md           # KitsuneOptimizer API
│   ├── scheduler.md           # Scheduler API
│   ├── executor.md            # Executor API
│   ├── graph.md               # Graph API
│   ├── task.md                # Task API
│   ├── amp.md                 # AMP API
│   ├── fusion.md              # Fusion API
│   ├── memory.md              # Memory API
│   └── profiler.md            # Profiler API
├── benchmarks/                 # Benchmark documentation
│   ├── results.md             # Performance results
│   └── methodology.md         # Benchmark methodology
├── contributing.md             # Contributing guidelines
├── code-of-conduct.md         # Code of conduct
├── changelog.md               # Changelog
├── stylesheets/               # Custom CSS
│   └── extra.css
└── javascripts/               # Custom JavaScript
    └── mathjax.js
```

## Writing Documentation

### Style Guide

- Use clear, concise language
- Include code examples
- Add cross-references with relative links
- Use admonitions for important notes (see below)

### Code Examples

Use fenced code blocks with language specification:

````markdown
```python
import kitsune
optimizer = kitsune.KitsuneOptimizer(base_optimizer)
```
````

### Admonitions

Use admonitions for special notes:

```markdown
!!! note "Optional Title"
    This is a note

!!! warning
    This is a warning

!!! tip
    This is a tip
```

Types: `note`, `abstract`, `info`, `tip`, `success`, `question`, `warning`, `failure`, `danger`, `bug`, `example`, `quote`

### API Documentation

API pages use mkdocstrings to auto-generate docs from docstrings:

```markdown
::: kitsune.KitsuneOptimizer
    options:
      show_root_heading: true
      show_source: true
```

### Links

Use relative paths for internal links:

```markdown
See the [Quick Start Guide](../getting-started/quickstart.md)
```

## Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

The deployment workflow is defined in [.github/workflows/docs.yml](../.github/workflows/docs.yml).

### Manual Deployment

To manually deploy:

```bash
mkdocs gh-deploy
```

## Configuration

Documentation configuration is in [mkdocs.yml](../mkdocs.yml).

Key sections:
- `theme`: Material theme configuration
- `plugins`: Enabled plugins (search, mkdocstrings, git-revision-date)
- `markdown_extensions`: Enabled Markdown extensions
- `nav`: Navigation structure

## Troubleshooting

### Build Errors

If you encounter build errors:

1. Check that all dependencies are installed
2. Verify all navigation links point to existing files
3. Run with `--strict` to see detailed errors
4. Check the [MkDocs documentation](https://www.mkdocs.org/)

### Missing API Docs

If API documentation isn't generating:

1. Ensure the package is installed (`pip install -e .`)
2. Check that docstrings exist in the source code
3. Verify the module path in the `:::` directive

### Broken Links

Run link checking:

```bash
mkdocs build
# Then use a link checker on the site/ directory
```

## Contributing

When contributing documentation:

1. Follow the existing structure and style
2. Test locally with `mkdocs serve`
3. Check for warnings with `mkdocs build --strict`
4. Update the navigation in `mkdocs.yml` if adding new pages
5. Submit a pull request

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Markdown Guide](https://www.markdownguide.org/)
