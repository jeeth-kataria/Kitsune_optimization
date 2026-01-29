# Running the Kitsune Documentation

## Quick Start

### Option 1: Simple Run Script (Recommended)

Just run:

```bash
./run_docs.sh
```

This will:
- Install dependencies if needed
- Start the documentation server
- Open at http://127.0.0.1:8000
- Enable live reload (auto-refresh on file changes)

### Option 2: Direct MkDocs Command

```bash
mkdocs serve
```

### Option 3: Using the Helper Script

```bash
./scripts/docs.sh serve
```

---

## What You'll See

When you run the documentation server, you'll see:

```
========================================
  Kitsune Documentation Server
========================================

Starting documentation server...
📚 Documentation will be available at: http://127.0.0.1:8000
⚡ Live reload enabled - changes will update automatically

Press Ctrl+C to stop the server

========================================

INFO     -  Building documentation...
INFO     -  [11:30:45] Watching paths for changes: 'docs', 'mkdocs.yml'
INFO     -  [11:30:45] Serving on http://127.0.0.1:8000/
```

Then open your browser to **http://127.0.0.1:8000**

---

## Other Useful Commands

### Build Static Site

Generate static HTML files (output in `site/` directory):

```bash
# Using run script
./scripts/docs.sh build

# Or directly
mkdocs build
```

### Build with Error Checking

Build with strict mode to catch warnings:

```bash
./scripts/docs.sh check
```

### Deploy to GitHub Pages

```bash
./scripts/docs.sh deploy
```

### Clean Build Artifacts

```bash
./scripts/docs.sh clean
```

---

## First Time Setup

If MkDocs isn't installed:

```bash
# Install all documentation dependencies
pip install -e ".[docs]"

# Or install manually
pip install mkdocs-material mkdocstrings[python] mkdocs-git-revision-date-localized-plugin
```

---

## Tips

- **Live Reload**: The server automatically refreshes your browser when you edit documentation files
- **Port in Use?**: If port 8000 is busy, use: `mkdocs serve -a localhost:8001`
- **Public Access**: For network access: `mkdocs serve -a 0.0.0.0:8000`

---

## Troubleshooting

**Problem**: `mkdocs: command not found`

**Solution**: Install dependencies:
```bash
pip install -e ".[docs]"
```

**Problem**: Port already in use

**Solution**: Use a different port:
```bash
mkdocs serve -a localhost:8001
```

**Problem**: Changes not updating

**Solution**: 
- Hard refresh browser (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)
- Restart the server
- Clear browser cache

---

## What's Included

The documentation includes:

- 🏠 **Homepage**: Features, benchmarks, quick start
- 📖 **Getting Started**: Installation & 5-minute tutorial
- 📚 **User Guide**: Stream parallelism, fusion, memory, AMP
- 🔧 **API Reference**: Complete API documentation (9 modules)
- 📊 **Benchmarks**: Performance results & methodology
- 🤝 **Contributing**: Development guidelines

---

## Happy Documenting! 📚
