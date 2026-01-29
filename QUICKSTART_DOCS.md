# 🚀 Quick Reference - Running Kitsune Documentation

## TL;DR

```bash
./run_docs.sh
```

Open: **http://127.0.0.1:8000**

---

## All Commands

```bash
# Run documentation server (recommended)
./run_docs.sh

# Or use MkDocs directly
mkdocs serve

# Or use helper script
./scripts/docs.sh serve
```

---

## What You Get

✅ Live documentation site at http://127.0.0.1:8000
✅ Auto-refresh when you edit files
✅ Full navigation with search
✅ Code examples with copy buttons
✅ Dark/light theme toggle

---

## Common Tasks

| Task | Command |
|------|---------|
| **View docs** | `./run_docs.sh` |
| **Build static site** | `mkdocs build` |
| **Check for errors** | `mkdocs build --strict` |
| **Deploy to GitHub** | `mkdocs gh-deploy` |
| **Clean build** | `rm -rf site/` |

---

## First Time?

If you get "mkdocs not found":

```bash
pip install -e ".[docs]"
```

---

## Stop Server

Press `Ctrl+C` in the terminal

---

## Need Help?

See full guide: [RUNNING_DOCS.md](RUNNING_DOCS.md)
