#!/bin/bash
# Production PyPI Publish Script - Official Release for torch-kitsune
# ⚠️ WARNING: This uploads to the REAL PyPI - versions cannot be deleted!

set -e  # Exit on any error

echo "🚨 PRODUCTION RELEASE - Final Checks"
echo "===================================="
echo ""
echo "Before proceeding, confirm:"
echo "  ✓ TestPyPI upload succeeded"
echo "  ✓ Test installation works (pip install from TestPyPI)"
echo "  ✓ Import works (import kitsune; print(kitsune.__version__))"
echo "  ✓ Version number is correct in pyproject.toml"
echo ""
read -p "Continue with production release? (y/N): " confirm

if [[ $confirm != [yY] ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "🧹 Step 1: Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info kitsune.egg-info

echo ""
echo "🔨 Step 2: Building distribution packages..."
python -m build

echo ""
echo "✅ Step 3: Validating package with twine..."
twine check dist/*

echo ""
echo "🚀 Step 4: Uploading to PyPI..."
echo "--------------------------------------------"
echo "When prompted:"
echo "  Username: __token__"
echo "  Password: <your PyPI API token>"
echo ""
read -p "Press Enter to upload to PyPI now, or Ctrl+C to exit..."

twine upload dist/*

echo ""
echo "🎉 SUCCESS! Package published to PyPI"
echo "===================================="
echo ""
echo "📦 Package URL: https://pypi.org/project/torch-kitsune/"
echo ""
echo "Users can now install with:"
echo "  pip install torch-kitsune"
echo ""
echo "📌 Next Steps:"
echo "  1. Tag this release in git:"
echo "     git tag v0.1.0"
echo "     git push origin v0.1.0"
echo ""
echo "  2. Create a GitHub Release:"
echo "     https://github.com/jeeth-kataria/Kitsune_optimization/releases/new"
echo ""
echo "  3. Announce on social media/forums!"
echo ""
