#!/bin/bash
# TestPyPI Publish Script - Dress Rehearsal for torch-kitsune
# This script builds, validates, and uploads to TestPyPI for safe testing

set -e  # Exit on any error

echo "🧹 Step 1: Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info kitsune.egg-info

echo ""
echo "🔨 Step 2: Building distribution packages..."
python -m build

echo ""
echo "✅ Step 3: Validating package with twine..."
twine check dist/*

echo ""
echo "📦 Step 4: Ready to upload to TestPyPI"
echo "--------------------------------------------"
echo "Run the following command to upload:"
echo ""
echo "  twine upload --repository testpypi dist/*"
echo ""
echo "When prompted:"
echo "  Username: __token__"
echo "  Password: <your TestPyPI API token>"
echo ""
echo "--------------------------------------------"
echo ""
echo "🧪 Step 5: After upload, test installation with:"
echo ""
echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ torch-kitsune"
echo ""
echo "  python -c \"import kitsune; print(kitsune.__version__)\""
echo ""
echo "Expected output: 0.1.0"
echo ""
echo "--------------------------------------------"
echo ""
read -p "Press Enter to upload to TestPyPI now, or Ctrl+C to exit..."

twine upload --repository testpypi dist/*

echo ""
echo "✨ Upload complete! Now test the installation as shown above."
