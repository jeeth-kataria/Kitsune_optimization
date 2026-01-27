# Visual Assets Generation Guide

This directory contains scripts to generate professional visual assets for Kitsune documentation.

## 📊 Performance Charts

Generate publication-quality performance charts:

```bash
# Install dependencies
pip install matplotlib numpy

# Generate all charts
python scripts/generate_charts.py
```

**Output:**
- `docs/assets/speedup_comparison.png` - Bar chart showing 2-2.2x speedup
- `docs/assets/optimization_breakdown.png` - Horizontal bars showing optimization impact
- `docs/assets/memory_savings.png` - Memory efficiency comparison

## 🏗️ Architecture Diagrams

Generate system architecture diagrams:

```bash
# Install dependencies
pip install graphviz
brew install graphviz  # macOS
# OR
apt-get install graphviz  # Linux

# Generate all diagrams
python scripts/generate_diagrams.py
```

**Output:**
- `docs/assets/architecture_system.png` - High-level system flow
- `docs/assets/architecture_memory.png` - Memory pool structure
- `docs/assets/architecture_streams.png` - CUDA stream timeline

## 🎬 Demo GIF

Record and generate a terminal demo:

### Option 1: Using asciinema + agg (Recommended)

```bash
# Install tools
brew install asciinema  # macOS
pip install asciinema-agg

# Record demo
cd KITSUNE_ALGO
asciinema rec demo.cast -c "bash scripts/demo_script.sh"

# Convert to GIF
agg demo.cast docs/assets/demo.gif

# Or upload to asciinema.org
asciinema upload demo.cast
```

### Option 2: Using Kap (macOS GUI)

1. Install Kap: `brew install --cask kap`
2. Open Kap and start recording
3. Run: `bash scripts/demo_script.sh`
4. Stop recording and export as GIF
5. Save to `docs/assets/demo.gif`

### Option 3: Using Terminalizer

```bash
# Install
npm install -g terminalizer

# Record
terminalizer record demo -c "bash scripts/demo_script.sh"

# Render
terminalizer render demo -o docs/assets/demo.gif
```

## 📁 File Structure

After generation, your assets directory should look like:

```
docs/
└── assets/
    ├── speedup_comparison.png
    ├── optimization_breakdown.png
    ├── memory_savings.png
    ├── architecture_system.png
    ├── architecture_memory.png
    ├── architecture_streams.png
    └── demo.gif
```

## 🎨 Customization

### Charts
Edit `scripts/generate_charts.py`:
- Colors: Modify `colors` lists
- Sizes: Adjust `figsize` parameters
- Data: Update benchmark values
- Style: Change `plt.style.use()`

### Diagrams
Edit `scripts/generate_diagrams.py`:
- Colors: Modify `fillcolor` attributes
- Layout: Change `rankdir` ('TB', 'LR', 'BT', 'RL')
- Nodes: Add/remove components
- Labels: Update text content

### Demo
Edit `scripts/demo_script.sh`:
- Timing: Adjust `sleep` durations
- Content: Modify benchmark code
- Colors: Change ANSI color codes

## 💡 Tips

1. **High Resolution**: Charts are generated at 300 DPI for print quality
2. **Transparency**: Use PNG format to preserve transparency
3. **File Size**: Optimize GIFs with `gifsicle`:
   ```bash
   brew install gifsicle
   gifsicle -O3 --colors 256 -o demo_optimized.gif demo.gif
   ```
4. **Accessibility**: Add alt text when embedding in README
5. **Dark Mode**: Generate separate dark-themed versions if needed

## ✅ Quality Checklist

Before committing assets:

- [ ] Charts are readable at small sizes (GitHub preview)
- [ ] Text is legible (no pixelation)
- [ ] Colors are accessible (sufficient contrast)
- [ ] File sizes are reasonable (<500KB per image)
- [ ] All assets are in `docs/assets/`
- [ ] Assets are referenced in README.md

## 🔗 Integration

Add to README.md:

```markdown
## Performance

![Speedup Comparison](docs/assets/speedup_comparison.png)

## Architecture

![System Architecture](docs/assets/architecture_system.png)

## Demo

![Kitsune Demo](docs/assets/demo.gif)
```
