"""
Architecture Diagram Generator for Kitsune Documentation

Generates system architecture diagrams using Graphviz:
1. System overview (high-level flow)
2. Memory pool architecture
3. CUDA stream scheduling timeline

Requirements:
    pip install graphviz

Usage:
    python generate_diagrams.py
    
Output:
    docs/assets/architecture_system.png
    docs/assets/architecture_memory.png
    docs/assets/architecture_streams.png
"""

from graphviz import Digraph
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_system_overview():
    """Generate high-level system architecture diagram."""
    dot = Digraph(comment='Kitsune System Architecture', format='png')
    dot.attr(rankdir='TB', splines='ortho', bgcolor='white')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # User layer
    dot.node('user', 'PyTorch Training Script\n(model, optimizer, loss)', 
             fillcolor='#e3f2fd', color='#1976d2', fontcolor='#1976d2')
    
    # API layer
    dot.node('api', 'KitsuneOptimizer API\n(Drop-in Wrapper)', 
             fillcolor='#f3e5f5', color='#7b1fa2', fontcolor='#7b1fa2')
    
    # Core components layer
    with dot.subgraph(name='cluster_core') as c:
        c.attr(label='Core Components', style='dashed', color='#666')
        c.node('graph', 'Graph Analyzer\n• Capture ops\n• Build DAG\n• Find fusion', 
               fillcolor='#fff3e0', color='#f57c00', fontcolor='#333')
        c.node('scheduler', 'Dataflow Scheduler\n• Stream dispatch\n• Dependencies\n• Priority queue',
               fillcolor='#fff3e0', color='#f57c00', fontcolor='#333')
        c.node('memory', 'Memory Manager\n• Pool allocation\n• Size bins\n• Zero-copy',
               fillcolor='#fff3e0', color='#f57c00', fontcolor='#333')
    
    # Optimization layer
    dot.node('fusion', 'Fusion Engine\n(Triton Kernels)', 
             fillcolor='#e8f5e9', color='#388e3c', fontcolor='#388e3c')
    
    # Execution layer
    dot.node('cuda', 'CUDA Execution\n• Multi-stream\n• Event sync\n• Graph cache',
             fillcolor='#c8e6c9', color='#2e7d32', fontcolor='#2e7d32')
    
    # Connections
    dot.edge('user', 'api', label='wrap optimizer')
    dot.edge('api', 'graph')
    dot.edge('api', 'scheduler')
    dot.edge('api', 'memory')
    dot.edge('graph', 'fusion', label='fusion patterns')
    dot.edge('scheduler', 'fusion', label='schedule tasks')
    dot.edge('memory', 'fusion', label='allocate tensors')
    dot.edge('fusion', 'cuda', label='dispatch kernels')
    
    output_path = OUTPUT_DIR / "architecture_system"
    dot.render(output_path, cleanup=True)
    print(f"✓ Saved system architecture to {output_path}.png")


def generate_memory_architecture():
    """Generate memory pool architecture diagram."""
    dot = Digraph(comment='Kitsune Memory Pool', format='png')
    dot.attr(rankdir='LR', bgcolor='white')
    dot.attr('node', shape='record', style='filled', fontname='Arial')
    
    # Memory pool structure
    pool_label = '{Memory Pool|<512b>512B bin|<1kb>1KB bin|<4kb>4KB bin|<16kb>16KB bin|<64kb>64KB bin|<256kb>256KB bin}'
    dot.node('pool', pool_label, fillcolor='#bbdefb', color='#1976d2')
    
    # Size bins with allocation status
    bins = [
        ('512b', '512B Bin', [('In Use', 3), ('Free', 5)]),
        ('1kb', '1KB Bin', [('In Use', 8), ('Free', 4)]),
        ('4kb', '4KB Bin', [('In Use', 12), ('Free', 8)]),
        ('16kb', '16KB Bin', [('In Use', 6), ('Free', 10)]),
        ('64kb', '64KB Bin', [('In Use', 4), ('Free', 12)]),
        ('256kb', '256KB Bin', [('In Use', 2), ('Free', 14)])
    ]
    
    for bin_id, bin_name, allocations in bins:
        # Create allocation visualization
        alloc_parts = []
        for status, count in allocations:
            color = '#ef5350' if status == 'In Use' else '#66bb6a'
            alloc_parts.append(f'<{status.replace(" ", "_").lower()}>{status}: {count}')
        
        alloc_label = '{' + bin_name + '|{' + '|'.join(alloc_parts) + '}}'
        fillcolor = '#ffcdd2' if allocations[0][1] > allocations[1][1] else '#c8e6c9'
        dot.node(bin_id, alloc_label, fillcolor=fillcolor)
        dot.edge(f'pool:{bin_id}', bin_id, style='dashed')
    
    # Stats
    stats_label = '{Statistics|Total Allocated: 35 blocks|Total Free: 53 blocks|Reuse Rate: 94%|Zero-Copy Transfers: 1,247}'
    dot.node('stats', stats_label, fillcolor='#fff9c4', color='#f57f17', shape='record')
    
    output_path = OUTPUT_DIR / "architecture_memory"
    dot.render(output_path, cleanup=True)
    print(f"✓ Saved memory architecture to {output_path}.png")


def generate_stream_timeline():
    """Generate CUDA stream scheduling timeline diagram."""
    dot = Digraph(comment='CUDA Stream Timeline', format='png')
    dot.attr(rankdir='LR', bgcolor='white', ranksep='0.5')
    dot.attr('node', shape='box', style='filled', fontname='Arial', height='0.5')
    
    # Timeline header
    dot.node('timeline', 'Time →', shape='plaintext', fontsize='14', fontcolor='#666')
    
    # Stream 0: Forward Pass
    with dot.subgraph(name='cluster_s0') as c:
        c.attr(label='Stream 0: Forward Pass', style='filled', color='#e3f2fd', fontcolor='#1976d2')
        c.node('s0_t1', 'Conv1', fillcolor='#1976d2', fontcolor='white')
        c.node('s0_t2', 'Conv2', fillcolor='#1976d2', fontcolor='white')
        c.node('s0_t3', 'Conv3', fillcolor='#1976d2', fontcolor='white')
        c.edge('s0_t1', 's0_t2', style='invis')
        c.edge('s0_t2', 's0_t3', style='invis')
    
    # Stream 1: Gradient Compute
    with dot.subgraph(name='cluster_s1') as c:
        c.attr(label='Stream 1: Gradient Compute', style='filled', color='#f3e5f5', fontcolor='#7b1fa2')
        c.node('s1_t1', 'dConv3', fillcolor='#7b1fa2', fontcolor='white')
        c.node('s1_t2', 'dConv2', fillcolor='#7b1fa2', fontcolor='white')
        c.node('s1_t3', 'dConv1', fillcolor='#7b1fa2', fontcolor='white')
        c.edge('s1_t1', 's1_t2', style='invis')
        c.edge('s1_t2', 's1_t3', style='invis')
    
    # Stream 2: Memory Prefetch
    with dot.subgraph(name='cluster_s2') as c:
        c.attr(label='Stream 2: Memory Prefetch', style='filled', color='#fff3e0', fontcolor='#f57c00')
        c.node('s2_t1', 'Batch N+1', fillcolor='#f57c00', fontcolor='white')
        c.node('s2_t2', 'Batch N+2', fillcolor='#f57c00', fontcolor='white')
        c.edge('s2_t1', 's2_t2', style='invis')
    
    # Stream 3: Fusion Kernels
    with dot.subgraph(name='cluster_s3') as c:
        c.attr(label='Stream 3: Fusion Kernels', style='filled', color='#e8f5e9', fontcolor='#388e3c')
        c.node('s3_t1', 'BN+ReLU', fillcolor='#388e3c', fontcolor='white')
        c.node('s3_t2', 'BN+ReLU', fillcolor='#388e3c', fontcolor='white')
        c.edge('s3_t1', 's3_t2', style='invis')
    
    output_path = OUTPUT_DIR / "architecture_streams"
    dot.render(output_path, cleanup=True)
    print(f"✓ Saved stream timeline to {output_path}.png")


def generate_all_diagrams():
    """Generate all architecture diagrams."""
    print("🏗️  Generating Kitsune architecture diagrams...")
    print(f"📁 Output directory: {OUTPUT_DIR}\n")
    
    try:
        generate_system_overview()
        generate_memory_architecture()
        generate_stream_timeline()
        
        print("\n✅ All diagrams generated successfully!")
        print(f"\nGenerated files:")
        print(f"  • {OUTPUT_DIR / 'architecture_system.png'}")
        print(f"  • {OUTPUT_DIR / 'architecture_memory.png'}")
        print(f"  • {OUTPUT_DIR / 'architecture_streams.png'}")
        print("\n💡 Add these diagrams to your README and documentation!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure graphviz is installed:")
        print("   - pip install graphviz")
        print("   - System graphviz: brew install graphviz (macOS) or apt-get install graphviz (Linux)")


if __name__ == "__main__":
    generate_all_diagrams()
