# Graph API

The graph module handles computation graph representation and analysis.

---

## Overview

The graph module provides data structures for representing and analyzing computation graphs:

- **ComputeGraph**: Main graph structure
- **GraphNode**: Individual operation nodes

## Key Features

- **Graph construction**: Build computation graphs from operations
- **Dependency analysis**: Identify data dependencies
- **Graph optimization**: Apply graph-level optimizations
- **Visualization**: Export graphs for visualization

---

## Usage Examples

### Graph Construction

```python
from kitsune.core.graph import ComputeGraph, GraphNode

# Create graph
graph = ComputeGraph()

# Add nodes
node1 = GraphNode(op_type='linear', name='layer1')
node2 = GraphNode(op_type='relu', name='activation1')

graph.add_node(node1)
graph.add_node(node2)

# Add edge
graph.add_edge(node1, node2)
```

### Graph Analysis

```python
from kitsune.core.graph import ComputeGraph

graph = ComputeGraph()
# ... build graph ...

# Analyze dependencies
dependencies = graph.get_dependencies(node)

# Find independent nodes
independent = graph.find_independent_nodes()

# Topological sort
sorted_nodes = graph.topological_sort()
```

### Graph Visualization

```python
from kitsune.core.graph import ComputeGraph

graph = ComputeGraph()
# ... build graph ...

# Export to DOT format
graph.export_dot('model_graph.dot')

# Or visualize directly
graph.visualize(output='graph.png')
```

---

## See Also

- [Task API](task.md) - Task representation
- [PyTorch Graph Capture](../user-guide/graph-capture.md) - Graph capture from PyTorch
