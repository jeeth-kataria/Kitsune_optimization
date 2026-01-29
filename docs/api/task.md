# Task API

The task module defines the representation of computation tasks in the execution graph.

---

## Overview

The `Task` class represents individual operations in the computation graph.

## Key Features

- **Operation representation**: Store operation type and parameters
- **Dependency tracking**: Manage task dependencies
- **Resource requirements**: Track memory and compute requirements
- **Execution state**: Monitor task execution status

---

## Usage Examples

### Creating Tasks

```python
from kitsune.core.task import Task

# Create a task
task = Task(
    op_type='matmul',
    inputs=[input_tensor],
    outputs=[output_tensor],
    metadata={'shape': (256, 256)}
)
```

### Task Dependencies

```python
from kitsune.core.task import Task

task1 = Task(op_type='linear', name='layer1')
task2 = Task(op_type='relu', name='activation')

# Add dependency
task2.add_dependency(task1)

# Check dependencies
if task2.has_dependency(task1):
    print("Task2 depends on Task1")
```

---

## See Also

- [Graph API](graph.md) - Graph representation
- [Scheduler API](scheduler.md) - Task scheduling
