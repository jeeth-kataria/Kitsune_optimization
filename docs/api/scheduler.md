# Scheduler API

The scheduler is responsible for managing execution graph scheduling and stream parallelism.

---

## Overview

The `Scheduler` manages task execution order and stream assignment to maximize parallelism while respecting dependencies.

## Key Features

- **Dependency analysis**: Automatically detect task dependencies
- **Dynamic scheduling**: Adjust execution order based on runtime conditions
- **Stream assignment**: Intelligently assign tasks to CUDA streams
- **Load balancing**: Distribute work evenly across streams

## StreamPool

The `StreamPool` manages a pool of CUDA streams for concurrent execution.

---

## Usage Examples

### Custom Scheduler Configuration

```python
from kitsune.core.scheduler import Scheduler
from kitsune.cuda.stream_pool import StreamPool

# Create stream pool
stream_pool = StreamPool(num_streams=8)

# Create scheduler
scheduler = Scheduler(
    num_streams=8,
    stream_pool=stream_pool,
    dynamic_scheduling=True
)

# Schedule tasks
for task in tasks:
    scheduler.schedule(task)

# Execute
scheduler.execute()
```

### Stream Assignment

```python
from kitsune.core.scheduler import Scheduler

scheduler = Scheduler(num_streams=4)

# Manual stream assignment
scheduler.assign_stream(task_id=0, stream_id=1)
scheduler.assign_stream(task_id=1, stream_id=2)

# Or use automatic scheduling
scheduler.auto_assign_streams(tasks)
```

---

## See Also

- [Optimizer API](optimizer.md) - Main optimizer interface
- [Task API](task.md) - Task representation
- [Executor API](executor.md) - Task execution
