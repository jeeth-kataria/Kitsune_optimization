# Executor API

The executor handles task execution across CUDA streams.

---

## Overview

The `Executor` class is responsible for executing scheduled tasks across multiple CUDA streams. It manages:

- Task execution on assigned streams
- Stream synchronization
- Error handling
- Performance tracking

## Key Features

- **Multi-stream execution**: Execute tasks concurrently across streams
- **Automatic synchronization**: Handle stream dependencies
- **Error recovery**: Graceful handling of execution errors
- **Performance monitoring**: Track execution metrics

## Basic Usage

```python
from kitsune.core.executor import Executor
from kitsune.cuda.stream_pool import StreamPool

stream_pool = StreamPool(num_streams=4)
executor = Executor(stream_pool=stream_pool)

# Execute tasks
executor.execute(tasks)
```

## Async Execution

```python
from kitsune.core.executor import Executor

executor = Executor(stream_pool=stream_pool)

# Submit tasks for async execution
future = executor.submit_async(task)

# Wait for completion
result = future.result()
```

---

## See Also

- [Scheduler API](scheduler.md) - Task scheduling
- [Task API](task.md) - Task representation
