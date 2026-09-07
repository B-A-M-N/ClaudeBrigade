"""Public bounded structured-call executor.

Native sidecar agents do not use this module; they are launched through
Claude Code's Agent tool and the claimed native lifecycle path.
"""

from enhanced_router.sidecar_executor import (
    CoprocessorExecutor,
    get_coprocessor_executor,
    shutdown_coprocessor_executor,
)

__all__ = [
    "CoprocessorExecutor",
    "get_coprocessor_executor",
    "shutdown_coprocessor_executor",
]
