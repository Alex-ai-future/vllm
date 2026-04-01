#!/bin/bash
# vLLM Docker entrypoint script with optional environment info collection

# If VLLM_LOG_ENV_ON_START is set to "1", collect environment info on startup
if [ "$VLLM_LOG_ENV_ON_START" = "1" ]; then
    echo "=== vLLM Environment Info ==="
    python -m vllm.collect_env 2>&1 || echo "Failed to collect environment info"
    echo "============================="
fi

# Execute the main command
exec "$@"
