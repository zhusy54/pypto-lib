#!/usr/bin/env bash
# Run a pypto-lib example with all required environment variables configured.
# Usage:
#   ./run_example.sh examples/beginner/hello_world.py
#   ./run_example.sh examples/beginner/hello_world.py -p a2a3sim
#   ./run_example.sh -l                              # list available examples

# ── Environment variables ────────────────────────────────────────────────────
export TORCH_DEVICE_BACKEND_AUTOLOAD=0  # disable torch_npu auto-loading (version mismatch)
export PATH=/usr/local/bin/ptoas-bin:$PATH
export PTOAS_ROOT=/usr/local/bin/ptoas-bin
export PTO_ISA_ROOT="${PTO_ISA_ROOT:-/data/z00934994/zhusy54/pypto-lib-deps/pto-isa}"
export SIMPLER_ROOT=/data/zhusy54/workspace/pypto-lib-deps/pypto/runtime
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}"

# export SIMPLER_ROOT="${SIMPLER_ROOT:-$WORKSPACE_DIR/simpler-zhusy54}"
# export PTOAS_ROOT="${PTOAS_ROOT:-$WORKSPACE_DIR/ptoas-bin}"

# ── Run ──────────────────────────────────────────────────────────────────────
echo "=== Environment ==="
echo "  PTOAS_ROOT=$PTOAS_ROOT"
echo "  PTO_ISA_ROOT=$PTO_ISA_ROOT"
echo "  SIMPLER_ROOT=$SIMPLER_ROOT"
echo "  ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "=== Running: $EXAMPLE $@ ==="
exec python3 "$@"
