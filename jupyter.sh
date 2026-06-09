#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Use a dedicated port (8889) to avoid colliding with the auto-grading
# project's Jupyter server on 8888, which would otherwise force an
# auto-incremented port and break the MCP connection.
uv run jupyter lab --port 8889 --IdentityProvider.token xaichem-mcp-token --ip 0.0.0.0
