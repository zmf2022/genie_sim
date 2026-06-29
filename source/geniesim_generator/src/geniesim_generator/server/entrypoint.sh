#!/bin/bash
set -e

# The asset library (geniesim_assets) is pip-installed on the host; the host's
# package directory is mounted read-only into /opt/geniesim_assets by
# compose.yaml. Adding /opt to PYTHONPATH makes `import geniesim_assets`
# resolve to that mount — no in-container install of the multi-GB library.
#
# The generator distribution still lives on a mount (so live source edits take
# effect without a rebuild) and gets editable-installed at startup.
GEN_DIR="${GENIESIM_GENERATOR_DIR:-/opt/geniesim_generator}"
ASSETS_MOUNT="/opt/geniesim_assets"

export PYTHONPATH="/opt${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== geniesim generator MCP server bootstrap ==="
echo "Generator distribution: ${GEN_DIR}"
echo "Assets mount:           ${ASSETS_MOUNT}"

if [ ! -f "${ASSETS_MOUNT}/__init__.py" ]; then
    echo "WARN: ${ASSETS_MOUNT}/__init__.py missing — set GENIESIM_ASSETS_DIR on the host" >&2
    echo "      to the dir of the pip-installed geniesim_assets package:" >&2
    echo '      export GENIESIM_ASSETS_DIR=$(python -c "import geniesim_assets, os; print(os.path.dirname(geniesim_assets.__file__))")' >&2
fi

if [ -f "${GEN_DIR}/pyproject.toml" ]; then
    pip install -e "${GEN_DIR}" --no-deps -q || echo "WARN: geniesim_generator editable install failed"
else
    echo "WARN: no geniesim_generator package at ${GEN_DIR}"
fi

python -c "import geniesim_assets, geniesim_generator; print('import check OK')" \
    || echo "WARN: package import check failed — server may not start"
echo "================"

exec "$@"
