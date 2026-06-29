Choose one of the following two options.

Prerequisite (both profiles): geniesim_assets must be pip-installed on the host.
The compose stack mounts the host's installed copy into the container — point
GENIESIM_ASSETS_DIR at its directory before bringing the stack up:

    export GENIESIM_ASSETS_DIR=$(python -c \
        "import geniesim_assets, os; print(os.path.dirname(geniesim_assets.__file__))")

Run docker compose from this directory (where compose.yaml lives):
    source/geniesim_generator/src/geniesim_generator/

    1. qwen embedding API:
        - Configure server/mcp_text_embedding/text_embedding_config.json (api_key, base_url, model)
        - docker compose --profile text up --build

    2. qwen embedding model (requires GPU and Nvidia Container Toolkit):
        - cd server/assets_searcher
        - bash download_model.sh  # Downloads model and scripts; see download_model.sh for details
        - cd ../..                # back to the compose.yaml dir

        - Adjust BATCH_SIZE and USE_RERANKER under services/mcp-server_vl/environment in compose.yaml for your hardware:
            - BATCH_SIZE: Batch size when building the vector index; larger uses more VRAM, smaller takes longer. Default 10 (e.g. RTX 4090).
            - USE_RERANKER: Whether to use the reranker (True/False). Reranking improves search quality but increases latency and VRAM use.
        - docker compose --profile vl up --build
