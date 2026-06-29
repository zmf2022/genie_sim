# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import json
import sys
import logging
from datetime import datetime
import traceback
import time
from typing import List, Dict, Union, Optional
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(SOURCE_DIR)

from geniesim_generator.server.assets_searcher.embeddings.text_embedding import DashscopeTextEmbeddings, TextEmbeddings
from geniesim_assets import (
    ASSETS_INDEX,
    ASSETS_INDEX_HASH,
)

CHROMA_DB_CACHE_PATH = os.path.join("/tmp", "chromadb_cache")

import shutil
import re

# Configure logging to use stderr to avoid interfering with JSON-RPC on stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class _EmbeddingFunctionAdapter(EmbeddingFunction):
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return self._embeddings.embed_documents(list(input))


class AssetVectorDB:
    """Asset vector database management class (based on ChromaDB)"""

    def __init__(
        self,
        config: Dict,
        db_path: str = CHROMA_DB_CACHE_PATH,
        auto_sync: bool = True,
        force_rebuild: bool = False,
    ):
        self.config = config
        self.embedding_model = config.get("model", "text-embedding-v4")
        self.api_key = config.get("api_key", None)
        self.base_url = config.get("base_url", None)
        self.dimension = config.get("dimension", 2048)
        self.dashscope_mode = config.get("dashscope_mode", True)

        if db_path is None:
            db_path = CHROMA_DB_CACHE_PATH
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
        if not os.path.exists(db_path) or not os.path.exists(f"{db_path}/assets_sync_state.json") and not force_rebuild:
            if os.path.exists(db_path):
                shutil.rmtree(db_path, ignore_errors=True)
            logger.info(f"Creating ChromaDB cache directory: {db_path}")
            os.makedirs(db_path, exist_ok=True, mode=0o777)
        else:
            logger.info(f"ChromaDB cache directory already exists: {db_path}")

        if not os.access(db_path, os.W_OK):
            os.chmod(db_path, 0o777)
            logger.info(f"Changed ChromaDB cache directory to writable: {db_path}")

        self.assets_index = ASSETS_INDEX
        self.db_path = db_path
        self.chroma_persist_directory = db_path
        self.sync_state_path = f"{db_path}/assets_sync_state.json"
        if self.dashscope_mode:
            self.embeddings = DashscopeTextEmbeddings(
                model=self.embedding_model,
                api_key=self.api_key,
                dimension=self.dimension,
            )
        else:
            self.embeddings = TextEmbeddings(
                model=self.embedding_model,
                api_key=self.api_key,
                base_url=self.base_url,
                dimension=self.dimension,
            )
        self.embeddings.validate_environment()

        self.collection_name = "assets_collection"
        self._client = None
        self._collection = None
        self.is_initialized = False

        if auto_sync:
            self.sync(force_rebuild)
        else:
            self._initialize_chroma()

    def _initialize_chroma(self):
        try:
            logger.info(f"Initializing ChromaDB, persist directory: {self.chroma_persist_directory}")
            self._client = chromadb.PersistentClient(path=self.chroma_persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=_EmbeddingFunctionAdapter(self.embeddings),
                metadata={"hnsw:space": "cosine"},
            )
            self.is_initialized = True
            logger.info(f"ChromaDB initialized successfully, current asset count: {self.count()}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            logger.error(traceback.format_exc())
            self.is_initialized = False

    def _load_sync_state(self) -> Dict:
        """Load sync state"""
        if os.path.exists(self.sync_state_path):
            try:
                with open(self.sync_state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {"last_sync_time": None, "assets_hash": None}

    def _save_sync_state(self, state: Dict):
        """Save sync state"""
        os.makedirs(os.path.dirname(self.sync_state_path), exist_ok=True)
        with open(self.sync_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_assets(self) -> List[Dict]:
        assets_data = []
        for asset_id, asset_info in self.assets_index.items():
            desc = asset_info.get("description", {})
            semantic_names = desc.get("semantic_name", [])
            full_descriptions = desc.get("full_description", [])

            # Handle possible list or string format
            semantic_name = ""
            if semantic_names:
                if isinstance(semantic_names, list):
                    semantic_name = semantic_names[0] if semantic_names else ""
                else:
                    semantic_name = str(semantic_names)

            full_description = ""
            if full_descriptions:
                if isinstance(full_descriptions, list):
                    full_description = full_descriptions[0] if full_descriptions else ""
                else:
                    full_description = str(full_descriptions)

            if not semantic_name and not full_description:
                logger.warning(f"Asset {asset_id} missing semantic name or full description, skipping")
                continue

            # Build asset data
            asset_data = {
                "asset_id": asset_id,
                "semantic_description": f"{semantic_name}: {full_description}",
            }

            # Preserve all other information
            for key, value in asset_info.items():
                if key not in ["semantic_description", "asset_id"]:
                    asset_data[key] = value

            assets_data.append(asset_data)

        logger.info(f"Loaded {len(assets_data)} assets")
        return assets_data

    def _create_document(self, asset_info: Dict) -> Optional[Dict]:
        asset_id = asset_info.get("asset_id", "")
        semantic_description = asset_info.get("semantic_description", "")
        if not asset_id or not semantic_description:
            logger.warning(f"Asset {asset_id} missing required fields, skipping")
            return None
        return {
            "id": asset_id,
            "document": semantic_description,
            "metadata": {
                "asset_id": asset_id,
                "semantic_description": semantic_description,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            },
        }

    def sync(self, force_rebuild: bool = False) -> Dict:
        sync_state = self._load_sync_state()
        current_assets_hash = ASSETS_INDEX_HASH

        self._initialize_chroma()
        if self.count() == 0:
            logger.warning("ChromaDB is empty, will rebuild")
            force_rebuild = True
        if (
            not force_rebuild
            and sync_state.get("assets_hash") == current_assets_hash
            and sync_state.get("last_sync_time")
        ):
            logger.info("Assets file unchanged, skipping sync")
            return {
                "synced": False,
                "reason": "no_changes",
                "asset_count": self.count(),
            }

        logger.info("Assets file changed, starting sync")
        if force_rebuild:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=_EmbeddingFunctionAdapter(self.embeddings),
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Vector database reset, current asset count: {self.count()}")

        assets = self._load_assets()
        existing_assets = self.get_all()
        assets_id_list = [asset["asset_id"] for asset in assets]
        for existing_asset in existing_assets:
            if existing_asset["asset_id"] not in assets_id_list:
                self._collection.delete(where={"asset_id": existing_asset["asset_id"]})
                logger.info(f"Asset deleted: {existing_asset['asset_id']}")

        if assets:
            to_be_added = []
            for asset in assets:
                in_db_asset = self._search_by_asset_id(asset["asset_id"], top_k=1, from_assets_dict=False)
                if in_db_asset:
                    asset_semantic_description = asset.get("semantic_description", "")
                    if in_db_asset[0].get("semantic_description") != asset_semantic_description:
                        self._collection.delete(where={"asset_id": asset["asset_id"]})
                        logger.info(f"Asset changed: {asset['asset_id']}")
                        to_be_added.append(asset)
                else:
                    to_be_added.append(asset)
                    logger.info(f"New asset found: {asset['asset_id']}")
            if to_be_added:
                ids = []
                documents = []
                metadatas = []
                for asset in to_be_added:
                    doc = self._create_document(asset)
                    if doc:
                        ids.append(doc["id"])
                        documents.append(doc["document"])
                        metadatas.append(doc["metadata"])
                logger.info(f"Adding {len(documents)} assets to ChromaDB")
                self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
                logger.info(f"Added {len(documents)} assets to ChromaDB")

        new_sync_state = {
            "last_sync_time": datetime.now().isoformat(),
            "assets_hash": current_assets_hash,
            "asset_count": self.count(),
        }
        self._save_sync_state(new_sync_state)

        logger.info(f"Sync completed, current asset count: {self.count()}")
        return {
            "synced": True,
            "asset_count": self.count(),
            "last_sync_time": new_sync_state["last_sync_time"],
        }

    def search(
        self,
        query: Union[str, List[str]],
        top_k: int = 10,
        exclude_regex: str = None,
        include_regex: str = None,
        scene_description: str = None,
    ) -> Union[List[Dict], List[List[Dict]]]:
        before = time.time()
        if not self.embedding_model:
            raise ValueError("Text embedding model or base_url not set, please check configuration")

        if not self.is_initialized or not self._collection:
            logger.warning("Vector database not initialized or empty")
            return [] if isinstance(query, str) else [[]]

        is_single_query = isinstance(query, str)
        queries = [query] if is_single_query else query

        all_results = []
        search_top_k = top_k if (exclude_regex is None or exclude_regex == "") else self.count()
        for query_text in queries:
            try:
                query_results = []
                id_results = self._search_by_asset_id(query_text, top_k=search_top_k)
                for id_result in id_results:
                    asset_id = id_result.get("asset_id")
                    asset_params = self.assets_index.get(asset_id, None)
                    if not asset_params:
                        continue
                    result = {
                        "asset_id": asset_id,
                        "description": asset_params.get("description", {}),
                    }
                    query_results.append(result)
                if len(query_results) >= top_k:
                    query_results = query_results[:top_k]
                else:
                    results = self._collection.query(
                        query_texts=[query_text],
                        n_results=self.count(),
                    )
                    if results and results["metadatas"] and results["metadatas"][0]:
                        for metadata in results["metadatas"][0]:
                            asset_id = metadata.get("asset_id")
                            if not asset_id:
                                continue
                            if exclude_regex and re.search(exclude_regex, asset_id):
                                continue
                            if include_regex and not re.search(include_regex, asset_id):
                                continue
                            already_in_query = False
                            for id_result in id_results:
                                if id_result.get("asset_id") == asset_id:
                                    already_in_query = True
                                    break
                            if already_in_query:
                                continue
                            asset_params = self.assets_index.get(asset_id, None)
                            if not asset_params or "description" not in asset_params:
                                continue
                            result = {
                                "asset_id": asset_id,
                                "info": asset_params,
                            }
                            query_results.append(result)
                            if len(query_results) >= top_k:
                                break

                all_results.append(query_results)

            except Exception as e:
                logger.error(f"Error searching query '{query_text}': {e}")
                all_results.append([])

        if isinstance(query, str):
            logger.info(f"Search '{query}' returned {len(all_results[0])} results")
            after = time.time()
            logger.info(f"Search took {after - before:.2f} seconds")
            return all_results[0]
        else:
            logger.info(f"Batch search {len(query)} queries, each returning up to {top_k} results")
            after = time.time()
            logger.info(f"Search took {after - before:.2f} seconds")
            return all_results

    def _search_by_asset_id(self, asset_id: str, top_k: int = 1, from_assets_dict=True) -> List[Dict]:
        if from_assets_dict:
            asset_params = self.assets_index.get(asset_id, None)
            if not asset_params:
                return []
            result = {
                "asset_id": asset_id,
                "description": asset_params.get("description", {}),
            }
            return [result]
        try:
            docs = self._collection.get(where={"asset_id": asset_id}, limit=top_k)
            if docs and "metadatas" in docs and docs["metadatas"]:
                return list(docs["metadatas"][:top_k])
        except Exception as e:
            logger.error(f"Error searching by asset_id: {e}")
        return []

    def search_by_asset_id(self, asset_id: str, top_k: int = 1) -> List[Dict]:
        results = self._search_by_asset_id(asset_id, top_k=top_k)
        results_output = []
        for id_result in results:
            asset_id = id_result.get("asset_id")
            asset_params = self.assets_index.get(asset_id, None)
            if not asset_params:
                continue
            result = {
                "asset_id": asset_id,
                "description": asset_params.get("description", {}),
            }
            results_output.append(result)
        return results_output

    def get(self, asset_id: str) -> Optional[Dict]:
        """
        Get asset information by asset_id

        Args:
            asset_id: Asset ID

        Returns:
            Asset information dictionary, returns None if not found
        """
        results = self._search_by_asset_id(asset_id, top_k=1)
        return results[0] if results else None

    def get_all(self) -> List[Dict]:
        if not self.is_initialized:
            self._initialize_chroma()
            if not self.is_initialized:
                return []
        try:
            docs = self._collection.get()
            if docs and "metadatas" in docs:
                return list(docs["metadatas"])
        except Exception as e:
            logger.error(f"Error getting all assets: {e}")
        return []

    def count(self) -> int:
        if not self.is_initialized or not self._collection:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def export_to_list(self, output_path: str = None) -> bool:

        if not self.is_initialized:
            self._initialize_chroma()
            if not self.is_initialized:
                logger.error("Unable to load vector database")
                return False

        # Get all assets
        all_assets = self.get_all()
        if not all_assets:
            logger.warning("No assets to export")
            return False

        # Convert to assets compatible format
        asset_list = []
        for asset in all_assets:
            asset_id = asset.get("asset_id", "")
            if not asset_id:
                continue
            asset_text = f"{asset.get('semantic_description', '')}; asset_id:{asset_id}"
            # Build assets structure
            asset_list.append(asset_text)

        os.makedirs(output_path, exist_ok=True)
        for i, asset in enumerate(asset_list):
            with open(os.path.join(output_path, f"asset_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(asset)
        return True

    def clear_cache(self):
        """Delete cache files"""
        import shutil

        cache_dirs = [self.chroma_persist_directory]
        cache_files = [self.sync_state_path]

        for dir_path in cache_dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    logger.info(f"Deleted directory: {dir_path}")
                else:
                    logger.info(f"Directory does not exist: {dir_path}")
            except Exception as e:
                logger.error(f"Error deleting directory {dir_path}: {e}")

        for file_path in cache_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                else:
                    logger.info(f"File does not exist: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")

        # Reset state
        self._collection = None
        self.is_initialized = False

    def status(self) -> Dict:
        """
        Get database status

        Returns:
            Status information dictionary
        """
        sync_state = self._load_sync_state()

        return {
            "initialized": self.is_initialized,
            "asset_count": self.count(),
            "last_sync_time": sync_state.get("last_sync_time"),
            "model": self.embedding_model,
            "chroma_persist_directory": self.chroma_persist_directory,
            "collection_name": self.collection_name,
        }


def use_example():
    # Example configuration
    from pathlib import Path
    import sys

    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))

    # Create database instance (auto sync YAML)
    logger.info("=== Creating database instance ===")
    config = json.load(open(f"{current_dir}/../mcp_text_embedding/text_embedding_config.json"))
    db = AssetVectorDB(config=config, auto_sync=True, force_rebuild=False)
    queries = ["table_001", "table"]
    batch_results = db.search(queries, top_k=10)

    for i, query_results in enumerate(batch_results):
        logger.info(f"\nQuery '{queries[i]}' results:")
        for j, result in enumerate(query_results):
            logger.info(f"  {j+1}. {result['asset_id']}...")


def force_rebuild():
    from pathlib import Path
    import sys

    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))

    from config.scene_gen_config_loader import SceneGenConfig

    config_path = "config/scene_gen_config.yaml"
    config = SceneGenConfig(config_path=config_path)

    # Create database instance (auto sync YAML)
    logger.info("=== Creating database instance ===")
    db = AssetVectorDB(
        config=config.get_text_embedding_config(),
        auto_sync=True,
    )
    db.sync(force_rebuild=True)


def export():
    from pathlib import Path
    import sys

    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))

    from config.scene_gen_config_loader import SceneGenConfig

    config_path = "config/scene_gen_config.yaml"
    config = SceneGenConfig(config_path=config_path)
    db = AssetVectorDB(
        config=config.get_text_embedding_config(),
        auto_sync=True,
    )
    db.export_to_list(f"{current_dir}/data_list")


if __name__ == "__main__":
    use_example()
    # force_rebuild()
    # export()
