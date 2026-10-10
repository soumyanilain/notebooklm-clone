"""Optional persistence for hosts with an ephemeral disk (Cloud Run, free Spaces, Render).

Enabled only when HF_DATASET_REPO (e.g. "username/notebooklm-data") and HF_TOKEN are
set. On start-up the notebooks folder is restored from that private dataset repo;
while running, huggingface_hub's CommitScheduler pushes changed files every few
minutes. Vector indexes are not synced: they are rebuilt from the chunk files.

Limitations (documented in the README): changes made in the last sync interval
before a restart can be lost, and only one running instance should write to a repo.
"""

from __future__ import annotations

import logging
import threading

from ..config import Settings

log = logging.getLogger(__name__)

IGNORE = ["*/vectordb/*", "*/.tmp_*", "*.tmp_*", "*/.deleted", "*.lock", ".cache/*", "*/.cache/*"]


class HFDatasetSync:
    def __init__(self, settings: Settings):
        if not settings.hf_token:
            raise RuntimeError("HF_DATASET_REPO is set but HF_TOKEN is missing.")
        self.settings = settings
        self.repo_id = settings.hf_dataset_repo
        self.scheduler = None
        from huggingface_hub import HfApi

        self.api = HfApi(token=settings.hf_token)

    def restore(self) -> None:
        from huggingface_hub import snapshot_download

        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=self.repo_id, repo_type="dataset", local_dir=str(self.settings.data_dir),
                              token=self.settings.hf_token, allow_patterns=["notebooks/*"])
            log.info("Restored notebooks from dataset %s", self.repo_id)
        except Exception:  # noqa: BLE001 - an empty repo is fine on first run
            log.warning("Nothing restored from %s (new or empty repo?)", self.repo_id, exc_info=True)

    def attach(self, store) -> None:
        from huggingface_hub import CommitScheduler

        self.scheduler = CommitScheduler(
            repo_id=self.repo_id, repo_type="dataset", folder_path=str(self.settings.data_dir),
            every=max(1, self.settings.hf_sync_minutes), private=True, token=self.settings.hf_token,
            allow_patterns=["notebooks/*"], ignore_patterns=IGNORE,
        )
        store.on_notebook_deleted.append(self._delete_remote)

    def _delete_remote(self, nb_id: str) -> None:
        def run() -> None:
            try:
                self.api.delete_folder(path_in_repo=f"notebooks/{nb_id}", repo_id=self.repo_id, repo_type="dataset",
                                       commit_message=f"Delete notebook {nb_id}")
            except Exception:  # noqa: BLE001 - folder may not have been pushed yet
                log.info("Remote delete skipped for %s", nb_id)

        threading.Thread(target=run, daemon=True).start()
