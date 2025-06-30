#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import typer
import yaml
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError

CHECKPOINT_RE = re.compile(r"ep\d+-ba(\d+)-rank0\.pt$")

app = typer.Typer(pretty_exceptions_show_locals=False, context_settings={"help_option_names": ["-h", "--help"]})


def find_local_checkpoints(base_dir: Path, model_dirs: List[str]) -> Dict[str, Path]:
    """Return {repo_path: local_path} for every `ep*-ba*-rank0.pt` under each model dir."""
    ckpts: Dict[str, Path] = {}
    for mdir in model_dirs:
        for path in (base_dir / mdir).glob("ep*-ba*-rank0.pt"):
            if path.name.startswith("latest-"):
                continue  # Skip alias
            if CHECKPOINT_RE.match(path.name):
                ckpts[f"{mdir}/{path.name}"] = path
    return ckpts


def find_remote_checkpoints(api: HfApi, repo_id: str, token: Optional[str]) -> Set[str]:
    """Return the set of path strings already present in the HF repo."""
    try:
        return set(list_repo_files(repo_id, token=token))
    except HfHubHTTPError as e:
        typer.secho(f"❌ Cannot list repo files: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def upload_file(
    api: HfApi,
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
    token: Optional[str],
    commit_msg: str = "Add checkpoint",
):
    print('Uploading ', local_path)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=token,
        repo_type="model",
        commit_message=commit_msg,
    )
    typer.echo(f"✅  Uploaded {path_in_repo}")


def catchup_upload(
    api: HfApi,
    repo_id: str,
    base_dir: Path,
    model_dirs: List[str],
    token: Optional[str],
):
    typer.echo("🔍  Running one-off catch-up scan …")

    local_ckpts = find_local_checkpoints(base_dir, model_dirs)
    remote_ckpts = find_remote_checkpoints(api, repo_id, token)

    to_upload = [
        (repo_path, local_path)
        for repo_path, local_path in local_ckpts.items()
        if repo_path not in remote_ckpts
    ]

    # Order by batch number (int) to keep history tidy
    to_upload.sort(key=lambda x: int(CHECKPOINT_RE.search(x[0]).group(1)))  # type: ignore

    for repo_path, local_path in to_upload:
        upload_file(api, repo_id, local_path, repo_path, token, commit_msg="Catch-up upload")

    if not to_upload:
        typer.echo("✨  Repo already up to date.")


def poll_loop(
    api: HfApi,
    repo_id: str,
    base_dir: Path,
    model_dirs: List[str],
    poll_interval: int,
    token: Optional[str],
):
    typer.echo(f"🔄  Entering polling loop (every {poll_interval}s) …\n")

    while True:
        try:
            local_ckpts = find_local_checkpoints(base_dir, model_dirs)
            remote_ckpts = find_remote_checkpoints(api, repo_id, token)

            new_items = [
                (rp, lp) for rp, lp in local_ckpts.items() if rp not in remote_ckpts
            ]
            new_items.sort(key=lambda x: int(CHECKPOINT_RE.search(x[0]).group(1)))  # type: ignore

            for repo_path, local_path in new_items:
                upload_file(api, repo_id, local_path, repo_path, token, commit_msg="Add checkpoint")

        except Exception as e:
            # Log and continue; do not kill the loop
            typer.secho(f"⚠️  Error in poll loop: {e}", fg=typer.colors.YELLOW, err=True)

        time.sleep(poll_interval)


# --------------------------- CLI -------------------------------------------------


def conf_callback(ctx: typer.Context, param: typer.CallbackParam, config: Optional[str] = None):
    """Merge YAML config into Typer defaults (same helper you already use)."""
    if config:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
        ctx.default_map = ctx.default_map or {}
        ctx.default_map.update(cfg)
    return config


@app.command()
def main(
    repo_id: str = typer.Option(..., help="HF repo to push to, e.g. answerdotai/huge-in-run-checkpoints"),
    base_dir: Path = typer.Option(
        ..., help="Root with model_dir/checkpoints"
    ),
    model_dirs: List[str] = typer.Option(
        ..., help="One or more sub-dirs to watch"
    ),
    token: Optional[str] = typer.Option(None, help="HF token (or set HF_TOKEN env var)"),
    poll_interval: int = typer.Option(60, help="Seconds between scans after catch-up"),
    once: bool = typer.Option(
        False, "--once", help="Exit after the catch-up pass (no polling)"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        callback=conf_callback,
        is_eager=True,
        help="YAML file with default values (CLI overrides)",
    ),
):  # fmt: skip
    """
    Upload all `ep*-ba*-rank0.pt` checkpoints found under *base_dir/model_dir/* to Hugging Face Hub.

    1.  Performs an initial catch-up (only missing files are pushed).
    2.  Unless `--once` is given, keeps polling local dirs for fresh checkpoints.
    """

    api = HfApi(token=token)

    catchup_upload(api, repo_id, base_dir, model_dirs, token)

    if once:
        typer.echo("🏁  Done (catch-up only).")
        return

    poll_loop(api, repo_id, base_dir, model_dirs, poll_interval, token)


if __name__ == "__main__":
    app()