"""LiteLLM proxy configuration compilation from the model registry.

Compiles ``config/models.yaml`` into a LiteLLM proxy YAML configuration.
Fixed logical models with multiple LiteLLM deployments receive
endpoint-qualified aliases.  Explicit ``managed-group`` models instead share
one logical alias so LiteLLM can select among their certified equivalent
deployments while Brigade pins the group and generation.

Never materializes secret values — uses ``os.environ/ENV_VAR_NAME`` syntax
so LiteLLM loads credentials from environment variables at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

import yaml

from enhanced_router.config_models import ModelSpec


def deployment_filter_enabled(explicit: bool | None = None) -> bool:
    """Return whether the pinned LiteLLM deployment filter is enabled.

    The setting is intentionally opt-in.  Keeping the decision in one helper
    means config generation and generation change detection cannot disagree
    when an operator toggles the callback between launches.
    """
    if explicit is not None:
        return explicit
    return os.environ.get(
        "BRIGADE_LITELLM_DISPATCH_FILTER", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def generate_litellm_config(
    models: dict[str, ModelSpec],
    *,
    referenced_ids: set[str] | None = None,
    enable_deployment_filter: bool | None = None,
) -> str:
    """Generate a LiteLLM proxy YAML configuration string.

    Parameters
    ----------
    models :
        Registry model definitions (already validated and loaded).
    referenced_ids :
        When given, discovered-catalog models (``catalog_source ==
        "discovered"``) not in this set are skipped -- provider discovery can
        add hundreds of models, and registering all of them with the LiteLLM
        child process needlessly costs minutes of startup time for models
        nothing has ever been assigned to use. Bundled/operator models are
        always included regardless of this filter. ``None`` (the default)
        includes everything, matching prior behavior.

    Returns
    -------
    str
        Complete YAML configuration text ready to write to disk.

    Note
    ----
    The ``BRIGADE_LITELLM_KEY`` environment variable must be set at runtime
    for the proxy ``master_key``.  Never materializes secrets in the YAML.
    """
    model_list: list[dict[str, Any]] = []

    # The callback is deliberately opt-in until the exact LiteLLM child
    # version has been certified in the deployment environment.  This
    # preserves conservative all-candidate admission when the child is
    # upgraded independently of ClaudeBrigade.
    enable_deployment_filter = deployment_filter_enabled(enable_deployment_filter)

    for model_id, spec in sorted(models.items()):
        if not spec.enabled:
            continue
        if (
            referenced_ids is not None
            and spec.catalog_source == "discovered"
            and model_id not in referenced_ids
        ):
            continue

        endpoint_specs = spec.endpoints or {"default": spec}
        for endpoint_id, endpoint in sorted(endpoint_specs.items()):
            if endpoint.backend != "litellm":
                continue
            if not endpoint.litellm_model:
                continue
            litellm_model = endpoint.litellm_model
            if spec.provider_id == "freeinference" and "/" not in litellm_model:
                litellm_model = f"openai/{litellm_model}"

            if spec.routing_mode == "managed-group":
                alias = f"brigade-{spec.deployment_group or model_id}"
            else:
                alias = f"brigade-{model_id}--{endpoint_id}" if spec.endpoints else f"brigade-{model_id}"
            credential_envs = _credential_env_names(endpoint.api_key_env)
            for credential_env in credential_envs:
                entry: dict[str, Any] = {
                    "model_name": alias,
                    "litellm_params": {"model": litellm_model},
                    "model_info": {
                        "logical_model_id": model_id,
                        "deployment_group": spec.deployment_group or model_id,
                        "deployment_id": endpoint_id,
                        "provider_id": endpoint.provider_id or spec.provider_id,
                        "priority": getattr(endpoint, "priority", 0),
                        "routing_mode": spec.routing_mode,
                    },
                }

                # API base — static URL or env var reference
                if endpoint.api_base:
                    entry["litellm_params"]["api_base"] = endpoint.api_base
                elif endpoint.api_base_env:
                    entry["litellm_params"]["api_base"] = f"os.environ/{endpoint.api_base_env}"

                # API keys remain env references. Named key slots become
                # equivalent LiteLLM deployments in the same model group.
                if credential_env:
                    entry["litellm_params"]["api_key"] = f"os.environ/{credential_env}"
                    if credential_env != endpoint.api_key_env:
                        entry["model_info"]["credential_env"] = credential_env

                model_list.append(entry)

    config: dict[str, Any] = {
        "model_list": model_list,
        "general_settings": {
            "master_key": "os.environ/BRIGADE_LITELLM_KEY",
        },
        "litellm_settings": {
            "drop_params": False,
            "num_retries": 0,
            "request_timeout": 600,
        },
        "model_group_settings": {
            "forward_client_headers_to_llm_api": sorted(
                {entry["model_name"] for entry in model_list}
            ),
        },
    }
    if enable_deployment_filter:
        config["litellm_settings"]["callbacks"] = [
            "brigade_litellm_dispatch.proxy_handler_instance"
        ]

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def _credential_env_names(api_key_env: str | None) -> list[str | None]:
    if not api_key_env:
        return [None]
    try:
        from enhanced_router.credential_store import materialized_env_name, slot_names

        slots = slot_names(api_key_env)
        materialized: list[str | None] = [
            materialized_env_name(api_key_env, slot)
            for slot in slots
            if os.environ.get(materialized_env_name(api_key_env, slot))
        ]
        if materialized:
            return materialized
    except Exception:
        pass
    return [api_key_env]


def config_digest(
    models: dict[str, ModelSpec],
    *,
    referenced_ids: set[str] | None = None,
    enable_deployment_filter: bool | None = None,
) -> str:
    """Return a deterministic SHA-256 hex digest of the litellm-relevant config.

    Only includes models with ``backend='litellm'``.  Used for change detection
    in catalog reload.

    ``referenced_ids`` applies the identical discovered-model filter that
    ``generate_litellm_config`` uses, so a scope-only change (e.g. a new run
    selecting a profile that references previously-excluded models) changes
    the digest and triggers a new generation, even though the underlying
    model definitions in *models* haven't changed at all.
    """
    entries: list[dict[str, Any]] = []
    for model_id, spec in sorted(models.items()):
        if spec.backend != "litellm" and not spec.endpoints:
            continue
        if (
            referenced_ids is not None
            and spec.catalog_source == "discovered"
            and model_id not in referenced_ids
        ):
            continue
        entries.append(
            {
                "model_id": model_id,
                "enabled": spec.enabled,
                "routing_mode": spec.routing_mode,
                "deployment_group": spec.deployment_group,
                "litellm_model": spec.litellm_model,
                "endpoints": {
                    endpoint_id: endpoint.model_dump()
                    for endpoint_id, endpoint in sorted(spec.endpoints.items())
                },
                "api_base": spec.api_base,
                "api_base_env": spec.api_base_env,
                "api_key_env": spec.api_key_env,
            }
        )
    raw = json.dumps(
        {
            "models": entries,
            # A callback toggle changes the child behavior even when the
            # model registry is unchanged, so it must create a new
            # blue-green generation rather than silently reusing the old
            # child.
            "deployment_filter_enabled": deployment_filter_enabled(
                enable_deployment_filter
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_litellm_config(
    config_text: str, output_path: str | pathlib.Path
) -> None:
    """Write the config file atomically with mode ``0o600``.

    Writes to a temporary file in the same directory, then renames to
    the target path to prevent partial reads by the LiteLLM process.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(output_path.parent),
        prefix=".litellm-",
        suffix=".yaml.tmp",
    )
    try:
        os.write(fd, config_text.encode("utf-8"))
        os.close(fd)
        # Restrict permissions before rename
        os.chmod(tmp_path, 0o600)
        # Atomic rename (POSIX)
        os.replace(tmp_path, str(output_path))
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
