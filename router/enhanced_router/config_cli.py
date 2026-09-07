"""Interactive saved configuration manager for ClaudeBrigade.

The CLI deliberately displays credential names and availability only. Secret
values are loaded from the OS keyring when available, with the same locked
providers.env parser as a compatibility fallback. They are never echoed or
written to model/sidecar YAML.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Literal, cast

import yaml

try:
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.prompt import Prompt as RichPrompt
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by minimal installations
    # Rich is a declared dependency, but keep a safe plain-mode fallback for
    # minimal embedded installations.  The typed Any boundary prevents the
    # optional import from poisoning every interactive call site in Pyright.
    box = cast(Any, None)
    Console = Group = Panel = RichPrompt = Table = Text = cast(Any, None)
    _RICH_AVAILABLE = False

from enhanced_router.bootstrap_env import (
    ALLOWED_PROVIDER_KEYS,
    PROVIDER_SECRET_KEYS,
    load_router_credentials,
)
from enhanced_router.certification import publish_contract_report
from enhanced_router.credential_store import (
    CredentialStoreUnavailable,
    available as credential_store_available,
    backend_label,
    delete_slot,
    read_slots,
    resolve,
    resolve_loaded,
    set_slot,
    slot_names,
)
from enhanced_router.env_parser import parse_env_file
from enhanced_router.registry import ModelRegistry
from enhanced_router.provider_config import migrate_provider_config
from enhanced_router.sidecar_config import migrate_sidecar_config
from enhanced_router.workflow_config import migrate_workflow_config
from enhanced_router.state import get_state

_ROLES = ("recon", "implementer", "adversary", "repairer")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PRINTED_REGISTRY_DIAGNOSTICS: set[tuple[str, tuple[str, ...]]] = set()


# ---------------------------------------------------------------------------
# Interactive presentation
# ---------------------------------------------------------------------------

_CONSOLE: Any = Console(highlight=False) if _RICH_AVAILABLE else None


def _rich_interactive() -> bool:
    """Return whether this invocation can use the full Rich UI.

    Rich is deliberately limited to real terminal sessions.  Hooks, tests,
    launcher JSON hand-offs, and redirected output keep the old line protocol
    so the configuration CLI remains scriptable and safe to embed.
    """
    return bool(
        _RICH_AVAILABLE
        and _CONSOLE is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
        and not os.environ.get("BRIGADE_PLAIN")
    )


def _rich_prompt_label(label: str) -> str:
    """Style a prompt label without treating user/config text as markup."""
    if not _RICH_AVAILABLE:
        return label
    from rich.markup import escape

    return f"[bold cyan]{escape(label)}[/]"


def _render_menu(
    title: str,
    sections: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    *,
    footer: str | None = None,
) -> None:
    """Render a grouped interactive menu, with a plain fallback.

    Keeping this in one helper prevents the wizard from having subtly
    different menu conventions across the inference, sidecar, and launch
    setup editors.
    """
    if not _rich_interactive():
        print(f"\n{title}")
        for heading, entries in sections:
            if heading:
                print(f"\n{heading}")
            for key, description in entries:
                print(f"  {key}. {description}")
        if footer:
            print(f"\n{footer}")
        return

    table = Table(
        box=box.SIMPLE,
        expand=True,
        show_header=False,
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("key", style="bold cyan", width=5, justify="right", no_wrap=True)
    table.add_column("description", overflow="fold")
    for heading, entries in sections:
        if heading:
            table.add_row("", Text(heading.upper(), style="bold yellow"))
        for key, description in entries:
            table.add_row(Text(str(key), style="bold cyan"), Text(str(description)))
    subtitle = footer or "Type a choice and press Enter"
    _CONSOLE.print(Panel(table, title=title, subtitle=subtitle, border_style="cyan"))


def _render_options(
    label: str,
    options: Sequence[tuple[str, str]],
    *,
    start: int = 0,
    end: int | None = None,
    suffix: str = "",
    navigation: Sequence[str] = (),
    current: set[str] | None = None,
) -> None:
    """Render one page of a numbered picker."""
    visible_end = len(options) if end is None else end
    if not _rich_interactive():
        print(f"\n{label}{suffix}")
        for number in range(start, visible_end):
            value, description = options[number]
            marker = ">" if current and value in current else " "
            print(f"{marker} {number + 1}. {value} — {description}")
        if navigation:
            print("  " + "   ".join(navigation))
        return

    table = Table(
        box=box.SIMPLE,
        expand=True,
        show_header=False,
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("", width=2, style="bold green", no_wrap=True)
    table.add_column("#", width=4, justify="right", style="bold cyan", no_wrap=True)
    table.add_column("value", style="white", no_wrap=True)
    table.add_column("details", overflow="fold")
    for number in range(start, visible_end):
        value, description = options[number]
        marker = "✓" if current and value in current else ""
        table.add_row(
            Text(marker, style="bold green"),
            str(number + 1),
            Text(str(value)),
            Text(str(description)),
        )
    if navigation:
        table.add_row("", "", Text(" · ".join(navigation), style="dim"), "")
    _CONSOLE.print(Panel(table, title=f"{label}{suffix}", border_style="cyan"))


def _status(message: str, style: str = "green") -> None:
    """Print a semantic status message in the interactive UI."""
    if _rich_interactive():
        _CONSOLE.print(Text("●", style=style), Text(f" {message}"))
    else:
        print(message)


def startup_choice() -> str:
    """Choose the launch mode for the outer ``claude-brigade`` command.

    The shell launcher delegates this one prompt here so the first screen and
    the configuration wizard share the same Rich presentation.  It writes
    the prompt to stderr because the launcher captures stdout for the choice.
    """
    if _RICH_AVAILABLE and sys.stdin.isatty() and sys.stderr.isatty():
        console = Console(stderr=True, highlight=False)
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("key", style="bold cyan", width=5, justify="right")
        table.add_column("description")
        table.add_row("1", "Current configuration — Claude Code + configured model lanes")
        table.add_row("2", "Configure controller, role models, fallbacks, and sidecars")
        console.print(Panel(table, title="ClaudeBrigade", subtitle="Choose a launch mode", border_style="cyan"))
        return str(RichPrompt.ask(
            "[bold cyan]Start with[/]",
            console=console,
            choices=["1", "2"],
            default="1",
            show_choices=False,
        )).strip()

    sys.stderr.write("Start with:\n")
    sys.stderr.write("  1) Current configuration -- Claude Code + configured model lanes\n")
    sys.stderr.write("  2) Configure your own controller/role models, fallbacks, and sidecar\n")
    sys.stderr.write("Choice [1]: ")
    sys.stderr.flush()
    return (input().strip() or "1")

# ---------------------------------------------------------------------------
# Typed route choice — one concrete model + provider + endpoint combination.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteKey:
    """Stable identity for one provider/model/endpoint route."""

    provider_id: str
    model_id: str
    endpoint_id: str

    def serialize(self) -> str:
        return "\x1f".join((self.provider_id, self.model_id, self.endpoint_id))

    def __iter__(self):
        return iter((self.provider_id, self.model_id, self.endpoint_id))


class EditorResult(str):
    """String-compatible result with explicit save/cancel state.

    The string compatibility keeps older callers that compare the returned
    profile ID working, while the launch wizard can now distinguish a saved
    object from a discarded draft.
    """

    status: Literal["saved", "cancelled", "deleted", "unchanged"]
    object_id: str | None

    def __new__(
        cls,
        object_id: str | None,
        status: Literal["saved", "cancelled", "deleted", "unchanged"],
    ) -> "EditorResult":
        result = str.__new__(cls, object_id or "")
        result.status = status
        result.object_id = object_id
        return result


@dataclass(frozen=True)
class RouteChoice:
    """One selectable route: a concrete model through a specific provider
    endpoint. A single logical model may appear as multiple RouteChoices
    when it can be reached through distinct providers or endpoints."""

    model_id: str
    endpoint_id: str
    provider_id: str
    provider_name: str
    model_name: str
    backend: str
    credential_configured: bool
    availability: str
    certified: bool
    context_tokens: int | None = None
    max_output_tokens: int | None = None
    routing_mode: str = "fixed"
    is_free: bool = False
    tools: bool = False
    catalog_status: str = "unknown"
    catalog_age_seconds: float | None = None

    def route_key(self) -> RouteKey:
        """Unique key for this concrete provider/model/endpoint route."""
        return RouteKey(self.provider_id, self.model_id, self.endpoint_id)


def route_id(route: RouteChoice) -> str:
    """Serialize a RouteChoice identity for picker selections and lookups."""
    return route.route_key().serialize()


def _saved_endpoint_matches(candidate_endpoint: str, saved_endpoint: str) -> bool:
    """Treat legacy ``auto`` as the single concrete default route."""
    return candidate_endpoint == saved_endpoint or (
        saved_endpoint in {"", "auto"} and candidate_endpoint == "default"
    )


@dataclass(frozen=True)
class PickerFilters:
    query: str = ""
    free_only: bool = False
    verified_only: bool = False
    context_min: int | None = None
    tools_only: bool = False
    model_id: str | None = None


@dataclass(frozen=True)
class ModelPickerRequest:
    purpose: str
    provider_id: str
    choices: list[RouteChoice]
    current_route: RouteKey | None = None
    allow_uncertified: bool = True
    filters: PickerFilters = PickerFilters()


@dataclass(frozen=True)
class ModelPickerResult:
    route: RouteChoice | None = None
    action: NavigationAction | None = None


@dataclass(frozen=True)
class ProviderPickerSummary:
    provider_id: str
    display_name: str
    logical_model_count: int
    route_count: int
    free_model_count: int
    certified_model_count: int
    catalog_status: str = "unknown"
    catalog_age_seconds: float | None = None

    def __getitem__(self, index: int) -> object:
        """Keep tuple-style reads compatible with older integrations."""
        return (
            self.provider_id,
            self.display_name,
            self.logical_model_count,
        )[index]


class NavigationAction(Enum):
    BACK = auto()
    CANCEL = auto()
    DONE = auto()
    REFRESH = auto()


@dataclass(frozen=True)
class ChoiceControls:
    """Controls for the _choose_nav function — which navigation actions are
    available and what the default selected value should be."""
    allow_back: bool = False
    allow_cancel: bool = False
    allow_done: bool = False
    allow_refresh: bool = False
    default_value: str | None = None


@dataclass(frozen=True)
class NavResult:
    """Result from a navigation-aware choice prompt. Either a value was
    selected, or a navigation action was triggered."""
    value: str | None = None
    action: NavigationAction | None = None

    @property
    def is_navigation(self) -> bool:
        return self.action is not None


def _config_dir(value: str | None) -> Path:
    return Path(
        value
        or os.environ.get("BRIGADE_CONFIG_DIR")
        or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "claude-brigade"
    ).expanduser()


def _credential_configured(
    key_env: str | None,
    config_dir: Path | None = None,
) -> bool:
    """Return whether a provider credential is available to this process.

    Keyring-backed credentials are materialized under router-owned aliases,
    so checking only the legacy environment variable would make configured
    providers disappear from the picker and refresh summary.
    """
    if not key_env:
        return True
    if os.environ.get(key_env):
        return True
    try:
        if resolve_loaded(key_env, config_dir):
            return True
    except Exception:
        pass
    # The interactive CLI runs outside the router process, so credentials
    # have not necessarily been materialized into BRIGADE_KEYRING_* names.
    # Read the keyring directly here without displaying or logging the value.
    try:
        if resolve(key_env, config_dir):
            return True
    except Exception:
        pass
    # providers.env remains a locked-down compatibility source.  This is
    # deliberately last: the router still prefers the OS credential store.
    try:
        path = (config_dir or _config_dir(None)) / "providers.env"
        if path.exists():
            return bool(parse_env_file(path, allowed_keys=ALLOWED_PROVIDER_KEYS).get(key_env))
    except Exception:
        pass
    return False


def _print_version_diagnostic() -> None:
    """Print the source path and git revision of this config_cli module.

    Only prints when BRIGADE_DEBUG is set, making it opt-in for diagnostics.
    """
    if not os.environ.get("BRIGADE_DEBUG"):
        return
    here = Path(__file__).resolve()
    rev = "unknown"
    try:
        parent = here.parent
        while parent != parent.parent:
            if (parent / ".git").exists():
                import subprocess
                result = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                )
                rev = result.stdout.strip() or "unknown"
                break
            parent = parent.parent
    except Exception:
        pass
    # Also try reading install-info.json for installed-revision info.
    install_info_path = here.parents[2] / "install-info.json"
    if install_info_path.exists():
        try:
            import json
            info = json.loads(install_info_path.read_text())
            installed_rev = info.get("source_revision", "") or info.get("git_commit", "")[:8] or ""
            if installed_rev:
                rev = f"{rev} (installed: {installed_rev})"
        except Exception:
            pass
    print(f"ClaudeBrigade config module: {here}", file=sys.stderr)
    print(f"  revision: {rev}", file=sys.stderr)


def _load_registry(config_dir: Path) -> tuple[ModelRegistry, tuple[str, ...]]:
    config_dir.mkdir(parents=True, exist_ok=True)
    bundled_providers = config_dir / "providers.yaml.example"
    if not bundled_providers.exists():
        bundled_providers = Path(__file__).resolve().parents[2] / "config" / "providers.yaml"
    migrate_provider_config(config_dir / "providers.yaml", bundled_providers)
    loaded = load_router_credentials(config_dir)
    # Diagnostic header: report which config_cli module is running.
    _print_version_diagnostic()
    # The values stay in this short-lived CLI process only. They are needed to
    # determine which configured transports are usable; they are never printed.
    os.environ.update(loaded.provider_env)
    registry = ModelRegistry(config_dir)
    registry.load_models()
    # Preserve older user-owned catalogs while making newly installed
    # sidecars/fastpath definitions resolvable.  The installer stores the
    # current bundled catalog as models.yaml.example on upgrades; source
    # checkouts use the repository config directory instead.  Only the
    # router-owned compatibility models are overlaid, and user definitions
    # always remain authoritative.
    bundled_models = config_dir / "models.yaml.example"
    if not bundled_models.exists():
        source_models = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
        if source_models.exists():
            bundled_models = source_models
    if bundled_models.exists() and bundled_models.resolve() != (config_dir / "models.yaml").resolve():
        bundled = ModelRegistry(bundled_models.parent)
        bundled.load_models(bundled_models)
        for model_id, spec in bundled.models.items():
            if spec.backend == "anthropic-passthrough" or spec.provider_id == "freeinference":
                registry.models.setdefault(model_id, spec)
    registry.load_profiles()
    registry.load_workflows()
    registry.load_providers()
    registry.load_fastpath()
    registry.load_sidecars()
    registry.load_sidecar_profiles()
    registry.load_launch_presets()
    # Tolerant, not fail-closed: a broken reference anywhere in the saved
    # config (e.g. a profile pointing at a model that was since removed)
    # must not prevent the wizard from even starting -- the operator needs
    # a working menu to *fix* it. Runtime startup (get_registry()) is a
    # separate, unrelated call path and still calls _validate_cross_refs()
    # directly, so the running router stays fail-closed on invalid config.
    diagnostics = tuple(
        f"saved '{diagnostic.section}' configuration has an issue: {diagnostic.message}"
        for diagnostic in registry.collect_config_diagnostics()
    ) + tuple(registry.profile_model_diversity_warnings())
    diagnostic_key = (str(config_dir.resolve()), diagnostics)
    if diagnostics and diagnostic_key not in _PRINTED_REGISTRY_DIAGNOSTICS:
        _PRINTED_REGISTRY_DIAGNOSTICS.add(diagnostic_key)
        for message in diagnostics:
            print(f"Warning: {message}")
    return registry, loaded.provider_keys


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _validate_config_draft(config_dir: Path) -> None:
    """Validate a complete operator config before accepting a YAML edit.

    Individual credential/catalog files are intentionally allowed to exist
    without the rest of the configuration during bootstrap.  Once the model
    and profile documents exist, however, a saved edit must pass the same
    cross-reference validation used by the runtime registry.
    """
    if not (config_dir / "models.yaml").exists() or not (config_dir / "profiles.yaml").exists():
        return
    draft = ModelRegistry(config_dir)
    draft.load_models()
    draft.load_profiles()
    optional_loaders = (
        ("workflows.yaml", draft.load_workflows),
        ("providers.yaml", draft.load_providers),
        ("fastpath.yaml", draft.load_fastpath),
        ("sidecars.yaml", draft.load_sidecars),
        ("sidecar_profiles.yaml", draft.load_sidecar_profiles),
        ("launch_presets.yaml", draft.load_launch_presets),
    )
    for filename, loader in optional_loaders:
        if (config_dir / filename).exists():
            loader()
    draft._validate_cross_refs()


def _atomic_restore(path: Path, previous: bytes | None, mode: int | None) -> None:
    """Restore one YAML file after a failed draft validation."""
    if previous is None:
        if path.exists() and not path.is_symlink():
            path.unlink()
        return
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.restore.", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(previous)
        os.replace(temporary_name, path)
        if mode is not None:
            os.chmod(path, mode)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_yaml(
    path: Path,
    data: dict[str, Any],
    *,
    validate_config_dir: Path | None = None,
) -> None:
    """Atomically write YAML and roll it back if the draft is invalid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked configuration: {path}")
    previous = path.read_bytes() if path.exists() else None
    previous_mode = (path.stat().st_mode & 0o777) if path.exists() else None
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(path.parent), text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
        if validate_config_dir is not None:
            try:
                _validate_config_draft(validate_config_dir)
            except Exception:
                _atomic_restore(path, previous, previous_mode)
                raise
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_provider_env(path: Path, values: dict[str, str]) -> None:
    """Atomically write owner-readable provider credentials without echoing them."""
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked credential file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(path.parent), text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                "# ClaudeBrigade provider credentials; managed by claude-brigade-config.\n"
                + "".join(
                    f"{key}={json.dumps(values[key])}\n"
                    for key in sorted(values)
                    if values[key]
                )
            )
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _prompt(label: str, default: str | None = None) -> str:
    if _rich_interactive():
        value = RichPrompt.ask(
            _rich_prompt_label(label),
            console=_CONSOLE,
            default=default if default else "",
            show_default=bool(default),
        )
        return value.strip() or (default or "")
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def _menu_prompt(label: str, default: str | None = None) -> str:
    """Read a menu command without hijacking tests that stub field prompts.

    The wizard's menu commands are intentionally a separate seam: callers
    embedding a sub-editor can stub ``_prompt`` for ordinary fields while
    still driving the menu with the original ``input`` sequence.
    """
    if _rich_interactive():
        return _prompt(label, default)
    suffix = f" [{default}]" if default else ""
    return input(f"{label}{suffix}: ").strip() or (default or "")


def _prompt_int(label: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        raw = _prompt(label, str(default))
        try:
            value = int(raw)
        except ValueError:
            _status("Enter a whole number.", "yellow")
            continue
        if minimum <= value <= maximum:
            return value
        _status(f"Enter a number from {minimum} to {maximum}.", "yellow")


def _prompt_float(label: str, default: float, minimum: float, maximum: float) -> float:
    while True:
        raw = _prompt(label, str(default))
        try:
            value = float(raw)
        except ValueError:
            _status("Enter a number.", "yellow")
            continue
        if minimum <= value <= maximum:
            return value
        _status(f"Enter a number from {minimum} to {maximum}.", "yellow")


def _choose_toggle(label: str, current: bool, *, enabled_text: str = "Enabled", disabled_text: str = "Disabled") -> bool:
    """Render an explicit Rich/plain On/Off choice and return its value."""
    selected = _choose(
        label,
        [("on", enabled_text), ("off", disabled_text)],
        default=1 if current else 2,
    )
    return selected == "on"


_PAGE_SIZE = 15


def _choose(label: str, options: list[tuple[str, str]], default: int = 1) -> str:
    """Numbered picker over *options*, paginated for lists longer than a
    screenful. Numbers are absolute across pages (option 17 is always
    option 17, on whichever page it's currently shown), so a remembered or
    typed-ahead number never shifts meaning when the page changes."""
    if not options:
        raise RuntimeError(f"No choices are available for {label}.")
    total_pages = (len(options) - 1) // _PAGE_SIZE + 1
    page = ((default - 1) // _PAGE_SIZE) if 1 <= default <= len(options) else 0
    while True:
        start = page * _PAGE_SIZE
        end = min(start + _PAGE_SIZE, len(options))
        suffix = f" (page {page + 1}/{total_pages})" if total_pages > 1 else ""
        navigation = ["n) next page", "p) previous page"] if total_pages > 1 else []
        _render_options(label, options, start=start, end=end, suffix=suffix, navigation=navigation)
        if total_pages > 1:
            # Navigation is rendered above; keep the command grammar the same
            # for existing scripts and muscle memory.
            pass
        raw = _prompt("Choose", str(default))
        lowered = raw.strip().lower()
        if total_pages > 1 and lowered == "n":
            page = min(page + 1, total_pages - 1)
            continue
        if total_pages > 1 and lowered == "p":
            page = max(page - 1, 0)
            continue
        try:
            index = int(raw)
        except ValueError:
            _status("Choose one of the listed numbers, or n/p to change page.", "yellow")
            continue
        if 1 <= index <= len(options):
            return options[index - 1][0]
        _status("Choose one of the listed numbers.", "yellow")


def _choose_nav(
    label: str,
    options: list[tuple[str, str]],
    controls: ChoiceControls | None = None,
    default: int = 1,
) -> NavResult:
    """Numbered picker over *options* with navigation actions (back/cancel/done).
    Returns the selected value or a navigation action. Blank input returns the
    nav action whose letter is used as the first navigation entry's key, or
    CANCEL when multiple nav actions are defined and none is the obvious default.
    """
    if not options:
        raise RuntimeError(f"No choices are available for {label}.")
    ctrl = controls or ChoiceControls()
    total_pages = (len(options) - 1) // _PAGE_SIZE + 1
    page = ((default - 1) // _PAGE_SIZE) if 1 <= default <= len(options) else 0

    # Build nav action keys
    nav_letters: dict[str, NavigationAction] = {}
    if ctrl.allow_back:
        nav_letters["b"] = NavigationAction.BACK
    if ctrl.allow_cancel:
        nav_letters["q"] = NavigationAction.CANCEL
    if ctrl.allow_done:
        nav_letters["d"] = NavigationAction.DONE
    if ctrl.allow_refresh:
        nav_letters["r"] = NavigationAction.REFRESH

    # Determine default nav action when no numeric input given
    default_nav: NavigationAction | None = None
    if ctrl.allow_back and not ctrl.allow_cancel and not ctrl.allow_done:
        default_nav = NavigationAction.BACK
    elif ctrl.allow_cancel and not ctrl.allow_back and not ctrl.allow_done:
        default_nav = NavigationAction.CANCEL
    elif ctrl.allow_done and not ctrl.allow_back and not ctrl.allow_cancel:
        default_nav = NavigationAction.DONE

    while True:
        start = page * _PAGE_SIZE
        end = min(start + _PAGE_SIZE, len(options))
        suffix = f" (page {page + 1}/{total_pages})" if total_pages > 1 else ""
        navigation = ["n) next page", "p) previous page"] if total_pages > 1 else []
        navigation.extend(f"{letter}) {action.name.lower()}" for letter, action in nav_letters.items())
        _render_options(
            label,
            options,
            start=start,
            end=end,
            suffix=suffix,
            navigation=navigation,
        )
        # A blank line is not an implicit selection. Existing assignments are
        # indicated by the highlighted page/description, but line mode still
        # requires an explicit number or navigation command.
        raw = _prompt("Choose", ctrl.default_value)
        lowered = raw.strip().lower()
        if total_pages > 1 and lowered == "n":
            page = min(page + 1, total_pages - 1)
            continue
        if total_pages > 1 and lowered == "p":
            page = max(page - 1, 0)
            continue
        if lowered in nav_letters:
            return NavResult(action=nav_letters[lowered])
        if not lowered:
            if default_nav is not None:
                return NavResult(action=default_nav)
            _status("Enter a number or a navigation command.", "yellow")
            continue
        try:
            index = int(raw)
        except ValueError:
            allowed = ", ".join(sorted(nav_letters.keys()))
            if total_pages > 1:
                allowed = "n, p, " + allowed
            _status(
                f"Enter one of the listed numbers{', or ' + allowed if allowed else ''}.",
                "yellow",
            )
            continue
        if 1 <= index <= len(options):
            return NavResult(value=options[index - 1][0])
        _status("Choose one of the listed numbers.", "yellow")


def generate_route_choices(
    registry: ModelRegistry,
    *,
    role: str | None = None,
    controller: bool = False,
) -> list[RouteChoice]:
    """Generate typed RouteChoices for every selectable route combination.

    Each model that is available through a provider with credentials yields
    one RouteChoice per configured endpoint. A model available through two
    different providers appears as separate RouteChoices.
    """
    choices: list[RouteChoice] = []
    for model_id, model in sorted(registry.models.items(), key=lambda item: item[0]):
        if not model.enabled:
            continue
        if model.backend == "anthropic-passthrough" and not controller:
            continue
        if role is not None:
            if model.allowed_roles:
                if role not in model.allowed_roles:
                    continue

        # RouteChoice is a concrete route. ``auto`` is a runtime policy, not a
        # second provider endpoint, so never synthesize it into this browser.
        if model.endpoints:
            endpoints = sorted(model.endpoints.items())
        else:
            endpoint_id = getattr(model, "default_endpoint", None) or "default"
            endpoints = [(endpoint_id, model)]

        for endpoint_id, endpoint_candidate in endpoints:
            provider_id = (getattr(endpoint_candidate, "provider_id", None)
                           or model.provider_id or "local")
            provider = registry.providers.get(provider_id)
            provider_name = provider.display_name if provider else provider_id
            key_env = (getattr(endpoint_candidate, "api_key_env", None)
                       or model.api_key_env)
            if provider is not None:
                key_env = key_env or provider.api_key_env
            configured = _credential_configured(key_env, registry.config_dir)
            backend = getattr(endpoint_candidate, "backend", model.backend)
            if backend == "direct-anthropic":
                configured = configured and bool(
                    getattr(endpoint_candidate, "api_base", None)
                    or model.api_base
                    or getattr(endpoint_candidate, "api_base_env", None)
                    or model.api_base_env
                )
            if backend == "litellm":
                configured = configured and bool(
                    getattr(endpoint_candidate, "litellm_model", None)
                    or model.litellm_model
                )
            if not configured:
                continue

            # Certification: a model is "certified" for a role when its
            # allowed_roles includes it (or controller_eligible is true).
            certified = False
            if controller:
                certified = model.capabilities.controller_eligible
            elif role is not None and model.allowed_roles:
                certified = role in model.allowed_roles

            context_tokens = (getattr(endpoint_candidate, "max_context_tokens", None)
                              or model.capabilities.max_context_tokens
                              or model.capabilities.context_tokens)
            max_output = (getattr(endpoint_candidate, "max_output_tokens", None)
                          or model.capabilities.max_output_tokens)

            availability = getattr(endpoint_candidate, "availability", "unknown") or "unknown"
            if availability == "unknown":
                availability = model.availability or "unknown"

            is_free = (
                model.capabilities.cost_class == "free"
                or model_id.lower().endswith(":free")
                or ":free" in model_id.lower()
            )
            catalog_status, catalog_age = _model_catalog_status(registry, model)
            choices.append(RouteChoice(
                model_id=model_id,
                endpoint_id=endpoint_id,
                provider_id=provider_id,
                provider_name=provider_name,
                model_name=model.display_name,
                backend=backend,
                credential_configured=configured,
                availability=availability,
                certified=certified,
                context_tokens=context_tokens,
                max_output_tokens=max_output,
                routing_mode=model.routing_mode,
                is_free=is_free,
                tools=bool(getattr(endpoint_candidate, "tools", False) or model.capabilities.tools),
                catalog_status=catalog_status,
                catalog_age_seconds=catalog_age,
            ))
    return choices


def _model_catalog_status(registry: ModelRegistry, model: Any) -> tuple[str, float | None]:
    if getattr(model, "catalog_source", "bundled") != "discovered":
        return "bundled", None
    config_dir = registry.config_dir
    if config_dir is None:
        return "cached", None
    state = _load_catalog_refresh_state(config_dir)
    provider_id = str(getattr(model, "provider_id", None) or "")
    refreshed_at = state.get(provider_id, {}).get("refreshed_at")
    if not isinstance(refreshed_at, (int, float)):
        return "stale", None
    age = max(0.0, time.time() - refreshed_at)
    return ("fresh" if age <= _CATALOG_REFRESH_TTL_SECONDS else "stale"), age


def eligible_providers(choices: Sequence[RouteChoice]) -> list[ProviderPickerSummary]:
    """Summarize distinct logical models and concrete routes by provider."""
    prov: dict[str, dict[str, Any]] = {}
    for rc in choices:
        entry = prov.setdefault(rc.provider_id, {
            "display_name": rc.provider_name,
            "models": set(),
            "free": set(),
            "certified": set(),
            "routes": 0,
            "status": rc.catalog_status,
            "age": rc.catalog_age_seconds,
        })
        entry["models"].add(rc.model_id)
        if rc.is_free:
            entry["free"].add(rc.model_id)
        if rc.certified:
            entry["certified"].add(rc.model_id)
        entry["routes"] += 1
        if rc.catalog_status == "fresh":
            entry["status"] = "fresh"
        if rc.catalog_age_seconds is not None:
            entry["age"] = rc.catalog_age_seconds
    result = [
        ProviderPickerSummary(
            provider_id=pid,
            display_name=value["display_name"],
            logical_model_count=len(value["models"]),
            route_count=value["routes"],
            free_model_count=len(value["free"]),
            certified_model_count=len(value["certified"]),
            catalog_status=value["status"],
            catalog_age_seconds=value["age"],
        )
        for pid, value in prov.items()
    ]
    result.sort(key=lambda item: item.display_name.lower())
    return result


def choose_provider(
    choices: Sequence[RouteChoice],
    *,
    purpose: str = "model",
    current_provider_id: str | None = None,
    on_refresh: Callable[[], Sequence[RouteChoice]] | None = None,
) -> NavResult | str:
    """Provider-first picker. Shows only providers that have eligible routes.
    Returns the selected provider_id or a navigation action.
    """
    current_choices = list(choices)
    while True:
        providers = eligible_providers(current_choices)
        if not providers:
            print(f"No eligible providers found for {purpose}.")
            return NavResult(action=NavigationAction.BACK)
        if current_provider_id:
            providers.sort(key=lambda item: (item.provider_id != current_provider_id, item.display_name.lower()))
        options = []
        for summary in providers:
            catalog = summary.catalog_status
            if summary.catalog_age_seconds is not None and catalog in {"fresh", "stale"}:
                catalog = f"{catalog} ({int(summary.catalog_age_seconds)}s old)"
            route_suffix = f" · {summary.route_count} routes" if summary.route_count != summary.logical_model_count else ""
            options.append((
                summary.provider_id,
                f"{summary.display_name} — {summary.logical_model_count} models · "
                f"{summary.free_model_count} free · {catalog}{route_suffix}",
            ))
        default = 1
        result = _choose_nav(
            f"Choose provider for {purpose}",
            options,
            ChoiceControls(allow_back=True, allow_cancel=True, allow_refresh=on_refresh is not None),
            default,
        )
        if result.action == NavigationAction.REFRESH and on_refresh is not None:
            current_choices = list(on_refresh())
            continue
        if result.action is not None:
            return result
        if result.value is not None:
            return result.value
        return NavResult(action=NavigationAction.BACK)


def choose_route(
    choices: Sequence[RouteChoice],
    *,
    provider_id: str,
    purpose: str = "model",
    current_model_id: str | None = None,
    current_route: RouteKey | None = None,
) -> NavResult | RouteChoice:
    """After provider selection, run the shared model picker."""
    provider_choices = [rc for rc in choices if rc.provider_id == provider_id]
    if not provider_choices:
        print(f"No eligible routes from provider '{provider_id}'.")
        return NavResult(action=NavigationAction.BACK)
    result = ModelPicker(ModelPickerRequest(
        purpose=purpose,
        provider_id=provider_id,
        choices=list(provider_choices),
        current_route=current_route or next(
            (rc.route_key() for rc in provider_choices
             if current_model_id and rc.model_id == current_model_id),
            None,
        ),
    )).pick()
    if result.action is not None:
        return NavResult(action=result.action)
    return result.route or NavResult(action=NavigationAction.BACK)


def _rank_route_matches(query: str, choices: Sequence[RouteChoice]) -> list[RouteChoice]:
    """Token-aware, punctuation-tolerant route search with structured filters."""
    filters = _parse_picker_filters(query)
    if not filters.query and not any((filters.free_only, filters.verified_only,
                                      filters.context_min is not None, filters.tools_only,
                                      filters.model_id)):
        return list(choices)
    tokens = _normalize_search(filters.query).split()

    def searchable(rc: RouteChoice) -> str:
        return _normalize_search(" ".join((
            rc.model_id,
            rc.model_name,
            rc.provider_id,
            rc.provider_name,
            rc.endpoint_id,
            "free" if rc.is_free else "",
            "verified" if rc.certified else "unverified",
            "tools" if _route_has_tools(rc) else "",
        )))

    def matches(rc: RouteChoice) -> bool:
        if filters.free_only and not rc.is_free:
            return False
        if filters.verified_only and not rc.certified:
            return False
        if filters.context_min is not None and (rc.context_tokens or 0) < filters.context_min:
            return False
        if filters.tools_only and not _route_has_tools(rc):
            return False
        if filters.model_id and _normalize_search(filters.model_id) not in _normalize_search(rc.model_id):
            return False
        text = searchable(rc)
        return all(token in text for token in tokens)

    def rank(rc: RouteChoice) -> tuple[int, int]:
        raw_id = rc.model_id.lower()
        raw_name = rc.model_name.lower()
        normalized_name = _normalize_search(rc.model_name)
        normalized_id = _normalize_search(rc.model_id)
        if filters.query.lower() == raw_id:
            score = 0
        elif filters.query.lower() == raw_name:
            score = 1
        elif tokens and all(token in normalized_name for token in tokens):
            score = 2
        elif tokens and any(normalized_id.startswith(token) for token in tokens):
            score = 3
        elif tokens and all(token in normalized_id for token in tokens):
            score = 4
        else:
            score = 5
        return score, choices.index(rc)

    ranked = [rc for rc in choices if matches(rc)]
    ranked.sort(key=rank)
    return ranked


_SEARCH_SEPARATORS = re.compile(r"[-_/():,]+")


def _normalize_search(value: str) -> str:
    return " ".join(_SEARCH_SEPARATORS.sub(" ", value.lower()).split())


def _parse_context_limit(value: str) -> int | None:
    match = re.fullmatch(r">=?\s*([0-9]+(?:\.[0-9]+)?)\s*([km]?)", value.lower())
    if not match:
        return None
    number = float(match.group(1))
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2)]
    return int(number * multiplier)


def _parse_picker_filters(query: str) -> PickerFilters:
    parts = query.strip().split()
    free_only = False
    verified_only = False
    tools_only = False
    context_min = None
    model_id = None
    text_parts: list[str] = []
    for part in parts:
        key, separator, value = part.partition(":")
        lowered = part.lower()
        if lowered == "free":
            free_only = True
        elif lowered in {"verified", "certified"}:
            verified_only = True
        elif separator and key.lower() == "free" and value.lower() in {"true", "yes", "1"}:
            free_only = True
        elif separator and key.lower() in {"verified", "certified"} and value.lower() in {"true", "yes", "1"}:
            verified_only = True
        elif separator and key.lower() == "tools" and value.lower() in {"true", "yes", "1"}:
            tools_only = True
        elif separator and key.lower() == "ctx":
            context_min = _parse_context_limit(value)
        elif separator and key.lower() == "id":
            model_id = value
        else:
            text_parts.append(part)
    return PickerFilters(
        query=" ".join(text_parts),
        free_only=free_only,
        verified_only=verified_only,
        context_min=context_min,
        tools_only=tools_only,
        model_id=model_id,
    )


def _route_has_tools(rc: RouteChoice) -> bool:
    return rc.tools


def _format_route(rc: RouteChoice, choices: Sequence[RouteChoice] | None = None) -> str:
    name = rc.model_name or rc.model_id
    if rc.is_free:
        name += " [FREE]"
    facts: list[str] = []
    facts.append(f"provider {rc.provider_name or rc.provider_id}")
    facts.append(f"endpoint {rc.endpoint_id}")
    if rc.context_tokens:
        facts.append(f"{rc.context_tokens:,} ctx")
    if rc.certified:
        facts.append("verified")
    else:
        facts.append("unverified")
    return f"{name} — " + " · ".join(facts)


class ModelPicker:
    """Shared provider-aware model picker with TTY and line-mode paths."""

    def __init__(self, request: ModelPickerRequest) -> None:
        self.request = request

    def pick(self) -> ModelPickerResult:
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                return self._pick_tty()
            except (ImportError, EOFError, KeyboardInterrupt):
                pass
        return self._pick_line()

    def _pick_tty(self) -> ModelPickerResult:
        from prompt_toolkit.completion import Completer, Completion  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.shortcuts import PromptSession  # pyright: ignore[reportMissingImports]

        choices = list(self.request.choices)
        # Keep the completion text human-readable.  The old implementation
        # inserted RouteKey.serialize() into the input buffer, which exposed
        # internal unit-separator bytes and made an ordinary typed model name
        # behave like Back.  Route identity stays in this map, not in the UI.
        def input_label(rc: RouteChoice) -> str:
            return f"{rc.model_id} [{rc.provider_id}/{rc.endpoint_id}]"

        by_label = {input_label(rc): rc for rc in choices}
        picker = self

        class RouteCompleter(Completer):
            def get_completions(self, document, _complete_event):
                query = document.text
                for rc in _rank_route_matches(query, choices):
                    if not picker.request.allow_uncertified and not rc.certified:
                        continue
                    label = input_label(rc)
                    yield Completion(label, display=_format_route(rc, choices),
                                     display_meta=rc.endpoint_id)

        session = PromptSession(completer=RouteCompleter(), complete_while_typing=True)
        try:
            selected = session.prompt(
                f"{choices[0].provider_name} models for {self.request.purpose}: ",
                bottom_toolbar="↑/↓ move · Enter select · Ctrl-U clear · b back · q cancel",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return ModelPickerResult(action=NavigationAction.CANCEL)
        if selected.lower() == "b":
            return ModelPickerResult(action=NavigationAction.BACK)
        if selected.lower() == "q":
            return ModelPickerResult(action=NavigationAction.CANCEL)
        if selected in by_label:
            return ModelPickerResult(route=by_label[selected])
        exact_ids = [
            rc for rc in _rank_route_matches(selected, choices)
            if rc.model_id == selected or rc.model_name == selected
        ]
        if len(exact_ids) == 1:
            return ModelPickerResult(route=exact_ids[0])
        return ModelPickerResult()

    def _pick_line(self) -> ModelPickerResult:
        choices = list(self.request.choices)
        page = 0
        filters = self.request.filters
        while True:
            query = filters.query
            ranked = _rank_route_matches(
                " ".join(filter(None, (
                    query,
                    "free:true" if filters.free_only else "",
                    "verified:true" if filters.verified_only else "",
                    f"ctx:>={filters.context_min}" if filters.context_min else "",
                    "tools:true" if filters.tools_only else "",
                    f"id:{filters.model_id}" if filters.model_id else "",
                ))), choices,
            )
            if not self.request.allow_uncertified:
                ranked = [rc for rc in ranked if rc.certified]
            total_pages = max(1, (len(ranked) + _PAGE_SIZE - 1) // _PAGE_SIZE)
            page = min(page, total_pages - 1)
            start = page * _PAGE_SIZE
            visible = ranked[start:start + _PAGE_SIZE]
            provider_name = choices[0].provider_name if choices else self.request.provider_id
            status = choices[0].catalog_status if choices else "unavailable"
            print(f"\n{provider_name} models for {self.request.purpose} — {len(ranked)} matching, {status} catalog")
            print(f"Filter: {query or '(none)'}")
            for index, rc in enumerate(visible, start + 1):
                marker = ">" if self.request.current_route == rc.route_key() else " "
                print(f"{marker} {index}. {_format_route(rc, choices)}")
            print("Commands: /text filter · c clear · n/p page · f free-only · v verified-only · b back · q cancel")
            raw = _prompt("Choose/filter").strip()
            lowered = raw.lower()
            if lowered == "b":
                return ModelPickerResult(action=NavigationAction.BACK)
            if lowered == "q":
                return ModelPickerResult(action=NavigationAction.CANCEL)
            if lowered == "c":
                filters = PickerFilters()
                page = 0
                continue
            if lowered == "f":
                filters = PickerFilters(**{**filters.__dict__, "free_only": not filters.free_only})
                page = 0
                continue
            if lowered == "v":
                filters = PickerFilters(**{**filters.__dict__, "verified_only": not filters.verified_only})
                page = 0
                continue
            if lowered == "n":
                page = min(page + 1, total_pages - 1)
                continue
            if lowered == "p":
                page = max(page - 1, 0)
                continue
            if raw.startswith("/"):
                filters = _parse_picker_filters(raw[1:])
                page = 0
                continue
            if raw.isdigit() and 1 <= int(raw) <= len(ranked):
                return ModelPickerResult(route=ranked[int(raw) - 1])
            if raw:
                filters = _parse_picker_filters(raw)
                page = 0


def _rank_query_matches(query: str, options: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Rank *options* against *query*, best match first.

    A plain substring filter treats "starts with the query" the same as
    "the query happens to appear buried in the description" -- with a
    provider-discovered catalog of hundreds of models, that buries the
    model the operator is actually typing towards under noise. Rank
    instead: exact ID match, then ID-prefix, then ID-substring, then
    label-substring; ties keep the original (already-sorted-by-ID) order.
    """
    query = query.lower()
    if not query:
        return list(options)

    def _rank(item: tuple[str, str]) -> int:
        model_id, label = item
        model_id_lower = model_id.lower()
        if model_id_lower == query:
            return 0
        if model_id_lower.startswith(query):
            return 1
        if query in model_id_lower:
            return 2
        if query in label.lower():
            return 3
        return 4

    ranked = [(index, item) for index, item in enumerate(options) if _rank(item) < 4]
    ranked.sort(key=lambda pair: (_rank(pair[1]), pair[0]))
    return [item for _, item in ranked]


def _distinct_providers(options: list[tuple[str, str]]) -> list[str]:
    """Extract each distinct 'provider=X' token from _model_choices labels,
    in first-seen order, for the provider-first filtering step."""
    seen: list[str] = []
    for _, label in options:
        for segment in label.split("; "):
            if segment.startswith("provider="):
                provider = segment[len("provider="):]
                if provider not in seen:
                    seen.append(provider)
    return seen


def _choose_model(
    label: str, options: list[tuple[str, str]], default_id: str = ""
) -> str:
    """Search a potentially large discovered catalog before choosing.

    Provider-first filtering narrows hundreds of discovered models down to
    one provider's handful before the operator has to type anything, and
    ranked search (see _rank_query_matches) puts the closest ID match
    first instead of preserving arbitrary catalog order.
    """
    providers = _distinct_providers(options)
    scoped = options
    if len(providers) > 1:
        provider_choice = _prompt(
            f"Filter {label} by provider ({', '.join(providers)}; blank = all)"
        ).strip()
        if provider_choice:
            narrowed = [item for item in options if f"provider={provider_choice}" in item[1]]
            if narrowed:
                scoped = narrowed
            else:
                print(f"No models from provider '{provider_choice}'; showing all providers instead.")
    while True:
        query = _prompt(f"Search {label} (blank lists all)").strip().lower()
        filtered = scoped if not query else _rank_query_matches(query, scoped)
        if not filtered:
            print(f"No matches for '{query}'. Press Enter with nothing typed to list all {len(scoped)} options.")
            continue
        default = next(
            (index for index, item in enumerate(filtered, 1) if item[0] == default_id),
            1,
        )
        return _choose(label, filtered, default)


def _model_provider(registry: ModelRegistry, model_id: str, endpoint: str = "auto") -> tuple[str, str, str, bool]:
    model = registry.get_model(model_id)
    candidates = []
    if endpoint != "auto" and endpoint in model.endpoints:
        candidates = [model.endpoints[endpoint]]
    else:
        candidates = list(model.endpoints.values()) or [model]
    for candidate in candidates:
        provider_id = getattr(candidate, "provider_id", None) or model.provider_id or "local"
        provider = registry.providers.get(provider_id)
        key_env = getattr(candidate, "api_key_env", None) or model.api_key_env
        if provider is not None:
            key_env = key_env or provider.api_key_env
        configured = _credential_configured(key_env, registry.config_dir)
        backend = getattr(candidate, "backend", model.backend)
        if backend == "direct-anthropic":
            configured = configured and bool(
                getattr(candidate, "api_base", None)
                or model.api_base
                or getattr(candidate, "api_base_env", None)
                or model.api_base_env
            )
        if backend == "litellm":
            configured = configured and bool(
                getattr(candidate, "litellm_model", None) or model.litellm_model
            )
        if configured:
            return provider_id, backend, key_env or "no key required", True
    provider_id = model.provider_id or "local"
    return provider_id, model.backend, model.api_key_env or "no key required", False


def _model_choices(
    registry: ModelRegistry,
    *,
    role: str | None = None,
    controller: bool = False,
) -> list[tuple[str, str]]:
    """List selectable models for a role (or the controller).

    A freshly-discovered model (see registry.apply_discovered_catalog) starts
    with an empty ``allowed_roles`` set, and ``capabilities.controller_eligible
    = False``, by design -- discovery never grants roles or controller
    eligibility on its own. Without special-casing that here, such a model
    could never be searched or selected for ANY role or as the controller:
    the filters below would always exclude it, and nothing else ever grants
    it. Treat "not yet decided" as offerable; picking it is what grants the
    role or controller eligibility (see configure_inference /
    _grant_controller_eligible). Once a model has been explicitly restricted
    to a non-empty set of roles, that restriction is still enforced normally.
    """
    choices: list[tuple[str, str]] = []
    for model_id, model in sorted(registry.models.items(), key=lambda item: item[0]):
        if not model.enabled or model.backend == "anthropic-passthrough" and not controller:
            continue
        ungranted = False
        if controller:
            if not model.capabilities.controller_eligible:
                ungranted = True
        elif role is not None:
            if model.allowed_roles:
                if role not in model.allowed_roles:
                    continue
            else:
                ungranted = True
        provider, backend, key_env, configured = _model_provider(registry, model_id)
        if not configured:
            continue
        label = f"{model.display_name}; {backend}; provider={provider}; key={key_env}"
        if ungranted:
            suffix = "controller eligibility" if controller else "the role"
            label += f"; UNCERTIFIED -- selecting this grants it {suffix}"
        choices.append((model_id, label))
    return choices


def _grant_discovered_role(config_dir: Path, registry: ModelRegistry, model_id: str, role: str) -> None:
    """Grant *role* to a discovered model that has no roles assigned yet.

    Persisted into discovered_models.yaml (the same file
    Registry._persist_discovered_catalog writes) so the grant survives a
    future `refresh` re-running discovery -- that function preserves any
    existing allowed_roles/capabilities it finds there rather than
    overwriting them. Also updates the in-memory registry so the rest of
    the current wizard session sees the grant immediately.
    """
    path = config_dir / "discovered_models.yaml"
    raw = _read_yaml(path)
    models = raw.setdefault("models", {})
    entry = models.get(model_id)
    if not isinstance(entry, dict):
        return
    roles = list(entry.get("allowed_roles") or [])
    if role not in roles:
        roles.append(role)
    entry["allowed_roles"] = roles
    if role in {"implementer", "repairer"}:
        caps = dict(entry.get("capabilities") or {})
        caps["mutation"] = True
        entry["capabilities"] = caps
    _write_yaml(path, raw, validate_config_dir=config_dir)

    spec = registry.models.get(model_id)
    if spec is not None:
        updates: dict[str, Any] = {"allowed_roles": spec.allowed_roles | {role}}
        if role in {"implementer", "repairer"}:
            updates["capabilities"] = spec.capabilities.model_copy(update={"mutation": True})
        registry.models[model_id] = spec.model_copy(update=updates)


def _grant_controller_eligible(config_dir: Path, registry: ModelRegistry, model_id: str) -> None:
    """Grant controller eligibility to a model that isn't certified for it.

    Same persistence pattern as _grant_discovered_role: written into
    discovered_models.yaml so it survives a future `refresh`, and reflected
    in-memory immediately. The launcher (bin/claude-brigade) independently
    re-checks `capabilities.controller_eligible or backend ==
    "anthropic-passthrough"` before it will actually launch with this model
    as the controller, so this grant is what makes that check pass -- without
    it, a model picked here would be rejected at launch time with "is not
    controller-compatible" even though the wizard let you select it.
    """
    path = config_dir / "discovered_models.yaml"
    raw = _read_yaml(path)
    models = raw.setdefault("models", {})
    entry = models.get(model_id)
    if isinstance(entry, dict):
        caps = dict(entry.get("capabilities") or {})
        caps["controller_eligible"] = True
        entry["capabilities"] = caps
        _write_yaml(path, raw, validate_config_dir=config_dir)

    spec = registry.models.get(model_id)
    if spec is not None:
        registry.models[model_id] = spec.model_copy(
            update={"capabilities": spec.capabilities.model_copy(update={"controller_eligible": True})}
        )


def _target_label(role: str | None, controller: bool) -> str:
    return "controller" if controller else str(role)


def _record_certification(
    config_dir: Path,
    model_id: str,
    target: str,
    probe_record: dict[str, Any],
    *,
    provider_id: str | None = None,
    endpoint_id: str = "auto",
    protocol_version: str = "model-probe-v1",
    legacy_alias: bool = False,
) -> None:
    """Persist probe evidence against the exact route and target."""
    path = config_dir / "model_certifications.yaml"
    raw = _read_yaml(path)
    certifications = raw.setdefault("certifications", {})
    per_model = certifications.setdefault(model_id, {})
    provider_id = provider_id or "unknown-provider"
    per_provider = per_model.setdefault(provider_id, {})
    per_endpoint = per_provider.setdefault(endpoint_id, {})
    per_endpoint[target] = {
        **probe_record,
        "provider_id": provider_id,
        "endpoint_id": endpoint_id,
        "target": target,
        "protocol_version": protocol_version,
    }
    # Read compatibility for pre-route callers only. New writes always carry
    # the route-qualified evidence above and never overwrite another route.
    if legacy_alias:
        per_model[target] = probe_record
    _write_yaml(path, raw, validate_config_dir=config_dir)


def _record_override(
    config_dir: Path,
    model_id: str,
    target: str,
    reason: str,
    *,
    provider_id: str | None = None,
    endpoint_id: str = "auto",
    legacy_alias: bool = False,
) -> None:
    """Persist an explicit, unverified operator decision to model_overrides.yaml.

    Distinct from model_certifications.yaml on purpose: this is a decision
    the operator made without passing evidence, not a claim that the model
    was tested and works.
    """
    path = config_dir / "model_overrides.yaml"
    raw = _read_yaml(path)
    overrides = raw.setdefault("overrides", {})
    per_model = overrides.setdefault(model_id, {})
    provider_id = provider_id or "unknown-provider"
    per_endpoint = per_model.setdefault(provider_id, {}).setdefault(endpoint_id, {})
    record = {
        "reason": reason,
        "provider_id": provider_id,
        "endpoint_id": endpoint_id,
        "granted_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    per_endpoint[target] = record
    if legacy_alias:
        per_model[target] = record
    _write_yaml(path, raw, validate_config_dir=config_dir)


def _apply_grant(config_dir: Path, registry: ModelRegistry, model_id: str, *, role: str | None, controller: bool) -> None:
    if controller:
        _grant_controller_eligible(config_dir, registry, model_id)
    elif role is not None:
        _grant_discovered_role(config_dir, registry, model_id, role)


def _apply_staged_grants(config_dir: Path, registry: ModelRegistry, grants: list[dict]) -> None:
    """Apply all staged grants from a setup session atomically.

    Each grant dict must have 'model_id', and either 'role' or 'controller'.
    """
    for grant in grants:
        _apply_grant(
            config_dir, registry, grant["model_id"],
            role=grant.get("role"), controller=grant.get("controller", False),
        )


def _confirm_and_grant(
    config_dir: Path, registry: ModelRegistry, model_id: str, *,
    role: str | None = None, controller: bool = False,
    endpoint_id: str = "auto", provider_id: str | None = None,
    staged_grants: list[dict] | None = None,
) -> bool:
    """Gate an uncertified model behind an explicit test-or-override choice.

    Returns True if the model is now usable for this target (already was,
    just got certified, or was explicitly overridden). Returns False if the
    operator backed out, so the caller should let them pick a different
    model instead of silently proceeding.

    When *staged_grants* is provided (a mutable list), grants are appended
    to the list rather than applied immediately. The caller applies them all
    at once via _apply_staged_grants(). This prevents back/cancel from
    leaving accidental authority grants behind.
    """
    target = _target_label(role, controller)
    spec = registry.get_model(model_id)
    route_provider_id = provider_id
    if not route_provider_id:
        endpoint = spec.endpoints.get(endpoint_id) if endpoint_id != "auto" else None
        route_provider_id = (
            getattr(endpoint, "provider_id", None) if endpoint is not None else None
        ) or spec.provider_id or "local"
    already_ok = spec.capabilities.controller_eligible if controller else (role in spec.allowed_roles if role else True)
    if already_ok:
        return True

    print(
        f"\n'{model_id}' is unverified for {target}. Discovery never grants roles or "
        "controller eligibility on its own -- pick how to proceed:"
    )
    print("  t) Run a compatibility test now (one real API call, uses your credentials)")
    print("  o) Override without testing (explicit, at your own risk)")
    print("  <anything else>) Cancel and pick a different model")
    choice = _prompt("Choice").strip().lower()

    if choice == "t":
        from enhanced_router.model_probe import probe_model

        print(f"Probing '{model_id}' (endpoint: {endpoint_id}) for {target} compatibility...")
        result = probe_model(model_id, spec, endpoint_id=endpoint_id, config_dir=str(config_dir))
        _record_certification(
            config_dir, model_id, target, result.to_record(),
            provider_id=route_provider_id, endpoint_id=endpoint_id,
            legacy_alias=provider_id is None,
        )
        if result.passed:
            print(f"Passed: {model_id} demonstrated tool-call support for {target}.")
            if staged_grants is not None:
                staged_grants.append({
                    "model_id": model_id, "role": role, "controller": controller,
                })
                print(f"(Grant for {target} will be saved when you confirm the profile.)")
            else:
                _apply_grant(config_dir, registry, model_id, role=role, controller=controller)
            return True
        print(f"Failed: {result.error or 'no tool call observed'}.")
        if _prompt("Override and use it anyway despite the failed test? [y/N]").strip().lower() != "y":
            return False
        choice = "o"

    if choice == "o":
        reason = _prompt("Reason for overriding without certification", "operator decision")
        confirm = _prompt(
            f"Type OVERRIDE to confirm using an unverified model for {target} "
            "(it may fail unpredictably, including mid-mutation)"
        ).strip()
        if confirm != "OVERRIDE":
            print("Not confirmed; cancelled.")
            return False
        _record_override(
            config_dir, model_id, target, reason,
            provider_id=route_provider_id, endpoint_id=endpoint_id,
            legacy_alias=provider_id is None,
        )
        if staged_grants is not None:
            staged_grants.append({
                "model_id": model_id, "role": role, "controller": controller,
            })
            print(f"(Grant for {target} will be saved when you confirm the profile.)")
        else:
            _apply_grant(config_dir, registry, model_id, role=role, controller=controller)
        return True

    return False


def _profile_model(profile: dict[str, Any], role: str) -> tuple[str, str, list[str]]:
    raw = profile.get(role, "")
    if isinstance(raw, dict):
        fallback_ids = []
        fallback_items = raw.get("fallback_routes")
        if fallback_items is None:
            fallback_items = raw.get("fallback_models", [])
        for item in fallback_items:
            if isinstance(item, str):
                fallback_ids.append(item)
            elif isinstance(item, dict) and isinstance(item.get("model"), str):
                fallback_ids.append(item["model"])
        return str(raw.get("model", "")), str(raw.get("endpoint", "auto")), fallback_ids
    return str(raw), "auto", []


def _profile_fallback_routes(profile: dict[str, Any], role: str) -> list[dict]:
    """Like _profile_model's third element, but keeps each fallback's own
    endpoint (not just its model ID) for handing to _edit_fallbacks."""
    raw = profile.get(role, "")
    if not isinstance(raw, dict):
        return []
    routes = []
    fallback_items = raw.get("fallback_routes")
    if fallback_items is None:
        fallback_items = raw.get("fallback_models", [])
    for item in fallback_items:
        if isinstance(item, str):
            routes.append({"model": item, "endpoint": "auto"})
        elif isinstance(item, dict) and isinstance(item.get("model"), str):
            route = {
                "model": item["model"],
                "endpoint": str(item.get("endpoint", "auto")),
            }
            if item.get("provider_id"):
                route["provider_id"] = str(item["provider_id"])
            routes.append(route)
    return routes


def _edit_fallbacks(
    config_dir: Path,
    registry: ModelRegistry,
    *,
    role: str | None,
    controller: bool = False,
    fallback_options: list[tuple[str, str]],
    previous: list[dict],
    fallback_choices: Sequence[RouteChoice] | None = None,
    staged_grants: list[dict] | None = None,
) -> list[dict]:
    """Interactive add/remove/reorder editor for a route's fallback ladder.

    Each fallback is a full {model, endpoint} pair (RouteCandidateSpec): a
    fallback can pin a different provider/endpoint than the primary route,
    not just fall back to the same model under 'auto'. Replaces a bare
    comma-separated free-text prompt, which had no way to express a
    per-fallback endpoint, reorder the ladder, or browse/search candidates.
    """
    fallbacks = [dict(item) for item in previous if isinstance(item, dict) and item.get("model")]
    if not fallback_options:
        return fallbacks
    target = _target_label(role, controller)
    while True:
        fallback_display_options = [
            (str(item["model"]), f"endpoint: {item.get('endpoint', 'auto')}")
            for item in fallbacks
        ]
        _render_options(
            f"Fallback ladder for {target}",
            fallback_display_options,
            navigation=[
                "a) add fallback",
                "r) remove",
                "m) move",
                "d) done",
            ],
        )
        if not fallbacks and not _rich_interactive():
            print("  (none)")
        action = _prompt("a) add fallback   r) remove   m) move   d) done").strip().lower()
        if action == "a":
            if fallback_choices is not None:
                used = {
                    (item.get("provider_id"), item.get("model"), item.get("endpoint", "auto"))
                    for item in fallbacks
                }
                candidates = [
                    rc for rc in fallback_choices
                    if (rc.provider_id, rc.model_id, rc.endpoint_id) not in used
                ]
                if not candidates:
                    print("No more distinct routes are available to add.")
                    continue
                provider_result = choose_provider(candidates, purpose=f"Fallback #{len(fallbacks) + 1} for {target}")
                if isinstance(provider_result, NavResult):
                    continue
                route_result = choose_route(
                    candidates,
                    provider_id=provider_result,
                    purpose=f"Fallback #{len(fallbacks) + 1} for {target}",
                )
                if isinstance(route_result, NavResult):
                    continue
                rc = route_result
                if not _confirm_assignment(config_dir, registry, rc, target, staged_grants or []):
                    continue
                fallbacks.append({
                    "model": rc.model_id,
                    "endpoint": rc.endpoint_id,
                    "provider_id": rc.provider_id,
                })
                continue
            already_used = {item["model"] for item in fallbacks}
            candidates = [item for item in fallback_options if item[0] not in already_used]
            if not candidates:
                print("No more distinct models are available to add.")
                continue
            model_id = _choose_model(f"Fallback #{len(fallbacks) + 1} for {target}", candidates)
            if not _confirm_and_grant(config_dir, registry, model_id, role=role, controller=controller):
                print(f"Dropping '{model_id}' (not confirmed).")
                continue
            model = registry.get_model(model_id)
            endpoints = [("auto", "registry endpoint selection")]
            endpoints.extend(
                (endpoint_id, f"configured {endpoint.backend} endpoint")
                for endpoint_id, endpoint in sorted(model.endpoints.items())
            )
            endpoint = _choose(f"Endpoint for fallback '{model_id}'", endpoints)
            fallbacks.append({"model": model_id, "endpoint": endpoint})
        elif action == "r":
            if not fallbacks:
                print("No fallbacks to remove.")
                continue
            index = _prompt_int("Remove which number", 1, 1, len(fallbacks))
            removed = fallbacks.pop(index - 1)
            print(f"Removed '{removed['model']}'.")
        elif action == "m":
            if len(fallbacks) < 2:
                print("Need at least two fallbacks to reorder.")
                continue
            source = _prompt_int("Move which number", 1, 1, len(fallbacks))
            destination = _prompt_int("Move to which position", source, 1, len(fallbacks))
            item = fallbacks.pop(source - 1)
            fallbacks.insert(destination - 1, item)
        elif action == "d":
            return fallbacks
        else:
            print("Choose a, r, m, or d.")


def _choose_or_create_id(kind: str, existing: list[str], default_new: str) -> str:
    """Numbered picker over existing saved *kind* entries, with a 'create
    new' option -- replaces a free-text name prompt that silently defaulted
    to the first existing entry and gave no browsable list beyond one
    printed line.
    """
    if not existing:
        name = _prompt(f"{kind} name", default_new)
    else:
        options = [(item, "existing") for item in existing] + [("(new)", "create a new one")]
        choice = _choose(f"Select a {kind} to edit, or create a new one", options)
        name = _prompt(f"New {kind} name", default_new) if choice == "(new)" else choice
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"{kind} IDs may contain letters, numbers, '.', '_' and '-'")
    return name


def _role_label(profile: dict, role: str) -> str:
    """Format a role's current assignment for the dashboard display."""
    raw = profile.get(role, "")
    if isinstance(raw, dict):
        model = raw.get("model", "(unset)")
        endpoint = raw.get("endpoint", "auto")
        provider = raw.get("provider_id") or "default provider"
        fallback_items = raw.get("fallback_routes")
        if fallback_items is None:
            fallback_items = raw.get("fallback_models") or []
        fallback_count = len(fallback_items) if isinstance(fallback_items, list) else 0
    elif isinstance(raw, str) and raw:
        model = raw
        endpoint = "auto"
        provider = "default provider"
        fallback_count = 0
    else:
        return "(unset)"
    suffix = f" · {fallback_count} fallback{'s' if fallback_count != 1 else ''}"
    return f"{model} · {provider} · endpoint: {endpoint}{suffix}"


def configure_inference(config_dir: Path, registry: ModelRegistry) -> EditorResult:
    raw = _read_yaml(config_dir / "profiles.yaml")
    profiles = raw.setdefault("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
        raw["profiles"] = profiles
    existing = sorted(str(item) for item in profiles)
    profile_id = _choose_or_create_id("Inference profile", existing, "my-profile")
    current = dict(profiles.get(profile_id) or {})

    staged_grants: list[dict] = []
    print(
        "\nThis edits the controller route and the durable worker-role routes "
        "(recon, implementer, adversary, repairer).\n"
        "Native Claude Code model lanes (main, background, haiku, sonnet, opus, fable, custom) "
        "are edited separately in this profile. Additional named native "
        "sidecars and bounded coprocessors are configured in the Sidecar lane."
    )

    while True:
        role_entries = [
            ("1", f"Controller — {_format_controller(model_id=current.get('controller_model','') or (current.get('controller') or {}).get('model','') or '(unset)')}"),
        ]
        role_entries.extend(
            (str(idx), f"{role.capitalize()} — {_role_label(current, role)}")
            for idx, role in enumerate(_ROLES, start=2)
        )
        _render_menu(
            f"Controller + worker-role profile: {profile_id}",
            [
                ("Role routes", role_entries),
                (
                    "Profile actions",
                    [
                        (str(len(_ROLES) + 2), "Review fallback ladders"),
                        (str(len(_ROLES) + 3), "Edit Claude Code model lanes (main/background/haiku/sonnet/opus/fable/custom)"),
                        ("s", "Save profile"),
                        ("b", "Back without saving"),
                    ],
                ),
            ],
            footer="Choose a role to route it to a provider/model/endpoint",
        )

        choice = _menu_prompt("Choose").strip().lower()
        if choice == "s":
            profiles[profile_id] = dict(current)
            _write_yaml(
                config_dir / "profiles.yaml", raw,
                validate_config_dir=config_dir,
            )
            if staged_grants:
                _apply_staged_grants(config_dir, registry, staged_grants)
            print(f"Saved inference profile '{profile_id}' to {config_dir / 'profiles.yaml'}.")
            print(f"Use it with: claude-brigade --brigade-profile {profile_id}")
            return EditorResult(profile_id, "saved")
        if choice in ("b", "q"):
            if staged_grants:
                print(f"Discarding {len(staged_grants)} pending grant(s) that were not saved.")
            print("Cancelled.")
            return EditorResult(None, "cancelled")

        if choice == str(len(_ROLES) + 2):
            # Edit fallback ladders
            _edit_all_fallbacks(config_dir, registry, current, staged_grants)
            continue
        if choice == str(len(_ROLES) + 3):
            _edit_native_model_slots(config_dir, registry, current, staged_grants)
            continue

        try:
            role_idx = int(choice) - 1
        except ValueError:
            print("Enter 1-7, s, or b.")
            continue
        if role_idx == 0:
            # Controller
            _edit_role_route(config_dir, registry, current, None, True,
                             None, staged_grants)
        elif 1 <= role_idx <= len(_ROLES):
            role = _ROLES[role_idx - 1]
            _edit_role_route(config_dir, registry, current, role, False,
                             None, staged_grants)
        else:
            print("Enter 1-7, s, or b.")


def _edit_native_model_slots(
    config_dir: Path,
    registry: ModelRegistry,
    current: dict[str, Any],
    staged_grants: list[dict],
) -> None:
    """Edit the Claude Code backing-model lanes in a profile."""
    slots = ("main", "background", "haiku", "sonnet", "opus", "fable", "custom")
    slot_roles = {
        "main": (None, True),
        "background": ("recon", False),
        "sonnet": ("implementer", False),
        "haiku": ("recon", False),
        "opus": ("adversary", False),
        "fable": ("adversary", False),
        "custom": (None, True),
    }
    slot_defaults = {
        "main": "controller",
        "background": "optional",
        "sonnet": "worker",
        "haiku": "worker",
        "opus": "worker",
        "fable": "critical",
        "custom": "optional",
    }
    slot_data = current.setdefault("slots", {})
    if not isinstance(slot_data, dict):
        slot_data = {}
        current["slots"] = slot_data
    while True:
        options: list[tuple[str, str]] = []
        for slot in slots:
            raw_entry = slot_data.get(slot)
            entry: dict[str, Any] = raw_entry if isinstance(raw_entry, dict) else {}
            options.append((slot, f"{entry.get('model', '(unset)')} · {entry.get('endpoint', 'auto')}"))
        _render_options(
            "Claude Code model slots",
            options,
            navigation=["number to edit", "d) done", "b) back"],
        )
        choice = _menu_prompt("Choose slot number, d when done, b to go back").strip().lower()
        if choice == "d":
            return
        if choice == "b":
            return
        if not choice.isdigit() or not 1 <= int(choice) <= len(slots):
            _status(f"Enter a number 1-{len(slots)}, d, or b.", "yellow")
            continue
        slot = slots[int(choice) - 1]
        role, controller = slot_roles[slot]
        choices = generate_route_choices(registry, role=role, controller=controller)
        if slot == "sonnet":
            choices = [
                item for item in choices
                if registry.get_model(item.model_id).capabilities.write_tool_certified is True
            ]
        if not choices:
            _status(f"No eligible model routes are available for slot '{slot}'.", "yellow")
            continue
        raw_saved = slot_data.get(slot)
        saved: dict[str, Any] = raw_saved if isinstance(raw_saved, dict) else {}
        route = _pick_registry_route(
            registry,
            choices,
            purpose=f"Claude Code {slot} slot",
            current_model_id=str(saved.get("model") or ""),
            current_endpoint_id=str(saved.get("endpoint") or "auto"),
            current_provider_id=str(saved.get("provider_id") or ""),
        )
        if route is None:
            continue
        raw_previous = saved.get("fallbacks")
        previous: list[dict[str, Any]] = (
            [item for item in raw_previous if isinstance(item, dict)]
            if isinstance(raw_previous, list) else []
        )
        fallbacks = _edit_fallbacks(
            config_dir,
            registry,
            role=role,
            controller=controller,
            fallback_options=[],
            previous=previous,
            fallback_choices=choices,
            staged_grants=staged_grants,
        )
        slot_data[slot] = {
            "model": route.model_id,
            "endpoint": route.endpoint_id,
            "provider_id": route.provider_id,
            "fallbacks": fallbacks,
            "reserve_class": str(saved.get("reserve_class") or slot_defaults[slot]),
        }


def _edit_role_route(
    config_dir: Path, registry: ModelRegistry,
    current: dict, role: str | None, controller: bool,
    all_route_choices: list[RouteChoice] | None,
    staged_grants: list[dict],
) -> None:
    """Provider-first route picker for one role or controller.

    Walks provider → model → endpoint → confirm. Returns when the user
    confirms an assignment or navigates back without changing anything.
    """
    purpose = _target_label(role, controller)
    def target_choices() -> list[RouteChoice]:
        generated = (
            generate_route_choices(registry, controller=True)
            if controller else generate_route_choices(registry, role=role)
        )
        return [rc for rc in generated if rc.credential_configured]

    role_choices = target_choices()

    # Get current assignment
    current_model = ""
    current_endpoint = "auto"
    saved_provider_id = ""
    if controller:
        controller_entry = current.get("controller") or {}
        current_model = current.get("controller_model", "") or controller_entry.get("model", "")
        saved_provider_id = str(controller_entry.get("provider_id") or "")
        current_endpoint = str(controller_entry.get("endpoint", "auto"))
    elif role:
        raw = current.get(role, {})
        if isinstance(raw, dict):
            current_model = raw.get("model", "")
            current_endpoint = raw.get("endpoint", "auto")
            saved_provider_id = str(raw.get("provider_id") or "")
        elif isinstance(raw, str):
            current_model = raw

    while True:
        # Step 1: choose provider
        current_provider = next(
            (rc.provider_id for rc in role_choices
             if rc.model_id == current_model
             and _saved_endpoint_matches(rc.endpoint_id, current_endpoint)
             and (not saved_provider_id or rc.provider_id == saved_provider_id)),
            None,
        )
        def refresh_target() -> Sequence[RouteChoice]:
            refresh_catalogs(registry, force=True)
            registry.load_models()
            return target_choices()

        try:
            if current_provider:
                prov_result = choose_provider(
                    role_choices, purpose=purpose, current_provider_id=current_provider,
                    on_refresh=refresh_target,
                )
            else:
                prov_result = choose_provider(
                    role_choices, purpose=purpose, on_refresh=refresh_target,
                )
        except TypeError as exc:
            # Compatibility for callers that replace the provider picker with
            # the pre-refresh two-argument test seam.
            if "on_refresh" not in str(exc):
                raise
            if current_provider:
                prov_result = choose_provider(
                    role_choices, purpose=purpose,
                    current_provider_id=current_provider,
                )
            else:
                prov_result = choose_provider(role_choices, purpose=purpose)
        role_choices = target_choices()
        if isinstance(prov_result, NavResult):
            return  # back/cancel
        provider_id = prov_result

        # Step 2: choose route (model + endpoint)
        current_route = next(
            (rc.route_key() for rc in role_choices
             if rc.provider_id == provider_id and rc.model_id == current_model
             and _saved_endpoint_matches(rc.endpoint_id, current_endpoint)),
            None,
        )
        route_kwargs: dict[str, Any] = {
            "provider_id": provider_id,
            "purpose": purpose,
            "current_model_id": current_model if current_model else None,
        }
        if current_route is not None:
            route_kwargs["current_route"] = current_route
        route_result = choose_route(role_choices, **route_kwargs)
        if isinstance(route_result, NavResult):
            if route_result.action == NavigationAction.BACK:
                continue  # back to providers
            return  # cancel

        rc = route_result

        # Step 3: confirm
        if _confirm_assignment(config_dir, registry, rc, purpose, staged_grants):
            # Changing the primary route must not silently destroy the
            # operator's existing ladder.  Preserve every fallback except an
            # exact duplicate of the newly selected provider/model/endpoint.
            previous_fallbacks = _profile_fallback_routes(
                current, "controller" if controller else str(role)
            )
            preserved_fallbacks = [
                item for item in previous_fallbacks
                if not (
                    item.get("model") == rc.model_id
                    and item.get("endpoint", "auto") == rc.endpoint_id
                    and (
                        not item.get("provider_id")
                        or item.get("provider_id") == rc.provider_id
                    )
                )
            ]
            # Save to draft
            if controller:
                current.pop("controller_model", None)
                current["controller"] = {
                    "model": rc.model_id,
                    "endpoint": rc.endpoint_id,
                    "provider_id": rc.provider_id,
                    "fallback_models": preserved_fallbacks,
                    "fallback_routes": preserved_fallbacks,
                }
            elif role:
                current[role] = {
                    "model": rc.model_id,
                    "endpoint": rc.endpoint_id,
                    "provider_id": rc.provider_id,
                    "fallback_models": preserved_fallbacks,
                    "fallback_routes": preserved_fallbacks,
                }
            print(f"Assigned {rc.model_id} via {rc.provider_name}/{rc.endpoint_id} to {purpose}.")
            return


def _confirm_assignment(
    config_dir: Path, registry: ModelRegistry,
    rc: RouteChoice, purpose: str,
    staged_grants: list[dict],
) -> bool:
    """Show the assignment details and ask the user to confirm.

    Probes the exact endpoint if certification is needed. Returns True
    when the assignment is accepted.
    """
    # Map purpose back to role/controller
    controller = purpose == "controller"
    role = None if controller else purpose

    print(f"\n── Assign {purpose.capitalize()} ──")
    print(f"  Provider:      {rc.provider_name}")
    print(f"  Model:         {rc.model_name}")
    print(f"  Model ID:      {rc.model_id}")
    print(f"  Endpoint:      {rc.endpoint_id}")
    print(f"  Certification: {'certified' if rc.certified else 'unverified'}")
    ctx = rc.context_tokens
    print(f"  Context:       {f'{ctx:,}' if ctx else 'unknown'}")
    print(f"  Backend:       {rc.backend}")

    print("\n  1. Assign")
    if not rc.certified:
        print("  2. Test compatibility first")
    print("  b. Back to models")
    print("  q. Cancel")

    choice = _menu_prompt("Choose").strip().lower()
    if choice == "1":
        if not rc.certified:
            # Go through _confirm_and_grant for the route
            return _confirm_and_grant(
                config_dir, registry, rc.model_id,
                role=role, controller=controller,
                endpoint_id=rc.endpoint_id, provider_id=rc.provider_id,
                staged_grants=staged_grants,
            )
        return True
    if choice == "2" and not rc.certified:
        return _confirm_and_grant(
            config_dir, registry, rc.model_id,
            role=role, controller=controller,
            endpoint_id=rc.endpoint_id, provider_id=rc.provider_id,
            staged_grants=staged_grants,
        )
    return False


def _format_controller(*, model_id: str = "") -> str:
    if model_id:
        return model_id
    return "(unset — uses Claude default)"


def _pick_registry_route(
    registry: ModelRegistry,
    choices: Sequence[RouteChoice],
    *,
    purpose: str,
    current_model_id: str = "",
    current_endpoint_id: str = "",
    current_provider_id: str = "",
) -> RouteChoice | None:
    """Run provider → model selection for non-role configuration surfaces."""
    choices = [rc for rc in choices if rc.credential_configured]
    if not choices:
        return None
    current_provider = next(
        (rc.provider_id for rc in choices
         if rc.model_id == current_model_id
         and _saved_endpoint_matches(rc.endpoint_id, current_endpoint_id)),
        None,
    )
    current_provider = current_provider_id or current_provider
    provider_kwargs: dict[str, Any] = {"purpose": purpose}
    if current_provider:
        provider_kwargs["current_provider_id"] = current_provider
    provider_result = choose_provider(choices, **provider_kwargs)
    if isinstance(provider_result, NavResult):
        return None
    current_route = next(
        (rc.route_key() for rc in choices
         if rc.provider_id == provider_result and rc.model_id == current_model_id
         and _saved_endpoint_matches(rc.endpoint_id, current_endpoint_id)),
        None,
    )
    route_kwargs: dict[str, Any] = {
        "provider_id": provider_result,
        "purpose": purpose,
        "current_model_id": current_model_id or None,
    }
    if current_route is not None:
        route_kwargs["current_route"] = current_route
    route_result = choose_route(choices, **route_kwargs)
    return route_result if isinstance(route_result, RouteChoice) else None


def _edit_all_fallbacks(
    config_dir: Path, registry: ModelRegistry,
    current: dict,
    staged_grants: list[dict],
) -> None:
    """Edit fallback ladders for all roles + controller."""
    targets = [("controller", None, True)] + [(r, r, False) for r in _ROLES]
    for label, role, controller in targets:
        model_id = ""
        if controller:
            model_id = current.get("controller_model", "") or (current.get("controller") or {}).get("model", "")
        else:
            raw = current.get(role, {})
            if isinstance(raw, dict):
                model_id = raw.get("model", "")
            elif isinstance(raw, str):
                model_id = raw
        if not model_id:
            continue
        if controller:
            all_route_choices = generate_route_choices(registry, controller=True)
        else:
            all_route_choices = generate_route_choices(registry, role=role)
        # Build fallback options excluding the exact primary route.
        current_entry = current.get("controller") if controller else current.get(role)
        route_key = next(
            (rc.route_key() for rc in all_route_choices
             if rc.model_id == model_id
             and _saved_endpoint_matches(
                 rc.endpoint_id,
                 current_entry.get("endpoint", "auto") if isinstance(current_entry, dict) else "auto",
             )
             and (not isinstance(current_entry, dict) or not current_entry.get("provider_id")
                  or rc.provider_id == current_entry.get("provider_id"))),
            None,
        )
        fallback_candidates = [
            rc for rc in all_route_choices
            if (route_key is None or rc.route_key() != route_key) and rc.credential_configured
        ]
        if not fallback_candidates:
            print(f"\n{label}: no fallback candidates available.")
            continue
        fb_options = [(rc.model_id, f"{rc.provider_name} · {rc.endpoint_id}")
                      for rc in fallback_candidates]
        previous = _profile_fallback_routes(current, label if not controller else "controller")
        try:
            fallback_routes = _edit_fallbacks(
                config_dir, registry, role=role, controller=controller,
                fallback_options=fb_options, previous=previous,
                fallback_choices=fallback_candidates,
                staged_grants=staged_grants,
            )
        except TypeError as exc:
            # Keep the small compatibility seam used by third-party callers
            # that monkeypatch the pre-ModelPicker helper.
            if "fallback_choices" not in str(exc) and "staged_grants" not in str(exc):
                raise
            fallback_routes = _edit_fallbacks(
                config_dir, registry, role=role, controller=controller,
                fallback_options=fb_options, previous=previous,
            )
        # Save back to current
        if controller:
            raw_controller = current.get("controller")
            ctrl_entry: dict[str, Any] = (
                raw_controller if isinstance(raw_controller, dict) else {
                    "model": model_id,
                    "endpoint": "auto",
                }
            )
            ctrl_entry["fallback_models"] = fallback_routes
            ctrl_entry["fallback_routes"] = fallback_routes
            current["controller"] = ctrl_entry
        else:
            existing = current.get(role, {})
            if isinstance(existing, dict):
                existing["fallback_models"] = fallback_routes
                existing["fallback_routes"] = fallback_routes
                current[role] = existing
            else:
                current[role] = {
                    "model": existing,
                    "endpoint": "auto",
                    "fallback_models": fallback_routes,
                    "fallback_routes": fallback_routes,
                }


def configure_fastpath(config_dir: Path, registry: ModelRegistry) -> None:
    """Configure the optional automatic fastpath coprocessor (fastpath.yaml).

    This is distinct from sidecars.yaml: the fastpath model runs
    When enabled, it may be invoked automatically via
    /internal/fastpath/route and /internal/fastpath/verify. A sidecars.yaml
    entry, by contrast, does nothing until a workflow phase explicitly opts
    in with `execution_kind: coprocessor_call` and `coprocessor: <id>`.
    """
    print("\n=== Sidecar & fastpath models: coprocessor ===")
    raw = _read_yaml(config_dir / "fastpath.yaml")
    fastpath = raw.setdefault("fastpath", {})
    if not isinstance(fastpath, dict):
        fastpath = {}
        raw["fastpath"] = fastpath
    enabled = _choose_toggle(
        "Fastpath coprocessor",
        bool(fastpath.get("enabled", True)),
        enabled_text="On — allow automatic route/verify calls",
        disabled_text="Off — bypass fastpath and use deterministic routing",
    )
    if not enabled:
        fastpath["enabled"] = False
        _write_yaml(config_dir / "fastpath.yaml", raw, validate_config_dir=config_dir)
        print(f"Fastpath coprocessor disabled in {config_dir / 'fastpath.yaml'}.")
        return
    # registry._validate_cross_refs() rejects a write-tool-certified fastpath
    # model outright ("fastpath model cannot be write-tool certified") -- but
    # only at the NEXT registry load. Without filtering here, the wizard
    # would happily save a choice that crashes the very next launch, the
    # same failure mode this whole session has been chasing.
    choices = [
        item for item in generate_route_choices(registry, role="recon")
        if not registry.get_model(item.model_id).capabilities.write_tool_certified
    ]
    if not choices:
        raise RuntimeError("No enabled credential-backed model is available for the fastpath coprocessor.")
    previous_model = str(fastpath.get("model_id") or "")
    route = _pick_registry_route(
        registry, choices, purpose="fastpath", current_model_id=previous_model,
        current_endpoint_id=str(fastpath.get("endpoint") or ""),
        current_provider_id=str(fastpath.get("provider_id") or ""),
    )
    if route is None:
        return
    fallback_routes = _edit_fallbacks(
        config_dir,
        registry,
        role="recon",
        fallback_options=[],
        previous=list(fastpath.get("fallback_routes") or []),
        fallback_choices=choices,
    )
    fastpath["model_id"] = route.model_id
    fastpath["endpoint"] = route.endpoint_id
    fastpath["provider_id"] = route.provider_id
    fastpath["fallback_routes"] = fallback_routes
    fastpath["fallback_models"] = [item["model"] for item in fallback_routes]
    fastpath["enabled"] = True
    fastpath.setdefault("modes", ["route", "verify"])
    # FastpathConfigSpec.timeout_seconds caps at 30 (gt=0, le=30) -- the
    # schema is the source of truth, not a second hardcoded bound here.
    fastpath["timeout_seconds"] = _prompt_float(
        "Fastpath timeout seconds", float(fastpath.get("timeout_seconds", 5)), 1, 30,
    )
    _write_yaml(config_dir / "fastpath.yaml", raw, validate_config_dir=config_dir)
    print(f"Saved fastpath coprocessor config to {config_dir / 'fastpath.yaml'}.")


def configure_native_sidecar_agent(config_dir: Path, registry: ModelRegistry) -> None:
    """Configure one tool-capable native sidecar worker.

    Native sidecars are model-backed Claude Code agents.  They are distinct
    from bounded coprocessors and from sidecar profiles, which only select
    already-defined workers for a launch.
    """
    print("\n=== Native sidecar worker: Claude Code agent with its own model ===")
    raw = _read_yaml(config_dir / "sidecars.yaml")
    agents = raw.setdefault("sidecar_agents", {})
    if not isinstance(agents, dict):
        agents = {}
        raw["sidecar_agents"] = agents

    existing = sorted(str(item) for item in agents)
    agent_id = _choose_or_create_id("Native sidecar worker", existing, "grounder")
    current = dict(agents.get(agent_id) or {})
    is_new = not current

    if is_new:
        role = _choose(
            "Primary sidecar role",
            [
                ("recon", "grounding and repository investigation"),
                ("implementer", "tool-capable implementation"),
                ("adversary", "read-only implementation review"),
                ("repairer", "tool-capable repair of accepted findings"),
            ],
            1,
        )
    else:
        raw_roles = current.get("roles")
        roles: list[str] = (
            [str(item) for item in raw_roles]
            if isinstance(raw_roles, list) else []
        )
        role = next((item for item in roles if item in _ROLES), "recon")

    mutating = bool(current.get("can_mutate")) or role in {"implementer", "repairer"}
    choices = generate_route_choices(registry, role=role)
    if mutating:
        choices = [
            item for item in choices
            if registry.get_model(item.model_id).capabilities.write_tool_certified
        ]
    if not choices:
        raise RuntimeError(
            f"No credential-backed model is available for native sidecar role '{role}'."
        )
    route = _pick_registry_route(
        registry,
        choices,
        purpose=f"native sidecar {agent_id}",
        current_model_id=str(current.get("model_id") or ""),
        current_endpoint_id=str(current.get("endpoint") or ""),
        current_provider_id=str(current.get("provider_id") or ""),
    )
    if route is None:
        return

    previous_fallbacks = current.get("fallback_routes") or [
        {"model": item, "endpoint": "auto"}
        for item in (current.get("fallback_models") or [])
        if isinstance(item, str)
    ]
    fallback_routes = _edit_fallbacks(
        config_dir,
        registry,
        role=role,
        fallback_options=[],
        previous=previous_fallbacks,
        fallback_choices=choices,
    )

    if is_new:
        current.update({
            "native_agent_name": f"brigade-{agent_id}",
            "public_model_alias": f"anthropic-brigade-{agent_id}",
            "roles": [role],
            "description": f"ClaudeBrigade {role} sidecar worker",
            "tools": ["Read", "Grep", "Glob", "Bash"]
            + (["Edit", "Write"] if mutating else []),
            "can_mutate": mutating,
            "isolation": "worktree" if mutating else "none",
            "background": True,
            "max_turns": 120 if mutating else 80,
            "effort": "high" if mutating else "medium",
        })
    current.update({
        "model_id": route.model_id,
        "endpoint": route.endpoint_id,
        "provider_id": route.provider_id,
        "fallback_routes": fallback_routes,
        "fallback_models": [item["model"] for item in fallback_routes],
    })
    agents[agent_id] = current
    _write_yaml(config_dir / "sidecars.yaml", raw, validate_config_dir=config_dir)
    print(f"Saved native sidecar worker '{agent_id}' to {config_dir / 'sidecars.yaml'}.")


def configure_coprocessor(config_dir: Path, registry: ModelRegistry) -> None:
    print("\n=== Coprocessor model: bounded structured MCP call ===")
    raw = _read_yaml(config_dir / "sidecars.yaml")
    sidecars = raw.setdefault("coprocessors", {})
    if not isinstance(sidecars, dict):
        sidecars = {}
        raw["coprocessors"] = sidecars
    existing = sorted(str(item) for item in sidecars)
    sidecar_id = _choose_or_create_id("Sidecar", existing, "verification_reviewer")
    current = dict(sidecars.get(sidecar_id) or {})
    enabled = _choose_toggle(
        f"Coprocessor '{sidecar_id}'",
        bool(current.get("enabled", True)),
        enabled_text="On — permit workflow and feedback calls",
        disabled_text="Off — keep configuration but do not make model calls",
    )
    if not enabled:
        current["enabled"] = False
        sidecars[sidecar_id] = current
        _write_yaml(config_dir / "sidecars.yaml", raw, validate_config_dir=config_dir)
        print(f"Coprocessor '{sidecar_id}' disabled in {config_dir / 'sidecars.yaml'}.")
        return
    choices = generate_route_choices(registry, role="recon")
    if not choices:
        raise RuntimeError("No enabled credential-backed sidecar model is available.")
    previous_model = str(current.get("model_id") or "")
    route = _pick_registry_route(
        registry, choices, purpose="sidecar", current_model_id=previous_model,
        current_endpoint_id=str(current.get("endpoint") or ""),
        current_provider_id=str(current.get("provider_id") or ""),
    )
    if route is None:
        return
    previous_fallbacks = current.get("fallback_routes") or []
    fallback_routes = _edit_fallbacks(
        config_dir,
        registry,
        role="recon",
        fallback_options=[],
        previous=previous_fallbacks,
        fallback_choices=choices,
    )
    mode = _choose(
        "Sidecar purpose",
        [("route", "route advisory"), ("verify", "verification review"), ("structured", "generic structured specialist")],
        next((i for i, item in enumerate(("route", "verify", "structured"), 1) if item == current.get("mode")), 3),
    )
    current.update({
        "model_id": route.model_id,
        "mode": mode,
        "endpoint": route.endpoint_id,
        "provider_id": route.provider_id,
        "fallback_routes": fallback_routes,
        "fallback_models": [item["model"] for item in fallback_routes],
        "enabled": True,
        "timeout_seconds": _prompt_float("Timeout seconds", float(current.get("timeout_seconds", 45)), 1, 600),
        "max_packet_bytes": _prompt_int("Maximum packet bytes", int(current.get("max_packet_bytes", 64_000)), 1_024, 256_000),
        "max_output_tokens": _prompt_int("Maximum output tokens", int(current.get("max_output_tokens", 2_048)), 64, 131_072),
    })
    current["system_prompt"] = _prompt("System prompt (optional)", str(current.get("system_prompt", "")))
    sidecars[sidecar_id] = current
    _write_yaml(config_dir / "sidecars.yaml", raw, validate_config_dir=config_dir)
    print(f"Saved coprocessor '{sidecar_id}' to {config_dir / 'sidecars.yaml'}.")
    print("Reference it from a workflow phase with: execution_kind: coprocessor_call and coprocessor: " + sidecar_id)


# Source compatibility for callers and older installed launchers. New code
# should use the unambiguous coprocessor name.
configure_sidecar = configure_coprocessor


def configure_global_coprocessor_lane(
    config_dir: Path,
    registry: ModelRegistry,
) -> None:
    """Toggle the global bounded-coprocessor lane.

    This is the master switch for launches that do not select a named
    sidecar profile.  It intentionally leaves native sidecar agents and the
    separately configured fastpath untouched; those have independent policy
    controls.
    """
    print("\n=== Global bounded coprocessor lane ===")
    enabled = _choose_toggle(
        "Bounded coprocessor lane",
        registry.coprocessors_enabled,
        enabled_text="On — allow configured workflow and feedback calls",
        disabled_text="Off — suppress bounded calls for launches without a profile",
    )
    raw = _read_yaml(config_dir / "sidecars.yaml")
    raw["coprocessors_enabled"] = enabled
    _write_yaml(config_dir / "sidecars.yaml", raw, validate_config_dir=config_dir)
    print(
        f"Global bounded coprocessor lane {'enabled' if enabled else 'disabled'} "
        f"in {config_dir / 'sidecars.yaml'}."
    )


def configure_sidecar_lane(
    config_dir: Path,
    registry: ModelRegistry,
    current_profile_id: str | None = None,
) -> EditorResult | None:
    """Open the sidecar lane editor without conflating its three layers."""
    while True:
        native_count = len(registry.sidecar_agents)
        coprocessor_count = len(registry.coprocessors)
        _render_menu(
            "Sidecar lane",
            [
                (
                    "Sidecar definitions",
                    [
                        ("1", f"Edit native sidecar workers — model-backed Claude agents ({native_count})"),
                        ("2", f"Edit bounded coprocessors — structured MCP calls ({coprocessor_count})"),
                    ],
                ),
                (
                    "Launch selection",
                    [
                        ("3", "Select a sidecar bundle for this launch — allow-list only"),
                        ("4", "Configure automatic fastpath coprocessor — optional"),
                        (
                            "5",
                            "Toggle global bounded coprocessor lane — "
                            f"{'ON' if registry.coprocessors_enabled else 'OFF'}",
                        ),
                    ],
                ),
                ("Navigation", [("b", "Back to setup"), ("q", "Cancel setup")]),
            ],
            footer="Definitions choose models; a bundle chooses which definitions this launch may use",
        )
        choice = _menu_prompt("Choose").strip().lower()
        if choice == "1":
            configure_native_sidecar_agent(config_dir, registry)
            registry, _ = _load_registry(config_dir)
        elif choice == "2":
            configure_coprocessor(config_dir, registry)
            registry, _ = _load_registry(config_dir)
        elif choice == "3":
            return configure_sidecar_profile(config_dir, registry)
        elif choice == "4":
            configure_fastpath(config_dir, registry)
            registry, _ = _load_registry(config_dir)
        elif choice == "5":
            configure_global_coprocessor_lane(config_dir, registry)
            registry, _ = _load_registry(config_dir)
        elif choice in {"b", "q"}:
            return EditorResult(current_profile_id, "unchanged" if current_profile_id else "cancelled")
        else:
            _status("Choose 1, 2, 3, 4, 5, b, or q.", "yellow")


def configure_sidecar_profile(config_dir: Path, registry: ModelRegistry) -> EditorResult:
    """Configure a named, reusable sidecar-profile bundle (sidecar_profiles.yaml).

    A sidecar profile limits which globally-defined native workers and
    coprocessors (sidecars.yaml) are available for a launch. It does not
    assign their models; that happens in the sidecar-definition editor.
    The profile also carries its own fastpath coprocessor config -- fastpath
    is scoped per sidecar-profile, not one bare global singleton, so
    different launch presets can run different fastpath models (or none)
    side by side.
    """
    print("\n=== Sidecar launch bundle: choose which configured workers this launch may use ===")
    print("This screen selects existing sidecar definitions; use r to add per-launch route overrides.")
    raw = _read_yaml(config_dir / "sidecar_profiles.yaml")
    profiles = raw.setdefault("sidecar_profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
        raw["sidecar_profiles"] = profiles
    existing = sorted(str(item) for item in profiles)
    profile_id = _choose_or_create_id("Sidecar profile", existing, "default")
    current = dict(profiles.get(profile_id) or {})

    available_agents = sorted(registry.sidecar_agents)
    available_coprocessors = sorted(registry.coprocessors)
    if not available_agents and not available_coprocessors:
        raise RuntimeError(
            "No sidecar definitions exist yet -- go back and edit a native sidecar worker "
            "or bounded coprocessor first."
        )

    # Numbered toggle selector for native-worker and coprocessor membership.
    previous_agents = {
        str(item) for item in current.get("sidecar_agent_ids", [])
        if isinstance(item, str)
    }
    previous_coprocessors = {
        str(item) for item in current.get("coprocessor_ids", [])
        if isinstance(item, str)
    }
    # Pre-migration profiles used sidecar_ids for bounded coprocessors.
    previous_coprocessors.update(
        str(item) for item in current.get("sidecar_ids", [])
        if isinstance(item, str)
    )
    toggled_ids = {f"agent:{item}" for item in previous_agents}
    toggled_ids.update(f"coprocessor:{item}" for item in previous_coprocessors)
    while True:
        sidecar_options: list[tuple[str, str]] = []
        for agent_id in available_agents:
            agent_entry = registry.sidecar_agents.get(agent_id)
            model_label = getattr(agent_entry, "model_id", "")
            roles = ", ".join(getattr(agent_entry, "roles", []) or [])
            sidecar_options.append(
                (f"agent:{agent_id}", f"native worker · {model_label} · roles: {roles or 'unassigned'}")
            )
        for coprocessor_id in available_coprocessors:
            coprocessor_entry = registry.coprocessors.get(coprocessor_id)
            model_label = getattr(coprocessor_entry, "model_id", "")
            mode_label = getattr(coprocessor_entry, "mode", "structured")
            sidecar_options.append(
                (f"coprocessor:{coprocessor_id}", f"bounded MCP call · {model_label} · mode: {mode_label}")
            )
        lane_state = "ON" if bool(current.get("coprocessors_enabled", True)) else "OFF"
        _render_options(
            f"Sidecars in profile: {profile_id} · bounded coprocessor lane {lane_state}",
            sidecar_options,
            navigation=[
                "type a number to toggle",
                "c) toggle bounded coprocessor lane",
                "r) edit native-worker route overrides",
                "d) done",
                "b) back",
            ],
            current=toggled_ids,
        )
        choice = _menu_prompt("Enter number, r for route overrides, d when done, b to go back").strip().lower()
        if choice == "d":
            current["sidecar_ids"] = sorted(toggled_ids)
            break
        if choice == "b":
            return EditorResult(None, "cancelled")
        if choice == "c":
            current["coprocessors_enabled"] = not bool(
                current.get("coprocessors_enabled", True)
            )
            continue
        if choice == "r":
            selected_agents = sorted(
                item.removeprefix("agent:")
                for item in toggled_ids
                if item.startswith("agent:")
            )
            _edit_sidecar_profile_route_overrides(
                config_dir, registry, current, selected_agents,
            )
            continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sidecar_options):
                selected_id = sidecar_options[idx][0]
                if selected_id in toggled_ids:
                    toggled_ids.remove(selected_id)
                else:
                    toggled_ids.add(selected_id)
            else:
                _status(f"Enter a number 1-{len(sidecar_options)}, d, or b.", "yellow")
        except ValueError:
            _status(f"Enter a number 1-{len(sidecar_options)}, d, or b.", "yellow")

    current["sidecar_agent_ids"] = sorted(
        item.removeprefix("agent:")
        for item in toggled_ids
        if item.startswith("agent:")
    )
    selected_coprocessors = sorted(
        item.removeprefix("coprocessor:")
        for item in toggled_ids
        if item.startswith("coprocessor:")
    )
    current["coprocessor_ids"] = selected_coprocessors
    current.setdefault("coprocessors_enabled", True)
    # Retain the legacy field for older readers and migration fixtures.
    current["sidecar_ids"] = selected_coprocessors

    if _prompt("Configure a dedicated fastpath coprocessor for this profile too? [y/N]").strip().lower() == "y":
        fastpath = dict(current.get("fastpath") or {})
        choices = [
            item for item in generate_route_choices(registry, role="recon")
            if not registry.get_model(item.model_id).capabilities.write_tool_certified
        ]
        if not choices:
            raise RuntimeError("No enabled credential-backed model is available for the fastpath coprocessor.")
        previous_model = str(fastpath.get("model_id") or "")
        route = _pick_registry_route(
            registry, choices, purpose="sidecar profile fastpath",
            current_model_id=previous_model,
            current_endpoint_id=str(fastpath.get("endpoint") or ""),
            current_provider_id=str(fastpath.get("provider_id") or ""),
        )
        if route is None:
            return EditorResult(None, "cancelled")
        fastpath["model_id"] = route.model_id
        fastpath["endpoint"] = route.endpoint_id
        fastpath["provider_id"] = route.provider_id
        fastpath.setdefault("enabled", True)
        fastpath.setdefault("modes", ["route", "verify"])
        fastpath["timeout_seconds"] = _prompt_float(
            "Fastpath timeout seconds", float(fastpath.get("timeout_seconds", 5)), 1, 30,
        )
        current["fastpath"] = fastpath
    elif "fastpath" in current and _prompt(
        "Remove this profile's dedicated fastpath (fall back to the global fastpath.yaml)? [y/N]"
    ).strip().lower() == "y":
        current.pop("fastpath", None)

    profiles[profile_id] = current
    _write_yaml(config_dir / "sidecar_profiles.yaml", raw, validate_config_dir=config_dir)
    print(f"Saved sidecar profile '{profile_id}' to {config_dir / 'sidecar_profiles.yaml'}.")
    print(f"Use it with: claude-brigade --sidecar-profile {profile_id}")
    return EditorResult(profile_id, "saved")


def _edit_sidecar_profile_route_overrides(
    config_dir: Path,
    registry: ModelRegistry,
    current: dict[str, Any],
    selected_agents: Sequence[str],
) -> None:
    """Edit exact route ladders for native workers in one launch profile.

    The global sidecar definition remains the default.  An override is an
    immutable launch-time route policy, so the same native worker identity can
    be used by two presets with different provider/model/endpoint ladders.
    """
    if not selected_agents:
        _status("Select at least one native worker before editing route overrides.", "yellow")
        return
    overrides = dict(current.get("agent_route_overrides") or {})
    for agent_id in selected_agents:
        spec = registry.sidecar_agents.get(agent_id)
        if spec is None:
            continue
        role = next((item for item in spec.roles if item in _ROLES), "recon")
        choices = [
            item for item in generate_route_choices(registry, role=role)
            if item.credential_configured
        ]
        if spec.can_mutate:
            choices = [
                item for item in choices
                if registry.get_model(item.model_id).capabilities.write_tool_certified is True
            ]
        if not choices:
            _status(f"No eligible route candidates for native worker '{agent_id}'.", "yellow")
            continue
        saved = overrides.get(agent_id) if isinstance(overrides.get(agent_id), dict) else {}
        primary = saved.get("primary") if isinstance(saved, dict) else None
        if not isinstance(primary, dict):
            primary = {
                "model": spec.model_id,
                "endpoint": spec.endpoint,
                "provider_id": spec.provider_id,
            }
        route = _pick_registry_route(
            registry,
            choices,
            purpose=f"route override for native sidecar {agent_id}",
            current_model_id=str(primary.get("model") or ""),
            current_endpoint_id=str(primary.get("endpoint") or "auto"),
            current_provider_id=str(primary.get("provider_id") or ""),
        )
        if route is None:
            continue
        previous = saved.get("fallbacks") if isinstance(saved, dict) else []
        if not isinstance(previous, list):
            previous = []
        fallbacks = _edit_fallbacks(
            config_dir,
            registry,
            role=role,
            fallback_options=[],
            previous=previous,
            fallback_choices=choices,
        )
        overrides[agent_id] = {
            "primary": {
                "model": route.model_id,
                "endpoint": route.endpoint_id,
                "provider_id": route.provider_id,
            },
            "fallbacks": fallbacks,
        }
    current["agent_route_overrides"] = overrides


def configure_launch_preset(config_dir: Path, registry: ModelRegistry) -> str:
    """Configure a named launch preset pairing a saved inference profile with
    a saved sidecar profile, so both switch together with one choice."""
    print("\n=== Launch presets: pair model lanes with a workflow composition ===")
    inference_profiles = sorted(_read_yaml(config_dir / "profiles.yaml").get("profiles") or {})
    if not inference_profiles:
        raise RuntimeError("No inference profiles are saved yet -- run the inference wizard first.")
    sidecar_profiles = sorted(_read_yaml(config_dir / "sidecar_profiles.yaml").get("sidecar_profiles") or {})

    raw = _read_yaml(config_dir / "launch_presets.yaml")
    presets = raw.setdefault("launch_presets", {})
    if not isinstance(presets, dict):
        presets = {}
        raw["launch_presets"] = presets
    existing = sorted(str(item) for item in presets)
    preset_id = _choose_or_create_id("Launch preset", existing, "default")
    current = dict(presets.get(preset_id) or {})

    inference_profile_id = _choose(
        "Inference profile",
        [(item, "saved inference profile") for item in inference_profiles],
        next((i for i, item in enumerate(inference_profiles, 1) if item == current.get("inference_profile_id")), 1),
    )
    sidecar_profile_options = [("none", "no sidecar profile (use global sidecars.yaml/fastpath.yaml)")] + [
        (item, "saved sidecar profile") for item in sidecar_profiles
    ]
    sidecar_profile_choice = _choose(
        "Sidecar profile",
        sidecar_profile_options,
        next((i for i, item in enumerate(("none", *sidecar_profiles), 1) if item == (current.get("sidecar_profile_id") or "none")), 1),
    )
    workflow_ids = ["(automatic task tier)", *sorted(registry.workflows)]
    workflow_choice = _choose(
        "Workflow composition",
        [
            (item, "select by task tier" if item == "(automatic task tier)" else "saved workflow")
            for item in workflow_ids
        ],
        next(
            (
                i for i, item in enumerate(workflow_ids, 1)
                if item == (current.get("workflow_id") or "(automatic task tier)")
            ),
            1,
        ),
    )
    current["inference_profile_id"] = inference_profile_id
    current["sidecar_profile_id"] = None if sidecar_profile_choice == "none" else sidecar_profile_choice
    current["workflow_id"] = None if workflow_choice == "(automatic task tier)" else workflow_choice

    presets[preset_id] = current
    _write_yaml(config_dir / "launch_presets.yaml", raw, validate_config_dir=config_dir)
    print(f"Saved launch preset '{preset_id}' to {config_dir / 'launch_presets.yaml'}.")
    print(f"Use it with: claude-brigade --launch-preset {preset_id}")
    return preset_id


# ---------------------------------------------------------------------------
# Unified launch-setup orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchSetupResult:
    """Result of a complete launch-setup wizard invocation.

    Returned by ``configure_launch_setup()`` to the caller (the
    ``bin/claude-brigade`` launcher), which uses the three IDs to
    set the corresponding environment variables for the upcoming
    Claude Code launch.
    """

    status: Literal["saved", "cancelled"]
    inference_profile_id: str | None = None
    sidecar_profile_id: str | None = None
    launch_preset_id: str | None = None


def configure_launch_setup(
    config_dir: Path,
    registry: ModelRegistry,
) -> LaunchSetupResult:
    """Unified launch-setup orchestrator.

    The user enters through a lane-choice menu, picks one of the two
    configuration lanes (controller/worker-role routes or additional
    sidecar definitions), works through it
    with the dashboard editor, then returns to a setup overview that
    shows both lanes and offers to review/save.

    This replaces the old embedded ``configure_inference()``-only
    wizard in ``bin/claude-brigade`` -- that one skipped sidecar
    profiles altogether except for a single "configure fastpath"
    checkbox.
    """
    inference_profile_id: str | None = None
    sidecar_profile_id: str | None = None
    launch_preset_id: str | None = None

    while True:
        if inference_profile_id:
            _show_inference_summary(registry, inference_profile_id)
        if sidecar_profile_id:
            _show_sidecar_summary(registry, sidecar_profile_id)

        setup_choices: list[tuple[str, str]] = []
        if not inference_profile_id:
            setup_choices.append(("1", "Controller + worker-role routes — models and fallbacks"))
        else:
            setup_choices.append(("1", f"Edit controller + worker-role routes ({inference_profile_id})"))
        if not sidecar_profile_id:
            setup_choices.append(("2", "Sidecar lane — edit workers, coprocessors, or launch selection"))
        else:
            setup_choices.append(("2", f"Edit sidecar lane ({sidecar_profile_id})"))

        if inference_profile_id and sidecar_profile_id:
            setup_choices.append(("3", "Create paired launch preset and continue"))
        elif inference_profile_id and not sidecar_profile_id:
            setup_choices.append(("3", "Skip sidecar for now — use global defaults"))
        _render_menu(
            "Configuration setup",
            [("Setup lanes", setup_choices), ("Navigation", [("b", "Back and discard changes"), ("q", "Cancel")])],
            footer="Configure controller/worker routes or sidecar definitions, then pair them into a launch preset",
        )

        choice = _menu_prompt("Choose").strip().lower()
        if choice == "1":
            editor_result = configure_inference(config_dir, registry)
            if isinstance(editor_result, EditorResult):
                if editor_result.status == "saved":
                    inference_profile_id = editor_result.object_id
            else:
                # Compatibility with integrations that still replace the
                # editor with a plain profile-id-returning callable.
                inference_profile_id = str(editor_result) if editor_result else None
            # Reload registry after profile changes
            registry, _ = _load_registry(config_dir)
        elif choice == "2":
            editor_result = configure_sidecar_lane(
                config_dir, registry, sidecar_profile_id,
            )
            if isinstance(editor_result, EditorResult):
                if editor_result.status == "saved":
                    sidecar_profile_id = editor_result.object_id
            elif editor_result:
                sidecar_profile_id = str(editor_result)
            registry, _ = _load_registry(config_dir)
        elif choice == "3" and inference_profile_id and sidecar_profile_id:
            launch_preset_id = _create_launch_preset_for_setup(
                config_dir, inference_profile_id, sidecar_profile_id,
            )
            print("\n── Setup complete ──")
            print(f"  Inference profile: {inference_profile_id}")
            print(f"  Sidecar profile:   {sidecar_profile_id}")
            print(f"  Launch preset:     {launch_preset_id or '(none)'}")
            print("All saved. Starting Claude Code with this preset.")
            return LaunchSetupResult(
                status="saved",
                inference_profile_id=inference_profile_id,
                sidecar_profile_id=sidecar_profile_id,
                launch_preset_id=launch_preset_id,
            )
        elif choice == "3":
            print("Skipping additional native sidecars. Starting with controller and worker-role routes only.")
            return LaunchSetupResult(
                status="saved",
                inference_profile_id=inference_profile_id,
                sidecar_profile_id=sidecar_profile_id,
                launch_preset_id=launch_preset_id,
            )
        elif choice in ("b", "q"):
            print("Setup cancelled.")
            return LaunchSetupResult(status="cancelled")
        else:
            print("Choose 1, 2, or q.")


def _show_inference_summary(registry: ModelRegistry, profile_id: str) -> None:
    """Display a summary of the inference profile for the setup overview."""
    try:
        profile = registry.get_profile(profile_id)
    except (KeyError, LookupError):
        print(f"\n  Controller + worker-role profile: {profile_id} (unable to load)")
        return
    def route_label(candidate: Any) -> str:
        provider = candidate.provider_id or "auto provider"
        return f"{provider} / {candidate.model} / {candidate.endpoint}"

    route = profile.controller_route()
    print(f"\n  Inference profile: {profile_id}")
    print("    Claude Code model lanes (six persistent lanes + optional custom):")
    slots = registry.slot_alias_manifest(profile_id)
    for slot in ("main", "background", "haiku", "sonnet", "opus", "fable", "custom"):
        entry = slots.get(slot)
        if entry is None:
            print(f"      {slot:<7} (unset)")
            continue
        print(
            f"      {slot:<7} {entry.get('provider_id') or 'auto provider'} / "
            f"{entry.get('model_id', '(unset)')} / {entry.get('endpoint', 'auto')} "
            f"({len(entry.get('fallbacks') or [])} fallback(s))"
        )
    print("    Controller + worker-role compatibility routes:")
    print(f"      Controller  {route_label(route.primary) if route else '(default)'}")
    for role in _ROLES:
        target = profile.route_target(role)
        print(
            f"      {role.capitalize():<11s} {route_label(target.primary)} "
            f"({len(target.fallbacks)} fallback(s))"
        )
    print()


def _show_sidecar_summary(registry: ModelRegistry, profile_id: str) -> None:
    """Display a summary of the sidecar profile for the setup overview."""
    try:
        config_dir = registry.config_dir
        if config_dir is None:
            raise RuntimeError("registry has no config directory")
        profiles = _read_yaml(config_dir / "sidecar_profiles.yaml")
        profile = profiles.get("sidecar_profiles", {}).get(profile_id, {})
    except Exception:
        print(f"\n  Sidecar profile: {profile_id} (unable to load)")
        return
    if not isinstance(profile, dict):
        print(f"\n  Sidecar profile: {profile_id} (unable to load)")
        return
    sidecar_agent_ids = profile.get("sidecar_agent_ids", [])
    coprocessor_ids = profile.get("coprocessor_ids")
    if coprocessor_ids is None:
        coprocessor_ids = profile.get("sidecar_ids", [])
    fastpath = profile.get("fastpath", None)
    fastpath_str = ""
    if isinstance(fastpath, dict):
        fastpath_str = f" · {fastpath.get('model_id', '(unset)')}"
    print(f"\n  Sidecar launch bundle: {profile_id}")
    print(f"    Fastpath: {'configured' if fastpath else 'global default'}{fastpath_str}")
    resolved_agents = registry.resolve_sidecar_agents(profile_id)
    for agent_id in sidecar_agent_ids:
        entry = resolved_agents.get(agent_id)
        if entry is not None:
            fallback_count = len(entry.fallback_routes)
            print(
                f"    native {agent_id}: {entry.provider_id or 'auto provider'} / "
                f"{entry.model_id} / {entry.endpoint} · "
                f"roles: {', '.join(entry.roles) or 'unassigned'} · "
                f"{fallback_count} fallback(s)"
            )
        else:
            print(f"    native {agent_id} (missing definition)")
    for coprocessor_id in coprocessor_ids:
        entry = registry.coprocessors.get(coprocessor_id)
        if entry is not None:
            print(f"    coprocessor {coprocessor_id}: {entry.model_id} · {entry.mode}")
        else:
            print(f"    coprocessor {coprocessor_id} (missing definition)")
    print()


def _create_launch_preset_for_setup(
    config_dir: Path,
    inference_profile_id: str,
    sidecar_profile_id: str,
    workflow_id: str | None = None,
) -> str:
    """Create or re-use a launch preset for the given profile pair.

    If an existing launch preset already pairs these two profiles,
    return its ID. Otherwise create a new one with a generated name.
    """
    raw = _read_yaml(config_dir / "launch_presets.yaml")
    presets = raw.setdefault("launch_presets", {})
    if not isinstance(presets, dict):
        presets = {}
        raw["launch_presets"] = presets

    # Check for an existing preset with this exact pair
    for preset_id, entry in presets.items():
        if isinstance(entry, dict):
            if (entry.get("inference_profile_id") == inference_profile_id
                    and entry.get("sidecar_profile_id") == sidecar_profile_id
                    and entry.get("workflow_id") == workflow_id):
                return preset_id

    preset_id = f"{inference_profile_id}-{sidecar_profile_id}"
    presets[preset_id] = {
        "inference_profile_id": inference_profile_id,
        "sidecar_profile_id": sidecar_profile_id,
    }
    if workflow_id:
        presets[preset_id]["workflow_id"] = workflow_id
    _write_yaml(
        config_dir / "launch_presets.yaml", raw,
        validate_config_dir=config_dir,
    )
    return preset_id


def configure_keys(config_dir: Path, registry: ModelRegistry) -> None:
    """Interactively save/remove provider keys in the OS credential store."""
    path = config_dir / "providers.env"
    legacy_values = parse_env_file(path, allowed_keys=ALLOWED_PROVIDER_KEYS) if path.exists() else {}
    key_options: set[str] = set()
    for provider in registry.providers.values():
        if provider.api_key_env:
            key_options.add(provider.api_key_env)
    for model in registry.models.values():
        if model.api_key_env:
            key_options.add(model.api_key_env)
        for endpoint in model.endpoints.values():
            if endpoint.api_key_env:
                key_options.add(endpoint.api_key_env)
    key_options.update(
        item for item in ALLOWED_PROVIDER_KEYS
        if item.endswith("_API_KEY")
    )
    secure_slots: dict[str, dict[str, str]] = {}
    secure_available = credential_store_available()
    if secure_available:
        for key in sorted(key_options & PROVIDER_SECRET_KEYS):
            try:
                secure_slots[key] = read_slots(key, config_dir)
            except CredentialStoreUnavailable:
                secure_available = False
                secure_slots = {}
                break
    choices = []
    for key in sorted(key_options):
        if secure_slots.get(key):
            state = "keyring slots: " + ", ".join(sorted(secure_slots[key]))
        elif legacy_values.get(key):
            state = "legacy file value"
        else:
            state = "missing"
        choices.append((key, state))
    selected = _choose("Provider credential to save or remove", choices)
    existing_slots = sorted(secure_slots.get(selected, {}))
    slot = _prompt("Key slot name", existing_slots[0] if existing_slots else "primary")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}", slot):
        raise ValueError("credential slot names may contain letters, numbers, '.', '_' and '-'")
    if secure_available:
        print(f"Enter the value for {selected}/{slot}. Input is hidden and will be stored in the OS keyring.")
    else:
        print(
            f"No usable OS keyring was found. Input is hidden and will be stored only in {path} "
            "with mode 0600."
        )
    value = getpass.getpass(f"{selected} (blank removes it): ")
    if secure_available:
        if value:
            set_slot(selected, slot, value, config_dir)
            legacy_values.pop(selected, None)
            _write_provider_env(path, legacy_values)
            print(f"Saved {selected}/{slot} in the OS keyring ({backend_label()}); active requests rotate slots round-robin.")
        else:
            delete_slot(selected, slot, config_dir)
            legacy_values.pop(selected, None)
            _write_provider_env(path, legacy_values)
            print(f"Removed {selected}/{slot} from the OS keyring and legacy file.")
    else:
        if value:
            legacy_values[selected] = value
            print(f"Saved {selected} in the protected legacy credential file.")
        else:
            legacy_values.pop(selected, None)
            print(f"Removed {selected} from the protected legacy credential file.")
        _write_provider_env(path, legacy_values)
        print(f"Credential file mode: {oct(path.stat().st_mode & 0o777)}")


def import_environment_credentials(config_dir: Path, registry: ModelRegistry) -> None:
    """Import explicitly loaded API keys without displaying their values."""
    key_options = set(PROVIDER_SECRET_KEYS)
    for provider in registry.providers.values():
        if provider.api_key_env:
            key_options.add(provider.api_key_env)
    available_keys = sorted(key for key in key_options if os.environ.get(key))
    if not available_keys:
        print("No provider API keys are present in the current environment.")
        return
    print("Loaded environment credentials: " + ", ".join(available_keys))
    if _prompt("Import these key names into the OS keyring? [y/N]").strip().lower() != "y":
        print("Nothing imported.")
        return
    if not credential_store_available():
        raise CredentialStoreUnavailable(
            "no usable OS keyring is available; use 'keys' to save one interactively"
        )
    for key in available_keys:
        set_slot(key, "imported", os.environ[key], config_dir)
    print(f"Imported {len(available_keys)} credentials into the OS keyring ({backend_label()}); imported slots participate in rotation.")


_CATALOG_REFRESH_TTL_SECONDS = 900.0  # 15 minutes
_CATALOG_REFRESH_TIMEOUT_SECONDS = 12.0
_CATALOG_REFRESH_MAX_WORKERS = 4
_CATALOG_REFRESH_STATE_FILENAME = "catalog_refresh_state.json"


@dataclass(frozen=True)
class CatalogRefreshResult:
    """One provider's outcome from a ``refresh_catalogs`` pass."""

    provider_id: str
    status: Literal["success", "skipped", "cached", "error"]
    model_count: int = 0
    digest: str | None = None
    detail: str = ""


def _load_catalog_refresh_state(config_dir: Path) -> dict[str, dict[str, Any]]:
    path = config_dir / _CATALOG_REFRESH_STATE_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_catalog_refresh_state(config_dir: Path, state: dict[str, dict[str, Any]]) -> None:
    path = config_dir / _CATALOG_REFRESH_STATE_FILENAME
    path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.chmod(path, 0o600)


def _catalog_error_detail(exc: Exception) -> str:
    """Return a short operator-facing discovery error."""
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        request_url = str(exc.request.url) if exc.request is not None else "catalog URL"
        return f"HTTP {exc.response.status_code} — requested {request_url}"
    if isinstance(exc, httpx.RequestError):
        return f"{exc.__class__.__name__}: {exc.request.url}"
    return str(exc).splitlines()[0][:180]


def _cached_catalog_available(registry: ModelRegistry, provider_id: str) -> bool:
    path = registry.config_dir / "discovered_models.yaml" if registry.config_dir else None
    if path is None or not path.exists():
        return False
    try:
        models = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models") or {}
    except (OSError, yaml.YAMLError):
        return False
    return any(str(model_id).startswith(f"{provider_id}/") for model_id in models)


def refresh_catalogs(
    registry: ModelRegistry,
    *,
    ttl_seconds: float = _CATALOG_REFRESH_TTL_SECONDS,
    force: bool = False,
) -> list[CatalogRefreshResult]:
    """Fetch configured OpenAI-compatible catalogs for keys that are present.

    Network fetches for distinct providers run concurrently (bounded pool,
    a short per-provider timeout) instead of serially at up to 30s each --
    the previous behavior meant a config with several credential-backed
    providers could block the wizard for minutes even when every provider
    responds quickly on its own. A provider refreshed within *ttl_seconds*
    is skipped (``force=True`` bypasses this) so re-entering the wizard
    doesn't repeat unnecessary network round-trips.
    """
    config_dir = registry.config_dir
    refresh_state = _load_catalog_refresh_state(config_dir) if config_dir else {}
    now = time.time()

    candidates: list[tuple[str, str]] = []  # (provider_id, endpoint_id)
    results: list[CatalogRefreshResult] = []
    for provider_id, provider in sorted(registry.providers.items()):
        if provider.discovery.get("type") != "openai-models":
            continue
        if not provider.api_key_env or not _credential_configured(
            provider.api_key_env, registry.config_dir
        ):
            results.append(CatalogRefreshResult(
                provider_id, "skipped",
                detail=f"missing {provider.api_key_env or 'credential'}",
            ))
            continue
        if not force:
            last_refreshed = refresh_state.get(provider_id, {}).get("refreshed_at")
            if isinstance(last_refreshed, (int, float)) and now - last_refreshed < ttl_seconds:
                results.append(CatalogRefreshResult(
                    provider_id, "cached",
                    detail=f"refreshed {int(now - last_refreshed)}s ago",
                ))
                continue
        endpoint_id = provider.discovery.get("endpoint_id") or (
            "openai" if "openai" in provider.endpoints else next(iter(provider.endpoints), None)
        )
        if not endpoint_id and not provider.discovery.get("url"):
            results.append(CatalogRefreshResult(provider_id, "skipped", detail="no catalog endpoint"))
            continue
        candidates.append((provider_id, endpoint_id or "openai"))

    fetched: dict[str, Any] = {}
    errors: dict[str, str] = {}
    if candidates:
        with ThreadPoolExecutor(max_workers=min(_CATALOG_REFRESH_MAX_WORKERS, len(candidates))) as pool:
            futures = {
                pool.submit(
                    registry.fetch_discovered_entries,
                    provider_id, endpoint_id=endpoint_id, timeout=_CATALOG_REFRESH_TIMEOUT_SECONDS,
                ): provider_id
                for provider_id, endpoint_id in candidates
            }
            for future in as_completed(futures):
                provider_id = futures[future]
                try:
                    fetched[provider_id] = future.result()
                except Exception as exc:
                    errors[provider_id] = _catalog_error_detail(exc)

    # Publishing mutates shared registry state (the discovered-catalog file
    # and in-memory models) -- serialize it on the main thread even though
    # the fetches above ran concurrently.
    for provider_id, _endpoint_id in candidates:
        if provider_id in errors:
            results.append(CatalogRefreshResult(provider_id, "error", detail=errors[provider_id]))
            continue
        entries = fetched.get(provider_id, [])
        try:
            digest, count = registry.apply_discovered_entries(entries, provider_id=provider_id)
        except Exception as exc:
            results.append(CatalogRefreshResult(provider_id, "error", detail=_catalog_error_detail(exc)))
            continue
        results.append(CatalogRefreshResult(provider_id, "success", model_count=count, digest=digest))
        refresh_state[provider_id] = {"refreshed_at": now, "digest": digest}

    if config_dir and any(r.status == "success" for r in results):
        _save_catalog_refresh_state(config_dir, refresh_state)

    # Print configured-provider summary before the per-provider detail lines.
    all_providers = sorted(registry.providers.keys())
    discovery_providers = {r.provider_id for r in results}
    no_discovery = [
        pid for pid in all_providers
        if pid not in discovery_providers
        and registry.providers[pid].discovery.get("type") != "openai-models"
    ]
    if _rich_interactive():
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("Provider", style="bold cyan")
        table.add_column("Status")
        table.add_column("Details", overflow="fold")
        for result in sorted(results, key=lambda item: item.provider_id):
            if result.status == "success":
                status = Text("refreshed", style="green")
                detail = f"{result.model_count} models · catalog {(result.digest or '')[:12]}"
            elif result.status == "cached":
                status = Text("cached", style="cyan")
                detail = result.detail
            elif result.status == "skipped":
                status = Text("skipped", style="yellow")
                detail = result.detail
            else:
                status = Text("failed", style="red")
                detail = result.detail
            table.add_row(result.provider_id, status, detail)
        for provider_id in no_discovery:
            table.add_row(provider_id, Text("not configured", style="dim"), "no live catalog configured")
        if not results and not no_discovery:
            table.add_row("—", Text("none", style="dim"), "No providers are configured")
        _CONSOLE.print(Panel(table, title="Provider catalog refresh", border_style="cyan"))
        if not any(result.status == "success" for result in results):
            _status("No credential-backed dynamic provider catalogs were refreshed.", "yellow")
        return results
    if all_providers:
        print(f"\nConfigured providers: {', '.join(all_providers)}")
    if results:
        refreshed = [r.provider_id for r in results if r.status == "success"]
        cached = [r.provider_id for r in results if r.status == "cached"]
        failed = [r.provider_id for r in results if r.status == "error"]
        missing = [
            r.provider_id for r in results
            if r.status == "skipped" and r.detail.startswith("missing ")
        ]
        print("Catalog refresh results:")
        print(f"  Refreshed: {', '.join(refreshed) if refreshed else 'none'}")
        print(f"  Cached: {', '.join(cached) if cached else 'none'}")
        print(f"  Failed: {', '.join(failed) if failed else 'none'}")
        print(f"  Missing credentials: {', '.join(missing) if missing else 'none'}")
    if no_discovery:
        print(f"  Not configured: {', '.join(no_discovery)}")
    print()

    for result in sorted(results, key=lambda r: r.provider_id):
        if result.status == "success":
            print(f"{result.provider_id}: saved {result.model_count} models (catalog {(result.digest or '')[:12]})")
        elif result.status == "cached":
            print(f"{result.provider_id}: skipped ({result.detail}, within TTL)")
        elif result.status == "skipped":
            print(f"{result.provider_id}: skipped ({result.detail})")
        else:
            cache_note = ""
            if _cached_catalog_available(registry, result.provider_id):
                refreshed_at = refresh_state.get(result.provider_id, {}).get("refreshed_at")
                if isinstance(refreshed_at, (int, float)):
                    stamp = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(refreshed_at))
                    cache_note = f"; using cached catalog from {stamp}"
                else:
                    cache_note = "; using cached catalog"
            print(f"{result.provider_id}: discovery failed ({result.detail}){cache_note}")
    if not any(r.status == "success" for r in results):
        print("No credential-backed dynamic provider catalogs were refreshed.")
    return results


_DELETE_TARGETS = {
    "native sidecar": ("sidecars.yaml", "sidecar_agents"),
    "coprocessor": ("sidecars.yaml", "coprocessors"),
    # Legacy name retained for old scripts/configs that still use the
    # pre-migration ``sidecars:`` section.
    "sidecar": ("sidecars.yaml", "sidecars"),
    "sidecar profile": ("sidecar_profiles.yaml", "sidecar_profiles"),
    "launch preset": ("launch_presets.yaml", "launch_presets"),
}


def _referencing_configs(config_dir: Path, kind: str, item_id: str) -> list[str]:
    """Return human-readable descriptions of saved configs that still
    reference *item_id* -- deleting it out from under them would leave a
    dangling reference that only surfaces later, as a fail-closed daemon
    startup error for an operator who may not remember this deletion.
    """
    blockers: list[str] = []
    if kind in {"sidecar", "native sidecar", "coprocessor"}:
        sidecar_profiles = _read_yaml(config_dir / "sidecar_profiles.yaml").get("sidecar_profiles") or {}
        for profile_id, profile in sidecar_profiles.items():
            if not isinstance(profile, dict):
                continue
            references = set(profile.get("sidecar_ids") or [])
            if kind == "native sidecar":
                references = set(profile.get("sidecar_agent_ids") or [])
                references.update(profile.get("agent_route_overrides", {}).keys())
            elif kind == "coprocessor":
                references = set(profile.get("coprocessor_ids") or [])
                references.update(profile.get("sidecar_ids") or [])
            if item_id in references:
                blockers.append(f"sidecar profile '{profile_id}'")
        workflows = _read_yaml(config_dir / "workflows.yaml").get("workflows") or {}
        for workflow_id, workflow in workflows.items():
            if not isinstance(workflow, dict):
                continue
            for phase in workflow.get("phases") or []:
                if not isinstance(phase, dict):
                    continue
                field = "sidecar_agent" if kind == "native sidecar" else "coprocessor"
                if kind == "sidecar":
                    field = "sidecar"
                if phase.get(field) == item_id:
                    blockers.append(f"workflow '{workflow_id}' phase '{phase.get('id', '?')}'")
        if kind == "coprocessor":
            feedback = _read_yaml(config_dir / "fastpath.yaml").get("fastpath") or {}
            if isinstance(feedback, dict) and feedback.get("coprocessor_id") == item_id:
                blockers.append("global fastpath")
            for profile_id, profile in sidecar_profiles.items():
                if not isinstance(profile, dict):
                    continue
                monitor = profile.get("feedback_monitor") or {}
                fastpath = profile.get("fastpath") or {}
                if isinstance(monitor, dict) and monitor.get("coprocessor_id") == item_id:
                    blockers.append(f"sidecar profile '{profile_id}' feedback monitor")
                if isinstance(fastpath, dict) and fastpath.get("coprocessor_id") == item_id:
                    blockers.append(f"sidecar profile '{profile_id}' fastpath")
    elif kind == "sidecar profile":
        launch_presets = _read_yaml(config_dir / "launch_presets.yaml").get("launch_presets") or {}
        for preset_id, preset in launch_presets.items():
            if isinstance(preset, dict) and preset.get("sidecar_profile_id") == item_id:
                blockers.append(f"launch preset '{preset_id}'")
    elif kind == "inference profile":
        launch_presets = _read_yaml(config_dir / "launch_presets.yaml").get("launch_presets") or {}
        for preset_id, preset in launch_presets.items():
            if isinstance(preset, dict) and preset.get("inference_profile_id") == item_id:
                blockers.append(f"launch preset '{preset_id}'")
        workflows = _read_yaml(config_dir / "workflows.yaml").get("workflows") or {}
        for workflow_id, workflow in workflows.items():
            if isinstance(workflow, dict) and workflow.get("default_profile") == item_id:
                blockers.append(f"workflow '{workflow_id}'")
    return blockers


def _delete_saved(config_dir: Path, *, kind: str) -> None:
    filename, key = _DELETE_TARGETS.get(kind, ("profiles.yaml", "profiles"))
    raw = _read_yaml(config_dir / filename)
    entries = raw.get(key) if isinstance(raw.get(key), dict) else {}
    if not entries:
        print(f"No saved {kind} configurations found.")
        return
    selected = _choose(f"Delete saved {kind}", [(str(item), "saved configuration") for item in sorted(entries)])
    blockers = _referencing_configs(config_dir, kind, selected)
    if blockers:
        print(f"Cannot delete '{selected}': still referenced by {', '.join(blockers)}.")
        return
    if _prompt(f"Delete '{selected}' permanently? [y/N]").strip().lower() != "y":
        print("Nothing deleted.")
        return
    del entries[selected]
    _write_yaml(config_dir / filename, raw, validate_config_dir=config_dir)
    print(f"Deleted {kind} '{selected}'.")


def _rich_show_saved(config_dir: Path, provider_keys: tuple[str, ...]) -> None:
    """Render saved configuration as a compact Rich dashboard."""
    profiles = _read_yaml(config_dir / "profiles.yaml").get("profiles") or {}
    fastpath = _read_yaml(config_dir / "fastpath.yaml").get("fastpath") or {}
    sidecar_yaml = _read_yaml(config_dir / "sidecars.yaml")
    native_sidecars = sidecar_yaml.get("sidecar_agents") or {}
    coprocessors = sidecar_yaml.get("coprocessors")
    if coprocessors is None:
        coprocessors = sidecar_yaml.get("sidecars") or {}
    sidecar_profiles = _read_yaml(config_dir / "sidecar_profiles.yaml").get("sidecar_profiles") or {}
    presets = _read_yaml(config_dir / "launch_presets.yaml").get("launch_presets") or {}

    overview = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    overview.add_column("field", style="bold cyan")
    overview.add_column("value")
    overview.add_row("Config directory", str(config_dir))
    overview.add_row(
        "Global coprocessor lane",
        "ON" if bool(sidecar_yaml.get("coprocessors_enabled", True)) else "OFF",
    )
    credential_names = [key for key in sorted(provider_keys) if key.endswith("_API_KEY")]
    overview.add_row("Loaded credentials", ", ".join(credential_names) or "none")
    if credential_store_available():
        overview.add_row("Credential backend", backend_label())

    role_table = Table(box=box.SIMPLE, expand=True)
    role_table.add_column("Profile", style="bold cyan")
    role_table.add_column("Controller")
    for role in _ROLES:
        role_table.add_column(role.capitalize())
    for profile_id, profile in sorted(profiles.items()):
        if not isinstance(profile, dict):
            continue
        controller_raw = profile.get("controller")
        controller = (
            _role_label({"controller": controller_raw}, "controller")
            if isinstance(controller_raw, dict)
            else profile.get("controller_model") or "Claude/default"
        )
        role_table.add_row(
            str(profile_id),
            str(controller),
            *[_role_label(profile, role) for role in _ROLES],
        )
    if not profiles:
        role_table.add_row("—", "none", *(["—"] * len(_ROLES)))

    sidecar_table = Table(box=box.SIMPLE, expand=True)
    sidecar_table.add_column("Kind", style="bold cyan")
    sidecar_table.add_column("ID")
    sidecar_table.add_column("Model")
    sidecar_table.add_column("Mode / policy")
    sidecar_table.add_column("Enabled")
    if fastpath:
        sidecar_table.add_row(
            "fastpath",
            "global",
            str(fastpath.get("model_id", "(unset)")),
            ", ".join(map(str, fastpath.get("modes", []))) or "route/verify",
            str(fastpath.get("enabled", True)),
        )
    for agent_id, entry in sorted(native_sidecars.items()):
        if isinstance(entry, dict):
            sidecar_table.add_row(
                "native sidecar",
                str(agent_id),
                f"{entry.get('model_id', '(unset)')} · {entry.get('provider_id') or 'default'}",
                (
                    f"{entry.get('endpoint', 'auto')} · "
                    f"{len(entry.get('fallback_routes') or entry.get('fallback_models') or [])} fallback(s) · "
                    f"{'mutating' if entry.get('can_mutate') else 'read-only'}"
                ),
                str(entry.get("enabled", True)),
            )
    for sidecar_id, entry in sorted(coprocessors.items()):
        if isinstance(entry, dict):
            sidecar_table.add_row(
                "coprocessor",
                str(sidecar_id),
                f"{entry.get('model_id', '(unset)')} · {entry.get('provider_id') or 'default'}",
                (
                    f"{entry.get('endpoint', 'auto')} · "
                    f"{len(entry.get('fallback_routes') or [])} fallback(s) · "
                    f"{entry.get('mode', 'structured')}"
                ),
                str(entry.get("enabled", True)),
            )
    if not fastpath and not native_sidecars and not coprocessors:
        sidecar_table.add_row("—", "none", "—", "—", "—")

    launch_table = Table(box=box.SIMPLE, expand=True)
    launch_table.add_column("Saved sidecar profile", style="bold cyan")
    launch_table.add_column("Workers")
    launch_table.add_column("Fastpath")
    for profile_id, entry in sorted(sidecar_profiles.items()):
        if isinstance(entry, dict):
            workers = [
                *(f"native:{item}" for item in (entry.get("sidecar_agent_ids") or [])),
                *(f"coprocessor:{item}" for item in (
                    entry.get("coprocessor_ids")
                    or entry.get("sidecar_ids")
                    or []
                )),
            ]
            launch_table.add_row(
                str(profile_id),
                ", ".join(map(str, workers)) or "none",
                "dedicated" if isinstance(entry.get("fastpath"), dict) else "global",
            )
    if not sidecar_profiles:
        launch_table.add_row("—", "none", "—")

    preset_table = Table(box=box.SIMPLE, expand=True)
    preset_table.add_column("Launch preset", style="bold cyan")
    preset_table.add_column("Inference profile")
    preset_table.add_column("Sidecar profile")
    preset_table.add_column("Workflow")
    for preset_id, entry in sorted(presets.items()):
        if isinstance(entry, dict):
            preset_table.add_row(
                str(preset_id),
                str(entry.get("inference_profile_id", "(unset)")),
                str(entry.get("sidecar_profile_id") or "global defaults"),
                str(entry.get("workflow_id") or "automatic task tier"),
            )
    if not presets:
        preset_table.add_row("—", "none", "—")

    _CONSOLE.print(
        Group(
            Panel(overview, title="ClaudeBrigade saved configuration", border_style="cyan"),
            Panel(role_table, title="Controller + worker-role profiles", border_style="blue"),
            Panel(sidecar_table, title="Native sidecars & coprocessors", border_style="magenta"),
            Panel(launch_table, title="Sidecar profiles", border_style="green"),
            Panel(preset_table, title="Launch presets", border_style="yellow"),
        )
    )


def show_saved(config_dir: Path, provider_keys: tuple[str, ...]) -> None:
    if _rich_interactive():
        _rich_show_saved(config_dir, provider_keys)
        return
    print(f"\nConfig directory: {config_dir}")
    credential_names = [key for key in sorted(provider_keys) if key.endswith("_API_KEY")]
    print("Loaded provider credentials: " + (", ".join(credential_names) or "none"))
    if credential_store_available():
        print("Credential backend: " + backend_label())
        for key in sorted(PROVIDER_SECRET_KEYS):
            names = slot_names(key, config_dir)
            if names:
                print(f"  {key} slots: {', '.join(names)}")
    print("\nController + worker-role routes -- controller + recon/implementer/adversary/repairer")
    profiles = _read_yaml(config_dir / "profiles.yaml").get("profiles") or {}
    for profile_id, value in sorted(profiles.items()):
        print(f"  {profile_id}:")
        controller_raw = value.get("controller")
        if isinstance(controller_raw, dict):
            controller_label = _role_label({"controller": controller_raw}, "controller")
        else:
            controller_label = value.get("controller_model") or "Claude/default"
        print(f"    controller: {controller_label}")
        for role in _ROLES:
            role_value = value.get(role)
            if isinstance(role_value, dict):
                model = role_value.get("model", "(unset)")
                provider = role_value.get("provider_id") or "default"
                endpoint = role_value.get("endpoint", "auto")
                model = f"{model} via {provider}/{endpoint}"
                raw_fallbacks = role_value.get("fallback_routes")
                if not isinstance(raw_fallbacks, list):
                    raw_fallbacks = role_value.get("fallback_models") or []
                fallback_ids = [
                    item if isinstance(item, str) else item.get("model", "")
                    for item in raw_fallbacks
                    if isinstance(item, (str, dict))
                ]
                fallback_ids = [item for item in fallback_ids if item]
                suffix = f" (fallbacks: {', '.join(fallback_ids)})" if fallback_ids else ""
            else:
                model = role_value or "(unset)"
                suffix = ""
            print(f"    {role}: {model}{suffix}")
    if not profiles:
        print("  (none)")

    print("\nSidecar & fastpath models -- coprocessor + workflow-triggered specialists")
    print(
        "  global bounded coprocessor lane: "
        f"{'enabled' if bool(_read_yaml(config_dir / 'sidecars.yaml').get('coprocessors_enabled', True)) else 'disabled'}"
    )
    fastpath = _read_yaml(config_dir / "fastpath.yaml").get("fastpath") or {}
    if fastpath:
        print(f"  fastpath coprocessor: model={fastpath.get('model_id', '(unset)')} enabled={fastpath.get('enabled', True)} modes={fastpath.get('modes', [])}")
    else:
        print("  fastpath coprocessor: (not configured)")
    sidecar_yaml = _read_yaml(config_dir / "sidecars.yaml")
    sidecar_agents = sidecar_yaml.get("sidecar_agents") or {}
    for agent_id, value in sorted(sidecar_agents.items()):
        print(
            f"  native sidecar '{agent_id}': model={value.get('model_id')} "
            f"via {value.get('provider_id') or 'default'}/{value.get('endpoint', 'auto')} "
            f"fallbacks={len(value.get('fallback_routes') or value.get('fallback_models') or [])} "
            f"native={value.get('native_agent_name')} mutate={value.get('can_mutate', False)}"
        )
    if not sidecar_agents:
        print("  native sidecar agents: (none)")
    sidecars = sidecar_yaml.get("coprocessors")
    if sidecars is None:
        sidecars = sidecar_yaml.get("sidecars") or {}
    for sidecar_id, value in sorted(sidecars.items()):
        print(
            f"  coprocessor '{sidecar_id}': model={value.get('model_id')} "
            f"via {value.get('provider_id') or 'default'}/{value.get('endpoint', 'auto')} "
            f"fallbacks={len(value.get('fallback_routes') or [])} "
            f"mode={value.get('mode', 'structured')} enabled={value.get('enabled', True)}"
        )
    if not sidecars:
        print("  coprocessors: (none)")

    sidecar_profiles = _read_yaml(config_dir / "sidecar_profiles.yaml").get("sidecar_profiles") or {}
    for profile_id, value in sorted(sidecar_profiles.items()):
        sidecar_ids = value.get("sidecar_ids") or []
        own_fastpath = value.get("fastpath")
        fastpath_label = f"model={own_fastpath.get('model_id')}" if isinstance(own_fastpath, dict) else "(uses global fastpath.yaml)"
        coprocessor_state = "on" if value.get("coprocessors_enabled", True) else "off"
        print(
            f"  sidecar profile '{profile_id}': sidecars=[{', '.join(sidecar_ids)}] "
            f"coprocessors={coprocessor_state} fastpath={fastpath_label}"
        )
    if not sidecar_profiles:
        print("  sidecar profiles: (none)")

    presets = _read_yaml(config_dir / "launch_presets.yaml").get("launch_presets") or {}
    for preset_id, value in sorted(presets.items()):
        print(
            f"  launch preset '{preset_id}': inference_profile={value.get('inference_profile_id')} "
            f"sidecar_profile={value.get('sidecar_profile_id') or '(none)'}"
        )
    if not presets:
        print("  launch presets: (none)")


def menu(config_dir: Path) -> int:
    registry, provider_keys = _load_registry(config_dir)
    while True:
        _render_menu(
            "ClaudeBrigade configuration",
            [
                (
                    "Controller + worker-role routes",
                    [
                        ("1", "Create/edit inference profile — models, endpoints, fallbacks"),
                        ("2", "Delete inference profile"),
                    ],
                ),
                (
                    "Native sidecars & coprocessors",
                    [
                        ("3", "Configure fastpath coprocessor — optional route/verify model"),
                        ("4", "Create/edit native sidecar worker — Claude Code Agent"),
                        ("5", "Create/edit bounded coprocessor — MCP structured call"),
                        ("6", "Toggle global bounded coprocessor lane"),
                        ("7", "Delete native sidecar worker"),
                        ("8", "Delete coprocessor"),
                        ("9", "Create/edit sidecar profile — allowed workers for a launch"),
                        ("10", "Delete sidecar profile"),
                    ],
                ),
                (
                    "Launch presets",
                    [
                        ("11", "Create/edit paired inference + sidecar preset"),
                        ("12", "Delete launch preset"),
                    ],
                ),
                (
                    "Providers & credentials",
                    [
                        ("13", "Add/edit provider API key"),
                        ("14", "Import already-loaded environment keys"),
                        ("15", "Refresh provider model catalogs"),
                    ],
                ),
                ("Inspect", [("16", "Show saved configuration"), ("q", "Quit")]),
            ],
            footer="Enter a number to open a section · q to quit",
        )
        choice = _menu_prompt("Choose", "1").strip().lower()
        try:
            if choice == "1":
                configure_inference(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "2":
                _delete_saved(config_dir, kind="inference profile")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "3":
                configure_fastpath(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "4":
                configure_native_sidecar_agent(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "5":
                configure_coprocessor(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "6":
                configure_global_coprocessor_lane(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "7":
                _delete_saved(config_dir, kind="native sidecar")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "8":
                _delete_saved(config_dir, kind="coprocessor")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "9":
                configure_sidecar_profile(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "10":
                _delete_saved(config_dir, kind="sidecar profile")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "11":
                configure_launch_preset(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "12":
                _delete_saved(config_dir, kind="launch preset")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "13":
                configure_keys(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "14":
                import_environment_credentials(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "15":
                refresh_catalogs(registry, force=True)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "16":
                show_saved(config_dir, provider_keys)
            elif choice in {"q", "quit", "exit"}:
                return 0
            else:
                print("Choose 1-16 or q.")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        except Exception as exc:
            print(f"Configuration was not saved: {exc}")


def _selected_profile_id(registry: ModelRegistry, requested: str | None = None) -> str | None:
    if requested:
        if requested not in registry.profiles:
            raise ValueError(f"unknown inference profile '{requested}'")
        return requested
    workflow = registry.workflows.get("normal")
    if workflow is not None and workflow.default_profile in registry.profiles:
        return workflow.default_profile
    return next(iter(sorted(registry.profiles)), None)


def show_slots_command(registry: ModelRegistry, profile_id: str | None = None) -> None:
    """Print the effective native model-slot projection."""
    profile_ids = [profile_id] if profile_id else sorted(registry.profiles)
    for selected in profile_ids:
        slots = registry.slot_alias_manifest(selected)
        print(f"[{selected}]")
        if not slots:
            print("  (legacy profile: role routes only)")
            continue
        for name, entry in slots.items():
            print(
                f"  {name:<7} -> {entry['model_id']} "
                f"({entry['public_model_alias']}, reserve={entry['reserve_class']})"
            )


def show_agents_command(
    registry: ModelRegistry,
    profile_id: str | None = None,
    sidecar_profile_id: str | None = None,
) -> None:
    """Print the effective named native-worker manifest."""
    selected = _selected_profile_id(registry, profile_id)
    manifest = registry.native_worker_manifest(selected, sidecar_profile_id)
    for native_name, entry in sorted(manifest.items()):
        print(
            f"{native_name}: model={entry.get('model_id')} "
            f"alias={entry.get('model_alias') or entry.get('public_model_alias')} "
            f"roles={','.join(entry.get('roles') or []) or '(none)'} "
            f"mutate={bool(entry.get('can_mutate'))} "
            f"isolation={entry.get('isolation', 'none')}"
        )


def explain_workflow_command(registry: ModelRegistry, tier: str) -> None:
    workflow = registry.workflows.get(tier)
    if workflow is None:
        raise ValueError(f"unknown workflow tier '{tier}'")
    print(f"workflow: {tier}")
    print(f"default profile: {workflow.default_profile}")
    for phase in workflow.phases:
        actor = phase.actor or ",".join(phase.roles) or "(unset)"
        dependencies = ",".join(phase.depends_on) or "none"
        worker = getattr(phase, "agent_id", None) or phase.sidecar_agent or phase.coprocessor or actor
        print(
            f"  {phase.id}: worker={worker} execution={phase.execution_kind} "
            f"depends_on={dependencies} fanout={phase.min_fanout}-{phase.max_fanout} "
            f"parallelism={phase.max_parallelism or phase.max_fanout} "
            f"mutation={phase.mutation}"
        )


def explain_effective_command(
    registry: ModelRegistry,
    profile_id: str | None = None,
) -> None:
    selected = _selected_profile_id(registry, profile_id)
    print(f"inference profile: {selected or '(none)'}")
    if selected:
        slots = registry.slot_alias_manifest(selected)
        for name, entry in slots.items():
            print(f"{name:<7} -> {entry['model_id']}")
    print("provider lanes:")
    for provider_id, provider in sorted(registry.providers.items()):
        limits = provider.limits
        print(
            f"  {provider_id}: max={limits.max_active_agents} "
            f"controller_reserve={limits.controller_reserve} "
            f"worker={limits.max_worker_concurrency} "
            f"priority={limits.priority_policy}"
        )
    print("native worker settings:")
    print("  CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS=3")
    print("  CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH=1")
    print("  CLAUDE_CODE_SUBAGENT_MODEL=(unset)")


def migrate_config_command(config_dir: Path) -> None:
    source_dir = Path(__file__).resolve().parents[2] / "config"
    provider_changed = migrate_provider_config(
        config_dir / "providers.yaml", source_dir / "providers.yaml"
    )
    sidecar_changed = migrate_sidecar_config(
        config_dir / "sidecars.yaml", source_dir / "sidecars.yaml"
    )
    workflow_changed = migrate_workflow_config(
        config_dir / "workflows.yaml", source_dir / "workflows.yaml"
    )
    print(
        "configuration migration: "
        f"providers={'updated' if provider_changed else 'unchanged'}, "
        f"sidecars={'updated' if sidecar_changed else 'unchanged'}, "
        f"workflows={'updated' if workflow_changed else 'unchanged'}"
    )


def certify_report_command(
    config_dir: Path,
    *,
    report_path: Path,
    provider_id: str,
    model_id: str,
    endpoint_id: str,
    configuration_hash: str | None = None,
    certification_id: str | None = None,
    litellm_version: str | None = None,
    expires_at: str | None = None,
) -> int:
    """Publish an explicitly reviewed compatibility report into SQLite.

    This command only consumes a pre-existing sanitized report. It never
    performs inference, grants model roles, or changes YAML configuration.
    The exact provider/model/endpoint identity is checked against the loaded
    registry before route evidence is published.
    """
    if not report_path.is_file():
        raise ValueError(f"report file does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON report {report_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("compatibility report must contain a JSON object")

    registry, _ = _load_registry(config_dir)
    try:
        spec = registry.get_model(model_id)
    except KeyError as exc:
        raise ValueError(f"unknown model in certification report: {model_id}") from exc

    if endpoint_id == "auto":
        if len(spec.endpoints) != 1:
            raise ValueError(
                "certification publication requires an exact --endpoint when "
                "the model has multiple endpoints"
            )
        endpoint_id = next(iter(spec.endpoints))
    endpoint = spec.endpoints.get(endpoint_id)
    if endpoint is None:
        raise ValueError(f"unknown endpoint {endpoint_id!r} for model {model_id!r}")
    expected_provider = endpoint.provider_id or spec.provider_id
    if expected_provider != provider_id:
        raise ValueError(
            f"endpoint {model_id}/{endpoint_id} belongs to provider "
            f"{expected_provider!r}, not {provider_id!r}"
        )

    rows = publish_contract_report(
        get_state(),
        provider_id=provider_id,
        model_id=model_id,
        endpoint_id=endpoint_id,
        configuration_hash=configuration_hash or registry.registry_hash(),
        report=report,
        certification_id=certification_id,
        litellm_version=litellm_version,
        expires_at=expires_at,
    )
    print(json.dumps({
        "status": "published",
        "provider_id": provider_id,
        "model_id": model_id,
        "endpoint_id": endpoint_id,
        "capabilities": [
            row["capability"] for row in rows if row["status"] == "pass"
        ],
        "rows": len(rows),
    }, indent=2))
    return 0
def _structured_command(argv: list[str]) -> int | None:
    """Handle the documented grouped read/migration commands."""
    if not argv or argv[0] not in {"workflow", "config"}:
        return None
    parser = argparse.ArgumentParser(prog=f"claude-brigade-config {argv[0]}")
    parser.add_argument("operation", choices=("explain", "migrate"))
    if argv[0] == "workflow":
        parser.add_argument("tier")
    else:
        parser.add_argument("--effective", action="store_true")
    parser.add_argument("--config-dir")
    parser.add_argument("--profile")
    args = parser.parse_args(argv[1:])
    config_dir = _config_dir(args.config_dir)
    if argv[0] == "config" and args.operation == "migrate":
        migrate_config_command(config_dir)
        return 0
    registry, _ = _load_registry(config_dir)
    if argv[0] == "workflow":
        if args.operation != "explain":
            parser.error("workflow supports only 'explain'")
        explain_workflow_command(registry, args.tier)
    elif args.operation == "explain":
        if not args.effective:
            parser.error("config explain requires --effective")
        explain_effective_command(registry, args.profile)
    else:
        parser.error("config supports only 'explain --effective' or 'migrate'")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    structured = _structured_command(raw_argv)
    if structured is not None:
        return structured
    parser = argparse.ArgumentParser(description="Interactively manage saved ClaudeBrigade inference and sidecar configuration")
    parser.add_argument(
        "command", nargs="?",
        choices=(
            "menu", "keys", "import-env", "refresh", "sidecar", "native-sidecar",
            "coprocessor", "coprocessor-lane", "inference",
            "fastpath", "sidecar-profile", "launch-preset", "launch-wizard", "show",
            "slots", "agents", "certify-report",
        ),
        default="menu",
    )
    parser.add_argument("--config-dir", help="BRIGADE_CONFIG_DIR override")
    parser.add_argument("--output-json", help="Write result as JSON to this path (for launcher integration)")
    parser.add_argument("--report", help="Sanitized JSON compatibility report for certify-report")
    parser.add_argument("--provider", dest="report_provider", help="Exact provider ID for certify-report")
    parser.add_argument("--model", dest="report_model", help="Exact logical model ID for certify-report")
    parser.add_argument("--endpoint", dest="report_endpoint", default="auto", help="Exact endpoint ID for certify-report")
    parser.add_argument("--configuration-hash", help="Registry/configuration hash captured by the report")
    parser.add_argument("--certification-id", help="Stable certification ID override")
    parser.add_argument("--litellm-version", help="LiteLLM version used for the report")
    parser.add_argument("--expires-at", help="Optional ISO timestamp for certification expiry")
    args = parser.parse_args(argv)
    config_dir = _config_dir(args.config_dir)
    if args.command == "menu":
        return menu(config_dir)
    registry, provider_keys = _load_registry(config_dir)

    def _output_result(data: dict) -> None:
        if args.output_json:
            with open(args.output_json, "w") as fh:
                json.dump(data, fh, default=str)
            os.chmod(args.output_json, 0o600)

    try:
        if args.command == "certify-report":
            if not args.report or not args.report_provider or not args.report_model:
                parser.error(
                    "certify-report requires --report, --provider, and --model"
                )
            return certify_report_command(
                config_dir,
                report_path=Path(args.report).expanduser(),
                provider_id=args.report_provider,
                model_id=args.report_model,
                endpoint_id=args.report_endpoint,
                configuration_hash=args.configuration_hash,
                certification_id=args.certification_id,
                litellm_version=args.litellm_version,
                expires_at=args.expires_at,
            )
        elif args.command == "launch-wizard":
            # Refresh catalogs first
            print("Refreshing model catalogs...")
            refresh_catalogs(registry)
            # Reload so newly discovered models are selectable
            registry, _ = _load_registry(config_dir)
            result = configure_launch_setup(config_dir, registry)
            out = {
                "status": result.status,
                "inference_profile_id": result.inference_profile_id,
                "sidecar_profile_id": result.sidecar_profile_id,
                "launch_preset_id": result.launch_preset_id,
            }
            _output_result(out)
            print(json.dumps(out, indent=2))
            return 0 if result.status == "saved" else 1
        elif args.command == "keys":
            configure_keys(config_dir, registry)
        elif args.command == "import-env":
            import_environment_credentials(config_dir, registry)
        elif args.command == "refresh":
            refresh_catalogs(registry, force=True)
        elif args.command == "sidecar":
            configure_sidecar_lane(config_dir, registry)
        elif args.command == "native-sidecar":
            configure_native_sidecar_agent(config_dir, registry)
        elif args.command == "coprocessor":
            configure_coprocessor(config_dir, registry)
        elif args.command == "coprocessor-lane":
            configure_global_coprocessor_lane(config_dir, registry)
        elif args.command == "inference":
            configure_inference(config_dir, registry)
        elif args.command == "fastpath":
            configure_fastpath(config_dir, registry)
        elif args.command == "sidecar-profile":
            configure_sidecar_profile(config_dir, registry)
        elif args.command == "launch-preset":
            configure_launch_preset(config_dir, registry)
        elif args.command == "slots":
            show_slots_command(registry)
        elif args.command == "agents":
            show_agents_command(registry)
        else:
            show_saved(config_dir, provider_keys)
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
