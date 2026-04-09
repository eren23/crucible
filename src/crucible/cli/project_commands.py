"""CLI handlers for ``crucible project`` — project-spec management.

Subcommands:
  - new:       create a new project spec from a template
  - list:      list existing project specs in .crucible/projects/
  - templates: list available built-in templates
  - validate:  load and validate an existing project spec

All rendered specs go through ``write_project_spec`` so they pass template
resolution, substitution, and post-render validation before hitting disk.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from crucible.core.config import list_project_specs, load_project_spec
from crucible.core.errors import ProjectTemplateError
from crucible.core.project_template import (
    iter_required_missing,
    list_templates,
    required_vars,
    write_project_spec,
)


def handle_project(args: argparse.Namespace) -> None:
    """Dispatch ``crucible project <subcommand>``."""
    cmd = getattr(args, "project_command", None)
    if cmd is None:
        print(
            "Usage: crucible project {new|list|templates|validate}",
            file=sys.stderr,
        )
        sys.exit(1)

    if cmd == "new":
        _cmd_new(args)
    elif cmd == "list":
        _cmd_list(args)
    elif cmd == "templates":
        _cmd_templates(args)
    elif cmd == "validate":
        _cmd_validate(args)
    else:
        print(f"Unknown project command: {cmd}", file=sys.stderr)
        sys.exit(1)


def _parse_set_overrides(overrides: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--set KEY=VALUE`` flags into a dict."""
    result: dict[str, str] = {}
    if not overrides:
        return result
    for raw in overrides:
        if "=" not in raw:
            raise ProjectTemplateError(
                f"--set expects KEY=VALUE, got: {raw!r}"
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise ProjectTemplateError(f"--set key is empty: {raw!r}")
        result[key] = value
    return result


def _cmd_new(args: argparse.Namespace) -> None:
    """Render a template and write the spec to .crucible/projects/<name>.yaml."""
    project_name = args.name
    template_name = args.template
    overwrite = bool(getattr(args, "force", False))
    non_interactive = bool(getattr(args, "no_prompt", False))

    try:
        overrides = _parse_set_overrides(getattr(args, "set_vars", None) or [])
    except ProjectTemplateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    # Figure out which required vars are still missing after CLI + env.
    supplied_keys = set(overrides.keys()) | set(os.environ.keys()) | {"PROJECT_NAME"}
    try:
        missing = [
            v for v in required_vars(template_name)
            if v not in supplied_keys
        ]
    except ProjectTemplateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if missing:
        if non_interactive:
            print(
                f"Error: missing required variables: {', '.join(missing)}. "
                f"Pass --set KEY=VALUE for each or export them in the environment.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Interactive prompt
        print(f"Template '{template_name}' needs the following variables:")
        for var in missing:
            try:
                value = input(f"  {var}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.", file=sys.stderr)
                sys.exit(130)
            if not value:
                print(
                    f"Error: {var} is required and no value given",
                    file=sys.stderr,
                )
                sys.exit(2)
            overrides[var] = value

    try:
        target = write_project_spec(
            template_name,
            project_name,
            overrides,
            overwrite=overwrite,
        )
    except ProjectTemplateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        rel = target.relative_to(Path.cwd())
    except ValueError:
        rel = target
    print(f"Created {rel}")
    print(f"  template: {template_name}")
    if overrides:
        print("  values:")
        for key, value in sorted(overrides.items()):
            print(f"    {key}={value}")
    print()
    print("Next steps:")
    print(f"  - Review and edit {rel} to match your training setup")
    print("  - crucible project validate " + project_name)
    print("  - crucible fleet bootstrap    # once a pod is ready")


def _cmd_list(args: argparse.Namespace) -> None:
    """List project specs in ``.crucible/projects/``."""
    del args  # unused
    specs = list_project_specs()
    if not specs:
        print("No project specs found in .crucible/projects/")
        print("Create one with: crucible project new <name> --template <template>")
        return
    print(f"Found {len(specs)} project spec(s):")
    for spec in specs:
        name = spec.get("name", "?")
        repo = spec.get("repo", "") or "(local)"
        launcher = spec.get("launcher", "") or "(none)"
        primary = spec.get("metrics_primary", "val_loss")
        print(f"  {name}")
        print(f"    repo:     {repo}")
        print(f"    launcher: {launcher}")
        print(f"    metric:   {primary}")


def _cmd_templates(args: argparse.Namespace) -> None:
    """List available built-in project templates."""
    del args  # unused
    templates = list_templates()
    if not templates:
        print("No project templates found.")
        return
    print(f"Available project templates ({len(templates)}):")
    for tmpl in templates:
        print(f"  {tmpl.name}")
        if tmpl.description:
            print(f"    {tmpl.description}")
        try:
            req = required_vars(tmpl.name)
        except ProjectTemplateError:
            req = []
        if req:
            print(f"    required vars: {', '.join(req)}")


def _cmd_validate(args: argparse.Namespace) -> None:
    """Load an existing spec and report success or parsing errors."""
    name = args.name
    try:
        spec = load_project_spec(name)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error loading spec {name!r}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"✓ {name} is valid")
    print(f"  repo:     {spec.repo or '(local)'}")
    print(f"  launcher: {spec.launcher or '(none)'}")
    print(f"  train:    {spec.train or '(none)'}")
    print(f"  metric:   {spec.metrics.primary} ({spec.metrics.direction})")
    if spec.env_forward:
        print(f"  forward:  {', '.join(spec.env_forward)}")
    if spec.env_set:
        keys = sorted(spec.env_set.keys())
        print(f"  env_set:  {', '.join(keys)}")
