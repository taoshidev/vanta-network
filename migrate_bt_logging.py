#!/usr/bin/env python3
"""
Migrate bt.logging calls to stdlib logging via shared_objects/log.py.

Usage:
    python migrate_bt_logging.py           # dry run, show unified diffs
    python migrate_bt_logging.py --apply   # apply changes in-place
    python migrate_bt_logging.py --apply path/to/file.py  # single file
"""
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
LOGGER_IMPORT = "from shared_objects.log import logger"

# bt.logging.METHOD -> logger.MAPPED_METHOD
METHOD_MAP = {
    "info": "info",
    "error": "error",
    "warning": "warning",
    "debug": "debug",
    "success": "info",   # no success level in stdlib
    "trace": "debug",    # no trace level in stdlib
}

METHODS_RE = "|".join(METHOD_MAP)


def replace_calls(content: str) -> str:
    def _map_method(m: re.Match) -> str:
        return f"logger.{METHOD_MAP[m.group(1)]}("

    return re.sub(
        rf"bt\.logging\.({METHODS_RE})\s*\(",
        _map_method,
        content,
    )


def replace_config(content: str) -> tuple[str, bool]:
    """Replace setup/config calls. Returns (new_content, needs_logging_import)."""
    needs_logging = False

    replacements = [
        # enable_X() -> setLevel
        (r"bt\.logging\.enable_info\s*\(\s*\)", "logger.setLevel(logging.INFO)"),
        (r"bt\.logging\.enable_debug\s*\(\s*\)", "logger.setLevel(logging.DEBUG)"),
        (r"bt\.logging\.enable_trace\s*\(\s*\)", "logger.setLevel(logging.DEBUG)"),
        (r"bt\.logging\.enable_default\s*\(\s*\)", "logger.setLevel(logging.INFO)"),
    ]

    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            needs_logging = True

    # set_debug(expr) / set_trace(expr) — conditional level based on boolean arg
    for method in ("set_debug", "set_trace"):
        pattern = rf"bt\.logging\.{method}\s*\(([^)]+)\)"
        if re.search(pattern, content):
            content = re.sub(
                pattern,
                lambda m: f"logger.setLevel(logging.DEBUG if ({m.group(1).strip()}) else logging.INFO)",
                content,
            )
            needs_logging = True

    # add_args(parser) — remove entire line (argparse hook, no stdlib equivalent)
    content = re.sub(r"[ \t]*bt\.logging\.add_args\s*\([^)]+\)[ \t]*\n", "", content)

    # _logger attribute access — direct logger reference
    content = re.sub(r"bt\.logging\._logger", "logger", content)

    return content, needs_logging


def find_last_toplevel_import(lines: list[str]) -> int:
    """Return the index of the last top-level import line (col-0 import/from)."""
    last = -1
    for i, line in enumerate(lines):
        if re.match(r"^(?:import|from)\s", line):
            last = i
    return last


def inject_imports(content: str, needs_logging: bool) -> str:
    lines = content.split("\n")

    has_logger_import = any(LOGGER_IMPORT in line for line in lines)
    has_logging_import = any(re.match(r"^import logging\b", line) for line in lines)

    inserts: list[str] = []
    if needs_logging and not has_logging_import:
        inserts.append("import logging")
    if not has_logger_import:
        inserts.append(LOGGER_IMPORT)

    if not inserts:
        return content

    last_import = find_last_toplevel_import(lines)

    # If logger is referenced before the last import (e.g. module-level enable_info
    # between imports), move the insertion point to just before that first reference.
    first_logger_ref = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(r"\blogger\b", line) and not line[0].isspace():
            first_logger_ref = i
            break

    if first_logger_ref != -1 and first_logger_ref <= last_import:
        insert_idx = first_logger_ref - 1
    else:
        insert_idx = last_import

    if insert_idx < 0:
        return "\n".join(inserts) + "\n" + content

    lines = lines[: insert_idx + 1] + inserts + lines[insert_idx + 1 :]
    return "\n".join(lines)


def migrate(path: Path, apply: bool) -> bool:
    """Return True if the file changed (or would change)."""
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False  # skip non-UTF-8 files

    if "bt.logging" not in original:
        return False

    content = replace_calls(original)
    content, needs_logging = replace_config(content)
    content = inject_imports(content, needs_logging)

    if content == original:
        return False

    if not apply:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{path.relative_to(ROOT)}",
            tofile=f"b/{path.relative_to(ROOT)}",
            n=2,
        )
        sys.stdout.writelines(diff)
    else:
        path.write_text(content)

    return True


def main() -> None:
    args = sys.argv[1:]
    apply = "--apply" in args
    targets = [Path(a).resolve() for a in args if not a.startswith("--")]

    if not apply:
        print("# DRY RUN — pass --apply to write changes\n")

    if targets:
        paths = targets
    else:
        paths = sorted(
            p for p in ROOT.rglob("*.py")
            if "__pycache__" not in p.parts
            and p.name not in ("migrate_bt_logging.py", "log.py")
        )

    changed: list[Path] = []
    for path in paths:
        if migrate(path, apply):
            changed.append(path)

    verb = "Modified" if apply else "Would modify"
    print(f"\n# {verb} {len(changed)} file(s):")
    for p in changed:
        print(f"#   {p.relative_to(ROOT)}")

    if not apply and changed:
        print("\n# Re-run with --apply to write changes.")
        print("# After applying, grep for remaining bt.logging references:")
        print("#   grep -r 'bt\\.logging' . --include='*.py'")


if __name__ == "__main__":
    main()
