"""
Migration runner for the validator.

Runs the migrations in ACTIVE_MIGRATIONS that haven't been executed yet.

Each migration must have a main() function that returns True on success, False on failure.

Usage:
    python runnable/run_migrations.py [--dry-run]
"""

import importlib.util
import os
import sys

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")
COMPLETED_FILE = os.path.join(MIGRATIONS_DIR, "migrations_completed.txt")

# Migrations to run, in order. Remove once every validator has run them.
ACTIVE_MIGRATIONS: list[str] = []


def get_completed_migrations() -> set[str]:
    """Load the set of completed migration names."""
    if not os.path.exists(COMPLETED_FILE):
        return set()
    with open(COMPLETED_FILE, "r") as f:
        return {line.strip() for line in f if line.strip()}


def mark_completed(migration_name: str) -> None:
    """Mark a migration as completed."""
    os.makedirs(os.path.dirname(COMPLETED_FILE), exist_ok=True)
    with open(COMPLETED_FILE, "a") as f:
        f.write(f"{migration_name}\n")


def get_pending_migrations() -> list[str]:
    """Get the active migrations that haven't been run yet, in ACTIVE_MIGRATIONS order."""
    completed = get_completed_migrations()
    return [m for m in ACTIVE_MIGRATIONS if m not in completed]


def run_migration(filename: str, dry_run: bool = False) -> bool:
    """Run a single migration script."""
    filepath = os.path.join(MIGRATIONS_DIR, filename)

    print(f"{'[DRY RUN] Would run' if dry_run else 'Running'}: {filename}")

    if dry_run:
        return True

    try:
        spec = importlib.util.spec_from_file_location(filename[:-3], filepath)
        if spec is None or spec.loader is None:
            print(f"  Failed to load migration: {filename}")
            return False

        module = importlib.util.module_from_spec(spec)
        sys.modules[filename[:-3]] = module
        spec.loader.exec_module(module)

        if hasattr(module, "main"):
            result = module.main()
            # If main() returns None, treat as success (for backwards compatibility)
            return result is None or result is True
        else:
            print(f"  Warning: {filename} has no main() function, skipping")
            return False

    except Exception as e:
        print(f"  Migration failed with error: {e}")
        return False


def main() -> bool:
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv

    if dry_run:
        print("*** DRY RUN MODE - No migrations will be executed ***\n")

    pending = get_pending_migrations()

    if not pending:
        print("No pending migrations.")
        return True

    print(f"Found {len(pending)} pending migration(s):\n")

    success_count = 0
    for migration in pending:
        if not os.path.exists(os.path.join(MIGRATIONS_DIR, migration)):
            print(f"  Skipping {migration}: file not found in {MIGRATIONS_DIR}\n")
            continue
        if run_migration(migration, dry_run):
            if not dry_run:
                mark_completed(migration)
            success_count += 1
            print(f"  Completed: {migration}\n")
        else:
            print(f"  FAILED: {migration}")
            print("Stopping migration runner due to failure.")
            return False

    print(f"\nAll {success_count} migration(s) completed successfully.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
