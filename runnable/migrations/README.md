# Migrations

Migrations live in per-version folders under `runnable/migrations/`. At validator startup, the runner runs the migrations in the folder matching `subnet_version` in `meta/meta.json`.

```
runnable/migrations/
  8.17.0/
    migrate_something.py
```

## Creating a migration

1. Create `runnable/migrations/<release version>/migrate_something.py`
2. Implement a `main()` function that returns `True` on success, `False` on failure:

```python
def main() -> bool:
    # Migration logic here
    return True
```

## How it works

- On startup, `neurons/validator.py` calls the runner. Under `--split-state` the state tier (`vanta_api/run_state_server.py`) calls it instead.
- The runner runs, alphabetically, each migration in `migrations/<subnet_version>/` that isn't already in `migrations_completed.txt`
- Successful migrations are recorded in `migrations_completed.txt` as `<version>/<filename>` (gitignored, local to each validator)
- If a migration fails, the runner stops and the validator starts without the remaining migrations

Once the version is bumped, older folders stop running. Delete them whenever you like; they don't need to keep working as the code changes.

## Running manually

```bash
# Run pending migrations
python3 runnable/run_migrations.py

# Dry run (shows what would run without executing)
python3 runnable/run_migrations.py --dry-run
```

## Development

Prefix work-in-progress migrations with `_` to prevent them from running.

## Re-running a migration

Remove its `<version>/<filename>` line from `runnable/migrations/migrations_completed.txt`. It only runs again while its folder matches the current version.
