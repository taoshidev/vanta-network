# Migrations

Migration scripts live in `runnable/migrations/`. The ones listed in `ACTIVE_MIGRATIONS` (`runnable/run_migrations.py`) run automatically when the validator starts.

## Creating a migration

1. Create a new Python file in `runnable/migrations/` (e.g., `migrate_something.py`)
2. Implement a `main()` function that returns `True` on success, `False` on failure:

```python
def main() -> bool:
    # Migration logic here
    return True
```

3. Add the filename to `ACTIVE_MIGRATIONS` in `runnable/run_migrations.py`

## How it works

- On startup, `neurons/validator.py` calls the runner. Under `--split-state` the state tier (`vanta_api/run_state_server.py`) calls it instead.
- The runner goes through `ACTIVE_MIGRATIONS` in order and runs each one that isn't already in `migrations_completed.txt`
- Successful migrations are recorded in `migrations_completed.txt` (gitignored, local to each validator)
- If a migration fails, the runner stops and the validator starts without the remaining migrations

Files in this directory that are not in `ACTIVE_MIGRATIONS` never run.

## Retiring a migration

Once every validator has run a migration (auto-update runs it within ~30 minutes of release), remove it from `ACTIVE_MIGRATIONS`. After that it never runs again, including on new validators, whose state already comes from up-to-date code. Retired files are kept for reference only and don't need to keep working as the code changes.

A new validator still runs whatever is in `ACTIVE_MIGRATIONS`, so active migrations should be a no-op on state that's already migrated.

## Running manually

```bash
# Run pending migrations
python3 runnable/run_migrations.py

# Dry run (shows what would run without executing)
python3 runnable/run_migrations.py --dry-run
```

## Re-running a migration

Remove its filename from `runnable/migrations/migrations_completed.txt` (and make sure it's in `ACTIVE_MIGRATIONS`), or create a new migration file with a different name.
