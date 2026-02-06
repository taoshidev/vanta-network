# Migrations

Migration scripts live in `runnable/migrations/` and run automatically before the validator restarts after an update.

## Creating a migration

1. Create a new Python file in `runnable/migrations/` (e.g., `migrate_something.py`)
2. Implement a `main()` function that returns `True` on success, `False` on failure:

```python
def main() -> bool:
    # Migration logic here
    return True
```

## How it works

- On update, `run.sh` calls `python3 runnable/run_migrations.py`
- The runner scans `runnable/migrations/` for `.py` files
- Migrations that aren't in `migrations_completed.txt` are run alphabetically
- Successful migrations are recorded in `migrations_completed.txt`
- If a migration fails, the validator restart is aborted

## Running manually

```bash
# Run pending migrations
python3 runnable/run_migrations.py

# Dry run (shows what would run without executing)
python3 runnable/run_migrations.py --dry-run
```

## Development

Prefix work-in-progress migrations with `_` to prevent them from running:

```
_migrate_wip.py      # Ignored by runner
migrate_something.py # Will run
```

Remove the prefix when the migration is ready.

## Re-running a migration

Remove its filename from `runnable/migrations/migrations_completed.txt`, or create a new migration file with a different name.
