# Migration Guide: PTN to Vanta

This guide explains how to migrate your validator from the old `ptn_api` directory structure to the new `vanta_api` structure.

## What Changed?

As part of the rebrand from Proprietary Trading Network (PTN) to Vanta Network:
- The `ptn_api` directory has been renamed to `vanta_api`
- All class names and imports have been updated accordingly
- The PM2 process name changed from `ptn` to `vanta`

**Important**: The `run.sh` script now includes automatic cleanup to stop the old `ptn` process when starting the new `vanta` process, preventing both from running simultaneously.

## Do I Need to Migrate?

**Yes**, but the migration is designed to be gradual and non-breaking:

- ✅ **Backwards Compatible**: The system will automatically fall back to `ptn_api/` if files are not found in `vanta_api/`
- ⚠️ **Deprecated Warning**: You'll see warnings in your logs if you're still using `ptn_api/`
- 📅 **Grace Period**: You have time to migrate at your convenience

## Files That Need Migration

One file needs to be migrated from `ptn_api/` to `vanta_api/`:

1. **`api_keys.json`** - Your API authentication keys

## Migration Methods

### Method 1: Automated Migration Script (Recommended)

Run the migration script from your repository root:

```bash
cd /path/to/proprietary-trading-network
./vanta_api/migrate_from_ptn.sh
```

The script will:
- ✅ Check for existing files in both directories
- ✅ Safely copy files to `vanta_api/`
- ✅ Prompt before overwriting existing files
- ✅ Leave original files in `ptn_api/` for safety

### Method 2: Manual Migration

If you prefer to migrate manually:

```bash
cd /path/to/proprietary-trading-network

# Copy api_keys.json
cp ptn_api/api_keys.json vanta_api/api_keys.json
```

## Verification

After migration, verify the files are in place:

```bash
ls -la vanta_api/
# You should see:
# - api_keys.json
```

Start your validator and check the logs. You should **NOT** see warnings about using deprecated `ptn_api/` paths.

## PM2 Process Migration

The PM2 process name has changed from `ptn` to `vanta`. The `run.sh` script handles this automatically:

✅ **Automatic Cleanup**: When you update and run the new version, the script will:
1. Check for the old `ptn` process
2. Stop it if found
3. Start the new `vanta` process

You can verify the migration with:
```bash
pm2 status
# You should see "vanta" listed, not "ptn"
```

If you still see the old `ptn` process running, stop it manually:
```bash
pm2 delete ptn
```

## After Migration

Once you've verified everything works correctly:

1. ✅ Files are in `vanta_api/`
2. ✅ No deprecation warnings in logs
3. ✅ Validator is functioning normally
4. ✅ Only `vanta` process running (not `ptn`)

You can safely delete the old `ptn_api/` directory:

```bash
# Optional: Only after verifying everything works
rm -rf ptn_api/
```

## Troubleshooting

### "Permission denied" when running the script

Make the script executable:
```bash
chmod +x vanta_api/migrate_from_ptn.sh
```

### Files already exist in vanta_api/

The script will ask if you want to overwrite. Choose `y` if you want to use the version from `ptn_api/`, or `n` to keep the existing version.

### Still seeing deprecation warnings after migration

1. Check that files exist in `vanta_api/`:
   ```bash
   ls -la vanta_api/*.json
   ```

2. Restart your validator process

3. Check file permissions:
   ```bash
   chmod 644 vanta_api/*.json
   ```

## Questions?

If you encounter any issues during migration, please reach out to the Taoshi team on Discord.
