# Regenerating Validator State Manually

When initiating a validator for the first time or recovering from unexpected downtime, it's crucial to synchronize your validator's positions to maintain consensus. Failure to do so may result in missing orders/positions, leading to a lower VTRUST score.

Note: as of 6/11/24, Taoshi added support for automatic validator restoration. Only follow these steps if you are unable to restore your validator using the automatic restoration process.

**Important**: As of the latest version, validator checkpoint files are now compressed (`.gz` format) for improved efficiency and reduced storage requirements. The restoration script automatically detects and handles both compressed and uncompressed checkpoint files.

**File Format Compatibility**: The restore script supports both formats:
- **Compressed files** (`.gz`): Preferred format, ~90% smaller (e.g., 15MB vs 150MB)
- **Uncompressed files** (`.json`): Legacy format, still fully supported

**Download Methods**: Depending on how you download the checkpoint file, it may be automatically decompressed:
- **Browser downloads**: Often automatically decompress `.gz` files to `.json`
- **HTTP clients**: May decompress based on `Accept-Encoding` headers
- **Command line tools**: `curl` and `wget` preserve compression when used correctly

If you receive a decompressed `.json` file instead of `.gz`, simply rename it to `validator_checkpoint.json` (without .gz extension) and the restore script will handle it correctly.
## Purpose

The steps detailed below regenerate the `validation/*` directory by fetching the latest trade data and metrics from the mainnet, ensuring your validator remains in sync with the network.


## Steps 
1. **Get Restoration File**: Ping a Taoshi team member in the Discord and they will send you a near realtime checkpoint file. (Distribution limited to once a week for verified validators)
   
2. **Prepare for Restoration**: Transfer the checkpoint file to the root level of the `proprietary-trading-network` directory on your validator:
   - **If you receive a compressed file**: Name it `validator_checkpoint.json.gz`
   - **If you receive an uncompressed file**: Name it `validator_checkpoint.json`
   - **Auto-detection**: The script will automatically detect which format you have
3. **Stop Validator**: Temporarily halt your validator with PM2 using `pm2 stop sn8 vanta`
4. **Run Restoration Script**: Within the `proprietary-trading-network` directory, execute:

    ```bash
    python3 restore_validator_from_backup.py
    ```

     The script will automatically detect your file format:
    ```
    INFO | Found compressed checkpoint file: /path/to/validator_checkpoint.json.gz
    ```
    or
    ```
    INFO | Found uncompressed checkpoint file: /path/to/validator_checkpoint.json
    ```
    
     Successful restoration is indicated by:
    ```
    INFO | regeneration complete in 19.78 seconds
    ```
     If restoration fails, consult the failure log for troubleshooting steps.

## Troubleshooting Common Issues

**Problem: File naming mismatches**

If you see an error like:
```
ERROR | File validator_checkpoint.json appears to contain compressed data but lacks .gz extension.
ERROR | Solution: Add .gz extension and rename to validator_checkpoint.json.gz
```
**Solution**: Your file contains compressed data but is named incorrectly. Rename it to `validator_checkpoint.json.gz`.

If you see an error like:
```
ERROR | File validator_checkpoint.json.gz has .gz extension but contains uncompressed data.
ERROR | Solution: Remove the .gz extension and rename to validator_checkpoint.json
```
**Solution**: Your file contains uncompressed data but is named incorrectly. Rename it to `validator_checkpoint.json`.

**Problem: No checkpoint file found**
```
ERROR | No checkpoint file found at validator_checkpoint.json or validator_checkpoint.json.gz
```
**Solution**: Ensure your checkpoint file is in the root directory with the correct name.

5. **Restart Validator**: Resume your PM2 processes with `pm2 start sn8` (will launch `vanta` automatically).


## Verify Backups (optional)

Confirm the creation of a `backups/` directory containing previous data (positions, eliminations, plagiarism). For example:
    ```
    (venv) jbonilla@MacBook-Air prop-net % ls -lh backups/20240325_010535


    total 16
    -rw-r--r--  1 jbonilla staff 164B Mar 25 01:05 eliminations.json
    drwxr-xr-x 148 jbonilla staff 4.6K Mar 25 00:36 miners
    -rw-r--r--  1 jbonilla staff 165B Mar 25 01:05 plagiarism.json
    ```

## Future Improvements

We're committed to refining this process through automation and your feedback. In the future, this process should be fully automated. In the meantime, please report any issues encountered during restoration.
