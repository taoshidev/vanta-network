def main():
    import pymysql
    from pymysql import cursors
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    from vali_objects.utils.live_price_server import LivePriceFetcherServer
    from vali_objects.utils.position_source import PositionSourceManager, PositionSource
    from vali_objects.utils.vali_utils import ValiUtils

    # Configuration
    DRY_RUN = False  # Set to True to test without updating database
    BATCH_SIZE = 1000  # Number of updates per batch

    # Check for dry-run argument
    if len(sys.argv) > 1 and sys.argv[1] in ['--dry-run', '-n']:
        DRY_RUN = True
        print("*** DRY RUN MODE - No database updates will be performed ***\n")

    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcherServer(secrets, disable_ws=True)
    source_type = PositionSource.DATABASE

    # Parse database connection from URL
    import urllib.parse as urlparse

    db_secrets = ValiUtils.get_taoshi_ts_secrets()
    db_url = db_secrets['secrets']['db_ptn_editor_url']
    parsed = urlparse.urlparse(db_url)

    db_config = {
        'host': parsed.hostname,
        'port': parsed.port or 3306,
        'user': parsed.username,
        'password': parsed.password,
        'database': parsed.path.lstrip('/'),
        'cursorclass': cursors.DictCursor
    }

    # Load positions
    position_source_manager = PositionSourceManager(source_type)
    hk_to_positions = position_source_manager.load_positions(
        end_time_ms=None,
        hotkeys=None)
    print(f"Loaded {sum(len(v) for v in hk_to_positions.values())} positions from {source_type} for {len(hk_to_positions)} hotkeys")

    # Track percentage changes for histogram and positions to update
    percentage_changes = []
    positions_to_update = []  # List of tuples (position_uuid, new_return_at_close, new_current_return)
    n_positions_changed = 0
    n_positions_checked = 0
    BATCH_SIZE = 1000  # Update database in batches

    total_positions = sum(len(positions) for positions in hk_to_positions.values())
    positions_processed = 0

    for hk_idx, (hk, positions) in enumerate(hk_to_positions.items(), 1):
        print(f"[{hk_idx}/{len(hk_to_positions)}] Processing hotkey {hk} with {len(positions)} positions...")
        for p in positions:
            positions_processed += 1

            # Progress update every 1000 positions
            if positions_processed % 1000 == 0:
                print(f"  Progress: {positions_processed}/{total_positions} positions ({(positions_processed/total_positions)*100:.1f}%), {n_positions_changed} changes found")

            if p.is_open_position:
                continue
            n_positions_checked += 1
            original_return = p.return_at_close
            p.rebuild_position_with_updated_orders(live_price_fetcher)
            new_return = p.return_at_close
            new_current = p.current_return

            if new_return != original_return:
                n_positions_changed += 1
                # Calculate percentage change
                if original_return != 0:
                    pct_change = ((new_return - original_return) / abs(original_return)) * 100
                else:
                    # Handle case where original return was 0
                    pct_change = (new_return - original_return) * 100
                percentage_changes.append(pct_change)

                # Add to update list
                positions_to_update.append((p.position_uuid, new_return, new_current))

                # Batch update when we reach BATCH_SIZE
                if len(positions_to_update) >= BATCH_SIZE:
                    if DRY_RUN:
                        print(f"[DRY RUN] Would update batch of {len(positions_to_update)} positions")
                    else:
                        print(f"Updating batch of {len(positions_to_update)} positions...")
                        try:
                            connection = pymysql.connect(**db_config)
                            with connection.cursor() as cursor:
                                # Use batch update with executemany for efficiency
                                update_query = """
                                    UPDATE position
                                    SET return_at_close = %s, curr_return = %s
                                    WHERE position_uuid = %s
                                """
                                # Reorder tuples for the query (return_at_close, curr_return, position_uuid)
                                update_data = [(ret, curr, uuid) for uuid, ret, curr in positions_to_update]
                                cursor.executemany(update_query, update_data)
                                connection.commit()
                                print(f"Successfully updated {len(positions_to_update)} positions")
                        except Exception as e:
                            print(f"Error updating batch: {e}")
                            connection.rollback()
                        finally:
                            connection.close()

                    # Clear the batch
                    positions_to_update = []

    # Update any remaining positions
    if positions_to_update:
        if DRY_RUN:
            print(f"[DRY RUN] Would update final batch of {len(positions_to_update)} positions")
        else:
            print(f"Updating final batch of {len(positions_to_update)} positions...")
            try:
                connection = pymysql.connect(**db_config)
                with connection.cursor() as cursor:
                    update_query = """
                        UPDATE position
                        SET return_at_close = %s, curr_return = %s
                        WHERE position_uuid = %s
                    """
                    # Reorder tuples for the query
                    update_data = [(ret, curr, uuid) for uuid, ret, curr in positions_to_update]
                    cursor.executemany(update_query, update_data)
                    connection.commit()
                    print(f"Successfully updated {len(positions_to_update)} positions")
            except Exception as e:
                print(f"Error updating final batch: {e}")
                connection.rollback()
            finally:
                connection.close()

    # Print summary statistics
    print(f"\n=== Summary ===")
    print(f"Total positions checked: {n_positions_checked}")
    print(f"Positions with changed returns: {n_positions_changed} ({(n_positions_changed/n_positions_checked)*100:.2f}%)")
    if DRY_RUN:
        print(f"[DRY RUN] No database updates performed - would have updated {n_positions_changed} positions")
    else:
        print(f"Database updates completed: {n_positions_changed} positions updated")

    if percentage_changes:
        print(f"\n=== Percentage Change Statistics ===")
        print(f"Mean change: {np.mean(percentage_changes):.4f}%")
        print(f"Median change: {np.median(percentage_changes):.4f}%")
        print(f"Std dev: {np.std(percentage_changes):.4f}%")
        print(f"Min change: {np.min(percentage_changes):.4f}%")
        print(f"Max change: {np.max(percentage_changes):.4f}%")

        # Create histogram of percentage changes
        plt.figure(figsize=(12, 6))

        # Clip extreme values for better visualization (optional)
        # You can adjust or remove these bounds based on your data
        clipped_changes = np.clip(percentage_changes, -10, 10)

        # Create histogram
        n_bins = 50
        plt.hist(clipped_changes, bins=n_bins, edgecolor='black', alpha=0.7)

        # Add vertical line at mean and median
        plt.axvline(np.mean(percentage_changes), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(percentage_changes):.4f}%')
        plt.axvline(np.median(percentage_changes), color='green', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(percentage_changes):.4f}%')
        plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        plt.xlabel('Percentage Change in return_at_close (%)')
        plt.ylabel('Number of Positions')
        plt.title(f'Distribution of Return Changes After Fee Update\n({n_positions_changed} positions with changes)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add text box with statistics
        stats_text = f'Total Changed: {n_positions_changed}\nMean: {np.mean(percentage_changes):.4f}%\nStd: {np.std(percentage_changes):.4f}%'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 verticalalignment='top', fontsize=10)

        plt.tight_layout()

        # Save the figure
        plt.savefig('return_change_histogram.png', dpi=300, bbox_inches='tight')
        print(f"\nHistogram saved as 'return_change_histogram.png'")

        # Show the plot
        plt.show()
    else:
        print("\nNo positions had return changes.")

if __name__ == "__main__":
    main()
