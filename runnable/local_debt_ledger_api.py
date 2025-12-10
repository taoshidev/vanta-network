#!/usr/bin/env python3
"""
Standalone script to test Debt Ledger API endpoint and visualize data for validation.

Usage:
    python local_debt_ledger_api.py --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY
    python local_debt_ledger_api.py --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY --save-plot output.png
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Dict, List, Any
import urllib.request
import urllib.error

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


class DebtLedgerTester:
    """Test and visualize debt ledger data from the API."""

    def __init__(self, api_host: str, api_port: int, api_key: str):
        self.api_host = api_host
        self.api_port = api_port
        self.api_key = api_key
        self.base_url = f"http://{api_host}:{api_port}"

    def fetch_debt_ledger(self, hotkey: str) -> Dict[str, Any]:
        """Fetch debt ledger data from the API."""
        url = f"{self.base_url}/debt-ledger/{hotkey}"

        print(f"Fetching debt ledger data from: {url}")

        try:
            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Bearer {self.api_key}')

            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    print(f"✓ Successfully fetched debt ledger data")
                    return data
                else:
                    print(f"✗ HTTP Error: {response.status}")
                    sys.exit(1)

        except urllib.error.HTTPError as e:
            print(f"✗ HTTP Error {e.code}: {e.reason}")
            print(f"Response: {e.read().decode('utf-8')}")
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"✗ Connection Error: {e.reason}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate the structure and content of debt ledger data."""
        print("\n=== Data Validation ===")

        issues = []

        # Check required fields
        required_fields = ['hotkey', 'checkpoints']
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")

        if 'checkpoints' in data:
            checkpoints = data['checkpoints']
            print(f"Total checkpoints: {len(checkpoints)}")

            if not checkpoints:
                issues.append("No checkpoints found in debt ledger")
            else:
                # Validate checkpoint structure
                required_checkpoint_fields = ['timestamp_ms', 'emissions', 'performance']

                for i, checkpoint in enumerate(checkpoints[:3]):  # Check first 3 checkpoints
                    for field in required_checkpoint_fields:
                        if field not in checkpoint:
                            issues.append(f"Checkpoint {i} missing field: {field}")

                    # Check emissions structure
                    if 'emissions' in checkpoint:
                        if 'chunk_alpha' not in checkpoint['emissions']:
                            issues.append(f"Checkpoint {i} emissions missing chunk_alpha")
                        if 'chunk_tao' not in checkpoint['emissions']:
                            issues.append(f"Checkpoint {i} emissions missing chunk_tao")
                        if 'alpha_balance_snapshot' not in checkpoint['emissions']:
                            issues.append(f"Checkpoint {i} emissions missing alpha_balance_snapshot")
                        if 'tao_balance_snapshot' not in checkpoint['emissions']:
                            issues.append(f"Checkpoint {i} emissions missing tao_balance_snapshot")

                    # Check performance structure
                    if 'performance' in checkpoint:
                        if 'portfolio_return' not in checkpoint['performance']:
                            issues.append(f"Checkpoint {i} performance missing portfolio_return")
                        if 'realized_pnl' not in checkpoint['performance']:
                            issues.append(f"Checkpoint {i} performance missing realized_pnl")
                        if 'unrealized_pnl' not in checkpoint['performance']:
                            issues.append(f"Checkpoint {i} performance missing unrealized_pnl")

                # Validate summary if present
                if 'summary' in data:
                    summary = data['summary']
                    print(f"Cumulative emissions (TAO): {summary.get('cumulative_emissions_tao', 0):.4f}")
                    print(f"Cumulative emissions (USD): ${summary.get('cumulative_emissions_usd', 0):,.2f}")
                    print(f"Portfolio return: {summary.get('portfolio_return', 0):.6f}")
                    print(f"Realized PnL: {summary.get('realized_pnl', 0):.2f}")
                    print(f"Unrealized PnL: {summary.get('unrealized_pnl', 0):.2f}")

        if issues:
            print("\n⚠️  Validation Issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✓ All validation checks passed")
            return True

    def print_summary(self, data: Dict[str, Any]):
        """Print summary statistics of the debt ledger."""
        print("\n=== Debt Ledger Summary ===")

        hotkey = data.get('hotkey', 'Unknown')
        checkpoints = data.get('checkpoints', [])
        summary = data.get('summary', {})

        print(f"Hotkey: {hotkey}")
        print(f"Total checkpoints: {len(checkpoints)}")

        if summary:
            print(f"\n--- Overall Summary ---")
            print(f"Cumulative Emissions:")
            print(f"  Alpha: {summary.get('cumulative_emissions_alpha', 0):.2f}")
            print(f"  TAO: {summary.get('cumulative_emissions_tao', 0):.4f}")
            print(f"  USD: ${summary.get('cumulative_emissions_usd', 0):,.2f}")
            print(f"Portfolio Return: {summary.get('portfolio_return', 0):.6f} ({summary.get('portfolio_return', 0)*100:.4f}%)")
            print(f"Weighted Score: {summary.get('weighted_score', 0):.6f}")
            print(f"Realized PnL: ${summary.get('realized_pnl', 0):.2f}")
            print(f"Unrealized PnL: ${summary.get('unrealized_pnl', 0):.2f}")

        if not checkpoints:
            print("\nNo checkpoints to summarize")
            return

        # Extract key metrics from checkpoints
        checkpoint_data = []
        for cp in checkpoints:
            checkpoint_data.append({
                'timestamp_ms': cp['timestamp_ms'],
                'chunk_alpha': cp['emissions']['chunk_alpha'],
                'chunk_tao': cp['emissions']['chunk_tao'],
                'chunk_usd': cp['emissions']['chunk_usd'],
                'alpha_balance_snapshot': cp['emissions']['alpha_balance_snapshot'],
                'portfolio_return': cp['performance']['portfolio_return'],
                'realized_pnl': cp['performance']['realized_pnl'],
                'unrealized_pnl': cp['performance']['unrealized_pnl'],
                'weighted_score': cp['derived']['weighted_score']
            })

        df = pd.DataFrame(checkpoint_data)
        df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

        # Calculate alpha balance validation metrics
        initial_alpha_balance = df['alpha_balance_snapshot'].iloc[0]
        final_alpha_balance = df['alpha_balance_snapshot'].iloc[-1]
        cumulative_alpha_emissions = df['chunk_alpha'].sum()
        balance_delta = final_alpha_balance - initial_alpha_balance
        difference = abs(cumulative_alpha_emissions - balance_delta)
        difference_pct = (difference / cumulative_alpha_emissions * 100) if cumulative_alpha_emissions != 0 else 0

        # ALPHA Balance Validation
        print(f"\n--- ALPHA Balance Validation ---")
        print(f"Initial ALPHA Balance: {initial_alpha_balance:.6f}")
        print(f"Final ALPHA Balance: {final_alpha_balance:.6f}")
        print(f"Balance Delta: {balance_delta:.6f}")
        print(f"Cumulative Emissions: {cumulative_alpha_emissions:.6f}")
        print(f"Difference: {difference:.6f} ({difference_pct:.2f}%)")
        if difference_pct < 1.0:
            print("✓ VALIDATION PASSED: Emissions match balance delta within 1%")
        elif difference_pct < 5.0:
            print("⚠️  VALIDATION WARNING: Emissions differ from balance delta by {:.2f}%".format(difference_pct))
        else:
            print("✗ VALIDATION FAILED: Emissions differ significantly from balance delta")

        # Latest values
        latest_cp = checkpoints[-1]
        print(f"\n--- Latest Checkpoint ---")
        # Use timestamp_utc if available, otherwise convert timestamp_ms to UTC
        if 'timestamp_utc' in latest_cp:
            timestamp_str = latest_cp['timestamp_utc']
        else:
            timestamp_str = datetime.utcfromtimestamp(latest_cp['timestamp_ms']/1000).strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"Timestamp (UTC): {timestamp_str}")
        print(f"Emissions (TAO): {latest_cp['emissions']['chunk_tao']:.4f}")
        print(f"Emissions (USD): ${latest_cp['emissions']['chunk_usd']:,.2f}")
        print(f"Portfolio Return: {latest_cp['performance']['portfolio_return']:.6f} ({latest_cp['performance']['portfolio_return']*100:.4f}%)")
        print(f"Realized PnL: ${latest_cp['performance']['realized_pnl']:.2f}")
        print(f"Unrealized PnL: ${latest_cp['performance']['unrealized_pnl']:.2f}")
        print(f"Weighted Score: {latest_cp['derived']['weighted_score']:.6f}")

        # Statistics over all checkpoints
        print(f"\n--- Statistics Over All Checkpoints ---")
        print(f"Emissions per Checkpoint (TAO):")
        print(f"  Min:  {df['chunk_tao'].min():.4f}")
        print(f"  Max:  {df['chunk_tao'].max():.4f}")
        print(f"  Mean: {df['chunk_tao'].mean():.4f}")
        print(f"  Total: {df['chunk_tao'].sum():.4f}")

        print(f"\nPortfolio Return:")
        print(f"  Min:  {df['portfolio_return'].min():.6f} ({df['portfolio_return'].min()*100:.4f}%)")
        print(f"  Max:  {df['portfolio_return'].max():.6f} ({df['portfolio_return'].max()*100:.4f}%)")
        print(f"  Mean: {df['portfolio_return'].mean():.6f} ({df['portfolio_return'].mean()*100:.4f}%)")

        print(f"\nWeighted Score:")
        print(f"  Min:  {df['weighted_score'].min():.6f}")
        print(f"  Max:  {df['weighted_score'].max():.6f}")
        print(f"  Mean: {df['weighted_score'].mean():.6f}")

        # Time range
        time_range = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
        print(f"\nTime Range: {time_range:.2f} hours ({time_range/24:.2f} days)")

    def plot_debt_ledger(self, data: Dict[str, Any], save_path: str = None):
        """Create comprehensive visualization of debt ledger data."""
        checkpoints = data.get('checkpoints', [])

        if not checkpoints:
            print("No checkpoints to plot")
            return

        # Extract data into DataFrame
        checkpoint_data = []
        for i, cp in enumerate(checkpoints):
            checkpoint_data.append({
                'checkpoint_num': i + 1,
                'timestamp_ms': cp['timestamp_ms'],
                'chunk_alpha': cp['emissions']['chunk_alpha'],
                'chunk_tao': cp['emissions']['chunk_tao'],
                'chunk_usd': cp['emissions']['chunk_usd'],
                'alpha_balance_snapshot': cp['emissions']['alpha_balance_snapshot'],
                'tao_balance_snapshot': cp['emissions']['tao_balance_snapshot'],
                'portfolio_return': cp['performance']['portfolio_return'],
                'realized_pnl': cp['performance']['realized_pnl'],
                'unrealized_pnl': cp['performance']['unrealized_pnl'],
                'weighted_score': cp['derived']['weighted_score'],
                'max_drawdown': cp['performance']['max_drawdown']
            })

        df = pd.DataFrame(checkpoint_data)
        df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Debt Ledger Analysis - {data.get("hotkey", "Unknown")[:16]}...\n'
                    f'Total Checkpoints: {len(checkpoints)} | '
                    f'Total TAO: {df["chunk_tao"].sum():.2f} | '
                    f'Total USD: ${df["chunk_usd"].sum():,.2f}',
                     fontsize=14, fontweight='bold')

        # 1. Cumulative ALPHA: Emissions vs Balance Delta
        ax1 = axes[0, 0]

        # Calculate cumulative alpha emissions
        cumulative_alpha_emissions = df['chunk_alpha'].cumsum()

        # Calculate cumulative balance delta (change from initial balance)
        initial_alpha_balance = df['alpha_balance_snapshot'].iloc[0]
        cumulative_balance_delta = df['alpha_balance_snapshot'] - initial_alpha_balance

        # Plot both lines
        line1 = ax1.plot(df['datetime'], cumulative_alpha_emissions,
                marker='o', markersize=3, linewidth=2, color='#e74c3c', alpha=0.8,
                label='Cumulative Emissions')[0]
        line2 = ax1.plot(df['datetime'], cumulative_balance_delta,
                marker='s', markersize=3, linewidth=2, color='#2ecc71', alpha=0.8,
                label='Balance Delta', linestyle='--')[0]

        # Calculate difference for validation
        final_emissions = cumulative_alpha_emissions.iloc[-1]
        final_balance_delta = cumulative_balance_delta.iloc[-1]
        difference = abs(final_emissions - final_balance_delta)
        difference_pct = (difference / final_emissions * 100) if final_emissions != 0 else 0

        ax1.set_title(f'Cumulative ALPHA: Emissions vs Balance Delta\n'
                     f'Difference: {difference:.4f} ALPHA ({difference_pct:.2f}%)',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative ALPHA')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc='upper left', fontsize=9)

        # Add annotations for final values
        ax1.annotate(f'Emissions: {final_emissions:.2f}',
                    xy=(df['datetime'].iloc[-1], final_emissions),
                    xytext=(-80, 15), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='#e74c3c'))

        ax1.annotate(f'Balance Δ: {final_balance_delta:.2f}',
                    xy=(df['datetime'].iloc[-1], final_balance_delta),
                    xytext=(-80, -25), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='#2ecc71'))

        # 2. Portfolio Return Over Time
        ax2 = axes[0, 1]
        ax2.plot(df['datetime'], df['portfolio_return'] * 100,
                marker='s', markersize=3, linewidth=2, color='#3498db', alpha=0.8)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Portfolio Return Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.tick_params(axis='x', rotation=45)

        # Color positive/negative returns
        colors = ['green' if x > 0 else 'red' for x in df['portfolio_return']]
        ax2.scatter(df['datetime'], df['portfolio_return'] * 100, c=colors, alpha=0.3, s=20)

        # 3. Weighted Score Over Time
        ax3 = axes[1, 0]
        ax3.plot(df['datetime'], df['weighted_score'],
                marker='D', markersize=3, linewidth=2, color='#2ecc71', alpha=0.8)
        ax3.set_title('Weighted Score Over Time', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Weighted Score')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax3.tick_params(axis='x', rotation=45)

        # Add trend line
        z = np.polyfit(range(len(df)), df['weighted_score'], 1)
        p = np.poly1d(z)
        ax3.plot(df['datetime'], p(range(len(df))), "r--", alpha=0.5, linewidth=1, label=f'Trend: {z[0]:+.6f}/checkpoint')
        ax3.legend()

        # 4. Recent Checkpoints Table
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Show last 10 checkpoints
        recent_df = df.tail(10)[['checkpoint_num', 'datetime', 'chunk_tao', 'portfolio_return', 'weighted_score']].copy()
        recent_df['chunk_tao'] = recent_df['chunk_tao'].apply(lambda x: f'{x:.4f}')
        recent_df['portfolio_return'] = recent_df['portfolio_return'].apply(lambda x: f'{x*100:.4f}%')
        recent_df['weighted_score'] = recent_df['weighted_score'].apply(lambda x: f'{x:.6f}')
        recent_df['datetime'] = recent_df['datetime'].apply(lambda x: x.strftime('%m/%d %H:%M'))

        # Create table
        table = ax4.table(cellText=recent_df.values,
                         colLabels=['CP#', 'Date/Time', 'TAO', 'Return%', 'Wtd Score'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(recent_df) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

        ax4.set_title('Recent Checkpoints (Last 10)', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved to: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Test Debt Ledger API endpoint and visualize data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python local_debt_ledger_api.py --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY
  python local_debt_ledger_api.py --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY --save-plot debt_ledger.png
  python local_debt_ledger_api.py --hotkey 5FRWVox3FD5Jc2VnS7FUCCf8UJgLKfGdEnMAN7nU3LrdMWHu --host localhost
        """
    )

    parser.add_argument('--hotkey', type=str, default="5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY",
                       help='Miner hotkey to query')
    parser.add_argument('--host', type=str, default='34.187.155.10',
                       help='API host (default: 34.187.155.10)')
    parser.add_argument('--port', type=int, default=48888,
                       help='API port (default: 48888)')
    parser.add_argument('--api-key', type=str, default='diQDNkoB3urHC9yOFo7iZOsTo09S?2hm9u',
                       help='API key for authentication')
    parser.add_argument('--save-plot', type=str, metavar='PATH',
                       help='Save plot to file instead of displaying')
    parser.add_argument('--save-json', type=str, metavar='PATH',
                       help='Save raw JSON response to file')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting (validation only)')

    args = parser.parse_args()

    # Create tester
    tester = DebtLedgerTester(args.host, args.port, args.api_key)

    # Fetch data
    data = tester.fetch_debt_ledger(args.hotkey)

    # Save raw JSON if requested
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Raw JSON saved to: {args.save_json}")

    # Validate data
    is_valid = tester.validate_data(data)

    # Print summary
    tester.print_summary(data)

    # Plot data
    if not args.no_plot:
        tester.plot_debt_ledger(data, save_path=args.save_plot)

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
