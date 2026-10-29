"""`vanta collateral`: Collateral operation commands."""
from datetime import datetime, timezone
import getpass
import json
import secrets
from typing import Any

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import FloatPrompt
from rich.table import Table

from vanta_cli.src.config import COLLATERAL_DEST_ADDRESS_MAINNET, COLLATERAL_DEST_ADDRESS_TESTNET, VANTA_API_BASE_URL_MAINNET, VANTA_API_BASE_URL_TESTNET
from vanta_cli.src.options import EPILOG, Amount, Network, WalletHotkey, WalletName, WalletPath, wallet_ask
from vanta_cli.src.utils.api import make_api_request
from vanta_cli.src.utils.console import console

app = typer.Typer(no_args_is_help=True, help="Vanta Network - Collateral operation commands", epilog=EPILOG)


@app.command("list", rich_help_panel="Collateral Management")
def collateral_list(
    wallet_name: WalletName = None,
    wallet_path: WalletPath = None,
    wallet_hotkey: WalletHotkey = None,
    network: Network = "finney",
):
    """
    List collateral balance for a miner address
    """
    wallet = wallet_ask(wallet_name, wallet_path, wallet_hotkey)

    console.print("[blue]Checking collateral balance[/blue]")

    # Determine the base URL based on network
    base_url = VANTA_API_BASE_URL_TESTNET if network == "test" else VANTA_API_BASE_URL_MAINNET

    # Make the API request
    miner_address = wallet.hotkey.ss58_address
    endpoint = f"/collateral/balance/{miner_address}"

    try:
        response = make_api_request(endpoint, method="GET", base_url=base_url)

        if response is None:
            console.print("[red]❌ Failed to retrieve collateral balance[/red]")
            return False

        # Display results in a nice table format
        balance_theta = response.get("balance_theta", 0.0)

        # Create a table for the results
        table = Table(title="Collateral Balance Information", show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        if wallet.name:
            table.add_row("Wallet Name", wallet.name)

        table.add_row("Miner Address", miner_address)
        table.add_row("Collateral Balance (THETA)", str(balance_theta))

        console.print(table)

        if response.get("balance_theta") is not None:
            return True
        else:
            error_message = (
                response.get("error") or
                "An unknown error occurred."
            )
            console.print(f"[red]❌ Error: {error_message}[/red]")
            return False

    except Exception as e:
        console.print(f"[red]❌ Error retrieving collateral balance: {e}[/red]")
        return False


@app.command("deposit", rich_help_panel="Collateral Operations")
def deposit(
    wallet_name: WalletName = None,
    wallet_path: WalletPath = None,
    wallet_hotkey: WalletHotkey = None,
    network: Network = "finney",
    amount: Amount = None,
):
    """
    Deposit collateral from the Vanτa Neτwork
    """
    wallet = wallet_ask(wallet_name, wallet_path, wallet_hotkey)

    if amount is None:
        amount = FloatPrompt.ask("Enter collateral deposit amount")

    from collateral_sdk import CollateralManager, Network # importing on compile time causes help menu to break
    manager = CollateralManager(Network.TESTNET if network == 'test' else Network.MAINNET)

    password = getpass.getpass(prompt='Enter your password: ')

    coldkey = wallet.get_coldkey(password)
    hotkey = wallet.hotkey

    # Set netuid based on network
    netuid = 116 if network == 'test' else 8

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching stake information...", total=None)
        source_stake: Any = manager.subtensor_api.staking.get_stake_info_for_coldkey(coldkey.ss58_address)

        progress.update(task, description="Checking Wallet Information...")
        balance: Any = manager.balance_of(hotkey.ss58_address) / 10 ** 9
        progress.stop()

    # Create wallet info table
    table_title = "Current Collateral Balance"
    if network == 'test':
        table_title += " (testnet)"

    table = Table(title=table_title, show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # Handle Balance object formatting
    try:
        if hasattr(balance, 'value'):
            balance_str = str(balance.value)
        elif hasattr(balance, 'free'):
            balance_str = str(balance.free)
        else:
            balance_str = repr(balance)
    except Exception:
        balance_str = repr(balance)

    # Add wallet information rows
    table.add_row("Coldkey", coldkey.ss58_address)
    table.add_row("Hotkey", hotkey.ss58_address)
    table.add_row("Current balance", f"{balance_str} Theta")

    console.print(table)

    # Display all stake information in a single table
    matching_stake = None
    if source_stake:
        # Create unified stake table
        stake_table = Table(title="Stake Information", show_header=True, header_style="bold cyan")
        stake_table.add_column("Hotkey", style="cyan", no_wrap=True)
        stake_table.add_column("Netuid", style="magenta", no_wrap=True)
        stake_table.add_column("Stake Amount", style="green", justify="right")
        stake_table.add_column("Locked", style="yellow", justify="right")
        stake_table.add_column("Registered", style="bold", justify="center")

        for stake_info in source_stake:
            if stake_info.netuid != netuid:
                continue

            # Format the hotkey address to show first 8 and last 6 characters
            formatted_hotkey = f"{stake_info.hotkey_ss58[:8]}...{stake_info.hotkey_ss58[-6:]}"

            # Highlight the target netuid row
            netuid_style = "bold green" if stake_info.netuid == netuid else "magenta"

            stake_table.add_row(
                formatted_hotkey,
                f"[{netuid_style}]{stake_info.netuid}[/{netuid_style}]",
                f"{float(stake_info.stake):.4f}",
                f"{float(stake_info.locked):.4f}",
                "✅" if stake_info.is_registered else "❌"
            )
        console.print(stake_table)

        # Set matching_stake
        matching_stake = next(
            (stake for stake in source_stake if (stake.hotkey_ss58 == hotkey.ss58_address and stake.netuid == netuid)),
            None
        )
    else:
        console.print("[yellow]No stake information available[/yellow]")


      # Check if source_stake is empty to avoid index error
    if not source_stake:
        console.print("[red]❌ No source stake found for this coldkey[/red]")
        manager.subtensor_api.close()
        return None

    if not matching_stake:
        console.print(f"[red]❌ No stake found for hotkey {hotkey} on netuid {netuid}[/red]")
        manager.subtensor_api.close()
        return None

    confirm = typer.confirm(f"Deposit {amount} Theta from {hotkey.ss58_address}?")
    if not confirm:
        console.print("[yellow]Deposit cancelled[/yellow]")
        manager.subtensor_api.close()
        return False

    # Use configured dest address
    dest_address = COLLATERAL_DEST_ADDRESS_TESTNET if network == 'test' else COLLATERAL_DEST_ADDRESS_MAINNET


    # Create an extrinsic for a stake transfer.
    extrinsic = manager.create_stake_transfer_extrinsic(
        amount=int(amount * 10**9),     # convert theta to rao_theta
        dest=dest_address,
        source_stake=matching_stake.hotkey_ss58,
        source_wallet=wallet,
        wallet_password=password
    )

    encoded = manager.encode_extrinsic(extrinsic)

    # Close the subtensor connection to prevent CLI from hanging
    manager.subtensor_api.close()

    result = {
        "encoded": encoded.hex(),
        "amount": amount,
        "coldkey": coldkey.ss58_address,
    }

    try:
        if result is None:
            console.print("[red]❌ Collateral setup failed[/red]")
            return False

        try:
            # Convert bytearray to hex string for JSON serialization
            encoded_data = result["encoded"]
            if isinstance(encoded_data, bytearray):
                encoded_data = encoded_data.hex()

            payload = {
                "extrinsic": encoded_data,
            }

            # Determine the base URL based on network
            base_url = VANTA_API_BASE_URL_TESTNET if network == "test" else VANTA_API_BASE_URL_MAINNET

            # Use the new API utility
            response = make_api_request("/collateral/deposit", payload, base_url=base_url)

            if response is None:
                console.print("[yellow]⚠️ API call failed[/yellow]")
                return False
            else:
                if response.get("successfully_processed"):
                    console.print(f"[green]✅ Deposited {amount} Theta[/green]")
                    return True
                else:
                    error_message = (
                        response.get("error_message") or
                        response.get("error") or
                        "An unknown error occurred."
                    )
                    console.print(f"[red]❌ Deposit failed: {error_message}[/red]")
                    return False

        except Exception as api_error:
            console.print(f"[yellow]⚠️ API call failed: {api_error}[/yellow]")
            return False

    except Exception as e:
        console.print(f"[red]❌ Error adding collateral: {e}[/red]")
        return False


@app.command("withdraw", rich_help_panel="Collateral Operations")
def withdraw(
    wallet_name: WalletName = None,
    wallet_path: WalletPath = None,
    wallet_hotkey: WalletHotkey = None,
    network: Network = "finney",
    amount: Amount = None,
):
    """
    Withdraw collateral from the Vanτa Neτwork
    """
    wallet = wallet_ask(wallet_name, wallet_path, wallet_hotkey)

    if amount is None:
        amount = FloatPrompt.ask("Enter collateral withdrawal amount")

    # Load wallet and get keys
    password = getpass.getpass(prompt='Enter your password: ')

    coldkey = wallet.get_coldkey(password=password)
    hotkey = wallet.hotkey

    # Determine which API base URL to use based on network
    base_url = VANTA_API_BASE_URL_TESTNET if network == "test" else VANTA_API_BASE_URL_MAINNET

    # Step 1: Query the withdrawal to get the slashed amount
    query_payload = {
        "amount": amount,
        "miner_hotkey": hotkey.ss58_address
    }

    try:
        query_response = make_api_request("/collateral/query-withdraw", query_payload, base_url=base_url)
    except Exception as e:
        console.print(f"[red]❌ Error querying withdrawal: {e}[/red]")
        return False

    if query_response is None:
        console.print("[red]❌ Failed to query withdrawal details[/red]")
        return False

    if not query_response.get("successfully_processed"):
        error_message = (
            query_response.get("error_message") or
            query_response.get("error") or
            "An unknown error occurred."
        )
        console.print(f"[red]❌ Query failed: {error_message}[/red]")
        return False

    drawdown = query_response.get("drawdown", 0)
    slashed_amount = query_response.get("slashed_amount", 0)
    withdrawal_amount = query_response.get("withdrawal_amount", amount)
    new_balance = query_response.get("new_balance", 0)

    # Step 2: Show the withdrawal details and confirm
    table_title = "Collateral Withdrawal"
    if network == "test":
        table_title += " (testnet)"

    table = Table(title=table_title, show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Coldkey", coldkey.ss58_address)
    table.add_row("Hotkey", hotkey.ss58_address)
    table.add_row("Requested", f"{amount} Theta")
    table.add_row("Current drawdown", f"{(1.0 - drawdown) * 100:.2f}%")
    table.add_row("Slashed (approx.)", f"[yellow]{slashed_amount} Theta[/yellow]")
    table.add_row("Net withdrawal", f"{withdrawal_amount} Theta")
    table.add_row("New balance", f"{new_balance} Theta")

    console.print(table)
    console.print("[dim]Values are approximate and may change depending on current drawdown.[/dim]")

    confirm = typer.confirm(f"Withdraw {amount} Theta from {hotkey.ss58_address}?")
    if not confirm:
        console.print("[yellow]Withdrawal cancelled[/yellow]")
        return False

    # Step 3: Proceed with the actual withdrawal
    nonce = secrets.token_urlsafe()
    timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Prepare withdrawal data for signing
    withdrawal_data = {
        "amount": amount,
        "miner_coldkey": coldkey.ss58_address,
        "miner_hotkey": hotkey.ss58_address,
        "nonce": nonce,
        "timestamp": timestamp,
    }

    # Create message to sign (sorted JSON)
    message = json.dumps(withdrawal_data, sort_keys=True)

    # Sign the message with coldkey
    signature = coldkey.sign(message.encode('utf-8')).hex()

    # Prepare payload for withdrawal (include signature)
    payload = {
        "amount": amount,
        "miner_coldkey": coldkey.ss58_address,
        "miner_hotkey": hotkey.ss58_address,
        "nonce": nonce,
        "timestamp": timestamp,
        "signature": signature
    }

    try:
        response = make_api_request("/collateral/withdraw", payload, base_url=base_url)

        if response is None:
            console.print("[red]❌ Withdrawal request failed[/red]")
            return False

        # Check if withdrawal was successful
        if response.get("successfully_processed"):
            returned_amount = response.get("returned_amount", withdrawal_amount)
            console.print(f"[green]✅ Withdrew {returned_amount} Theta[/green]")
            return True
        else:
            error_message = (
                response.get("error_message") or
                response.get("error") or
                "An unknown error occurred."
            )
            console.print(f"[red]❌ Withdrawal failed: {error_message}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Error during withdrawal: {e}[/red]")
        return False
