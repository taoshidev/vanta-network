from typing import Annotated, Optional

from rich.prompt import FloatPrompt, IntPrompt, Prompt
from bittensor_cli.cli import CLIManager, Options, version_callback
from bittensor_cli.src import (
    WalletOptions as WO,
    WalletValidationTypes as WV,
)
from bittensor_cli.src.bittensor.utils import (
    console,
)
from rich.tree import Tree
import typer
from vanta_cli.src.config import VANTA_CLI_VERSION
from vanta_cli.src.commands.collateral import (
    list as list_collateral,
    deposit as deposit_collateral,
    withdraw as withdraw_collateral
)
from vanta_cli.src.commands.asset import (
    select as select_asset
)
from vanta_cli.src.commands.entity import (
    register as register_entity,
    create_subaccount as create_subaccount_entity,
    apikey as apikey_entity,
)
from vanta_cli.src.commands.profile import (
    set as profile_set,
)

_epilog = "Made with [bold red]:heart:[/bold red] by Vanτa Neτwork"

def vanta_version_callback(value: bool) -> None:
    """
    Prints the current version
    """
    if value:
        typer.echo(f"Vanta CLI version: {VANTA_CLI_VERSION}")
        version_callback(value)
        raise typer.Exit()

def commands_callback(value: bool) -> None:
    """
    Prints a tree of commands for the app
    """
    if value:
        cli = VantaCLIManager()
        console.print(cli.generate_command_tree())
        raise typer.Exit()


class VantaOptions:
    vanta_network = typer.Option(
        "finney",
        "--network",
        "--subtensor.network",
        help="The subtensor network to connect to.",
    )
    amount = typer.Option(
        None,
        "--amount",
        help="Amount of Theta to use for collateral",
    )
    prompt = typer.Option(
        True,
        "--prompt",
        help="Whether to prompt for confirmation",
    )


class VantaCLIManager(CLIManager):

    collateral_app: typer.Typer
    asset_app: typer.Typer
    entity_app: typer.Typer
    profile_app: typer.Typer

    def __init__(self):
        super().__init__()

        # Override btcli typer config
        self.app.info.callback = self.vanta_main_callback
        self.app.info.epilog = _epilog
        for a in self.app.registered_groups:
            a.epilog = _epilog

        self.collateral_app = typer.Typer(epilog=_epilog)
        self.asset_app = typer.Typer(epilog=_epilog)
        self.entity_app = typer.Typer(epilog=_epilog)
        self.profile_app = typer.Typer(epilog=_epilog)

        self.app.add_typer(
            self.collateral_app,
            name="collateral",
            short_help="Vanta Network - Collateral operation commands",
            no_args_is_help=True
        )
        self.app.add_typer(
            self.asset_app,
            name="asset",
            short_help="Vanta Network - Asset selection commands",
            no_args_is_help=True
        )
        self.app.add_typer(
            self.entity_app,
            name="entity",
            short_help="Vanta Network - Entity management commands",
            no_args_is_help=True
        )
        self.app.add_typer(
            self.profile_app,
            name="profile",
            short_help="Vanta Network - Profile management commands",
            no_args_is_help=True
        )

        self.collateral_app.command(
            "list", rich_help_panel="Collateral Management"
        )(self.collateral_list)
        self.collateral_app.command(
            "deposit", rich_help_panel="Collateral Operations"
        )(self.collateral_deposit)
        self.collateral_app.command(
            "withdraw", rich_help_panel="Collateral Operations"
        )(self.collateral_withdraw)

        self.asset_app.command(
            "select", rich_help_panel="Asset class selection"
        )(self.asset_select)

        self.entity_app.command(
            "register", rich_help_panel="Entity Management"
        )(self.entity_register)
        self.entity_app.command(
            "create-subaccount", rich_help_panel="Entity Management"
        )(self.entity_create_subaccount)
        self.entity_app.command(
            "apikey", rich_help_panel="Entity Management"
        )(self.entity_apikey)

        self.profile_app.command(
            "set", rich_help_panel="Profile Management"
        )(self.profile_set)

    def generate_command_tree(self) -> Tree:
        """
        Generates a rich.Tree of the commands, subcommands, and groups of this app
        """

        def build_rich_tree(data: dict, parent: Tree) -> None:
            for group, content in data.get("groups", {}).items():
                group_node = parent.add(
                    f"[bold cyan]{group}[/]"
                )  # Add group to the tree
                for command in content.get("commands", []):
                    group_node.add(f"[green]{command}[/]")  # Add commands to the group
                build_rich_tree(content, group_node)  # Recurse for subgroups

        def traverse_group(group: typer.Typer) -> dict:
            tree = {}
            if commands := [
                cmd.name for cmd in group.registered_commands if not cmd.hidden
            ]:
                tree["commands"] = commands
            for group in group.registered_groups:
                if "groups" not in tree:
                    tree["groups"] = {}
                if not group.hidden:
                    if group_transversal := traverse_group(group.typer_instance):
                        tree["groups"][group.name] = group_transversal

            return tree

        groups_and_commands = traverse_group(self.app)
        root = Tree("[bold magenta]BTCLI Commands[/]")  # Root node
        build_rich_tree(groups_and_commands, root)
        return root

    # Override btcli main callback
    def vanta_main_callback(
        self,
        version: Annotated[
            Optional[bool],
            typer.Option(
                "--version", callback=vanta_version_callback, help="Show Vanta and Bittensor CLI version"
            ),
        ] = None,
        commands: Annotated[
            Optional[bool],
            typer.Option(
                "--commands", callback=commands_callback, help="Show Vanta and Bittensor CLI commands"
            ),
        ] = None,
    ):
        """
        Command line interface (CLI) for Bittensor subnet 8 Vanta Network. Uses the values in the configuration file. These values can be
            overridden by passing them explicitly in the command line.
        """
        self.main_callback()

    def collateral_list(
        self,
        wallet_name: Optional[str] = Options.wallet_name,
        wallet_path: Optional[str] = Options.wallet_path,
        wallet_hotkey: Optional[str] = Options.wallet_hotkey_ss58,
        network: str = VantaOptions.vanta_network,
        quiet: bool = Options.quiet,
        verbose: bool = Options.verbose,
        json_output: bool = Options.json_output,
    ):
        """
        List collateral balance for a miner address
        """
        self.verbosity_handler(quiet, verbose, json_output)

        ask_for = [WO.NAME, WO.HOTKEY]
        wallet = self.wallet_ask(
            wallet_name,
            wallet_path,
            wallet_hotkey,
            ask_for=ask_for,
            validate=WV.WALLET_AND_HOTKEY,
        )

        return self._run_command(
            list_collateral.collateral_list(
                wallet,
                network,
                quiet,
                verbose,
                json_output
            )
        )

    def collateral_deposit(
        self,
        wallet_name: Optional[str] = Options.wallet_name,
        wallet_path: Optional[str] = Options.wallet_path,
        wallet_hotkey: Optional[str] = Options.wallet_hotkey_ss58,
        network: str = VantaOptions.vanta_network,
        amount: Optional[float] = VantaOptions.amount,
        quiet: bool = Options.quiet,
        verbose: bool = Options.verbose,
        json_output: bool = Options.json_output,
    ):
        """
        Deposit collateral from the Vanτa Neτwork
        """

        self.verbosity_handler(quiet, verbose, json_output)

        ask_for = [WO.NAME, WO.HOTKEY]
        wallet = self.wallet_ask(
            wallet_name,
            wallet_path,
            wallet_hotkey,
            ask_for=ask_for,
            validate=WV.WALLET_AND_HOTKEY,
        )

        if amount is None:
            amount = FloatPrompt.ask("Enter collateral deposit amount")

        return self._run_command(
            deposit_collateral.deposit(
                wallet,
                network,
                amount,
                quiet,
                verbose,
                json_output
            )
        )

    def collateral_withdraw(
        self,
        wallet_name: Optional[str] = Options.wallet_name,
        wallet_path: Optional[str] = Options.wallet_path,
        wallet_hotkey: Optional[str] = Options.wallet_hotkey_ss58,
        network: str = VantaOptions.vanta_network,
        amount: Optional[float] = VantaOptions.amount,
        prompt: bool = VantaOptions.prompt,
        quiet: bool = Options.quiet,
        verbose: bool = Options.verbose,
        json_output: bool = Options.json_output,
    ):
        """
        Withdraw collateral from the Vanτa Neτwork
        """
        self.verbosity_handler(quiet, verbose, json_output)

        ask_for = [WO.NAME, WO.HOTKEY]
        wallet = self.wallet_ask(
            wallet_name,
            wallet_path,
            wallet_hotkey,
            ask_for=ask_for,
            validate=WV.WALLET_AND_HOTKEY,
        )

        if amount is None:
            amount = FloatPrompt.ask("Enter collateral withdrawal amount")

        return self._run_command(
            withdraw_collateral.withdraw(
                wallet,
                network,
                amount,
                prompt,
                quiet,
                verbose,
                json_output
            )
        )

    def asset_select(
        self,
        wallet_name: Optional[str] = Options.wallet_name,
        wallet_path: Optional[str] = Options.wallet_path,
        wallet_hotkey: Optional[str] = Options.wallet_hotkey_ss58,
        network: str = VantaOptions.vanta_network,
        prompt: bool = VantaOptions.prompt,
        quiet: bool = Options.quiet,
        verbose: bool = Options.verbose,
        json_output: bool = Options.json_output,
    ):
        self.verbosity_handler(quiet, verbose, json_output)

        ask_for = [WO.NAME, WO.HOTKEY]
        wallet = self.wallet_ask(
            wallet_name,
            wallet_path,
            wallet_hotkey,
            ask_for=ask_for,
            validate=WV.WALLET_AND_HOTKEY,
        )

        assets = ["crypto", "forex", "equities", "commodities"]

        for idx, asset in enumerate(assets, start=1):
            console.print(f"{idx}. {asset}")

        choice = IntPrompt.ask(
            "\nEnter the [bold]number[/bold] of the asset class you want to select",
            choices=[str(i) for i in range(1, len(assets) + 1)],
            show_choices=False,
        )
        asset_choice = assets[choice - 1]

        return self._run_command(
            select_asset.select(
                wallet,
                network,
                asset_choice,
                prompt,
                quiet,
                verbose,
                json_output
            )
        )

    def entity_register(
        self,
        wallet_name: Optional[str] = Options.wallet_name,
        wallet_path: Optional[str] = Options.wallet_path,
        wallet_hotkey: Optional[str] = Options.wallet_hotkey_ss58,
        network: str = VantaOptions.vanta_network,
        prompt: bool = VantaOptions.prompt,
        quiet: bool = Options.quiet,
        verbose: bool = Options.verbose,
        json_output: bool = Options.json_output,
    ):
        """
        Register a new entity on the Vanta Network
        """
        self.verbosity_handler(quiet, verbose, json_output)

        ask_for = [WO.NAME, WO.HOTKEY]
        wallet = self.wallet_ask(
            wallet_name,
            wallet_path,
            wallet_hotkey,
            ask_for=ask_for,
            validate=WV.WALLET_AND_HOTKEY,
        )

        return self._run_command(
            register_entity.register(
                wallet,
                network,
                prompt,
                quiet,
                verbose,
                json_output
            )
        )

    def entity_create_subaccount(
        self,
        wallet_name: Optional[str] = Options.wallet_name,
        wallet_path: Optional[str] = Options.wallet_path,
        wallet_hotkey: Optional[str] = Options.wallet_hotkey_ss58,
        network: str = VantaOptions.vanta_network,
        account_size: Optional[float] = typer.Option(
            None,
            "--account-size",
            help="Account size in USD"
        ),
        asset_class: Optional[str] = typer.Option(
            None,
            "--asset-class",
            help="Asset class selection (crypto, forex, equities)"
        ),
        prompt: bool = VantaOptions.prompt,
        quiet: bool = Options.quiet,
        verbose: bool = Options.verbose,
        json_output: bool = Options.json_output,
    ):
        """
        Create a new subaccount for an entity
        """
        self.verbosity_handler(quiet, verbose, json_output)

        ask_for = [WO.NAME, WO.HOTKEY]
        wallet = self.wallet_ask(
            wallet_name,
            wallet_path,
            wallet_hotkey,
            ask_for=ask_for,
            validate=WV.WALLET_AND_HOTKEY,
        )

        # Prompt for account_size if not provided
        if account_size is None:
            account_size = FloatPrompt.ask("Enter subaccount size in USD")

        # Prompt for asset_class if not provided
        if asset_class is None:
            assets = ["crypto", "forex", "equities"]
            console.print("\nAvailable asset classes:")
            for idx, asset in enumerate(assets, start=1):
                console.print(f"{idx}. {asset}")

            choice = IntPrompt.ask(
                "\nEnter the number of the asset class",
                choices=[str(i) for i in range(1, len(assets) + 1)],
                show_choices=False,
            )
            asset_class = assets[choice - 1]

        return self._run_command(
            create_subaccount_entity.create_subaccount(
                wallet,
                network,
                account_size,
                asset_class,
                prompt,
                quiet,
                verbose,
                json_output
            )
        )

    def entity_apikey(
        self,
        wallet_name: Optional[str] = Options.wallet_name,
        wallet_path: Optional[str] = Options.wallet_path,
        wallet_hotkey: Optional[str] = Options.wallet_hotkey_ss58,
        network: str = VantaOptions.vanta_network,
        quiet: bool = Options.quiet,
        verbose: bool = Options.verbose,
        json_output: bool = Options.json_output,
    ):
        """
        Request or retrieve the API key for your entity miner
        """
        self.verbosity_handler(quiet, verbose, json_output)

        ask_for = [WO.NAME, WO.HOTKEY]
        wallet = self.wallet_ask(
            wallet_name,
            wallet_path,
            wallet_hotkey,
            ask_for=ask_for,
            validate=WV.WALLET_AND_HOTKEY,
        )

        return self._run_command(
            apikey_entity.apikey(
                wallet,
                network,
                quiet,
                verbose,
                json_output,
            )
        )

    def profile_set(
        self,
        wallet_name: Optional[str] = Options.wallet_name,
        wallet_path: Optional[str] = Options.wallet_path,
        wallet_hotkey: Optional[str] = Options.wallet_hotkey_ss58,
        network: str = VantaOptions.vanta_network,
        prompt: bool = VantaOptions.prompt,
        quiet: bool = Options.quiet,
        verbose: bool = Options.verbose,
        json_output: bool = Options.json_output,
    ):
        """
        Sets or updates profile configuration value on Vanta Network
        """
        self.verbosity_handler(quiet, verbose, json_output)

        ask_for = [WO.NAME, WO.HOTKEY]
        wallet = self.wallet_ask(
            wallet_name,
            wallet_path,
            wallet_hotkey,
            ask_for=ask_for,
            validate=WV.WALLET_AND_HOTKEY,
        )

        console.print("\nWhich config setting would you like to update?\n")
        profile_keys = ["display_name"]
        for idx, key in enumerate(profile_keys, start=1):
            console.print(f"{idx}. {key}")

        choice = IntPrompt.ask(
            "\nEnter the [bold]number[/bold] of the profile setting you want to update",
            choices=[str(i) for i in range(1, len(profile_keys) + 1)],
            show_choices=False,
        )
        profile_key = profile_keys[choice - 1]
        value = Prompt.ask(
            f"What value would you like to assign to [red]{profile_key}[/red]?"
        )

        return self._run_command(
            profile_set.set(
                wallet,
                network,
                profile_key,
                value,
                prompt,
            )
        )

    def vanta_run(self):
        self.app()


def main():
    manager = VantaCLIManager()
    manager.vanta_run()

if __name__ == "__main__":
    main()
