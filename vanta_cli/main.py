import json
from pathlib import Path
from typing import Annotated, Optional

from rich.tree import Tree
import typer
from vanta_cli.src.config import VANTA_CLI_VERSION
from vanta_cli.src.options import EPILOG
from vanta_cli.src.utils.console import console
from vanta_cli.src.commands import asset, collateral, entity

# meta/meta.json lives at the repo root; this file is at vanta_cli/main.py.
_META_JSON_PATH = Path(__file__).resolve().parent.parent / "meta" / "meta.json"

app = typer.Typer(
    rich_markup_mode="rich",
    epilog=EPILOG,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(collateral.app, name="collateral")
app.add_typer(asset.app, name="asset")
app.add_typer(entity.app, name="entity")
# Temporarily disabled - re-enable with dashboard visibility
# app.add_typer(profile.app, name="profile")


def _get_vanta_network_version() -> str:
    """Reads the subnet version from meta/meta.json."""
    try:
        with open(_META_JSON_PATH) as f:
            return json.load(f)["subnet_version"]
    except (OSError, json.JSONDecodeError, KeyError):
        return "unknown"


def generate_command_tree() -> Tree:
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

    groups_and_commands = traverse_group(app)
    root = Tree("[bold magenta]Vanta CLI Commands[/]")  # Root node
    build_rich_tree(groups_and_commands, root)
    return root


def vanta_version_callback(value: bool) -> None:
    """
    Prints the current version
    """
    if value:
        typer.echo(f"Vanta CLI version: {VANTA_CLI_VERSION}")
        typer.echo(f"Vanta Network version: {_get_vanta_network_version()}")
        raise typer.Exit()


def commands_callback(value: bool) -> None:
    """
    Prints a tree of commands for the app
    """
    if value:
        console.print(generate_command_tree())
        raise typer.Exit()


@app.callback()
def vanta_main_callback(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version", callback=vanta_version_callback, is_eager=True, help="Show Vanta CLI version"
        ),
    ] = None,
    commands: Annotated[
        Optional[bool],
        typer.Option(
            "--commands", callback=commands_callback, is_eager=True, help="Show Vanta CLI commands"
        ),
    ] = None,
):
    """
    Command line interface (CLI) for Bittensor subnet 8 Vanta Network.
    """


def main():
    app()


if __name__ == "__main__":
    main()
