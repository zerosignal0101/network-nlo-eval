"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Network NLO Eval."""


if __name__ == "__main__":
    main(prog_name="network-nlo-eval")  # pragma: no cover
