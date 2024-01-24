from pathlib import Path

from cinnamon_generic.api.commands import setup_registry

if __name__ == '__main__':
    setup_registry(
        directory=Path(__file__).parent.parent.resolve(),
        registrations_to_file=True
    )
