from argparse import ArgumentParser
from pathlib import Path

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry, run_component
from cinnamon_generic.components.data_loader import DataLoader

if __name__ == '__main__':
    setup_registry(directory=Path(__file__).parent.parent.resolve(),
                   registrations_to_file=True)

    parser = ArgumentParser()
    parser.add_argument('--name', '-n', default='data_loader', type=str)
    parser.add_argument('--tags', '-t', nargs='+', default={'category=A'}, type=str)
    parser.add_argument('--namespace', '-ns', default='nle/tos', type=str)
    args = parser.parse_args()

    logging_utility.logger.info(f'''
    Running data loader with: 
    {args}
    ''')

    loader = DataLoader.build_component(name=args.name,
                                        tags=args.tags,
                                        namespace=args.namespace)
    data = loader.run()
    logging_utility.logger.info(data)
    logging_utility.logger.info(loader.kb)
