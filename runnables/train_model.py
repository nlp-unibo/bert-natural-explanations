from argparse import ArgumentParser
from pathlib import Path

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry, run_component
from components.command import Command

if __name__ == '__main__':
    setup_registry(directory=Path(__file__).parent.parent.resolve(),
                   registrations_to_file=True)

    parser = ArgumentParser()
    parser.add_argument('--name', '-n', default='pipeline', type=str)
    parser.add_argument('--tags', '-t', nargs='+', default={
        'routine.data_loader.category=LTD',
        'routine.hf',
        'routine.kb',
        'routine.model.hf',
        'routine.model.hf_model_name=distilbert-base-uncased',
        'routine.model.kb_sampler.gain',
        'routine.model.kb_sampler.sampling_size=1',
        'routine.model.memory',
        'routine.model.ss_coefficient=1.0'
    },
                        type=str)
    parser.add_argument('--namespace', '-ns', default='nle/tos', type=str)
    parser.add_argument('--serialize', '-s', default='False', type=str)
    args = parser.parse_args()

    args.serialize = False if args.serialize.casefold() == 'false' else True

    logging_utility.logger.info(f'''
    Running model training with: 
    {args}
    ''')

    command = Command.build_component(name='command',
                                      namespace='nle')
    args, run_name = command.run(parse_args=args)

    result = run_component(name=args.name,
                           tags=args.tags,
                           namespace=args.namespace,
                           serialize=args.serialize,
                           run_args={'is_training': True},
                           run_name=run_name)
