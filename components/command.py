import os
from typing import Any, Tuple, List

from cinnamon_core.core.component import Component
from cinnamon_core.core.registry import RegistrationKey, Registry
from cinnamon_core.utility import logging_utility


class Command(Component):

    def compute_run_name(
            self,
            tags
    ):
        run_name = []
        flatten_tags = f'{os.linesep}'.join(tags)
        for real_tag, replace_tag in self.tag_mapping.items():
            flatten_tags = flatten_tags.replace(real_tag, replace_tag)

        flatten_tags = set(flatten_tags.split(os.linesep))

        for field_extractor in self.run_name_fields:
            run_value = field_extractor(flatten_tags)
            if run_value is not None:
                run_name.append(run_value)

        return '_'.join(run_name)

    def run(
            self,
            parse_args
    ) -> Tuple[Any, str]:
        parse_key = RegistrationKey(name=parse_args.name,
                                    tags=parse_args.tags,
                                    namespace=parse_args.namespace)
        keys = [key for key in Registry.REGISTRY if key == parse_key]
        assert len(keys) == 1, f'Expected to find only one key but found {keys}'

        found_key = keys[0]
        parse_args.tags = found_key.tags

        run_name = self.compute_run_name(tags=found_key.tags)

        return parse_args, run_name


class MultipleRunsCommand(Command):

    def run(
            self,
            parse_args
    ) -> Tuple[List[RegistrationKey], List[str]]:
        parse_key = RegistrationKey(name=parse_args.name,
                                    tags=parse_args.tags,
                                    namespace=parse_args.namespace)
        keys = [key for key in Registry.REGISTRY if key.partial_match(parse_key)]
        keys = [key for key in keys if Registry.build_configuration_from_key(key).validate(strict=False)]
        logging_utility.logger.info(
            f'Retrieved {len(keys)} keys. {os.linesep}{os.linesep.join([str(key) for key in keys])}')

        run_names = [self.compute_run_name(key.tags) for key in keys]

        return keys, run_names
