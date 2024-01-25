import re
from typing import Type, Dict, List, Callable

from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import Registry, register
from components.command import Command, MultipleRunsCommand


# TODO: check if there are additional mappings and fields to add
class CommandConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='tag_mapping',
                   type_hint=Dict[str, str],
                   value={
                       'strong_supervision': 'sup'
                   },
                   description='Tag or partial tag names that are replaced with corresponding map value.'
                               'This mapping is used to automatically build a  component run name.')

        config.add(name='run_name_fields',
                   type_hint=List[Callable[[str], str]],
                   value=[
                       lambda tags: [item for item in ['baseline', 'memory'] if item in tags][0],
                       lambda tags: 'sup' if 'sup=True' in tags else None,
                   ],
                   description='List of tag filters to extract run name fields.')

        return config


@register
def register_commands():
    Registry.add_and_bind(config_class=CommandConfig,
                          component_class=Command,
                          name='command',
                          namespace='nle')
    Registry.add_and_bind(config_class=CommandConfig,
                          component_class=MultipleRunsCommand,
                          name='command',
                          tags={'multiple'},
                          namespace='nle')
