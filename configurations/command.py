import re
from typing import Type, Dict, List, Callable

from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import Registry, register
from components.command import Command, MultipleRunsCommand


class CommandConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='tag_mapping',
                   type_hint=Dict[str, str],
                   value={
                       'routine.model.ss_coefficient': 'SS',
                       'routine.model.kb_sampler.sampling_size': 'sampling_size',
                       'routine.model.hf_model_name': 'model_card',
                       'routine.data_loader.topics': 'topics',
                       'routine.data_splitter.topics': 'topics',
                       'routine.data_loader.category': 'category'
                   },
                   description='Tag or partial tag names that are replaced with corresponding map value.'
                               'This mapping is used to automatically build a  component run name.')

        config.add(name='run_name_fields',
                   type_hint=List[Callable[[str], str]],
                   value=[
                       lambda tags: [tag.split('.')[-1] for tag in tags if 'topics=' in tag or 'category=' in tag][0],
                       lambda tags: [tag.split('model_card=')[-1] for tag in tags if 'model_card' in tag][0],
                       lambda tags: 'kb' if len([tag for tag in tags if 'kb' in tag]) else None,
                       lambda tags: None if not len([tag for tag in tags if 'sampler' in tag]) else [tag.split('kb_sampler.')[-1] for tag in tags if 'sampler' in tag][0],
                       lambda tags: None if not len([tag for tag in tags if 'sampling_size' in tag]) else [tag for tag in tags if 'sampling_size' in tag][0],
                       lambda tags: None if not len([tag for tag in tags if 'SS' in tag]) else [tag for tag in tags if 'SS=' in tag][0],
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
