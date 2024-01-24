from typing import Type, Dict

from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import Registry, RegistrationKey, register
from cinnamon_generic.components.processor import ProcessorPipeline
from cinnamon_generic.configurations.pipeline import OrderedPipelineConfig
from components.processor import HFTokenizer, ModelProcessor


class HFTokenizerConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='hf_model_name',
                   value='prajjwal1/bert-small',
                   is_required=True,
                   type_hint=str,
                   description="HuggingFace's model card name.")
        config.add(name='tokenization_args',
                   value={
                       'truncation': True,
                       'add_special_tokens': True,
                       'max_length': 200
                   },
                   type_hint=Dict,
                   description='Additional tokenization arguments.')

        return config


class ModelProcessorConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='num_workers',
                   value=0,
                   type_hint=int,
                   description='Number of processes to use for data loading.')
        config.add(name='batch_size',
                   value=16,
                   type_hint=int,
                   is_required=True,
                   description='Batch size for aggregating samples.')

        return config


@register
def register_processors():
    # Tokenizer
    Registry.add_and_bind(config_class=HFTokenizerConfig,
                          component_class=HFTokenizer,
                          name='processor',
                          tags={'tokenizer'},
                          namespace='nle/tos')

    # Model
    Registry.add_and_bind_variants(config_class=ModelProcessorConfig,
                                   component_class=ModelProcessor,
                                   name='processor',
                                   tags={'model'},
                                   namespace='nle/tos')

    # Pipeline
    Registry.add_and_bind_variants(config_class=OrderedPipelineConfig,
                                   component_class=ProcessorPipeline,
                                   config_constructor=OrderedPipelineConfig.from_keys,
                                   config_kwargs={
                                       'keys': [
                                           RegistrationKey(name='processor',
                                                           tags={'tokenizer'},
                                                           namespace='nle/tos'),
                                           RegistrationKey(name='processor',
                                                           tags={'model'},
                                                           namespace='nle/tos')
                                       ],
                                       'names': [
                                           'tokenizer',
                                           'model_processor'
                                       ]
                                   },
                                   name='processor',
                                   namespace='nle/tos')