from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import Registry, register
from components.processor import ResultsProcessor, THClassifierProcessor, ClassWeightProcessor


@register
def register_processor_configurations():
    # Model
    Registry.add_and_bind(config_class=Configuration,
                          component_class=ClassWeightProcessor,
                          name='processor',
                          tags={'weights'},
                          namespace='nle')

    Registry.add_and_bind(config_class=Configuration,
                          component_class=THClassifierProcessor,
                          name='processor',
                          tags={'classifier', 'th'},
                          namespace='nle')

    # Routine
    Registry.add_and_bind(config_class=Configuration,
                          component_class=ResultsProcessor,
                          name='processor',
                          tags={'results'},
                          namespace='nle')
