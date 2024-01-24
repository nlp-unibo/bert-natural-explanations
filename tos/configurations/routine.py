from cinnamon_core.core.registry import RegistrationKey, Registry, register
from cinnamon_generic.components.routine import CVRoutine
from configurations.routine import NLERoutineConfig


class ToSRoutineConfig(NLERoutineConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.data_loader = RegistrationKey(name='data_loader',
                                             namespace='nle/tos')

        config.data_splitter = RegistrationKey(name='data_splitter',
                                               namespace='nle/tos')

        config.get('model').variants = [
            RegistrationKey(name='model',
                            tags={'hf', 'baseline'},
                            namespace='nle/tos')
        ]

        config.pre_processor = RegistrationKey(name='processor',
                                               namespace='nle/tos')

        config.callbacks = RegistrationKey(name='callback',
                                           namespace='nle/tos')

        config.metrics = RegistrationKey(name='metrics',
                                         tags={'clf_f1'},
                                         namespace='nle')

        return config


@register
def register_routines():
    Registry.add_and_bind_variants(config_class=ToSRoutineConfig,
                                   component_class=CVRoutine,
                                   name='routine',
                                   namespace='nle/tos')
