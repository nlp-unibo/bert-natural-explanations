from cinnamon_core.core.registry import RegistrationKey, Registry, register
from cinnamon_generic.components.routine import CVRoutine
from configurations.routine import NLERoutineConfig


class IBMRoutineConfig(NLERoutineConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.data_loader = RegistrationKey(name='data_loader',
                                             namespace='nle/ibm')

        config.data_splitter = RegistrationKey(name='data_splitter',
                                               namespace='nle/ibm')

        config.metrics = RegistrationKey(name='metrics',
                                         tags={'clf_f1'},
                                         namespace='nle')

        config.add_condition(name='post_topics',
                             condition=lambda c: c.data_loader.topics == c.data_splitter.topics)

        return config

    @classmethod
    def get_hf_baseline(
            cls
    ):
        config = cls.get_default()

        config.model = RegistrationKey(name='model',
                                       tags={'hf', 'baseline'},
                                       namespace='nle/ibm')

        config.pre_processor = RegistrationKey(name='processor',
                                               tags={'hf'},
                                               namespace='nle/ibm')

        config.callbacks = RegistrationKey(name='callback',
                                           tags={'hf'},
                                           namespace='nle/ibm')

        return config

    @classmethod
    def get_hf_kb_config(
            cls
    ):
        config = cls.get_default()

        config.model = RegistrationKey(name='model',
                                       tags={'hf', 'memory'},
                                       namespace='nle/ibm')

        config.pre_processor = RegistrationKey(name='processor',
                                               tags={'kb', 'hf'},
                                               namespace='nle/ibm')

        config.callbacks = RegistrationKey(name='callback',
                                           tags={'hf', 'memory'},
                                           namespace='nle/ibm')

        config.metrics = RegistrationKey(name='metrics',
                                         tags={'memory'},
                                         namespace='nle/ibm')

        return config

    @classmethod
    def get_baseline_kb_config(
            cls
    ):
        config = cls.get_default()

        config.model = RegistrationKey(name='model',
                                       tags={'memory'},
                                       namespace='nle/ibm')

        config.pre_processor = RegistrationKey(name='processor',
                                               tags={'kb'},
                                               namespace='nle/ibm')

        config.callbacks = RegistrationKey(name='callback',
                                           tags={'memory'},
                                           namespace='nle/ibm')

        config.metrics = RegistrationKey(name='metrics',
                                         tags={'memory'},
                                         namespace='nle/ibm')

        return config

    @classmethod
    def get_baseline_config(
            cls
    ):
        config = cls.get_default()

        config.model = RegistrationKey(name='model',
                                       tags={'baseline', 'lstm'},
                                       namespace='nle/ibm')

        config.pre_processor = RegistrationKey(name='processor',
                                               namespace='nle/ibm')

        config.callbacks = RegistrationKey(name='callback',
                                           namespace='nle/ibm')

        return config


@register
def register_routines():
    Registry.add_and_bind_variants(config_class=IBMRoutineConfig,
                                   config_constructor=IBMRoutineConfig.get_hf_baseline,
                                   component_class=CVRoutine,
                                   name='routine',
                                   tags={'hf', 'baseline'},
                                   namespace='nle/ibm')

    Registry.add_and_bind_variants(config_class=IBMRoutineConfig,
                                   config_constructor=IBMRoutineConfig.get_hf_kb_config,
                                   component_class=CVRoutine,
                                   name='routine',
                                   tags={'kb', 'hf'},
                                   namespace='nle/ibm')

    Registry.add_and_bind_variants(config_class=IBMRoutineConfig,
                                   config_constructor=IBMRoutineConfig.get_baseline_kb_config,
                                   component_class=CVRoutine,
                                   name='routine',
                                   tags={'kb', 'baseline'},
                                   namespace='nle/ibm')

    Registry.add_and_bind_variants(config_class=IBMRoutineConfig,
                                   config_constructor=IBMRoutineConfig.get_baseline_config,
                                   component_class=CVRoutine,
                                   name='routine',
                                   tags={'baseline', 'lstm'},
                                   namespace='nle/ibm')
