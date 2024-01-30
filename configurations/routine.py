from cinnamon_core.core.registry import RegistrationKey
from cinnamon_generic.configurations.routine import CVRoutineConfig


class NLERoutineConfig(CVRoutineConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.metrics = RegistrationKey(name='metrics',
                                         tags={'clf_f1'},
                                         namespace='nle')
        config.helper = RegistrationKey(name='helper',
                                        tags={'default'},
                                        namespace='th')
        config.model_processor = RegistrationKey(name='processor',
                                                 tags={'th', 'classifier'},
                                                 namespace='nle')
        config.routine_processor = RegistrationKey(name='routine_processor',
                                                   tags={'average', 'fold'},
                                                   namespace='generic')

        config.seeds = [
            2023,
            15451,
            1337,
            2001,
            2080,
            666,
            1993,
            2048,
            42,
            69
        ]

        return config
