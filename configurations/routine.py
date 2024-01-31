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
            15371,
            15372,
            15373
        ]

        return config
