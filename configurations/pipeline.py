from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import RegistrationKey
from cinnamon_generic.configurations.calibrator import TunableConfiguration


class ResultsPipelineConfig(TunableConfiguration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='routine',
                   value=None,
                   is_required=True,
                   is_child=True)

        config.add(name='routine_processor',
                   value=RegistrationKey(name='processor',
                                         tags={'results'},
                                         namespace='nle'),
                   is_required=True,
                   is_child=True)

        return config
