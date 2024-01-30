from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import RegistrationKey, Registry, register
from cinnamon_generic.configurations.data_loader import DataLoaderConfig
from tos.components.data_loader import ToSLoader


class ToSLoaderConfig(DataLoaderConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.name = 'ToS'
        config.has_val_split = False
        config.has_test_split = False

        config.add(name='filename',
                   value='dataset_30.csv',
                   type_hint=str,
                   description='Name of the .csv file storing data',
                   is_required=True)
        config.add(name='file_manager_key',
                   value=RegistrationKey(
                       name='file_manager',
                       tags={'default'},
                       namespace='generic'
                   ),
                   type_hint=RegistrationKey,
                   description="registration info of built FileManager component."
                               " Used for filesystem interfacing")
        config.add(name='samples_amount',
                   value=-1,
                   type_hint=int,
                   description='Number of samples to take for each data split')
        config.add(name='category',
                   variants=[
                       'A',
                       # 'CH',
                       # 'CR',
                       # 'LTD',
                       # 'TER'
                   ],
                   is_required=True)

        return config


@register
def register_data_loaders():
    Registry.add_and_bind_variants(config_class=ToSLoaderConfig,
                                   component_class=ToSLoader,
                                   name='data_loader',
                                   namespace='nle/tos')
