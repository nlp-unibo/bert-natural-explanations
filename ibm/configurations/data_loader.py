from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import RegistrationKey, Registry, register
from cinnamon_generic.configurations.data_loader import DataLoaderConfig
from ibm.components.data_loader import IBMLoader


class IBMLoaderConfig(DataLoaderConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.name = 'IBM2015'
        config.has_val_split = False
        config.has_test_split = False

        config.add(name='filename',
                   value='dataset',
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
        config.add(name='topics',
                   variants=[
                       1,
                       2,
                       3,
                       4
                   ],
                   is_required=True)

        return config


@register
def register_data_loaders():
    Registry.add_and_bind_variants(config_class=IBMLoaderConfig,
                                   component_class=IBMLoader,
                                   name='data_loader',
                                   namespace='nle/ibm')
