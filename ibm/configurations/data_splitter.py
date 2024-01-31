from cinnamon_core.core.registry import Registry, register, RegistrationKey
from cinnamon_generic.components.data_splitter import PrebuiltCVSplitter
from cinnamon_generic.configurations.data_splitter import PrebuiltCVSplitterConfig


class IBMPrebuiltCVSplitterConfig(PrebuiltCVSplitterConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add(name='topics',
                   is_required=True,
                   variants=[
                       1,
                       2,
                       3,
                       4
                   ],
                   type_hint=int,
                   description='Dataset topics variant.')

        config.prebuilt_filename = f'ibm_{config.topics}_folds.pkl'
        config.file_manager_key = RegistrationKey(
            name='file_manager',
            tags={'default'},
            namespace='generic'
        )
        config.held_out_key = 'test'
        config.splitter_args['n_splits'] = 4
        config.y_key = 'C_Label'
        config.X_key = 'C_Label'

        return config


@register
def register_data_splitters():
    Registry.add_and_bind(config_class=IBMPrebuiltCVSplitterConfig,
                          component_class=PrebuiltCVSplitter,
                          name='data_splitter',
                          namespace='nle/ibm')
