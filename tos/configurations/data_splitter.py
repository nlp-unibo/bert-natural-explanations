from cinnamon_core.core.registry import Registry, register
from cinnamon_generic.components.data_splitter import PrebuiltCVSplitter
from cinnamon_generic.configurations.data_splitter import PrebuiltCVSplitterConfig


class TosPrebuiltCVSplitterConfig(PrebuiltCVSplitterConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.prebuilt_filename = 'tos_folds.pkl'
        config.file_manager_key = None
        config.held_out_key = 'test'
        config.splitter_args['n_splits'] = 10
        config.y_key = 'document'

        return config


@register
def register_data_splitters():
    Registry.add_and_bind(config_class=TosPrebuiltCVSplitterConfig,
                          component_class=PrebuiltCVSplitter,
                          name='data_splitter',
                          namespace='nle/tos')
