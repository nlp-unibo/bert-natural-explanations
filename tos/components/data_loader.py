from pathlib import Path
from typing import Tuple, Optional, Any, Iterable, List

import numpy as np
import pandas as pd

from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.data_loader import DataLoader
from cinnamon_generic.components.file_manager import FileManager


class ToSLoader(DataLoader):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kb = None

    def load_kb(
            self,
            filepath: Path
    ) -> Iterable[str]:
        sentences = []

        with open(filepath, 'r') as f:
            for line in f:
                sentences.append(line)

        return sentences

    def load_data(
            self
    ):
        file_manager = FileManager.retrieve_component_instance(name='file_manager',
                                                               tags={'default'},
                                                               namespace='generic')
        df_path: Path = file_manager.dataset_directory.joinpath(self.name)
        if not df_path.exists():
            df_path.mkdir(parents=True)

        df_path = df_path.joinpath(self.filename)
        df = pd.read_csv(df_path)

        if self.category:
            kb_name = f'{self.category}_KB.txt'
        else:
            kb_name = 'KB.txt'

        kb_path = df_path.with_name(kb_name)
        self.kb = self.load_kb(filepath=kb_path)

        return df

    def get_splits(
            self
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        df = self.load_data()

        return df, None, None

    def parse(
            self,
            data: Optional[Any] = None,
    ) -> Optional[FieldDict]:

        if data is None:
            return data

        return_field = FieldDict()
        return_field.add(name='text',
                         value=data['text'].values,
                         type_hint=Iterable[str],
                         tags={'text'},
                         description='Clause text to classify')
        return_field.add(name='label',
                         value=data[self.category].values,
                         type_hint=Iterable[str],
                         tags={'label'},
                         description='Unfair clause category')
        return_field.add(name='kb',
                         value=self.kb,
                         type_hint=List[str],
                         tags={'metadata'},
                         description="Natural language explanations of unfairness")
        targets = data[f'{self.category}_targets'].values
        targets = [[int(item) for item in t.replace('[', '').replace(']', '').split(',')] if t is not np.nan else [] for
                   t in targets]
        return_field.add(name='targets',
                         value=targets,
                         type_hint=Iterable[List[int]],
                         tags={'metadata'},
                         description='Ground-truth explanation indexes associated to sample.'
                                     'The targets are used by strong supervision to guide a model to'
                                     'correctly select explanations.')
        return return_field
