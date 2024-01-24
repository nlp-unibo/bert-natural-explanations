import os
from pathlib import Path
from typing import Optional

from cinnamon_core.core.component import Component
from cinnamon_core.utility import logging_utility
from cinnamon_core.utility.json_utility import save_json


class ResultsPipeline(Component):

    def run(
            self,
            serialize: bool = False,
            serialization_path: Optional[Path] = None,
            is_training: bool = True,
    ):
        # Run routine
        routine_result = self.routine.run(is_training=is_training,
                                          serialization_path=serialization_path)
        routine_result.steps = None

        routine_result_str = os.linesep.join([f'{key} --> {value}' for key, value in routine_result.items()])
        logging_utility.logger.info(f'Results: {os.linesep}{routine_result_str}')

        # Parse routine_result
        routine_result = self.routine_processor.run(data=routine_result)
        if serialization_path is not None and serialization_path.exists():
            save_json(serialization_path.joinpath('result.json'), routine_result.to_value_dict())
