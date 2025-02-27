import os
import re

import numpy as np

from eval_corrections.load_data.base_dataset import Entry, Dataset


class _ChildEntry(Entry):
    def __init__(self, entry_id: int, original_label: int, proposed_labels: np.ndarray,
                 is_manually_evaluated: bool = True):
        """
        Initializes a _ChildEntry instance with additional information.

        :param entry_id: Identifier for the entry.
        :param original_label: The original label of the entry.
        :param proposed_labels: Proposed labels_option for the entry as a numpy array.
        :param is_manually_evaluated: Indicates whether the evaluation was performed manually.
        """
        super().__init__(entry_id, original_label, proposed_labels)
        self.is_manually_evaluated = is_manually_evaluated.item()


class Real(Dataset):
    def __init__(self, split: str = 'validation'):
        """
        Initializes a Dataset instance.

        :param split: The dataset split to load (default is 'validation').
        """
        super().__init__(dataset_name='imagenet2012_real', split=split)

    def set_entries(self, manual_ids_filename: str = 'manual_real_imgs.npy') -> None:
        """
        Processes annotations and sets entries as numpy arrays.
        Lazy loads annotations if not already loaded.
        """
        if self.annotations is None:
            self.load_annotations()

        current_dir = os.path.dirname(__file__)
        manual_ids = np.load(os.path.join(current_dir, manual_ids_filename))

        for annotation in self.annotations:
            filename = annotation['file_name'].numpy().decode('utf-8')

            entry = _ChildEntry(
                entry_id=filename,
                original_label=annotation['original_label'].numpy(),
                proposed_labels=np.array(annotation['real_label'].numpy()),
                is_manually_evaluated=np.isin(filename, manual_ids)
            )
            self.entries.append(entry)
        self.entries = np.array(self.entries)
