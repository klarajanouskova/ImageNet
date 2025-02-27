import re

import numpy as np
import pandas as pd

from eval_corrections.load_data.base_dataset import Entry, Dataset


class _ChildEntry(Entry):
    def __init__(self, entry_id: str, original_label: int, proposed_labels: np.ndarray,
                 unclear_multi_labels: np.ndarray, wrong_multi_labels: np.ndarray, is_problematic: bool):
        """
        Initializes a _ChildEntry instance with additional information.

        :param entry_id: Identifier for the entry.
        :param original_label: The original label of the entry.
        :param proposed_labels: Proposed labels_option for the entry as a numpy array.
        :param unclear_multi_labels: Unclear labels_option for the entry as a numpy array.
        :param wrong_multi_labels: Wrong labels_option for the entry as a numpy array.
        :param is_problematic: Is the entry problematic.
        """
        super().__init__(entry_id, original_label, proposed_labels)
        self.is_manually_evaluated = True
        self.unclear_multi_labels = unclear_multi_labels
        self.wrong_multi_labels = wrong_multi_labels
        self.is_problematic = is_problematic


class Multilabel(Dataset):
    def __init__(self, split: str = 'validation'):
        """
        Initializes a Dataset instance.

        :param split: The dataset split to load (default is 'validation').
        """
        super().__init__(dataset_name='imagenet2012_multilabel', split=split)

    def set_entries(self) -> None:
        """
        Processes annotations and sets _ChildEntry instances as numpy arrays.
        Lazy loads annotations if not already loaded.
        """
        if self.annotations is None:
            self.load_annotations()

        for annotation in self.annotations:
            filename = annotation['file_name'].numpy().decode('utf-8')

            entry = _ChildEntry(
                entry_id=filename,
                original_label=annotation['original_label'].numpy(),
                proposed_labels=np.array(annotation['correct_multi_labels'].numpy()),
                unclear_multi_labels=annotation['unclear_multi_labels'].numpy(),
                wrong_multi_labels=annotation['wrong_multi_labels'].numpy(),
                is_problematic=annotation['is_problematic'].numpy()
            )
            self.entries.append(entry)
        self.entries = np.array(self.entries)

    def entries_to_dataframe(self) -> pd.DataFrame:
        """
        Converts the entries array into a Pandas DataFrame including additional annotation_type.

        :return: DataFrame with columns named by _ChildEntry attributes.
        """
        data = []
        for entry in self.entries:
            data.append({
                'id': entry.id,
                'category': entry.category,
                'original_label': entry.original_label,
                'proposed_labels': ', '.join(
                    map(str, entry.proposed_labels)) if entry.proposed_labels is not None else '',
                'manually_validated': entry.is_manually_evaluated,
                'unclear_multi_labels': ', '.join(
                    map(str, entry.unclear_multi_labels)) if entry.unclear_multi_labels is not None else '',
                'wrong_multi_labels': ', '.join(
                    map(str, entry.wrong_multi_labels)) if entry.wrong_multi_labels is not None else '',
                'is_problematic': entry.is_problematic,
            })

        return pd.DataFrame(data)
