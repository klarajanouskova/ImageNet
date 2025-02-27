import numpy as np
import pandas as pd
import tensorflow_datasets as tfds


class Entry:
    def __init__(self, entry_id: str, original_label: int, proposed_labels: np.ndarray | None,
                 add_category: bool = True):
        """
        Initializes an Entry instance.

        :param entry_id: Identifier for the entry.
        :param original_label: The original label of the entry.
        :param proposed_labels: Proposed labels_option for the entry as a numpy array.
        :param add_category: Flag to determine if category addition should be called, defaults to True.
        """
        self.id = entry_id
        self.original_label = original_label
        self.proposed_labels = proposed_labels

        self.is_duplicate: bool = False
        self.is_manually_evaluated: bool | None = None

        if add_category:
            self.category = self.__determine_category()

    def __determine_category(self) -> str:
        """
        Determines the category of the entry based on the labels_option.

        :return: A string representing the category.
        """
        if len(self.proposed_labels) == 0:
            return 'Z'
        elif len(self.proposed_labels) == 1:
            if self.proposed_labels[0] == self.original_label:
                return 'A'
            else:
                return 'B'
        else:
            return 'M'


class Dataset:
    def __init__(self, dataset_name: str = None, split: str = 'validation'):
        """
        Initializes a Dataset instance.

        :param dataset_name: Name of the dataset to load (default is None for loading from DataFrame).
        :param split: The dataset split to load (default is 'validation').
        """
        self.dataset_name = dataset_name
        self.split = split
        self.annotations = None
        self.entries = []

    def load_annotations(self) -> None:
        """
        Loads dataset annotations using TensorFlow Datasets.
        """
        if self.dataset_name is not None:
            self.annotations = tfds.load(name=self.dataset_name, split=self.split)

    def set_entries(self) -> None:
        pass

    def entries_to_dataframe(self) -> pd.DataFrame:
        """
        Converts the entries array into a Pandas DataFrame.

        :return: DataFrame with columns named by Entry attributes.
        """
        data = []
        for entry in self.entries:
            data.append({
                'id': entry.id,
                'category': entry.category,
                'original_label': entry.original_label,
                'proposed_labels': ', '.join(
                    map(str, entry.proposed_labels)) if entry.proposed_labels is not None else '',
                'manually_validated': entry.is_manually_evaluated
            })

        return pd.DataFrame(data)
