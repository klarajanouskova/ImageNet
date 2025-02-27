import json
import os

import numpy as np
import pandas as pd

from eval_corrections.load_data.base_dataset import Entry, Dataset


class _ChildEntry(Entry):
    def __init__(self, entry_id: str, original_label: int, cl_label: int, url: str, mturk: dict):
        """
        Initializes a _ChildEntry instance with additional information.

        :param entry_id: Identifier for the entry.
        :param original_label: The original label of the entry.
        :param cl_label: Confidence level framework's label for the entry.
        :param url: Link to the entry's image.
        :param mturk: MTurk results for the entry as a dict.
        """
        super().__init__(entry_id, original_label, None, add_category=False)
        self.cl_label = cl_label
        self.is_manually_evaluated = True
        self.url = url
        self.mturk = mturk
        self.category, self.proposed_labels = self.__determine_category()

    def __determine_category(self, majority_count: int = 3) -> (str, np.ndarray | None):
        """
        Determines the category of the entry based on the mturk decision.

        :return: A tuple: string representing the category, numpy array of proposed labels_option.
        """

        if self.mturk['given'] >= majority_count:
            return 'A', np.array([self.original_label])
        elif self.mturk['guessed'] >= majority_count:
            return 'B', np.array([self.cl_label])
        elif self.mturk['both'] >= majority_count:
            return 'M', np.array([self.original_label, self.cl_label])
        elif self.mturk['neither'] >= majority_count:
            return 'Z', None
        else:
            return 'X', None


class LabelErrors(Dataset):
    def set_entries(self) -> None:
        self.set_entries_from_json(file_path ='label_err_mturk.json')

    def set_entries_from_json(self, file_path: str) -> None:
        """
        Load entries from a JSON file and set them as attributes of the instance.
        """
        current_dir = os.path.dirname(__file__)

        with open(os.path.join(current_dir, file_path), 'r') as file:
            data = json.load(file)
            self.entries = [
                _ChildEntry(
                    entry_id=f"ILSVRC2012_val_{int(record['id']):08d}.JPEG",
                    original_label=int(record["given_original_label"]),
                    cl_label=int(record["our_guessed_label"]),
                    url=record["url"],
                    mturk=record["mturk"]
                ) for record in data
            ]
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
                'proposed_labels': ', '.join(map(str, entry.proposed_labels.tolist()))
                if entry.proposed_labels is not None else '',
                'manually_validated': entry.is_manually_evaluated,
                'cl_label': str(entry.cl_label),
                'mturk': entry.mturk,
            })

        return pd.DataFrame(data)
