import os

import numpy as np
import pandas as pd

from eval_corrections.load_data.base_dataset import Entry, Dataset


class _ChildEntry(Entry):
    def __init__(self, entry_id: str, original_label: int, proposed_labels: np.ndarray, annotation_type: str = None):
        """
        Initializes a _ChildEntry instance with additional information.

        :param entry_id: Identifier for the entry.
        :param original_label: The original label of the entry.
        :param proposed_labels: Proposed labels_option for the entry as a numpy array.
        :param annotation_type: The type of the annotation in the original work.
        """
        super().__init__(entry_id=entry_id,
                         original_label=original_label,
                         proposed_labels=None,
                         add_category=False)

        self.is_manually_evaluated = True
        self.category, self.proposed_labels = self.__determine_category(annotation_type, proposed_labels)

    def __determine_category(self, annotation_type: str, proposed_labels: np.ndarray) -> (str, np.ndarray | None):
        """
        Determines the category of the entry based on the annotation type.

        :return: A string representing the category.
        """
        if annotation_type is None:
            if len(proposed_labels) == 1:
                if self.original_label == proposed_labels[0]:
                    return 'A', proposed_labels
                else:
                    return 'B', proposed_labels
            else:
                return 'M', proposed_labels
        elif annotation_type == 'easy':
            return 'A', proposed_labels
        elif annotation_type == 'amb':
            return 'X', None
        else:
            return 'Z', None


class FinegrainedAnnotations(Dataset):
    def set_entries(self) -> None:
        self.set_entries_from_pkl(file_path_annotation_categories = 'annotation_categories.pkl',
                                  file_path_annotation_contains = 'annotation_contains.pkl',
                                  file_path_annotation_classify = 'annotation_classify.pkl')

    def set_entries_from_pkl(self, file_path_annotation_categories: str, file_path_annotation_contains: str,
                             file_path_annotation_classify: str) -> None:
        """
        Set entries from a pickle file.
        """
        current_dir = os.path.dirname(__file__)

        annotation_categories = pd.read_pickle(os.path.join(current_dir, file_path_annotation_categories))
        annotation_contains = pd.read_pickle(os.path.join(current_dir, file_path_annotation_contains))
        annotation_classify = pd.read_pickle(os.path.join(current_dir, file_path_annotation_classify))

        for index, row in annotation_categories.iterrows():
            filename = index
            annotation = row['annotation']

            if annotation == 'fu':
                original_label = annotation_classify.loc[filename, 'imagenet_label']
                objects = annotation_classify.loc[filename, 'objects']
                proposed_labels = np.array(
                    list(set([sorted((-v[0], k) for k, v in o.items())[0][1] for o in objects])))

                self.entries.append(_ChildEntry(entry_id=filename,
                                                original_label=original_label,
                                                proposed_labels=proposed_labels))
            else:
                original_label = annotation_contains.loc[filename, 'imagenet_label']
                proposed_labels = np.array([original_label])

                self.entries.append(_ChildEntry(entry_id=filename,
                                                original_label=original_label,
                                                proposed_labels=proposed_labels,
                                                annotation_type=annotation))
        self.entries = np.array(self.entries)
