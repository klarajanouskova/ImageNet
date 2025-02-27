import pandas as pd
import numpy as np
from typing import List, Optional, Union, Set


class DatasetSlicer:
    """
    A class to handle slicing of a dataset and processing operations.

    Attributes:
        dfs: A list of pandas DataFrames to be processed.
        intersected: A list of DataFrames with intersected images.
        not_intersected_flat: A list of DataFrames with non-intersected images.
        intersected_same_cat: A list of DataFrames with intersected images in the same category.
        intersected_diff_cat: A list of DataFrames with intersected images in different categories.
        verified: A list of DataFrames of images that have been verified.
        inconsistent_flat: A list of DataFrames of images with inconsistent labels_option.
        verified_flat: A concatenated DataFrame of all verified images.
    """
    def __init__(self, dfs: List[pd.DataFrame]):
        """
        Initializes the DatasetSlicer with a list of DataFrames.

        Args:
            dfs (List[pd.DataFrame]): A list of pandas DataFrames to be processed.
        """
        self.dfs = dfs

        self.intersected: Optional[List[pd.DataFrame]] = None
        self.not_intersected_flat: Union[pd.DataFrame, None] = None

        self.intersected_same_cat: Optional[List[pd.DataFrame]] = None
        self.intersected_diff_cat: Optional[List[pd.DataFrame]] = None

        self.intersected_flat: Union[pd.DataFrame, None] = None

        self.verified: Optional[List[pd.DataFrame]] = None
        self.inconsistent_flat: Union[pd.DataFrame, None] = None

        self.verified_flat: Union[pd.DataFrame, None] = None

    def get_all_ids(self, df_list: Optional[List[pd.DataFrame]] = None) -> Set[str]:
        """
        Retrieves all IDs from a list of DataFrames.

        Args:
            df_list (Optional[List[pd.DataFrame]]): A list of pandas DataFrames to extract IDs from.

        Returns:
            List[str]: A list of IDs from the provided DataFrames.
        """
        if df_list is None:
            nums = np.arange(1, 50001)
            arr = np.core.defchararray.add(
                np.core.defchararray.add("ILSVRC2012_val_", np.char.zfill(nums.astype(str), 8)),
                ".JPEG"
            )
            return set(arr)

        ids = []
        for df in df_list:
            ids += df['id'].tolist()
        return set(ids)

    def get_not_intersected_ids(self, intersected_ids: Set, all_ids: Set = None) -> List[str]:
        """
        Returns a list of IDs that are present in `dataset_ids` but not in `intersected_ids`.

        Args:
            intersected_ids (Set): A set of IDs to be excluded from the output.
        """
        union_set = set(self.get_all_ids()) if not all_ids else all_ids
        return list(union_set - intersected_ids)

    def get_all_intersected_ids(self) -> Set[str]:
        """
        Retrieves all IDs from the intersected DataFrames.

        Returns:
            List[str]: A list of IDs from the intersected DataFrames.
        """
        return self.get_all_ids(self.intersected)

    def get_all_same_cat_ids(self) -> Set[str]:
        """
        Retrieves all IDs from the intersected DataFrames with the same category.

        Returns:
            List[str]: A list of IDs from the intersected DataFrames with the same category.
        """
        return self.get_all_ids(self.intersected_same_cat)

    def get_all_verified_ids(self) -> Set[str]:
        """
        Retrieves all image IDs from the verified DataFrames.

        Returns:
            List[str]: A list of image IDs from the verified DataFrames.
        """
        return self.get_all_ids(self.verified)

    def concat_verified(self) -> pd.DataFrame:
        if self.verified is None:
            raise ValueError("The list of DataFrames is not initialized.")

        if not self.verified:
            raise ValueError("The list of DataFrames is empty.")

        self.verified_flat = pd.concat(self.verified, ignore_index=True)

        col_order = ["id", "category", "validation", "original_label", "proposed_labels"]
        self.verified_flat = self.verified_flat.loc[:, [col for col in col_order if col in self.verified_flat.columns]]

        return self.verified_flat
