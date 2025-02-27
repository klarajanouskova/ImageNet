import itertools
import pandas as pd
from typing import List, Tuple, Union


def filter_by_categories(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """
    Filter the DataFrame to include only rows with specified categories.

    Args:
    - df (pd.DataFrame): The DataFrame to filter.
    - categories (list of str): The categories to remain.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    return df[df['category'].isin(categories)]


def filter_inconsistent_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to remove inconsistent category data and sets validation types.

    Args:
        df (pd.DataFrame): The input DataFrame to be filtered and processed.

    Returns:
        pd.DataFrame: The processed DataFrame with inconsistent categories filtered out and validation type set.
    """
    return set_validation_type(__filter_inconsistent_rows(df.copy(), pattern='category'))


def filter_inconsistent_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows where columns matching the pattern have inconsistent values and consolidate them.

    Args:
    - df (pd.DataFrame): The DataFrame to filter.

    Returns:
    - pd.DataFrame: The filtered DataFrame with consistent rows and consolidated columns.
    """
    return __filter_inconsistent_rows(df.copy(), pattern='proposed_labels')


def __filter_inconsistent_rows(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """
    Filters out rows from a DataFrame that contain inconsistent values based on a specified pattern.

    Args:
        df (pd.DataFrame): The input DataFrame to be filtered.
        pattern (str): The pattern used to identify and process specific columns in the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with duplicate-labeled columns removed and rows with -1 values in the specified
        pattern column filtered out.
    """
    df = __remove_duplicate_cols(df, pattern)
    df = df[df[pattern] != -1]

    return df


def __remove_duplicate_cols(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """
    Remove duplicate columns starting with 'original_label' and consolidate them into a single column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    pattern (str): The pattern to match the column names that need to be consolidated.

    Returns:
    pd.DataFrame: The modified DataFrame with consolidated 'original_label' column.
    """
    column_names = [col for col in df.columns if col.startswith(pattern)]

    df[pattern] = df[column_names].apply(__check_values, axis=1)
    df.drop(columns=column_names, inplace=True)

    return df


def __check_values(row):
    """
    Check if all lists in a row are identical, considering NaN values.

    Parameters:
    row (pd.Series): A row of values from the DataFrame, each cell containing a list stored as a string.

    Returns:
    list or int: The identical list if all are the same, otherwise -1.
    """
    first_item = row.iloc[0]

    if pd.isna(first_item):
        if all(pd.isna(item) for item in row):
            return first_item
        else:
            return -1
    else:
        if all(first_item == item for item in row):
            return first_item
        else:
            return -1


def intersect_and_combine(dfs: List[pd.DataFrame], columns: List[str],
                          rows_to_omit: Union[List, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes a list of DataFrames and a list of column names, finds intersections by all columns,
    and returns a new DataFrame combining all columns from all DataFrames with unique column names.

    Args:
    - dfs (list of pd.DataFrame): List of DataFrames to process.
    - columns (list of str): List of column names to find intersections.
    - rows_to_omit (list or None): Rows to omit from the final DataFrame.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: DataFrame with combined columns from all input DataFrames and the
    intersections DataFrame.
    """
    if not dfs:
        raise ValueError("The list of dataframes is empty.")

    if not columns:
        raise ValueError("The list of columns is empty.")

    for df in dfs:
        if not all(col in df.columns for col in columns):
            raise ValueError("Not all specified columns are present in all DataFrames.")

    intersections = dfs[0][columns].copy()

    for df in dfs[1:]:
        intersections = intersections.merge(df[columns], on=columns, how='inner')

    if rows_to_omit is not None:
        df_merged = pd.merge(intersections, rows_to_omit, on=columns, how='left', indicator=True)
        intersections = df_merged[df_merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    combined_df = __combine_dataframes(dfs, intersections, columns)
    pattern = 'original_label'
    combined_df = __remove_duplicate_cols(combined_df.copy(), pattern)
    combined_df[pattern] = combined_df[pattern].astype(int)

    return combined_df, intersections


def __combine_dataframes(dfs: List[pd.DataFrame], combined_df: pd.DataFrame,
                         combine_columns: List[str]) -> pd.DataFrame:
    """
    Combine multiple DataFrames into one by merging on specified columns.

    Args:
    - dfs (list of pd.DataFrame): List of DataFrames to combine.
    - combined_df (pd.DataFrame): DataFrame with intersections.
    - combine_columns (list of str): Columns to merge on.

    Returns:
    - pd.DataFrame: The combined DataFrame.
    """
    combined_result = combined_df.copy()

    for idx, df in enumerate(dfs):
        columns_to_rename = {col: f"{col}_{idx}" for col in df.columns if col not in combine_columns}
        df_renamed = df.rename(columns=columns_to_rename)
        combined_result = combined_result.merge(df_renamed, on=combine_columns, how='inner')

    return combined_result


def find_all_intersections(dfs: List[pd.DataFrame], combination_length: int, columns: List[str],
                           prev_intersections: Union[pd.DataFrame, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find all intersections of DataFrames by combinations of a given length.

    Args:
    - dfs (list of pd.DataFrame): List of DataFrames to process.
    - combination_length (int): Length of DataFrame combinations to consider.
    - columns (list of str): Columns to find intersections.
    - prev_intersections (pd.DataFrame or None): Previously found intersections to omit.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: Combined DataFrame and intersections DataFrame.
    """
    combined_df = pd.DataFrame()
    intersections = pd.DataFrame()

    df_combinations = __get_combinations(dfs, combination_length)
    for combination in df_combinations:
        combined_, intersections_ = intersect_and_combine(combination, columns, rows_to_omit=prev_intersections)

        combined_df = pd.concat([combined_df, combined_], ignore_index=True)
        intersections = pd.concat([intersections, intersections_], ignore_index=True)

    return combined_df, intersections


def __get_combinations(dfs: List[pd.DataFrame], combination_length: int) -> List[Tuple[pd.DataFrame]]:
    """
    Get all combinations of DataFrames of a specified length.

    Args:
    - dfs (list of pd.DataFrame): List of DataFrames to combine.
    - combination_length (int): Length of combinations.

    Returns:
    - list of tuples: List of DataFrame combinations.
    """
    return list(itertools.combinations(dfs, combination_length))


def set_validation_type(df: pd.DataFrame, new_col_name: str = 'validation') -> pd.DataFrame:
    """
    Add a new column to the DataFrame based on the values in each row for columns matching the pattern.

    Args:
    - df (pd.DataFrame): The DataFrame to process.
    - new_col_name (str): The name for a new column.

    Returns:
    - pd.DataFrame: DataFrame with the new column added.
    """
    pattern: str = 'manually_validated'
    column_names = [col for col in df.columns if col.startswith(pattern)]
    df.loc[:, new_col_name] = df[column_names].apply(__calculate_occurrence, axis=1)
    df.drop(columns=column_names, inplace=True)

    col = df.pop(new_col_name)
    df.insert(2, new_col_name, col)

    return df


def __calculate_occurrence(row: pd.Series) -> str:
    """
    Calculate the occurrence pattern of True and False values in a row.

    Args:
    - row (pd.Series): The row to analyze.

    Returns:
    - str: A string representing the occurrence pattern of True and False values.
    """
    num_true = (row == True).sum()
    num_false = (row == False).sum()

    return '+' * num_true + '*' * num_false
