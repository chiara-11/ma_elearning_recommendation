import ast
import datetime
import json
import os
from enum import Enum

import pandas as pd

import config
import training_general


def read_data_file(filename: str) -> pd.DataFrame:
    """Returns pandas DataFrame stored under given filename in data folder."""
    return pd.read_csv(config.DATA_FOLDER / filename)


def save_as_csv(df: pd.DataFrame, filename: str, *, save_idx: bool) -> None:
    """Saves dataframe as csv in data folder."""
    df.to_csv(config.DATA_FOLDER / filename, index=save_idx)


def save_as_json(mydict: dict, filename: str) -> None:
    """Saves dictionary as json in data folder."""
    with open(config.DATA_FOLDER / filename, "w") as f:
        json.dump(mydict, f, indent=2)


def save_predictions(df: pd.DataFrame, conf: dict, *, save_idx: bool) -> None:
    """Saves dataframe as csv and conf dictionary as json in results folder with
    timestamp."""

    def enum_to_json(obj: Enum) -> str:
        if isinstance(obj, Enum):
            return str(obj)  # obj.value
        raise TypeError(f"Type {type(obj)} not serializable")  # noqa: EM102

    # get paths
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    save_info = conf["saving_file"]
    folder = save_info["folder"]
    if suff := save_info.get("filename_suffix"):
        filename = f"{save_info['filename']}_{suff}_{now}"
    else:
        filename = f"{save_info['filename']}_{now}"
    path_csv = config.RESULTS_FOLDER / folder / f"{filename}.csv"
    path_json = config.RESULTS_FOLDER / folder / f"{filename}.json"

    # save df as csv
    df.to_csv(path_csv, index=save_idx)

    # save conf as json
    with open(path_json, "w") as file:
        json.dump(conf, file, default=enum_to_json, indent=4)

    print(f"Saved predictions df and conf with filename {filename} in folder {folder}")


def save_evaluation_df(
    df: pd.DataFrame, save_info: dict, model_name: str, *, save_idx: bool
) -> None:
    """Saves evaluation dataframe as csv in results folder with timestamp."""
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    folder = save_info["folder"]
    if suff := save_info.get("filename_suffix"):
        filename = f"{model_name}_{suff}_{now}.csv"
    else:
        filename = f"{model_name}_{now}.csv"

    # save df as csv
    df.to_csv(config.RESULTS_FOLDER / folder / filename, index=save_idx)
    print(f"Saved evaluation df with filename {filename} in folder {folder}")


def read_predictions_and_conf(
    folder: str, filename: str, *, latest: bool
) -> tuple[pd.DataFrame, dict]:
    """Returns pandas DataFrame and conf stored under given filename in results
    folder."""
    if latest:
        filename = _get_latest_filename(folder, filename)
    print(f"Read file {filename}")

    # read csv
    df = pd.read_csv(config.RESULTS_FOLDER / folder / f"{filename}.csv")
    # read conf
    with open(config.RESULTS_FOLDER / folder / f"{filename}.json") as file:
        conf = json.load(file)

    return df, conf


def process_predictions(pred_df: pd.DataFrame, conf: dict) -> pd.DataFrame:
    pred_df = pred_df.set_index(["class_id", "ut_id", "student_id", "ref_class"])

    list_cols = ["y_true"] + [
        training_general.build_model_name(model_params)
        for model_params in conf["models"]
    ]
    return convert_str_cols_to_lists(pred_df, list_cols)


def read_evaluation_df(
    folder: str, filename: str, *, latest: bool, print_filename: bool = True
) -> pd.DataFrame:
    """Returns evaluation dataframe stored under given filename in results folder."""
    if latest:
        filename = _get_latest_filename(folder, filename) + ".csv"
    if print_filename:
        print(f"Read file {filename}")

    return pd.read_csv(config.RESULTS_FOLDER / folder / filename)


def _get_latest_filename(folder: str, filename: str) -> str:
    # find latest file with filename as prefix and correct file type
    dates = [
        datetime.datetime.strptime(file[-19:-4], "%Y%m%d_%H%M%S")  # noqa: DTZ007
        for file in os.listdir(config.RESULTS_FOLDER / folder)
        if file[-4:] == ".csv" and file[:-20] == filename
    ]
    if len(dates) == 0:
        print("There is no file with correct file type and specified prefix.")
        raise ImportError
    return f"{filename}_{max(dates).strftime('%Y%m%d_%H%M%S')}"


def convert_str_cols_to_lists(df: pd.DataFrame, list_cols: list[str]) -> pd.DataFrame:
    def _convert_to_list(col: pd.Series) -> pd.Series:
        return col.apply(ast.literal_eval)

    for col in list_cols:
        df[col] = _convert_to_list(df[col])
    return df


def read_action_logs(filename: str = "action_logs.csv") -> pd.DataFrame:
    return read_data_file(filename)


def read_unit_test_scores(
    filename: str = "training_unit_test_scores.csv",
) -> pd.DataFrame:
    return read_data_file(filename)


def read_assignment_relationships(
    filename: str = "assignment_relationships.csv",
) -> pd.DataFrame:
    df = read_data_file(filename)
    return df.drop_duplicates()


def read_assignment_details(filename: str = "assignment_details.csv") -> pd.DataFrame:
    df = read_data_file(filename)
    return df.set_index("assignment_log_id")  # one row for each assignment


def read_sequence_relationships(
    filename: str = "sequence_relationships.csv",
) -> pd.DataFrame:
    df = read_data_file(filename)
    return df.drop_duplicates()


def read_sequence_details(
    filename: str = "sequence_details.csv", *, second_save: bool = False
) -> pd.DataFrame:
    df = read_data_file(filename)
    df["sequence_problem_ids"] = convert_str_col_to_list(
        df, "sequence_problem_ids", second_save=second_save
    )
    return df


def read_problem_details(
    filename: str = "problem_details.csv", *, second_save: bool = False
) -> pd.DataFrame:
    df = read_data_file(filename)
    df = df.set_index("problem_id")  # one row for each problem id
    df["problem_text_bert_pca"] = convert_str_col_to_list(
        df, "problem_text_bert_pca", second_save=second_save
    )
    return df


def load_all_data() -> (
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]
):
    # result dataframes
    action_logs = read_action_logs()
    unit_test_scores = read_unit_test_scores()
    # assignment information
    assignment_relationships = read_assignment_relationships()
    assignment_details = read_assignment_details()
    # sequence information
    sequence_relationships = read_sequence_relationships()
    sequence_details = read_sequence_details()
    # problem information
    problem_details = read_problem_details()
    return (
        action_logs,
        unit_test_scores,
        assignment_relationships,
        assignment_details,
        sequence_relationships,
        sequence_details,
        problem_details,
    )


def convert_str_col_to_list(
    df: pd.DataFrame, col: str, *, second_save: bool = False
) -> pd.Series:
    if second_save:  # if df has been read and saved again
        return df[col].apply(lambda list_str: list_str[2:-2].split("', '"))
    return df[col].apply(lambda list_str: list_str[1:-1].split(","))
