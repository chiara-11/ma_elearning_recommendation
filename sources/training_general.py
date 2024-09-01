import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

sys.path.append(Path("../../sources").resolve())

import config
import utils
from config import LimType, InfoCols, RegMetrics, ClassMetrics, RecMethod
from collaborative_filtering import cf
from content_based_recommendation import cb
from data_preparation import determine_reference_classes
from item_response_theory import irt
from knowledge_tracing import kt


def check_conf(conf_dict: dict, *, save_file: bool) -> None:
    # check lim parameter
    lim_list = conf_dict.get("lim")
    assert lim_list, "There is no limit specified."
    assert isinstance(lim_list, list), "The limit must be of type list."
    assert all(
        (isinstance(lim, float) and 0 <= lim <= 1) or (isinstance(lim, LimType))
        for lim in lim_list
    ), "Each limit in the list should be a float between 0 and 1 or of type LimType."

    # check eval columns parameter
    eval_groups = conf_dict.get("eval_groups")
    assert eval_groups, "There are no evaluation column groups specified."
    assert isinstance(
        eval_groups, list
    ), "The parameter eval_groups should be of type list."
    assert all(
        group in config.EVAL_COL_GROUPS for group in eval_groups
    ), f"The groups specified in eval_groups should be in {config.EVAL_COL_GROUPS}."

    if "reg_metrics" in eval_groups:
        # check regression metrics parameters
        reg_metrics = conf_dict.get("reg_metrics")
        assert not reg_metrics or (
            isinstance(reg_metrics, list)
            and all(isinstance(met, RegMetrics) for met in reg_metrics)
        ), "The specified regression metrics are not valid."

    if "class_metrics" in eval_groups:
        # check classification metrics parameters
        class_metrics = conf_dict.get("class_metrics")
        assert not class_metrics or (
            isinstance(class_metrics, list)
            and all(isinstance(met, ClassMetrics) for met in class_metrics)
        ), "The specified classification metrics are not valid."

    if "info_cols" in eval_groups:
        # check info columns parameter
        info_cols = conf_dict.get("info_cols")
        assert not info_cols or (
            isinstance(info_cols, list)
            and all(isinstance(col, InfoCols) for col in info_cols)
        ), "The specified info columns are not valid."

    # check parameter with_ref_class:
    with_rc = conf_dict.get("with_ref_class")
    assert (
        with_rc is True or with_rc is False
    ), "There should be a parameter with_ref_class which is either True or False."

    # check method parameter
    method = conf_dict.get("method")
    assert method, "There is no recommendation method specified."
    assert isinstance(
        method, RecMethod
    ), "The specified recommendation method is not valid."

    # check method specific parameters
    _check_conf_model_params(method, conf_dict.get("models"), with_rc=with_rc)

    # check saving configuration
    if save_file:
        save_conf = conf_dict.get("saving_file")
        assert save_conf, "There is no path for saving the prediction df."
        folder = save_conf.get("folder")
        assert isinstance(folder, str), (
            "There is no folder specified. For storing in the root results folder use"
            "an empty string."
        )
        assert Path(
            config.RESULTS_FOLDER / folder
        ).exists(), "The given folder does not exist in the results folder."
        filename = save_conf.get("filename")
        assert filename, "There is no filename specified for saving the prediction df."
        filename_suffix = save_conf.get("filename_suffix")
        assert not filename_suffix or isinstance(
            filename_suffix, str
        ), "The specified filename suffix is in the wrong format."


def _check_conf_model_params(
    method: RecMethod, model_list: list[dict], *, with_rc: bool
) -> None:
    if method == RecMethod.CF:
        cf.check_conf_cf(model_list, with_rc=with_rc)
    elif method == RecMethod.CB:
        cb.check_conf_cb(model_list, with_rc=with_rc)
    elif method == RecMethod.IRT:
        irt.check_conf_irt(model_list, with_rc=with_rc)
    elif method == RecMethod.KT:
        kt.check_conf_kt(model_list, with_rc=with_rc)
    else:
        raise NotImplementedError


def prepare_df(conf: dict, df: pd.DataFrame) -> pd.DataFrame:
    if conf["method"] == RecMethod.CB:
        return cb.prepare_df_for_cb(df)
    if conf["method"] == RecMethod.KT:
        return kt.prepare_df_for_kt(df)
    raise NotImplementedError


def create_dataframes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    ass_seq = determine_reference_classes.create_ass_seq(df).set_index(["ut_class"])

    # list of students per class
    stud_per_class = (
        df.loc[df["unit_test"] == 1]
        .groupby("class_id")["student_id"]
        .unique()
        .apply(list)
    )

    return df.set_index("student_id"), ass_seq, stud_per_class


def initialize_pred_df(index: pd.MultiIndex | pd.Index, conf: dict) -> pd.DataFrame:
    model_cols = [build_model_name(model_params) for model_params in conf["models"]]
    info_cols = [col.value for col in conf["info_cols"]]
    return pd.DataFrame(index=index, columns=["y_true", *model_cols, *info_cols])


def build_model_name(model_params: dict) -> str:
    name = model_params["model_type"].value
    if "wrc" in model_params:
        name += "_wrc"
    elif "worc" in model_params:
        name += "_worc"
    return (
        name
        + "_"
        + "_".join(
            [
                f"{k}_{v}"
                for k, v in model_params.items()
                if k not in ["model_type", "wrc", "worc"]
            ]
        )
    )


def initialize_results_df(conf: dict, y_true: pd.Series) -> pd.DataFrame:
    model_names = [build_model_name(model_params) for model_params in conf["models"]]
    results_df = pd.DataFrame(index=["y_true", *model_names], columns=y_true.index)
    results_df.loc["y_true"] = y_true
    return results_df


def results_df_to_pred_lists(results_df: pd.DataFrame) -> dict:
    return {model: results_df.loc[model].to_list() for model in results_df.index}


def evaluate_predictions_and_save(pred_df: pd.DataFrame, conf: dict) -> None:
    for model_params in conf["models"]:
        model_name = build_model_name(model_params)
        print(f"Start evaluating {model_name} ({datetime.datetime.now()})")  # noqa: DTZ005
        model_eval_df = _evaluate_model(pred_df, conf, model_name)
        utils.save_evaluation_df(
            model_eval_df, conf["saving_file"], model_name, save_idx=True
        )


def _evaluate_model(pred_df: pd.DataFrame, conf: dict, model_name: str) -> pd.DataFrame:
    def compute_regression_evaluation_scores(
        y_true: pd.Series, y_pred: pd.Series, metrics: list[config.RegMetrics]
    ) -> float:
        # if y_true and y_pred do not have the same length, the difference between
        # non-existing indices is NaN which becomes 0 when the sum is computed
        # therefore, we do not have to restict y_pred to the values existing in y_true
        metric_values = {
            config.RegMetrics.MAE: mean_absolute_error(y_true, y_pred),
            config.RegMetrics.MSE: mean_squared_error(y_true, y_pred),
        }
        return [np.round(metric_values[met], config.ROUND_DECIMALS) for met in metrics]

    def compute_classification_evaluation_scores(
        y_true: pd.Series, y_pred: pd.Series, metrics: list[config.ClassMetrics]
    ) -> float:
        # conservative choice for zero_division:
        # precision: if no positive values are predicted, we assume that the model
        # performs bad on them
        metric_values = {
            config.ClassMetrics.ACC: accuracy_score(y_true, y_pred),
            config.ClassMetrics.F1: f1_score(y_true, y_pred, zero_division=0),
            config.ClassMetrics.PREC: precision_score(y_true, y_pred, zero_division=0),
            config.ClassMetrics.REC: recall_score(y_true, y_pred, zero_division=0),
        }
        return [np.round(metric_values[met], config.ROUND_DECIMALS) for met in metrics]

    # get model predictions
    model_df = (
        pred_df[[col.value for col in conf["info_cols"]] + ["y_true", model_name]]
        .rename(columns={model_name: "y_pred"})
        .dropna(subset=["y_pred"])
    )

    # regression metrics
    reg_met = [met.value for met in conf["reg_metrics"]]
    model_df[reg_met] = model_df.apply(
        lambda row: compute_regression_evaluation_scores(
            row["y_true"], row["y_pred"], metrics=conf["reg_metrics"]
        ),
        axis=1,
    ).to_list()

    # classification metrics
    lim_vals = [lim for lim in conf["lim"] if isinstance(lim, float)]
    for lim in lim_vals:
        lim_str = int(lim * 100)
        class_met = [f"{met.value}_lim_{lim_str}" for met in conf["class_metrics"]]
        model_df[class_met] = model_df.apply(
            lambda row: compute_classification_evaluation_scores(
                row["y_true"],
                (pd.Series(row["y_pred"]) > lim).astype(int),  # noqa: B023
                metrics=conf["class_metrics"],
            ),
            axis=1,
        ).to_list()

    # dynamic lim
    if config.LimType.DYNAMIC in conf["lim"]:
        class_met = [f"{met.value}_lim_dynamic" for met in conf["class_metrics"]]
        model_df[class_met] = model_df.apply(
            lambda row: compute_classification_evaluation_scores(
                row["y_true"],
                (pd.Series(row["y_pred"]) > row["mean_iu_perf"]).astype(int),
                metrics=conf["class_metrics"],
            ),
            axis=1,
        ).to_list()

    return model_df
