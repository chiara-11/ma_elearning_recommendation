import sys
from pathlib import Path

import pandas as pd

sys.path.append(Path("../../sources").resolve())

import config
import training_general
import utils


def get_total_results(
    models: dict[str, list[dict]],
    idx: pd.DataFrame | None,
    metrics: list[str],
    group_cols: list[str] | None = None,
    *,
    wrc: bool = False,
    print_info: bool = False,
) -> pd.DataFrame:
    df_list = []
    for method, model_list in models.items():
        eval_df = _get_comparison_df(
            method,
            model_list,
            idx,
            metrics,
            group_cols=group_cols,
            wrc=wrc,
            print_info=print_info,
        )
        eval_df["method"] = method
        df_list.append(eval_df)
    # read baseline df
    baseline = _read_worc_eval("baseline", "baseline", metrics)
    baseline = baseline.mean().round(config.ROUND_DECIMALS).to_frame().transpose()
    baseline.index = ["baseline"]
    baseline["method"] = "baseline"

    df_list.append(baseline)
    return (
        pd.concat(df_list)
        .reset_index(names="model_name")
        .set_index(["method", "model_name"])
    )


def _get_comparison_df(
    folder: str,
    model_list: list[dict],
    idx: pd.DataFrame | None,
    metrics: list[str],
    group_cols: list[str] | None,
    *,
    wrc: bool = False,
    print_info: bool = False,
) -> pd.DataFrame:
    model_names = [training_general.build_model_name(model) for model in model_list]
    df = pd.DataFrame(index=model_names, columns=metrics, data=0.0)

    for model in model_names:
        if wrc:
            eval_df = _read_wrc_eval(folder, model, idx, metrics, print_info=print_info)
        else:
            eval_df = _read_worc_eval(folder, model, metrics, print_info=print_info)
        if group_cols:
            eval_df = eval_df.groupby(group_cols).mean()
        df.loc[model] = eval_df.mean().round(config.ROUND_DECIMALS)
    return df


def _read_wrc_eval(
    folder: str,
    model_name: str,
    idx: pd.DataFrame,
    metrics: list[str],
    *,
    print_info: bool = False,
) -> pd.DataFrame:
    part_len = [354080, 237190, 389061, 194170]
    eval_df = pd.DataFrame()
    for num in [1, 2, 3, 4]:
        eval_part = utils.read_evaluation_df(
            folder, f"{model_name}_part{num}", latest=True, print_filename=print_info
        )
        if len(eval_part) != part_len[num - 1] and print_info:
            print(
                f"Part {num} of model {model_name} has length {len(eval_part)}, {part_len[num-1] - len(eval_part)} records missing"
            )
        eval_df = pd.concat([eval_df, eval_part])
    # restrict to specified combinations
    eval_df = idx.merge(
        eval_df, how="left", on=["class_id", "ut_id", "ref_class"]
    ).set_index(["class_id", "ut_id", "student_id"])
    if len(eval_df) != 30580 and print_info:
        print(
            f"Model {model_name} has length {len(eval_df)}, {30580 - len(eval_df)} records missing"
        )
    return eval_df[metrics]


def _read_worc_eval(
    folder: str, filename: str, metrics: list[str], *, print_info: bool = False
) -> pd.DataFrame:
    eval_df = (
        utils.read_evaluation_df(
            folder, filename, latest=True, print_filename=print_info
        )
        .set_index(["class_id", "ut_id", "student_id"])
        .round(config.ROUND_DECIMALS)
    )
    if len(eval_df) != 30580 and print_info:
        print(
            f"Model {filename} has length {len(eval_df)}, {30580 - len(eval_df)} records missing"
        )
    return eval_df[metrics]


def get_mean_df(model_list: list[tuple[str]]) -> pd.DataFrame:
    multi_index = pd.MultiIndex.from_tuples(model_list, names=["method", "model_name"])
    mean_df = pd.DataFrame(
        index=multi_index,
        columns=["Accuracy_all", "F1_all", "Accuracy_corr", "F1_corr"],
    )

    for model in model_list:
        mean_comp, mean_corr = _get_mean_with_and_without_correction(
            *model, ["acc_lim_50", "f1_lim_50"]
        )
        mean_df.loc[model] = list(mean_comp) + list(mean_corr)
    return mean_df


def _get_mean_with_and_without_correction(
    folder: str, filename: str, metrics: list[str]
) -> tuple[pd.Series, pd.Series]:
    eval_df = (
        utils.read_evaluation_df(folder, filename, latest=True)
        .set_index(["class_id", "ut_id", "student_id"])
        .round(config.ROUND_DECIMALS)[["mean_ut_perf", *metrics]]
    )
    return eval_df[metrics].mean().round(config.ROUND_DECIMALS), eval_df.loc[
        eval_df["mean_ut_perf"] != 0, metrics
    ].mean().round(config.ROUND_DECIMALS)


def get_model_eval_wrc(folder: str, model_name: str) -> pd.DataFrame:
    eval_complete = pd.DataFrame()
    for part in [1, 2, 3, 4]:
        eval_part = utils.read_evaluation_df(
            folder, f"{model_name}_part{part}", latest=True
        )
        eval_complete = pd.concat([eval_complete, eval_part])
    return eval_complete.set_index(["class_id", "ut_id", "student_id", "ref_class"])


def restrict_eval_to_rc(eval_df: pd.DataFrame, idx_df: pd.DataFrame) -> pd.DataFrame:
    return idx_df.merge(
        eval_df.reset_index(), how="left", on=["class_id", "ut_id", "ref_class"]
    ).set_index(["class_id", "ut_id", "student_id"])
