import sys
from pathlib import Path

import pandas as pd

sys.path.append(Path("../../sources").resolve())

import training_general
from content_based_recommendation import cb
from config import RecMethod
from item_response_theory import irt
from knowledge_tracing import kt


def get_idx_pred_df(rc_dict: dict) -> pd.DataFrame:
    # get all possible combinations of class and test sequence
    idx_list = []
    for cid, cid_dict in rc_dict.items():
        for ts, ts_dict in cid_dict["details"].items():
            if ts_dict["reference_classes"]:
                idx_list.extend([(cid, ts, stud) for stud in cid_dict["students"]])
    return pd.MultiIndex.from_tuples(
        idx_list, names=["class_id", "ut_id", "student_id"]
    )


def _get_idx_pred_df_cid(ts_cid: list[str], studs_cid: list[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [ts_cid, studs_cid], names=["ut_id", "student_id"]
    )


def _get_idx_pred_df_ts(studs: list[str]) -> pd.Index:
    return pd.Index(studs, name="student_id")


def perform_predictions_for_cid(
    conf: dict, cid: str, cid_dict: dict, df: pd.DataFrame, ass_seq: pd.DataFrame
) -> pd.DataFrame:
    # get students, ts and rc for cid
    studs_cid, ts_cid = _get_cid_variables(cid_dict)

    df_filt, ass_seq_filt = _filter_data(df.loc[studs_cid], ass_seq.loc[[cid]])

    # create predictions dataframe for whole class (predictions df for each ts is stored)
    pred_df_cid = training_general.initialize_pred_df(
        index=_get_idx_pred_df_cid(ts_cid, studs_cid), conf=conf
    )

    # perform for each test sequence
    for ts in ts_cid:
        # for ts in ["CD76U7XEG"]:
        # print("Test sequence:", ts)

        pred_df_cid.loc[ts] = (
            _perform_predictions_for_ts(conf, ass_seq_filt.loc[[ts]], df_filt)
            .reindex(pred_df_cid.loc[ts].index)
            .to_numpy()
        )

    return pred_df_cid


def _get_cid_variables(cid_dict: dict) -> tuple[list[str], list[str]]:
    # get all possible test sequence
    ts_cid = [
        ts
        for ts, ts_dict in cid_dict["details"].items()
        if ts_dict["reference_classes"]
    ]
    return cid_dict["students"], ts_cid


def _filter_data(
    df_cid: pd.DataFrame, ass_seq_cid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Set assignment_log_id as index for df filtered to students in cid.

    Set ut_seq as index for ass_seq restricted to sequences where cid is ut_class.
    """
    return df_cid.reset_index().set_index(
        "assignment_log_id"
    ), ass_seq_cid.reset_index().set_index(["ut_seq"])


def _perform_predictions_for_ts(
    conf: dict, ass_seq_ts: pd.DataFrame, df_filt: pd.DataFrame
) -> pd.DataFrame:
    df_ut, df_iu = _get_ts_data(ass_seq_ts, df_filt)

    # students that completed ts
    studs_cid_ts = list(set(df_ut.index))

    # create empty predictions dataframe for the ts
    pred_df_ts = training_general.initialize_pred_df(
        index=_get_idx_pred_df_ts(studs_cid_ts), conf=conf
    )

    # perform for one student
    for stud in studs_cid_ts:
        # for stud in ["1IB0KDMKQM"]:
        # print(f"------------ Student {stud} -----------")
        df_iu_stud = df_iu.loc[[stud]].set_index("problem_id")
        df_ut_stud = df_ut.loc[[stud]].set_index("problem_id")

        # initialize results_df
        results_df = training_general.initialize_results_df(
            conf, df_ut_stud["first_answer"]
        )

        # perform predictions for each model
        for model_params in conf["models"]:
            model_name = training_general.build_model_name(model_params)
            # print(model_name)
            results_df.loc[model_name] = _perform_predictions_for_stud(
                conf["method"], model_params, df_ut_stud, df_iu_stud
            )

        results_df = results_df.dropna(how="all")

        if len(results_df) > 1:
            # store predictions as lists in pred_df_ts
            pred_df_ts.loc[stud] = training_general.results_df_to_pred_lists(results_df)

            # get info values
            if "info_cols" in conf["eval_groups"]:
                info_values = {
                    "num_ut_probs": len(df_ut_stud),
                    "num_iu_probs": len(df_iu_stud),
                    "mean_ut_perf": df_ut_stud["first_answer"].mean(),
                    "mean_iu_perf": df_iu_stud["first_answer"].mean(),
                }
                pred_df_ts.loc[stud, [col.value for col in conf["info_cols"]]] = [
                    info_values[col.value] for col in conf["info_cols"]
                ]

    return pred_df_ts


def _get_ts_data(
    ass_seq_ts: pd.DataFrame, df_filt: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # get ut assignment data
    df_ut = df_filt.loc[ass_seq_ts["ut_ass"].unique()].set_index("student_id")

    # get iu assignment data
    df_iu = df_filt.loc[ass_seq_ts["iu_ass"].unique()].set_index("student_id")

    return df_ut, df_iu


def _perform_predictions_for_stud(
    method: RecMethod, model_params: dict, ut_stud: pd.DataFrame, iu_stud: pd.DataFrame
) -> pd.DataFrame:
    if method == RecMethod.CB:
        return cb.perform_content_based_recommendation(model_params, ut_stud, iu_stud)
    if method == RecMethod.IRT:
        return irt.perform_item_response_theory_worc(model_params, ut_stud, iu_stud)
    if method == RecMethod.KT:
        return kt.perform_knowledge_tracing_worc(model_params, ut_stud, iu_stud)
    raise NotImplementedError
