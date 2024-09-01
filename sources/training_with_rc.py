import sys
from itertools import chain
from pathlib import Path

import pandas as pd

sys.path.append(Path("../../sources").resolve())

import training_general
from collaborative_filtering import cf
from config import RecMethod
from item_response_theory import irt
from knowledge_tracing import kt


def get_idx_pred_df(rc_dict: dict) -> pd.MultiIndex:
    # get all possible combinations of class, test sequence and reference class
    idx_list = []
    for cid, cid_dict in rc_dict.items():
        for ts, ts_dict in cid_dict["details"].items():
            if ts_dict["reference_classes"]:
                for stud in cid_dict["students"]:
                    idx_list.extend(
                        [(cid, ts, stud, rc) for rc in ts_dict["reference_classes"]]
                    )
    return pd.MultiIndex.from_tuples(
        idx_list, names=["class_id", "ut_id", "student_id", "ref_class"]
    )


def _get_idx_pred_df_cid(
    ts_cid: list[str], studs_cid: list[str], rc_cid: list[str]
) -> pd.MultiIndex:
    # create predictions dataframe for whole class (predictions df for each ts is stored)
    return pd.MultiIndex.from_product(
        [ts_cid, studs_cid, rc_cid], names=["ut_id", "student_id", "ref_class"]
    )


def _get_idx_pred_df_ts(
    students: list[str], reference_classes: list[str]
) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [students, reference_classes], names=["student_id", "ref_class"]
    )


def perform_predictions_for_cid(
    conf: dict,
    cid: str,
    cid_dict: dict,
    df: pd.DataFrame,
    ass_seq: pd.DataFrame,
    stud_per_class: pd.Series,
) -> pd.DataFrame:
    # get students, ts and rc for cid
    studs_cid, ts_cid, rc_cid = _get_cid_variables(cid_dict)
    # get all students in reference classes
    studs_rc = set(chain.from_iterable(stud_per_class.loc[rc_cid]))

    # filter to data for relevant students
    df_filt, ass_seq_filt = _filter_data(
        df.loc[list(set(studs_cid) | studs_rc)],
        ass_seq.loc[[cid, *rc_cid]],
        cid_dict["problems"],
    )

    # create predictions dataframe for whole class (predictions df for each ts is stored)
    pred_df_cid = training_general.initialize_pred_df(
        index=_get_idx_pred_df_cid(ts_cid, studs_cid, rc_cid), conf=conf
    )

    # perform for each test sequence with reference classes
    for ts in ts_cid:
        # for ts in ["1CRN82227G"]:
        # print("Test sequence", ts)
        ts_dict = cid_dict["details"][ts]

        pred_df_cid.loc[ts] = (
            _perform_predictions_for_ts(
                conf, cid, ts_dict, ass_seq_filt.loc[ts], df_filt
            )
            .reindex(pred_df_cid.loc[ts].index)
            .to_numpy()
        )

    return pred_df_cid


def _get_cid_variables(cid_dict: dict) -> tuple[list[str], list[str], list[str]]:
    # get all possible test sequence
    ts_cid = [
        ts
        for ts, ts_dict in cid_dict["details"].items()
        if ts_dict["reference_classes"]
    ]
    return cid_dict["students"], ts_cid, cid_dict["ref_classes_complete"]


def _filter_data(
    df_cid: pd.DataFrame, ass_seq_cid: pd.DataFrame, cid_probs: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # filter to problems that students in cid completed (UT+IU) as other problems are irrelevant
    df_filt = (
        df_cid.loc[df_cid["problem_id"].isin(cid_probs)]
        .reset_index()
        .set_index("assignment_log_id")
    )

    # reindex
    ass_seq_filt = ass_seq_cid.reset_index().set_index(["ut_seq", "ut_class"])
    # restricting df_filt to problems might have removed assignments
    # if their problems do not overlap with the cid problems
    ass_df = set(df_filt.index)
    ass_seq_filt = ass_seq_filt.loc[
        (ass_seq_filt["ut_ass"].isin(ass_df)) & (ass_seq_filt["iu_ass"].isin(ass_df))
    ].copy()

    return df_filt, ass_seq_filt


def _perform_predictions_for_ts(
    conf: dict,
    cid: str,
    ts_dict: dict[list[str], list[str]],
    ass_seq_filt_ts: pd.DataFrame,
    df_filt: pd.DataFrame,
) -> pd.DataFrame:
    # get reference classes
    # reference classes might have been dropped when restricting to certain problems
    ref_classes = list(set(ts_dict["reference_classes"]) & set(ass_seq_filt_ts.index))
    # get relevant in unit sequence ids (that cid worked on and belong to ts)
    iu_seq = ts_dict["iu_seq"]

    # if no reference classes exist, return empty df
    if len(ref_classes) == 0:
        return training_general.initialize_pred_df(
            index=_get_idx_pred_df_ts([], []), conf=conf
        )

    # get data for test sequence
    iu_cid, iu_rc, ut_cid, ut_rc = _get_ts_data(
        cid, ref_classes, iu_seq, ass_seq_filt_ts, df_filt
    )

    # students that completed ts
    studs_cid_ts = ut_cid.index.unique()

    # create empty predictions dataframe for the ts
    pred_df_ts = training_general.initialize_pred_df(
        index=_get_idx_pred_df_ts(studs_cid_ts, ref_classes), conf=conf
    )

    # perform for each student in target class cid
    for stud in studs_cid_ts:
        # for stud in ["1IB0KDMKQM"]:
        # print(f"------------- Student {stud} --------------")

        # restrict to stud data and only keep one response per problem that is the minimum (remove duplicated problems and keep worse response)
        iu_stud = (
            iu_cid.loc[[stud]]
            .sort_values("first_answer")
            .drop_duplicates(subset=["problem_id"], keep="first")
            .set_index("problem_id")
        )
        ut_stud = ut_cid.loc[[stud]].set_index("problem_id")

        pred_df_ts.loc[stud] = (
            _perform_predictions_for_stud(
                conf, ut_stud, iu_stud, ut_rc, iu_rc, ref_classes
            )
            .reindex(pred_df_ts.loc[stud].index)
            .to_numpy()
        )

        # info columns
        if "info_cols" in conf["eval_groups"]:
            info_values_const = {
                "num_ut_probs": len(ut_stud),
                "num_iu_probs": iu_stud.index.nunique(),
                "mean_ut_perf": ut_stud["first_answer"].mean(),
                "mean_iu_perf": iu_stud["first_answer"].mean(),
                # "max_num_iu_probs_rc": np.nan
            }
            pred_df_ts.loc[
                stud,
                [
                    col.value
                    for col in conf["info_cols"]
                    if col.value in info_values_const
                ],
            ] = [
                info_values_const[col.value]
                for col in conf["info_cols"]
                if col.value in info_values_const
            ]

    return pred_df_ts


def _get_ts_data(
    cid: str,
    ref_classes: list[str],
    iu_seq: list[str],
    ass_seq_filt_ts: pd.DataFrame,
    df_filt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # get ut assignments and iu assignments for cid and ref_classes
    # filtern auf ts und classes
    ass_seq_ts = ass_seq_filt_ts.loc[[cid, *ref_classes]].copy()
    # filtern auf iu_seq
    ass_seq_ts = ass_seq_ts.loc[ass_seq_ts["iu_seq"].isin(iu_seq)].copy()

    # get ut assignment data
    df_ut = (
        df_filt.loc[ass_seq_ts["ut_ass"].unique()]
        .rename(columns={"class_id": "ut_class"})
        .set_index(["ut_class", "student_id"])
    )

    # get iu assignment data
    # sort such that if multiple answers exist, the last one given can be chosen
    df_iu = (
        df_filt.loc[ass_seq_ts["iu_ass"].unique()]
        .merge(
            ass_seq_ts.reset_index().set_index("iu_ass")["ut_class"],
            how="left",
            left_index=True,
            right_index=True,
        )
        .sort_values(by="timestamp", ascending=True)
        .set_index(["ut_class", "student_id"])
    )

    # separate matrices to cid and reference classes
    return df_iu.loc[cid], df_iu.drop(index=cid), df_ut.loc[cid], df_ut.drop(index=cid)


def _perform_predictions_for_stud(
    conf: dict,
    ut_stud: pd.Series,
    iu_stud: pd.Series,
    ut_rc: pd.DataFrame,
    iu_rc: pd.DataFrame,
    ref_classes: list[str],
) -> pd.DataFrame:
    # initialize predictions df for student
    pred_df_stud = training_general.initialize_pred_df(ref_classes, conf)

    # get mean iu performance
    iu_mean = iu_stud["first_answer"].mean()

    # initialize results df
    res_df = training_general.initialize_results_df(conf, ut_stud["first_answer"])

    for rc in ref_classes:
        # for rc in ["D3EXBNF3N"]:
        # print("Ref class", rc)
        results_df = res_df.copy()

        ut_rc_rc = ut_rc.loc[rc].copy()
        iu_rc_rc = iu_rc.loc[rc].copy()
        iu_rc_rc = iu_rc_rc[iu_rc_rc["problem_id"].isin(iu_stud.index)]

        if len(iu_rc_rc) > 0:
            for model_params in conf["models"]:
                model_name = training_general.build_model_name(model_params)
                # print(model_name)
                (
                    results_df.loc[model_name],
                    info_values_diff,
                ) = _perform_predictions_for_rc(
                    conf["method"], model_params, ut_stud, iu_stud, ut_rc_rc, iu_rc_rc
                )

            results_df = results_df.dropna(how="all")

            # replace nan values (no prediction possible) by mean performance of student during
            # in unit assignments
            results_df[pd.isna(results_df)] = iu_mean

            if len(results_df) > 1:
                # store predictions as lists in pred_df_stud
                pred_df_stud.loc[rc] = training_general.results_df_to_pred_lists(
                    results_df
                )

                if "info_cols" in conf["eval_groups"]:
                    for col in conf["info_cols"]:
                        if col.value in info_values_diff:
                            pred_df_stud.loc[rc, col.value] = info_values_diff[
                                col.value
                            ]

    return pred_df_stud


def _perform_predictions_for_rc(
    method: RecMethod,
    model_params: dict,
    ut_stud: pd.Series,
    iu_stud: pd.Series,
    ut_rc: pd.DataFrame,
    iu_rc: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if method == RecMethod.CF:
        return cf.perform_collaborative_filtering(
            ut_stud, iu_stud, ut_rc, iu_rc, model_params
        )
    if method == RecMethod.IRT:
        return irt.perform_item_response_theory_wrc(
            ut_stud, iu_stud, ut_rc, iu_rc, model_params
        )
    if method == RecMethod.KT:
        return kt.perform_knowledge_tracing_wrc(
            ut_stud, iu_stud, ut_rc, iu_rc, model_params
        )
    raise NotImplementedError
