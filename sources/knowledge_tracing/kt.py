import sys
from pathlib import Path

import pandas as pd

sys.path.append(Path("../../sources").resolve())

import utils
from config import KTModelType
from knowledge_tracing import bkt
from knowledge_tracing import lkt


def check_conf_kt(model_list: list[dict], *, with_rc: bool) -> None:
    """
    Conf dictionary for knowledge tracing needs a list of dictionaries, where each
    specifies a model.

    At the moment, the method is only implemented for recommendations without reference
    classes.
    """
    # check if there are models in the model_list
    assert (
        model_list
    ), "KT needs at least one model specification in 'models' but none are specified."

    for model in model_list:
        # check model specification
        _check_model_dict(model, with_rc=with_rc)


def _check_model_dict(model_params: dict, *, with_rc: bool) -> None:
    """Each model dictionary needs a parameter model_type of type KTModelType and model
    parameters related to that."""
    # check model type
    model_type = model_params.get("model_type")
    assert (
        model_type
    ), "The model specification needs a parameter model_type of type KTModelType."
    assert isinstance(
        model_type, KTModelType
    ), "The parameter model_type has to be of type KTModelType."

    # check with rc specification
    if with_rc:
        wrc = model_params.get("wrc")
        assert wrc is True, "There has to be a parameter 'wrc' with value True."
    else:
        worc = model_params.get("worc")
        assert worc is True, "There has to be a parameter 'worc' with value True."

    if model_type in [KTModelType.BKT, KTModelType.BKT_FORGET]:
        bkt.check_model_dict_bkt(model_params, with_rc=with_rc)
    if model_type == KTModelType.PFA:
        lkt.check_model_dict_pfa(model_params, with_rc=with_rc)


def prepare_df_for_kt(data: pd.DataFrame) -> pd.DataFrame:
    # read problem_details
    prob_det = utils.read_problem_details(second_save=False)

    # restrict prob_det to problems in data
    prob_det = prob_det.loc[list(set(data["problem_id"]))].copy()

    # map skill codes to first two levels and extract all levels
    prob_det = _map_skill_codes(prob_det)

    # merge problem_details to data
    data_cols = [
        "assignment_log_id",
        "problem_id",
        "timestamp",
        "first_answer",
        "sequence_id",
        "student_id",
        "class_id",
        "teacher_id",
        "unit_test",
    ]

    skill_cols = [
        "problem_skill_code_domain",
        "problem_skill_code_1",
        "problem_skill_code_2",
    ]
    data = data[data_cols].merge(
        prob_det[skill_cols], how="left", left_on="problem_id", right_index=True
    )

    # replace nan values in skill columns
    return _replace_skill_nan(data, skill_cols)


def _map_skill_codes(prob_det: pd.DataFrame) -> pd.DataFrame:
    # map skill codes to first two levels
    prob_det["problem_skill_code_domain"] = prob_det["problem_skill_code"].apply(
        lambda skill_code: ".".join(skill_code.split(".")[:2])
        if isinstance(skill_code, str)
        else skill_code
    )

    # extract levels of skill codes and store in one column each
    prob_det[[f"problem_skill_code_{i + 1}" for i in range(4)]] = (
        prob_det["problem_skill_code"]
        .apply(
            lambda skill_code: skill_code.split(".")
            if isinstance(skill_code, str)
            else [skill_code] * 4
        )
        .to_list()
    )

    return prob_det


def _replace_skill_nan(data: pd.DataFrame, skill_cols: list[str]) -> pd.DataFrame:
    # rows without skill code
    no_skill = data.loc[pd.isna(data["problem_skill_code_domain"])].copy()

    # number of sequences per problem
    num_seq = no_skill.groupby("problem_id")["sequence_id"].nunique()

    # read sequence details and restrict to relevant sequences
    seq_det = utils.read_sequence_details()
    seq_det = seq_det.loc[
        (seq_det["sequence_id"].isin(set(data["sequence_id"])))
        & (
            seq_det["sequence_folder_path_level_1"]
            == "EngageNY/Eureka Math (© by Great Minds®) *"
        )
    ].copy()

    # sequences including problems without skill code
    seq_det_ns = seq_det.loc[
        seq_det["sequence_id"].isin(no_skill["sequence_id"].unique())
    ].set_index("sequence_id")

    # sequences per level combination for sequences with problems without skill code
    level_cols = [
        "sequence_folder_path_level_2",
        "sequence_folder_path_level_3",
        "sequence_folder_path_level_4",
    ]
    levels_ns = seq_det_ns.groupby(level_cols).size().index
    seq_per_lev = seq_det.groupby(level_cols)["sequence_id"].unique().loc[levels_ns]

    # get most used skill code for level combinations
    level_to_skill = pd.DataFrame(index=seq_per_lev.index, columns=skill_cols)
    for idx in level_to_skill.index:
        if (
            len(
                skill_df := data.loc[
                    data["sequence_id"].isin(seq_per_lev[idx]), skill_cols
                ]
            )
            > 0
        ):
            level_to_skill.loc[idx] = (
                skill_df.value_counts().sort_values(ascending=False).index[0]
            )
    seq_det_ns = seq_det_ns.merge(
        level_to_skill, how="left", left_on=level_cols, right_index=True
    )

    # write skill codes to problem with only one sequence id
    prob_mask = data["problem_id"].isin(num_seq[num_seq == 1].index)
    for idx, row in seq_det_ns[skill_cols].iterrows():
        data.loc[prob_mask & (data["sequence_id"] == idx), skill_cols] = row.to_numpy()

    # rows with still no skill code (because they have two different sequences per
    # problem
    no_skill_mask = pd.isna(data["problem_skill_code_domain"])
    no_skill_2 = data.loc[no_skill_mask].copy()
    # all of them have the same two sequences which themselves have the same skill codes
    sid = no_skill_2.groupby("problem_id")["sequence_id"].unique().to_numpy()[0][0]
    data.loc[no_skill_mask, skill_cols] = seq_det_ns.loc[sid, skill_cols].to_numpy()

    return data


def perform_knowledge_tracing_worc(
    model_params: dict, ut_stud: pd.DataFrame, iu_stud: pd.DataFrame
) -> pd.Series:
    model_type = model_params["model_type"]
    if model_type in [KTModelType.BKT, KTModelType.BKT_FORGET]:
        return bkt.perform_bkt_worc(model_params, ut_stud, iu_stud)
    if model_type == KTModelType.PFA:
        return lkt.perform_pfa_worc(model_params, ut_stud, iu_stud)
    if model_type == KTModelType.LFA:
        return lkt.perform_lfa_worc(model_params, ut_stud, iu_stud)
    raise NotImplementedError


def perform_knowledge_tracing_wrc(
    ut_stud: pd.DataFrame,
    iu_stud: pd.DataFrame,
    ut_rc: pd.DataFrame,
    iu_rc: pd.DataFrame,
    model_params: dict,
) -> tuple[pd.Series, dict]:
    model_type = model_params["model_type"]
    if model_type in [KTModelType.BKT, KTModelType.BKT_FORGET]:
        return bkt.perform_bkt_wrc(ut_stud, iu_stud, ut_rc, iu_rc, model_params)
    if model_type == KTModelType.PFA:
        return lkt.perform_pfa_wrc(ut_stud, iu_stud, ut_rc, iu_rc, model_params)
    if model_type == KTModelType.LFA:
        return lkt.perform_lfa_wrc(ut_stud, iu_stud, ut_rc, iu_rc, model_params)
    raise NotImplementedError
