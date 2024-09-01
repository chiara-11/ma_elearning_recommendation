import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(Path("../../sources").resolve())

import config
from config import CFModelType


def check_conf_cf(model_list: list[dict], *, with_rc: bool) -> None:
    """
    Conf dictionary for collaborative filtering needs a list of dictionaries, where each
    specifies a model.

    The method does not work without reference classes.
    """
    # check parameter with_ref_class
    assert with_rc is True, "Collaborative filtering only works with reference classes."

    # check if there are models in the model_list
    assert model_list, (
        "Collaborative filtering needs at least one model specification"
        "in 'models' but none are specified."
    )

    for model in model_list:
        # check model specification
        _check_model_dict(model)


def _check_model_dict(model_params: dict) -> None:
    """Each model dictionary needs a parameter model_type of type CFModelType and model
    parameters related to that."""
    # check model type
    model_type = model_params.get("model_type")
    assert (
        model_type
    ), "The model specification needs a parameter model_type of type CFModelType."
    assert isinstance(
        model_type, CFModelType
    ), "The parameter model_type has to be of type CFModelType."

    if model_type == CFModelType.KNN:
        # check similarity metric
        sim_metric = model_params.get("sim")
        assert sim_metric, "A similarity metric should be specified."
        assert (
            sim_metric in config.CF_KNN_SIM
        ), f"The similarity metric should be one of {config.CF_KNN_SIM}."

        # check if similarities should be weighted
        weight_method = model_params.get("weight")
        assert (
            not weight_method or weight_method in config.CF_KNN_SIM_WEIGHT
        ), f"If a similarity weighting method is specified it should be one of {config.CF_KNN_SIM_WEIGHT}."
        if weight_method == "significance":
            min_num = model_params.get("T")
            assert (
                min_num
            ), "The parameter T should be specified for the significance weighting."
            assert (
                min_num > 0
            ), f"Parameter T should be an integer greater than 0 but is {min_num}."

        # check prediction method
        pred_method = model_params.get("pred")
        assert pred_method, "A prediction method should be specified."
        assert (
            pred_method in config.CF_KNN_PRED
        ), f"The prediction method should be one of {config.CF_KNN_PRED}."

        # check parameter k
        k = model_params.get("k")
        assert k, "There is no model parameter k specified."
        assert isinstance(k, int), "Model parameter k has to be of type int."


def perform_collaborative_filtering(
    ut_stud: pd.DataFrame,
    iu_stud: pd.DataFrame,
    ut_rc: pd.DataFrame,
    iu_rc: pd.DataFrame,
    model_params: dict,
) -> tuple[pd.Series, dict]:
    # prepare data
    iu_stud = iu_stud["first_answer"]
    ut_stud = ut_stud["first_answer"]

    iu_rc, ut_rc = _get_student_problem_matrices(iu_rc, ut_rc)
    iu_rc = iu_rc.reindex(columns=iu_stud.index)
    ut_rc = ut_rc.reindex(columns=ut_stud.index)

    model_type = model_params["model_type"]
    if model_type == CFModelType.KNN:
        y_pred = _perform_user_based_cf(iu_stud, ut_rc, iu_rc, model_params)
    elif model_type == CFModelType.KNN_ITEM:
        y_pred = _perform_item_based_cf(
            iu_stud, ut_rc, iu_rc, ut_stud.index, model_params
        )
    else:
        raise NotImplementedError

    info_values_diff = {
        "num_stud_rc": len(iu_rc),
        "max_num_iu_probs_rc": (~pd.isna(iu_rc)).sum(axis=1).max(),
        "mean_iu_perf_rc": iu_rc.mean(axis=1).mean(),
        "mean_ut_perf_rc": ut_rc.mean(axis=1).mean(),
    }

    return y_pred, info_values_diff


def _get_student_problem_matrices(
    iu_rc: pd.DataFrame, ut_rc: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # get ut assignment data and convert to student-item matrix for ut assignments
    ut_rc = ut_rc.pivot_table(
        index=["student_id"], columns="problem_id", values="first_answer", aggfunc="min"
    )

    # get iu assignment data and convert to student-item matrix for iu assignments
    # sort such that if multiple answers exist, the last one given can be chosen
    iu_rc = iu_rc.pivot_table(
        index=["student_id"],
        columns="problem_id",
        values="first_answer",
        aggfunc="last",
    )

    # separate matrices to cid and reference classes
    return iu_rc, ut_rc


def _perform_user_based_cf(
    iu_stud: pd.DataFrame, ut_rc: pd.DataFrame, iu_rc: pd.DataFrame, model_params: dict
) -> pd.Series:
    # compute similarities with every student in reference class
    sim_method = model_params["sim"]
    similarities = _compute_similarities(sim_method, iu_stud, iu_rc)

    weight_method = model_params.get("weight")
    if weight_method == "significance":
        # apply significance weighting (checks for number of common exercises)
        similarities = _apply_significance_weighting(
            model_params["T"], similarities, iu_rc
        )

    # merge number of completed ut problems per student to similarities
    similarities = (
        similarities.rename("similarity")
        .to_frame()
        .merge(
            (~ut_rc.isna()).sum(axis=1).rename("num_ut"),
            how="left",
            left_index=True,
            right_index=True,
        )
        .sort_values(by=["num_ut", "similarity"], ascending=[False, False])
    )

    # perform predictions for reference class
    return _perform_predictions_user_based(
        similarities, ut_rc, iu_rc, iu_stud, model_params
    )


def _perform_item_based_cf(
    iu_stud: pd.Series,
    ut_rc: pd.DataFrame,
    iu_rc: pd.DataFrame,
    ut_probs: pd.Series,
    model_params: dict,
) -> pd.Series:
    # transpose rc data to achieve item-student-matrices
    iu_rc = iu_rc.transpose()
    ut_rc = ut_rc.transpose()

    # initialize return series
    y_pred = pd.Series(0.0, index=ut_probs)
    # perform predictions for each ut problem individually
    for prob in ut_probs:
        # restrict iu rc data to students (columns) that are part of ut_rc_prob
        ut_rc_prob = ut_rc.loc[prob].dropna()
        iu_rc_prob = iu_rc[
            [col for col in ut_rc_prob.index if col in iu_rc.columns]
        ].copy()

        # iu_rc is already limited to iu problems that have been completed by stud,
        # hence no problems have to be removed
        # compute similarities with every iu problem
        sim_method = model_params["sim"]
        similarities = _compute_similarities(sim_method, ut_rc_prob, iu_rc_prob)

        weight_method = model_params.get("weight")
        if weight_method == "significance":
            # apply significance weighting (checks for number of common students)
            similarities = _apply_significance_weighting(
                model_params["T"], similarities, iu_rc_prob
            )

        # sort similarities in descending order
        similarities = similarities.sort_values(ascending=False)

        # perform predictions for ut problem
        y_pred[prob] = _perform_predictions_item_based(
            similarities, ut_rc_prob, iu_rc_prob, iu_stud, model_params
        )

    return y_pred


def _compute_similarities(
    method: str, iu_stud: pd.Series, iu_rc: pd.DataFrame
) -> pd.Series:
    """
    The similarities for user-based CF (item-based CF) between the main student (main
    problem) and every student in the reference class (every iu problem) are computed.

    In case of item-based CF, iu_stud is ut_rc_prob.
    """
    if method == "manhattan":
        return pd.Series(
            [1 - _compute_manhattan_dist(iu_stud, row) for _, row in iu_rc.iterrows()],
            index=iu_rc.index,
        )
    raise NotImplementedError


def _compute_manhattan_dist(main_stud: pd.Series, ref_stud: pd.Series) -> float:
    """
    Function computes normalized manhattan distance.

    User-based CF:
    Only problems that are completed by both students are considered, that is, problems
    only completed by the main student are dropped.

    Item-based CF:
    For item-based CF, main_stud should be called main_ut_prob and ref_stud should be
    called iu_prob.
    Only students that worked on both problems are considered, that is, students that
    only worked on the main problem are dropped.
    """
    num_probs = len(ref_stud.dropna())
    if num_probs == 0:
        return 1
    return np.abs(main_stud - ref_stud).sum() / num_probs


def _apply_significance_weighting(
    min_num: int, sim: pd.Series, iu_rc: pd.DataFrame
) -> pd.Series:
    """
    Apply the significance weighting and return the adjusted similarities.

    Get the size of the base on which the similarity was computed. Then, take the
    minimum of this value and the specified min_num and divide the value by min_num.
    Those are the significance weights with which the similarities are multiplied.

    Size of the similarity base:
    User-based: For each rc student, get the number of iu problems completed by this
    student and by the main student.
    Item-based: For each iu problem, get the number of rc students that worked on this
    problem and on the main ut problem.

    iu_rc is already restricted to iu problems of main student (students for main
    problem), therefore it is sufficient to check the number of problems of the other
    students (the number of students of the iu problems).
    """
    weights = np.minimum((~iu_rc.isna()).sum(axis=1), min_num) / min_num
    return sim * weights


def _perform_predictions_user_based(
    sim: pd.Series,
    ut_rc: pd.DataFrame,
    iu_rc: pd.DataFrame,
    iu_stud: pd.Series,
    model_params: dict,
) -> pd.Series:
    # get k most similar reference students
    # remove students with similarity 0 (they don't help for predictions and can be on
    # top due to sorting by number of equal problems)
    sim = sim.loc[sim["similarity"] != 0].copy()
    if len(sim) == 0:
        return np.nan
    # take students with most completed ut problems and then with highest similarity
    # score
    top_k_sim = sim.iloc[: model_params["k"]]["similarity"]

    # compute prediction for all problems using most similar students
    pred_method = model_params["pred"]
    if pred_method == "weightavg":
        return _compute_predictions_weighted_average(
            ut_rc.loc[top_k_sim.index], top_k_sim
        ).round(config.ROUND_DECIMALS)
    if pred_method == "resnick":
        return _compute_predictions_resnick(
            ut_rc.loc[top_k_sim.index],
            top_k_sim,
            iu_rc.loc[top_k_sim.index].mean(axis=1),
            iu_stud.mean(),
        ).round(config.ROUND_DECIMALS)
    raise NotImplementedError


def _perform_predictions_item_based(
    sim: pd.Series,
    ut_rc_prob: pd.DataFrame,
    iu_rc_prob: pd.DataFrame,
    iu_stud: pd.Series,
    model_params: dict,
) -> float:
    # get k most similar iu exercises
    # remove exercises with similarity 0 (they don't help for predictions)
    sim = sim[sim != 0]
    if len(sim) == 0:
        # if no neighbors can be chosen, use the mean of the ut problem
        return ut_rc_prob.mean()
    # take iu exercises with highest similarity score
    top_k_sim = sim[: model_params["k"]]

    # compute prediction for ut problem using most similar iu problems
    pred_method = model_params["pred"]
    if pred_method == "weightavg":
        return _compute_predictions_weighted_average(
            iu_stud[top_k_sim.index], top_k_sim
        ).round(config.ROUND_DECIMALS)
    if pred_method == "resnick":
        return _compute_predictions_resnick(
            iu_stud[top_k_sim.index],
            top_k_sim,
            iu_rc_prob.loc[top_k_sim.index].mean(axis=1),
            ut_rc_prob.mean(),
        ).round(config.ROUND_DECIMALS)
    raise NotImplementedError


def _compute_predictions_weighted_average(
    ut_matrix_top_k: pd.DataFrame | pd.Series, sim: pd.Series
) -> pd.Series:
    """
    Prediction method calculates the weighted mean.

    For each problem (only one in case of item-based), proceed as follows for user-based
    CF (item-based CF): For each similar student (problem), multiply answer value by
    similarity. Then, sum over all similar students (problem) and divide by the absolute
    sum of similarities (~weighted mean). In case of item-based CF, ut_matrix_top_k is
    iu_stud_top_k.
    """
    return ut_matrix_top_k.multiply(sim, axis=0).sum(axis=0) / sim.abs().sum()


def _compute_predictions_resnick(
    ut_matrix_top_k: pd.DataFrame | pd.Series,
    sim: pd.Series,
    mean_top_k: pd.Series,
    mean_stud: float,
) -> pd.Series:
    """
    Prediction method uses Resnick's formula.

    For each problem (only one in case of item-based), proceed as follows for user-based
    CF (item-based CF): For each similar student (problem), multiply difference between
    answer value and mean value of student's iu data (of problem's rc data) by
    similarity. Then, sum over all similar students (problems) and divide by sum of
    similarities (~weighted mean). Finally, add this value to the mean of main student's
    iu data (of main problem's rc data). In case of item-based CF, ut_matrix_top_k is
    iu_stud_top_k and mean_stud is mean_ut_prob.
    """
    return (
        mean_stud
        + (
            ut_matrix_top_k.sub(mean_top_k, axis=0).multiply(sim, axis=0).sum(axis=0)
            / sim.abs().sum()
        )
    ).clip(0, 1)
