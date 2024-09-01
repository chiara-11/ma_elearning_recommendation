import sys
from pathlib import Path

import numpy as np
import pandas as pd
from py_irt.dataset import Dataset
from py_irt.models import OneParamLog
from py_irt.models import TwoParamLog

sys.path.append(Path("../../sources").resolve())

import config
from config import IRTModelType


EXPERT_VALUES = pd.read_csv(config.DATA_FOLDER / "expert_data_irt.csv").set_index(
    "problem_id"
)


def check_conf_irt(model_list: list[dict], *, with_rc: bool) -> None:
    """
    Conf dictionary for item response theory needs a list of dictionaries, where each
    specified a model.

    The method is implemented for recommendation with and without reference classes.
    """
    # check if there are models in the model_list
    assert (
        model_list
    ), "IRT needs at least one model specificatoin in 'models' but none are specified."

    for model in model_list:
        # check model specification
        _check_model_dict(model, with_rc=with_rc)


def _check_model_dict(model_params: dict, *, with_rc: bool) -> None:
    """Each model dictionary needs a parameter model_type of type IRTModelType, a method
    for determining the learner ability, a method for determining the item difficulties,
    and model parameters related to that."""
    # check model type
    model_type = model_params.get("model_type")
    assert (
        model_type
    ), "The model specification needs a parameter model_type of type IRTModelType."
    if with_rc:
        assert (
            model_type == IRTModelType.WRC
        ), "The parameter model_type has to be IRTModelType.WRC if reference classes are used."
    else:
        assert (
            model_type == IRTModelType.WORC
        ), "The parameter model_type has to be IRTModelType.WORC if no reference classes are used."

    # check ability method
    ability_method = model_params.get("ability")
    assert (
        ability_method
    ), "An ability method should be specified (parameter 'ability')."
    assert (
        ability_method in config.IRT_ABILITY_METHODS
    ), f"The parameter ability should be one of {config.IRT_ABILITY_METHODS}."

    # check ability interval
    ability_interval = model_params.get("ab_int")
    assert not ability_interval or isinstance(
        ability_interval, int
    ), "The parameter ab_int should be a positive integer."

    # check difficulty method
    difficulty_method = model_params.get("difficulty")
    assert (
        difficulty_method
    ), "A difficulty method should be specified (parameter 'difficulty')."
    if with_rc:
        assert (
            difficulty_method in config.IRT_DIFFICULTY_METHODS_WRC
        ), f"The parameter difficulty should be one of {config.IRT_DIFFICULTY_METHODS_WRC}."
    else:
        assert (
            difficulty_method in config.IRT_DIFFICULTY_METHODS_WORC
        ), f"The parameter difficulty should be one of {config.IRT_DIFFICULTY_METHODS_WORC}."

    # check difficulty interval
    difficulty_interval = model_params.get("diff_int")
    assert not difficulty_interval or isinstance(
        difficulty_interval, int
    ), "The parameter diff_int should be a positive integer."

    if (ability_method == "elo") or (difficulty_method == "elo"):
        assert (
            difficulty_method == "elo"
        ), "If the ability method is elo, the difficulty method has to be elo, too."
        assert (
            ability_method == "elo"
        ), "If the difficulty method is elo, the ability method has to be elo, too."
        elo_weight = model_params.get("W")
        assert elo_weight, "There has to be specified a weight for the elo method."
        assert isinstance(
            elo_weight, float
        ), "The elo weight should be a float (recommended between 0 and 1)."


def perform_item_response_theory_wrc(
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

    ability_stud, difficulties, discriminations = _get_parameters_wrc(
        model_params, iu_stud, iu_rc, ut_rc
    )

    # compute item characteristic curve
    y_pred = _compute_item_response_function(
        ability_stud, discriminations, difficulties, 0
    ).round(config.ROUND_DECIMALS)

    # get info column values
    info_values_diff = {
        "num_stud_rc": len(iu_rc),
        "max_num_iu_probs_rc": (~pd.isna(iu_rc)).sum(axis=1).max(),
        "mean_iu_perf_rc": iu_rc.mean(axis=1).mean(),
        "mean_ut_perf_rc": ut_rc.mean(axis=1).mean(),
    }

    return y_pred, info_values_diff


def perform_item_response_theory_worc(
    model_params: dict, ut_stud: pd.Series, iu_stud: pd.Series
) -> pd.Series:
    # get relevant data
    ut_stud = ut_stud["first_answer"]
    iu_stud = iu_stud["first_answer"]

    # compute learner ability
    ability_method = model_params["ability"]
    ability_stud = _compute_learner_ability(ability_method, iu_stud)
    if ab_bound := model_params.get("ab_int"):
        ability_stud = _map_to_interval(ab_bound, ability_stud)

    # compute item difficulty
    difficulty_method = model_params["difficulty"]
    difficulties = _compute_item_difficulty_worc(
        difficulty_method, ut_stud.index, model_params
    )
    if diff_bound := model_params.get("diff_int"):
        difficulties = _map_to_interval(diff_bound, difficulties)

    # compute item characteristic curve
    return _compute_item_response_function(ability_stud, 1, difficulties, 0).round(
        config.ROUND_DECIMALS
    )


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

    return iu_rc, ut_rc


def _get_parameters_wrc(
    model_params: dict, iu_stud: pd.Series, iu_rc: pd.DataFrame, ut_rc: pd.DataFrame
) -> tuple[float, pd.Series, float | pd.Series]:
    ability_method = model_params["ability"]
    difficulty_method = model_params["difficulty"]

    if ability_method == "package":
        return _compute_params_with_package(iu_stud, iu_rc, ut_rc, difficulty_method)
    if ability_method == "elo":
        weight = model_params["W"]
        return _compute_params_with_elo(iu_stud, iu_rc, ut_rc, weight)

    # compute learner ability
    ability_stud = _compute_learner_ability(ability_method, iu_stud)
    if ab_bound := model_params.get("ab_int"):
        ability_stud = _map_to_interval(ab_bound, ability_stud)

    # compute item difficulty
    difficulties = _compute_item_difficulty_wrc(
        difficulty_method, ut_rc, iu_rc, model_params
    )
    if diff_bound := model_params.get("diff_int"):
        difficulties = _map_to_interval(diff_bound, difficulties)

    return ability_stud, difficulties, 1


def _compute_params_with_package(
    iu_stud: pd.Series, iu_rc: pd.DataFrame, ut_rc: pd.DataFrame, method: str
) -> tuple[float, pd.Series, float]:
    train = _get_train_data(iu_stud, iu_rc, ut_rc)
    result = _fit_irt_model(train, method)

    if method == "2pl":
        disc = pd.Series(result["disc"][-ut_rc.shape[1] :], index=ut_rc.columns)
    else:
        disc = 1
    return (
        result["ability"][-1],
        pd.Series(result["diff"][-ut_rc.shape[1] :], index=ut_rc.columns),
        disc,
    )


def _get_train_data(
    iu_stud: pd.Series, iu_rc: pd.DataFrame, ut_rc: pd.DataFrame
) -> pd.DataFrame:
    train = iu_rc.copy()
    train.loc["stud"] = iu_stud
    return pd.concat([train, ut_rc], axis=1)


def _fit_irt_model(df: pd.DataFrame, method: str) -> dict:
    data = Dataset.from_pandas(
        df.reset_index(), subject_column="student_id", item_columns=df.columns
    )
    if method == "1pl":
        trainer = OneParamLog.train(data, epochs=300)
    elif method == "2pl":
        trainer = TwoParamLog.train(data, epochs=300)
    else:
        raise NotImplementedError
    return trainer.irt_model.export()


def _compute_params_with_elo(
    iu_stud: pd.Series, iu_rc: pd.DataFrame, ut_rc: pd.DataFrame, weight: float
) -> tuple[float, pd.Series, float]:
    # initialize learner abilities and item difficulties
    theta = pd.Series(0.0, index=[*list(set(iu_rc.index) | set(ut_rc.index)), "stud"])
    diff = pd.Series(0.0, index=list(iu_rc.columns) + list(ut_rc.columns))

    # reshape iu_rc such that only entries having values are preserved
    # then shuffle the values such that a random order is achieved
    iu_rc_resp = iu_rc.stack().sample(frac=1, random_state=42).copy()  # noqa: PD013

    # make index of iu_stud multidimensional
    new_idx = pd.MultiIndex.from_arrays(
        [["stud"] * len(iu_stud), iu_stud.index], names=["student_id", "problem_id"]
    )
    iu_stud_resp = iu_stud.copy()
    iu_stud_resp.index = new_idx

    # reshape ut_rc such that only entries having values are preserved
    # then shuffle the values such that a random order is achieved
    ut_rc_resp = ut_rc.stack().sample(frac=1, random_state=42).copy()  # noqa: PD013

    # concatenate responses in wished order
    # first difficulties of iu items and abilities of rc studs are updated
    # second, ability of stud is updated
    # finally, difficulties of ut items are updated
    resp_seq = pd.concat([iu_rc_resp, iu_stud_resp, ut_rc_resp])

    # use each entry one after another and update the parameters
    for (user, item), resp in resp_seq.items():
        theta[user], diff[item] = _elo_update_step(
            theta[user], diff[item], resp, weight
        )

    return theta["stud"], diff[ut_rc.columns], 1


def _elo_update_step(
    theta_user: float, diff_item: float, resp: int, weight: float
) -> tuple[float, float]:
    # compute expected response using Rasch model
    exp_resp = _compute_item_response_function(theta_user, 1, diff_item, 0)

    # update theta
    theta_user = theta_user + weight * (resp - exp_resp)

    # update difficulty
    diff_item = diff_item + weight * (exp_resp - resp)

    return theta_user, diff_item


def _compute_learner_ability(method: str, iu_stud: pd.Series) -> float:
    if method == "mean":
        return _compute_learner_ability_mean(iu_stud)
    raise NotImplementedError


def _compute_learner_ability_mean(iu_stud: pd.Series) -> float:
    return iu_stud.mean()


def _compute_item_difficulty_wrc(
    method: str, ut_rc: pd.DataFrame, iu_rc: pd.DataFrame, model_params: dict
) -> pd.Series:
    if method == "mean":
        return _compute_item_difficulty_wrc_mean(ut_rc)
    if method == "pc":
        return _compute_item_difficulty_wrc_proportion_correct(
            ut_rc, iu_rc, model_params
        )
    raise NotImplementedError


def _compute_item_difficulty_wrc_mean(ut_rc: pd.DataFrame) -> pd.Series:
    return 1 - ut_rc.mean(axis=0)


def _compute_item_difficulty_wrc_proportion_correct(
    ut_rc: pd.DataFrame, iu_rc: pd.DataFrame, model_params: dict
) -> pd.Series:
    # compute mean ability in rc
    # first compute ability of each student (using mean of responses), then use average
    # of those abilities
    ability_rc = iu_rc.mean(axis=1)
    if ab_bound := model_params.get("ab_int"):
        ability_rc = _map_to_interval(ab_bound, ability_rc)

    # get proportion of correct answers per ut problem
    # prop_corr has to be greater than 0 such that we can divide by it and less than 1
    # such that log can be computed
    # 0.01 yields a value of ~4.6 for the first term and 0.99 a value of ~-4.6
    # for consistency, we should set all values to be between 0.01 and 0.99
    prop_corr = ut_rc.mean(axis=0).clip(0.01, 0.99)

    # compute item difficulty estimates
    return np.log((1 - prop_corr) / prop_corr) + ability_rc.mean()


def _map_to_interval(bound: int, values: float | pd.Series) -> float | pd.Series:
    """Values are transformed from the interval [0, 1] to the specified interval
    [-bound, bound]."""
    return (values - 0.5) * (bound / 0.5)


def _compute_item_difficulty_worc(
    method: str, probs: pd.Index, model_params: float
) -> pd.Series:
    if method == "constant":
        return pd.Series(model_params["difficulty_value"], index=probs)
    if method == "expert":
        return EXPERT_VALUES.loc[probs, "irt_difficulty"]
    raise NotImplementedError


def _compute_item_response_function(
    theta: float, a: pd.Series | float, b: pd.Series | float, c: pd.Series | float
) -> pd.Series:
    # Probabiliy that item i is answered correctly given learner ability theta
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))
