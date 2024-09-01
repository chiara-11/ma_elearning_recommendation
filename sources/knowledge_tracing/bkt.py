import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(Path("../../sources").resolve())

import config
from config import KTModelType


EXPERT_VALUES_SKILLS = pd.read_csv(
    config.DATA_FOLDER / "expert_data_bkt_skills.csv"
).set_index("problem_skill_code_2")
EXPERT_VALUES_PROBS = pd.read_csv(
    config.DATA_FOLDER / "expert_data_bkt_probs.csv"
).set_index("problem_id")


def check_model_dict_bkt(model_params: dict, *, with_rc: bool) -> None:
    # check knowl_param_method
    knowl_param_method = model_params.get("knowl_param_method")
    assert knowl_param_method, "A parameter method for the knowledge parameters should be specified (parameter 'knowl_param_method')."
    if with_rc:
        assert (
            knowl_param_method in config.KT_KNOWLEDGE_PARAM_METHODS["wrc"]
        ), f"The knowledge parameter method should be one of {config.KT_KNOWLEDGE_PARAM_METHODS['wrc']}."
    else:
        assert (
            knowl_param_method in config.KT_KNOWLEDGE_PARAM_METHODS["worc"]
        ), f"The knowledge parameter method should be one of {config.KT_KNOWLEDGE_PARAM_METHODS['worc']}."

    # check perf_param_method
    perf_param_method = model_params.get("perf_param_method")
    assert perf_param_method, "A parameter method for the performance parameters should be specified (parameter 'perf_param_method')."
    assert (
        perf_param_method in config.KT_PERFORMANCE_PARAM_METHODS[knowl_param_method]
    ), f"The performance parameter method should be one of {config.KT_PERFORMANCE_PARAM_METHODS[knowl_param_method]}."

    model_type = model_params.get("model_type")
    if model_type == KTModelType.BKT and knowl_param_method == "constant":
        _check_constant_values(model_params, with_forget=False)
    if model_type == KTModelType.BKT_FORGET and knowl_param_method == "constant":
        _check_constant_values(model_params, with_forget=True)


def _check_constant_values(model_params: dict, *, with_forget: bool) -> None:
    if with_forget:
        param_list = ["p_init", "p_learn", "p_slip", "p_guess", "p_forget"]
    else:
        param_list = ["p_init", "p_learn", "p_slip", "p_guess"]

    for param in param_list:
        param_val = model_params.get(param)
        assert (
            param_val or param_val == 0
        ), f"The parameter {param} should be specified."
        assert isinstance(param_val, float), f"The parameter {param} should be a float."
        assert (
            0 <= param_val <= 1
        ), f"The parameter {param} should be a float between 0 and 1."


def perform_bkt_worc(
    model_params: dict, ut_stud: pd.DataFrame, iu_stud: pd.DataFrame
) -> pd.Series:
    skill_col = "problem_skill_code_2"

    # skills existing in ut data
    ut_skills = ut_stud[skill_col].unique()

    # only problems from considered skills are needed and sort by timestamp
    iu_stud = _restrict_and_sort_iu_data(iu_stud, ut_skills, skill_col)

    # parameters for ut_skills
    probs_known = _get_init_probs_known(
        model_params, ut_skills, iu_rc=None, skill_col=None
    )

    if len(iu_stud) > 0:
        # add transit, forget, slip and guess probabilities
        iu_stud, probs_known, p_slip, p_guess = _add_parameters_to_iu_data(
            model_params,
            iu_stud,
            skill_col,
            probs_known,
            iu_rc=None,
            ut_skills=ut_skills,
        )

        # update p_known per skill using iu data
        probs_known = _update_probs_known(probs_known, iu_stud, skill_col)
    else:
        p_slip = None
        p_guess = None

    # add parameters to ut data
    ut_stud = _add_parameters_to_ut_data(
        model_params, ut_stud, probs_known, p_slip, p_guess, skill_col
    )

    # Probability that problem will be answered correct
    # Student knows skill and does not slip OR student does not know skill but guesses correctly
    return _perform_predictions(ut_stud).round(config.ROUND_DECIMALS)


def perform_bkt_wrc(
    ut_stud: pd.DataFrame,
    iu_stud: pd.DataFrame,
    ut_rc: pd.DataFrame,
    iu_rc: pd.DataFrame,
    model_params: dict,
) -> tuple[pd.Series, dict]:
    skill_col = "problem_skill_code_2"

    # skills existing in ut data
    ut_skills = ut_stud[skill_col].unique()

    # only problems from considered skills are needed and sort by timestamp
    iu_stud = _restrict_and_sort_iu_data(iu_stud, ut_skills, skill_col)

    # sort iu rc data
    iu_rc = (
        iu_rc.loc[iu_rc[skill_col].isin(ut_skills)]
        .reset_index()
        .sort_values([skill_col, "student_id", "timestamp"], ascending=True)[
            ["student_id", "problem_id", "first_answer", skill_col]
        ]
    )

    # parameters for ut_skills
    probs_known = _get_init_probs_known(model_params, ut_skills, iu_rc, skill_col)

    if len(iu_stud) > 0:
        # add transit, slip and guess probabilities
        iu_stud, probs_known, p_slip, p_guess = _add_parameters_to_iu_data(
            model_params, iu_stud, skill_col, probs_known, iu_rc, ut_skills
        )

        # update p_known per skill using iu data
        probs_known = _update_probs_known(probs_known, iu_stud, skill_col)
    else:
        p_slip = None
        p_guess = None

    # add parameters to ut data
    ut_stud = _add_parameters_to_ut_data(
        model_params, ut_stud, probs_known, p_slip, p_guess, skill_col
    )

    # Probability that problem will be answered correct
    # Student knows skill and does not slip OR student does not know skill but guesses correctly
    y_pred = _perform_predictions(ut_stud).round(config.ROUND_DECIMALS)

    # get info column values
    info_values_diff = {
        "num_stud_rc": iu_rc["student_id"].nunique(),
        "max_num_iu_probs_rc": iu_rc["student_id"].value_counts().max(),
        "mean_iu_perf_rc": iu_rc.groupby("student_id")["first_answer"].mean().mean(),
        "mean_ut_perf_rc": ut_rc.groupby("student_id")["first_answer"].mean().mean(),
    }

    return y_pred, info_values_diff


def _restrict_and_sort_iu_data(
    iu_stud: pd.DataFrame, ut_skills: list[str], skill_col: str
) -> pd.DataFrame:
    # only problems from considered skills are needed and sort by timestamp
    return iu_stud.loc[iu_stud[skill_col].isin(ut_skills)].sort_values(
        [skill_col, "timestamp"], ascending=True
    )[["first_answer", skill_col]]


def _get_init_probs_known(
    model_params: dict,
    ut_skills: list[str],
    iu_rc: pd.DataFrame | None,
    skill_col: str | None,
) -> pd.Series:
    param_method = model_params.get("knowl_param_method")
    if param_method == "constant":
        return pd.Series(model_params["p_init"], index=ut_skills).rename("p_known")
    if param_method in ["expert", "ep"]:
        return EXPERT_VALUES_SKILLS.loc[ut_skills, "bkt_init_known"].rename("p_known")
    if param_method in ["rc", "ep_rc"]:
        return _compute_init_probs_known_wrc(iu_rc, skill_col, ut_skills).rename(
            "p_known"
        )
    raise NotImplementedError


def _compute_init_probs_known_wrc(
    iu_rc: pd.DataFrame, skill_col: str, ut_skills: list[str]
) -> pd.Series:
    replace_val = 0.3
    # only keep first problem per student and skill
    iu_first = iu_rc.drop_duplicates([skill_col, "student_id"], keep="first")

    # number of students per skill
    num_stud_per_skill = iu_first.groupby(skill_col).size().sort_values()

    # proportion of correct answers per skill
    prop_corr = iu_first.groupby(skill_col)["first_answer"].mean()

    # replace init probabilities for skills with too small data base
    skills_sample = num_stud_per_skill[num_stud_per_skill < 3].index
    prop_corr.loc[skills_sample] = replace_val
    return prop_corr.reindex(ut_skills).fillna(replace_val).round(2).clip(0.1, 0.7)


def _add_parameters_to_iu_data(
    model_params: dict,
    iu_stud: pd.DataFrame,
    skill_col: str,
    probs_known: pd.Series,
    iu_rc: pd.DataFrame | None,
    ut_skills: list[str] | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, pd.DataFrame | None]:
    with_forget = model_params.get("model_type") == KTModelType.BKT_FORGET
    knowl_param_method = model_params.get("knowl_param_method")
    perf_param_method = model_params.get("perf_param_method")

    if knowl_param_method == "constant":
        iu_stud["bkt_slip"] = model_params["p_slip"]
        iu_stud["bkt_guess"] = model_params["p_guess"]
        iu_stud["bkt_learn_prob"] = model_params["p_learn"]
        iu_stud["bkt_forget"] = model_params["p_forget"] if with_forget else 0.0
        return iu_stud, probs_known, None, None

    if knowl_param_method == "expert":
        iu_stud = _add_knowledge_parameters(iu_stud, skill_col, with_forget=with_forget)
        return (
            _add_performance_parameters(iu_stud, perf_param_method, skill_col),
            probs_known,
            None,
            None,
        )

    if knowl_param_method == "rc":
        iu_stud = iu_stud.merge(
            _compute_learn_probs(iu_rc, skill_col, ut_skills),
            how="left",
            left_on=skill_col,
            right_index=True,
        )
        if with_forget:
            iu_stud = iu_stud.merge(
                _compute_forget_probs(iu_rc, skill_col, ut_skills).clip(
                    0, iu_stud["bkt_learn_prob"].max()
                ),
                how="left",
                left_on=skill_col,
                right_index=True,
            )
        else:
            iu_stud["bkt_forget"] = 0.0
        if perf_param_method == "rc":
            p_slip = _compute_slip_probs(iu_rc, skill_col, ut_skills)
            iu_stud = iu_stud.merge(
                p_slip, how="left", left_on=skill_col, right_index=True
            )
        else:
            p_slip = None
        return (
            _add_performance_parameters(iu_stud, perf_param_method, skill_col),
            probs_known,
            p_slip,
            None,
        )

    if knowl_param_method in ["ep", "ep_rc"]:
        # prepare iu data
        if knowl_param_method == "ep":
            # use student data for student-specific parameter values
            iu = iu_stud.reset_index()
            iu["student_id"] = "stud"
        else:  # knowl_param_method == "ep_rc"
            # use reference class' data
            iu = iu_rc.copy()

        # compute parameters
        params_df = compute_empirical_probability_parameters(iu, skill_col, ut_skills)
        # add parameters to iu_stud
        return (
            iu_stud.merge(
                params_df.drop(columns="p_known"),
                how="left",
                left_on=skill_col,
                right_index=True,
            ),
            params_df["p_known"],
            params_df["bkt_slip"],
            params_df["bkt_guess"],
        )

    raise NotImplementedError


def _add_knowledge_parameters(
    iu_stud: pd.DataFrame, skill_col: str, *, with_forget: bool
) -> pd.DataFrame:
    iu_stud = iu_stud.merge(
        EXPERT_VALUES_SKILLS[["bkt_learn_prob", "bkt_forget"]],
        how="left",
        left_on=skill_col,
        right_index=True,
    )
    if not with_forget:
        iu_stud["bkt_forget"] = 0.0
    return iu_stud


def _add_performance_parameters(
    iu_stud: pd.DataFrame, method: str, skill_col: str
) -> pd.DataFrame:
    if method == "expert_skill":
        return iu_stud.merge(
            EXPERT_VALUES_SKILLS[["bkt_slip", "bkt_guess"]],
            how="left",
            left_on=skill_col,
            right_index=True,
        )
    if method == "expert_prob":
        return iu_stud.merge(
            EXPERT_VALUES_PROBS, how="left", left_on="problem_id", right_index=True
        )
    if method == "rc":
        return iu_stud.merge(
            EXPERT_VALUES_SKILLS["bkt_guess"],
            how="left",
            left_on=skill_col,
            right_index=True,
        )
    raise NotImplementedError


def _compute_learn_probs(
    iu_rc: pd.DataFrame, skill_col: str, ut_skills: list[str]
) -> pd.Series:
    replace_val = 0.2
    # shift column first_answer to store next answer in same row
    iu_rc["next_answer"] = iu_rc.groupby([skill_col, "student_id"])[
        "first_answer"
    ].shift(-1)

    # restrict to rows where first answer is 0 and next answer exists
    iu_rest = iu_rc[(iu_rc["first_answer"] == 0) & (~pd.isna(iu_rc["next_answer"]))]

    # number of considered problems per skill
    num_prob_per_skill = iu_rest.groupby(skill_col).size().sort_values()

    # proportion of transitions from 0 to 1
    p_learn = iu_rest.groupby(skill_col)["next_answer"].mean().rename("bkt_learn_prob")

    # replace transition probabilities for skills with too small data base
    skills_sample = num_prob_per_skill[num_prob_per_skill < 3].index
    p_learn.loc[skills_sample] = replace_val
    return p_learn.reindex(ut_skills).fillna(replace_val).round(2).clip(0.1, 0.5)


def _compute_forget_probs(
    iu_rc: pd.DataFrame, skill_col: str, ut_skills: list[str]
) -> pd.Series:
    replace_val = 0.1
    # shift column first_answer to store next answer in same row
    iu_rc["next_answer"] = iu_rc.groupby([skill_col, "student_id"])[
        "first_answer"
    ].shift(-1)

    # restrict to rows where first answer is 1 and next answer exists
    iu_rest = iu_rc[(iu_rc["first_answer"] == 1) & (~pd.isna(iu_rc["next_answer"]))]

    # number of considered problems per skill
    num_prob_per_skill = iu_rest.groupby(skill_col).size().sort_values()

    # proportion of transitions from 1 to 0
    p_forget = 1 - iu_rest.groupby(skill_col)["next_answer"].mean().rename("bkt_forget")

    # replace transition probabilities for skills with too small data base
    skills_sample = num_prob_per_skill[num_prob_per_skill < 3].index
    p_forget.loc[skills_sample] = replace_val
    return p_forget.reindex(ut_skills).fillna(replace_val).round(2).clip(0, 0.3)


def _compute_slip_probs(
    iu_rc: pd.DataFrame, skill_col: str, ut_skills: list[str]
) -> pd.Series:
    replace_val = 0.1
    # number of considered problems per skill
    num_prob_per_skill = iu_rc.groupby(skill_col).size().sort_values()

    # compute proportion of wrong answers
    diff = (
        1 - iu_rc.groupby(skill_col)["first_answer"].mean().rename("bkt_diff")
    ).to_frame()

    # map difficulties to slip probabilities
    diff["bkt_slip"] = 0.05
    diff.loc[diff["bkt_diff"] > 0.2, "bkt_slip"] = 0.1
    diff.loc[diff["bkt_diff"] > 0.4, "bkt_slip"] = 0.15
    diff.loc[diff["bkt_diff"] > 0.6, "bkt_slip"] = 0.2

    # replace slip probabilities for skills with too small data base
    skills_sample = num_prob_per_skill[num_prob_per_skill < 3].index
    diff.loc[skills_sample, "bkt_slip"] = replace_val
    return diff["bkt_slip"].reindex(ut_skills).fillna(replace_val)


def compute_empirical_probability_parameters(
    iu: pd.DataFrame, skill_col: str, ut_skills: list[str]
) -> pd.DataFrame:
    # initialize dataframe for storing empirical probabilities
    params_df = pd.DataFrame(
        data=0.0,
        index=ut_skills,
        columns=["p_known", "bkt_learn_prob", "bkt_forget", "bkt_slip", "bkt_guess"],
    )
    # get response sequence per skill
    resp_seq_per_skill = iu.groupby([skill_col, "student_id"])["first_answer"].apply(
        list
    )

    # for each skill compute empirical probabilities
    for skill in resp_seq_per_skill.index.get_level_values(skill_col).unique():
        resp_list = [pd.Series(ri) for ri in resp_seq_per_skill[skill]]
        params_df.loc[skill] = _get_empirical_probabilities(resp_list)

    # clip values such that they are in a realistic range
    clip_dict = {
        "p_known": (0.1, 0.7),
        "bkt_learn_prob": (0.1, 0.5),
        "bkt_slip": (0.01, 0.3),
        "bkt_guess": (0.0, 0.4),
    }
    for col, clip_vals in clip_dict.items():
        params_df[col] = params_df[col].clip(*clip_vals)

    return params_df


def _get_empirical_probabilities(
    resp_list: list[pd.Series],
) -> tuple[float, float, float, float, float]:
    know_list = [_get_knowledge_annotation(ri) for ri in resp_list]
    return _compute_empirical_probabilities(know_list, resp_list)


def _get_knowledge_annotation(resp_seq: pd.Series) -> pd.Series:
    # get accuracies for each knowledge sequence
    # index indicates where 1 starts
    # (num_matching_zeros + num_matching_ones) / len(resp_seq)
    # num_matching_zeros = number of zeros up to first 1 (position i)
    # num_matching_ones = number of ones from first 1 on (position i)
    acc = pd.Series(
        [
            (len(resp_seq[:i]) - sum(resp_seq[:i]) + sum(resp_seq[i:])) / len(resp_seq)
            for i in range(len(resp_seq) + 1)
        ]
    )
    # get indices of best matching sequences
    max_idx = acc[acc == max(acc)].index

    # determine best matching sequence
    # if len(max_idx) = 1, then the corresponding sequence is used
    # otherwise the mean of the best sequences is taken
    best_seq = pd.Series(0, index=range(len(resp_seq)))
    for idx in max_idx:
        best_seq[idx:] += 1
    return best_seq / len(max_idx)


def _compute_empirical_probabilities(
    know_list: list[pd.Series], resp_list: list[pd.Series]
) -> tuple[float, float, float, float, float]:
    # initial knowledge probability
    p_init = np.mean([know_seq[0] for know_seq in know_list])

    # transition probability
    p_learn = _compute_param_value("p_learn", know_list, resp_list)

    # guessing probability
    p_guess = _compute_param_value("p_guess", know_list, resp_list)

    # slipping probability
    p_slip = _compute_param_value("p_slip", know_list, resp_list)

    return p_init, p_learn, 0.0, p_slip, p_guess


def _compute_param_value(
    param: str, know_list: list[pd.Series], resp_list: list[pd.Series]
) -> float:
    numerators = [
        _compute_numerator_for_one_stud(param, know_list[i], resp_list[i])
        for i in range(len(know_list))
    ]
    denominators = [
        _compute_denominator_for_one_stud(param, know_list[i])
        for i in range(len(know_list))
    ]
    return (
        np.round(sum(numerators) / sum(denominators), config.ROUND_DECIMALS)
        if sum(denominators) != 0
        else 0.0
    )


def _compute_numerator_for_one_stud(
    param: str, know_seq: pd.Series, resp_seq: pd.Series
) -> float:
    if param == "p_learn":
        # sum over i from 1 to len(K)
        # (1 - K[i-1]) * K[i]
        return sum(((1 - know_seq).shift(1) * know_seq)[1:])
    if param == "p_guess":
        return sum(resp_seq * (1 - know_seq))
    if param == "p_slip":
        return sum((1 - resp_seq) * know_seq)
    raise NotImplementedError


def _compute_denominator_for_one_stud(param: str, know_seq: pd.Series) -> float:
    if param == "p_learn":
        # sum over i from 1 to len(K)
        # (1 - K[i-1])
        return sum((1 - know_seq)[:-1])
    if param == "p_guess":
        return sum(1 - know_seq)
    if param == "p_slip":
        return sum(know_seq)
    raise NotImplementedError


def _add_parameters_to_ut_data(
    model_params: dict,
    ut_stud: pd.DataFrame,
    probs_known: pd.Series,
    p_slip: pd.Series | None,
    p_guess: pd.Series | None,
    skill_col: str,
) -> pd.DataFrame:
    param_method = model_params.get("perf_param_method")
    # exception when no iu student data was useful
    if param_method in ["rc", "ep", "ep_rc"] and p_slip is None:
        param_method = "expert_skill"

    if param_method == "constant":
        ut_stud["slip"] = model_params["p_slip"]
        ut_stud["guess"] = model_params["p_guess"]
        return ut_stud.merge(
            probs_known, how="left", left_on=skill_col, right_index=True
        )

    if param_method == "expert_skill":
        return (
            ut_stud[["first_answer", skill_col]]
            .merge(
                EXPERT_VALUES_SKILLS[["bkt_slip", "bkt_guess"]],
                how="left",
                left_on=skill_col,
                right_index=True,
            )
            .merge(probs_known, how="left", left_on=skill_col, right_index=True)
            .rename(columns={"bkt_slip": "slip", "bkt_guess": "guess"})
        )

    if param_method == "expert_prob":
        return (
            ut_stud[["first_answer", skill_col]]
            .merge(EXPERT_VALUES_PROBS, how="left", left_index=True, right_index=True)
            .merge(probs_known, how="left", left_on=skill_col, right_index=True)
            .rename(columns={"bkt_slip": "slip", "bkt_guess": "guess"})
        )

    if param_method == "rc":
        return (
            ut_stud[["first_answer", skill_col]]
            .merge(p_slip, how="left", left_on=skill_col, right_index=True)
            .merge(
                EXPERT_VALUES_SKILLS[["bkt_guess"]],
                how="left",
                left_on=skill_col,
                right_index=True,
            )
            .merge(probs_known, how="left", left_on=skill_col, right_index=True)
            .rename(columns={"bkt_slip": "slip", "bkt_guess": "guess"})
        )

    if param_method in ["ep", "ep_rc"]:
        return (
            ut_stud[["first_answer", skill_col]]
            .merge(p_slip, how="left", left_on=skill_col, right_index=True)
            .merge(p_guess, how="left", left_on=skill_col, right_index=True)
            .merge(probs_known, how="left", left_on=skill_col, right_index=True)
            .rename(columns={"bkt_slip": "slip", "bkt_guess": "guess"})
        )

    raise NotImplementedError


def _update_probs_known(
    probs_known: pd.Series, iu_stud: pd.DataFrame, skill_col: str
) -> pd.Series:
    # update p_known per skill using iu data
    for skill in iu_stud[skill_col].unique():
        skill_df = iu_stud.loc[iu_stud[skill_col] == skill].copy()
        p_known = probs_known.loc[skill]

        for _, row in skill_df.iterrows():
            p_known = _update_p_known(
                row["first_answer"],
                p_known,
                row["bkt_slip"],
                row["bkt_guess"],
                row["bkt_learn_prob"],
                row["bkt_forget"],
            )
        probs_known.loc[skill] = p_known
    return probs_known


def _update_p_known(
    resp: float,
    p_known: float,
    p_slip: float,
    p_guess: float,
    p_learn: float,
    p_forget: float,
) -> float:
    # Probability that student knew skill based on given response
    p_known_given_resp = _compute_p_known_given_resp(
        resp, p_known, p_slip, p_guess
    ).round(4)
    # Probability that student knows skill now
    # clip such that it is never divided by zero
    return (
        _compute_new_p_known(p_known_given_resp, p_learn, p_forget)
        .round(4)
        .clip(0.05, 0.95)
    )


def _compute_p_known_given_resp(
    resp: float, p_known: float, p_slip: float, p_guess: float
) -> float:
    # compute probability that student knew skill based on given response
    if resp == 1:  # correct response
        # Probability that student knew skill and did not slip
        p_known_and_correct = p_known * (1 - p_slip)
        # Probability that student did not know skill but guessed correctly
        p_not_known_and_correct = (1 - p_known) * p_guess
        # Probability that student knew skill based on correct response
        return p_known_and_correct / (p_known_and_correct + p_not_known_and_correct)
    else:  # wrong response  # noqa: RET505
        # Probability that student knew skill but slipped
        p_known_and_wrong = p_known * p_slip
        # Probability that student did not know skill and did not guess correctly
        p_not_known_and_wrong = (1 - p_known) * (1 - p_guess)
        # Probability that student knew skill based on wrong response
        return p_known_and_wrong / (p_known_and_wrong + p_not_known_and_wrong)


def _compute_new_p_known(
    p_known_given_resp: float, p_learn: float, p_forget: float
) -> float:
    # Probability that student knows skill now
    # Student knew skill and did not forget OR student did not know skill but learned during problem
    return p_known_given_resp * (1 - p_forget) + (1 - p_known_given_resp) * p_learn


def _perform_predictions(ut_stud: pd.DataFrame) -> pd.Series:
    return (
        ut_stud["p_known"] * (1 - ut_stud["slip"])
        + (1 - ut_stud["p_known"]) * ut_stud["guess"]
    )
