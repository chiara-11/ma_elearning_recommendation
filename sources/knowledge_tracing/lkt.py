import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.append(Path("../../sources").resolve())

import config


def check_model_dict_pfa(model_params: dict, *, with_rc: bool) -> None:
    beta = model_params.get("beta")
    assert beta, "Parameter beta should be specified."
    if with_rc:
        assert (
            beta in config.PFA_BETA
        ), f"Parameter beta should be one of {config.PFA_BETA}."
    else:
        assert beta == "skill", "Parameter beta must be 'skill'."


def perform_pfa_wrc(
    ut_stud: pd.DataFrame,
    iu_stud: pd.DataFrame,
    ut_rc: pd.DataFrame,
    iu_rc: pd.DataFrame,
    model_params: dict,
) -> tuple[pd.Series, dict]:
    skill_col = "problem_skill_code_2"

    iu_stud["student_id"] = "stud"
    ut_stud["student_id"] = "stud"

    beta = model_params["beta"]
    if beta == "item":
        enc_col = "problem_id"
    elif beta == "skill":
        enc_col = skill_col

    # prepare train and test data
    X_train, X_test, y_train, _ = _prepare_data_pfa(  # noqa: N806
        iu_stud, ut_stud, iu_rc, ut_rc, skill_col, enc_col, with_rc=True
    )

    if y_train.nunique() == 1:  # noqa: PD101
        y_pred = [y_train[0]] * len(ut_stud)
    else:
        # logistic regression
        model = LogisticRegression(fit_intercept=False, solver="lbfgs", max_iter=1000)
        model.fit(X_train, y_train)

        # prediction
        y_pred = model.predict_proba(X_test)[:, 1].round(config.ROUND_DECIMALS)

    # get info column values
    info_values_diff = {
        "num_stud_rc": iu_rc.index.nunique(),
        "max_num_iu_probs_rc": iu_rc.index.value_counts().max(),
        "mean_iu_perf_rc": iu_rc.groupby("student_id")["first_answer"].mean().mean(),
        "mean_ut_perf_rc": ut_rc.groupby("student_id")["first_answer"].mean().mean(),
    }

    return y_pred, info_values_diff


def perform_pfa_worc(
    model_params: dict,  # noqa: ARG001
    ut_stud: pd.DataFrame,
    iu_stud: pd.DataFrame,
) -> tuple[pd.Series, dict]:
    skill_col = "problem_skill_code_2"

    if iu_stud["first_answer"].nunique() == 1:  # noqa: PD101
        return [iu_stud["first_answer"].iloc[0]] * len(ut_stud)

    iu_stud["student_id"] = "stud"
    ut_stud["student_id"] = "stud"

    # prepare train and test data
    X_train, X_test, y_train, _ = _prepare_data_pfa(  # noqa: N806
        iu_stud, ut_stud, None, None, skill_col, skill_col, with_rc=False
    )

    # logistic regression
    model = LogisticRegression(fit_intercept=False, solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)

    # prediction
    return model.predict_proba(X_test)[:, 1].round(config.ROUND_DECIMALS)


def perform_lfa_wrc(
    ut_stud: pd.DataFrame,
    iu_stud: pd.DataFrame,
    ut_rc: pd.DataFrame,
    iu_rc: pd.DataFrame,
    model_params: dict,  # noqa: ARG001
) -> tuple[pd.Series, dict]:
    skill_col = "problem_skill_code_2"

    iu_stud["student_id"] = "stud"
    ut_stud["student_id"] = "stud"

    # prepare train and test data
    X_train, X_test, y_train, _ = _prepare_data_lfa(  # noqa: N806
        iu_stud, ut_stud, iu_rc, ut_rc, skill_col, with_rc=True
    )

    if y_train.nunique() == 1:  # noqa: PD101
        y_pred = [y_train[0]] * len(ut_stud)
    else:
        # logistic regression
        model = LogisticRegression(fit_intercept=False, solver="lbfgs", max_iter=1000)
        model.fit(X_train, y_train)

        # prediction
        y_pred = model.predict_proba(X_test)[:, 1].round(config.ROUND_DECIMALS)

    # get info column values
    info_values_diff = {
        "num_stud_rc": iu_rc.index.nunique(),
        "max_num_iu_probs_rc": iu_rc.index.value_counts().max(),
        "mean_iu_perf_rc": iu_rc.groupby("student_id")["first_answer"].mean().mean(),
        "mean_ut_perf_rc": ut_rc.groupby("student_id")["first_answer"].mean().mean(),
    }

    return y_pred, info_values_diff


def perform_lfa_worc(
    model_params: dict,  # noqa: ARG001
    ut_stud: pd.DataFrame,
    iu_stud: pd.DataFrame,
) -> tuple[pd.Series, dict]:
    skill_col = "problem_skill_code_2"

    if iu_stud["first_answer"].nunique() == 1:  # noqa: PD101
        return [iu_stud["first_answer"].iloc[0]] * len(ut_stud)

    iu_stud["student_id"] = "stud"
    ut_stud["student_id"] = "stud"

    # prepare train and test data
    X_train, X_test, y_train, _ = _prepare_data_lfa(  # noqa: N806
        iu_stud, ut_stud, None, None, skill_col, with_rc=False
    )

    # logistic regression
    model = LogisticRegression(fit_intercept=False, solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)

    # prediction
    return model.predict_proba(X_test)[:, 1].round(config.ROUND_DECIMALS)


def _prepare_data_pfa(
    iu_stud: pd.DataFrame,
    ut_stud: pd.DataFrame,
    iu_rc: pd.DataFrame | None,
    ut_rc: pd.DataFrame | None,
    skill_col: str,
    enc_col: str,
    *,
    with_rc: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # add number of successes and fails
    iu_stud, ut_stud = _add_num_succ_and_fail(iu_stud, ut_stud, skill_col)

    if with_rc:
        iu_rc, ut_rc = _add_num_succ_and_fail(iu_rc, ut_rc, skill_col)
        # get total train data
        train = pd.concat([iu_stud, iu_rc, ut_rc]).reset_index(drop=True)
    else:
        train = iu_stud

    # extend number of successes and failures to columns for each skill
    # skills that are only in ut_stud are ignored
    skills = train[skill_col].unique()
    train, ut_stud = _extend_num_cols_pfa(train, ut_stud, skills, skill_col)
    succ_cols = [f"num_succ_{sk}" for sk in skills]
    fail_cols = [f"num_fail_{sk}" for sk in skills]

    train_feat, test_feat = _perform_one_hot_encoding(train, ut_stud, enc_col)
    return (
        pd.concat([train_feat, train[succ_cols + fail_cols]], axis=1),
        pd.concat([test_feat, ut_stud[succ_cols + fail_cols]], axis=1),
        train["first_answer"],
        ut_stud["first_answer"],
    )


def _add_num_succ_and_fail(
    iu: pd.DataFrame, ut: pd.DataFrame, skill_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    iu = iu.reset_index()
    ut = ut.reset_index()
    # get number of prior successes and failures for iu data
    rel_cols = ["student_id", "problem_id", skill_col, "timestamp", "first_answer"]
    iu = iu[rel_cols].sort_values(["student_id", "timestamp"]).drop(columns="timestamp")
    iu["num_success"] = (
        iu.groupby(["student_id", skill_col])["first_answer"].cumsum()
        - iu["first_answer"]
    )
    iu["num_failure"] = (
        iu.groupby(["student_id", skill_col]).cumcount() - iu["num_success"]
    )

    # get total number of successes and failures in iu for iu data
    num_df = iu.drop_duplicates(subset=["student_id", skill_col], keep="last").copy()
    num_df["num_success"] += num_df["first_answer"]
    num_df["num_failure"] += 1 - num_df["first_answer"]

    # add total numbers as prior numbers to ut data
    ut = ut[rel_cols].drop(columns="timestamp")
    ut = ut.merge(
        num_df.drop(columns=["problem_id", "first_answer"]),
        how="left",
        on=["student_id", skill_col],
    ).fillna(0)
    return iu, ut


def _extend_num_cols_pfa(
    train: pd.DataFrame, ut_stud: pd.DataFrame, skills: list[str], skill_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for sk in skills:
        mask_sk = train[skill_col] == sk
        train[f"num_succ_{sk}"] = mask_sk * train["num_success"]
        train[f"num_fail_{sk}"] = mask_sk * train["num_failure"]
        mask_sk_ut = ut_stud[skill_col] == sk
        ut_stud[f"num_succ_{sk}"] = mask_sk_ut * ut_stud["num_success"]
        ut_stud[f"num_fail_{sk}"] = mask_sk_ut * ut_stud["num_failure"]
    return train, ut_stud


def _prepare_data_lfa(
    iu_stud: pd.DataFrame,
    ut_stud: pd.DataFrame,
    iu_rc: pd.DataFrame | None,
    ut_rc: pd.DataFrame | None,
    skill_col: str,
    *,
    with_rc: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # add number of successes and fails
    iu_stud, ut_stud = _add_num_completed(iu_stud, ut_stud, skill_col)

    if with_rc:
        iu_rc, ut_rc = _add_num_completed(iu_rc, ut_rc, skill_col)
        # get total train data
        train = pd.concat([iu_stud, iu_rc, ut_rc]).reset_index(drop=True)
    else:
        train = iu_stud

    # extend number of successes and failures to columns for each skill
    # skills that are only in ut_stud are ignored
    skills = train[skill_col].unique()
    train, ut_stud = _extend_num_col_lfa(train, ut_stud, skills, skill_col)
    comp_cols = [f"num_comp_{sk}" for sk in skills]

    # one hot encoding for students
    train_feat, test_feat = _perform_one_hot_encoding(train, ut_stud, "student_id")
    # one hot encoding for skill col
    train_feat2, test_feat2 = _perform_one_hot_encoding(train, ut_stud, skill_col)
    train_feat = pd.concat([train_feat, train_feat2], axis=1)
    test_feat = pd.concat([test_feat, test_feat2], axis=1)

    return (
        pd.concat([train_feat, train[comp_cols]], axis=1),
        pd.concat([test_feat, ut_stud[comp_cols]], axis=1),
        train["first_answer"],
        ut_stud["first_answer"],
    )


def _add_num_completed(
    iu: pd.DataFrame, ut: pd.DataFrame, skill_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    iu = iu.reset_index()
    ut = ut.reset_index()
    # get number of prior tasks completed for iu data
    rel_cols = ["student_id", "problem_id", skill_col, "timestamp", "first_answer"]
    iu = iu[rel_cols].sort_values(["student_id", "timestamp"]).drop(columns="timestamp")
    iu["num_completed"] = iu.groupby(["student_id", skill_col]).cumcount()
    # get total number of successes and failures in iu for iu data
    num_df = iu.drop_duplicates(subset=["student_id", skill_col], keep="last").copy()

    # add total numbers as prior numbers to ut data
    ut = ut[rel_cols].drop(columns="timestamp")
    ut = ut.merge(
        num_df.drop(columns=["problem_id", "first_answer"]),
        how="left",
        on=["student_id", skill_col],
    ).fillna(0)
    return iu, ut


def _extend_num_col_lfa(
    train: pd.DataFrame, ut_stud: pd.DataFrame, skills: list[str], skill_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for sk in skills:
        mask_sk = train[skill_col] == sk
        train[f"num_comp_{sk}"] = mask_sk * train["num_completed"]
        mask_sk_ut = ut_stud[skill_col] == sk
        ut_stud[f"num_comp_{sk}"] = mask_sk_ut * ut_stud["num_completed"]
    return train, ut_stud


def _perform_one_hot_encoding(
    train: pd.DataFrame, ut_stud: pd.DataFrame, col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # fit encoder and transform train data
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_feat = encoder.fit_transform(train[[col]])
    enc_cols = encoder.categories_[0]
    # transform test data
    test_feat = encoder.transform(ut_stud[[col]])

    return pd.DataFrame(data=train_feat, columns=enc_cols), pd.DataFrame(
        data=test_feat, columns=enc_cols
    )
