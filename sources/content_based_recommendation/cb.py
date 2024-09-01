import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier

sys.path.append(Path("../../sources").resolve())

import config
import utils
from config import CBModelType


def check_conf_cb(model_list: list[dict], *, with_rc: bool) -> None:
    """
    Conf dictionary for content based recommendation needs a list of dictionaries, where
    each specifies a model.

    At the moment, the method is only implemented for recommendations without reference
    classes.
    """
    # check parameter with_ref_class
    assert with_rc is False, "CB is only implemented without reference classes."

    # check if there are models in the model_list
    assert (
        model_list
    ), "CB needs at least one model specification in 'models' but none are specified."

    for model in model_list:
        # check model specification
        _check_model_dict(model)


def _check_model_dict(model_params: dict) -> None:
    """Each model dictionary needs a parameter model_type of type CBModelType and model
    parameters related to that."""
    # check model type
    model_type = model_params.get("model_type")
    assert (
        model_type
    ), "The model specification needs a parameter model_type of type CBModelType."
    assert isinstance(
        model_type, CBModelType
    ), "The parameter model_type has to be of type CBModelType."

    # check used columns
    used_cols = model_params.get("used_columns")
    assert used_cols, "CB needs parameter used_columns."
    assert (
        used_cols in config.CB_USED_COLS
    ), f"The used_columns parameter should be one of {list(config.CB_USED_COLS.keys())}"

    # check further model parameters
    if model_type == CBModelType.DTC:
        _check_model_params_dtc(model_params)
    if model_type == CBModelType.KNN:
        _check_model_params_knn(model_params)
    if model_type == CBModelType.LOGREG:
        _check_model_params_logreg(model_params)
    if model_type == CBModelType.RFC:
        _check_model_params_rfc(model_params)
    if model_type == CBModelType.XGB:
        _check_model_params_xgb(model_params)


def _check_model_params_dtc(model_params: dict) -> None:
    # check parameter max_depth if existing
    max_depth = model_params.get("max_depth")
    assert not max_depth or isinstance(
        max_depth, int
    ), "The specified parameter max_depth should be a positive integer or None."


def _check_model_params_knn(model_params: dict) -> None:
    # check parameter k
    k = model_params.get("k")
    assert (
        k
    ), "The parameter k (number of neighbors) must be specified. The default is 5."
    assert isinstance(k, int), "The specified parameter k should be a positive integer."


def _check_model_params_logreg(model_params: dict) -> None:
    # check parameter max_iter
    max_iter = model_params.get("max_iter")
    assert max_iter, "The parameter max_iter must be specified. The default is 100."
    assert isinstance(
        max_iter, int
    ), "The specified parameter max_iter should be a positive integer."


def _check_model_params_rfc(model_params: dict) -> None:
    # check parameter n_estimators
    n_estimators = model_params.get("n_estimators")
    assert (
        n_estimators
    ), "The parameter n_estimators must be specified. The default is 100."
    assert isinstance(
        n_estimators, int
    ), "The specified parameter n_estimators should be a positive integer."

    # check parameter max_depth if existing
    max_depth = model_params.get("max_depth")
    assert not max_depth or isinstance(
        max_depth, int
    ), "The specified parameter max_depth should be a positive integer or None."


def _check_model_params_xgb(model_params: dict) -> None:
    # check parameter n_estimators
    n_estimators = model_params.get("n_estimators")
    assert n_estimators, "The parameter n_estimators must be specified."
    assert isinstance(
        n_estimators, int
    ), "The specified parameter n_estimators should be a positive integer."

    # check parameter max_depth
    max_depth = model_params.get("max_depth")
    assert max_depth, "The parameter max_depth must be specified. The default is 6."
    assert isinstance(
        max_depth, int
    ), "The specified parameter max_depth should be a positive integer."

    # check parameter lr
    learning_rate = model_params.get("lr")
    assert learning_rate, "The parameter lr must be specified."


def prepare_df_for_cb(data: pd.DataFrame) -> pd.DataFrame:
    # read problem_details
    prob_det = utils.read_problem_details(second_save=False)

    # restrict prob_det to problems in data
    prob_det = prob_det.loc[list(set(data["problem_id"]))].copy()

    # extend bert pca lists to columns
    prob_det = _extend_bert_pca_lists(prob_det)

    # map skill codes to first two levels and extract all levels
    prob_det = _map_skill_codes(prob_det)

    # merge problem_details to df
    data = data.merge(prob_det, how="left", left_on="problem_id", right_index=True)

    # replace nan values in skill columns
    skill_cols = [
        "problem_skill_code_mapped",
        "problem_skill_code_1",
        "problem_skill_code_2",
    ]
    return _replace_skill_nan(data, skill_cols)


def _extend_bert_pca_lists(prob_det: pd.DataFrame) -> pd.DataFrame:
    # extend bert pca lists to columns
    bert_df = pd.DataFrame(
        prob_det["problem_text_bert_pca"].tolist(), index=prob_det.index
    ).astype(float)
    bert_df.columns = [f"bert_pca_{i + 1}" for i in range(len(bert_df.columns))]
    return prob_det.merge(bert_df, how="left", left_index=True, right_index=True).drop(
        columns="problem_text_bert_pca"
    )


def _map_skill_codes(prob_det: pd.DataFrame) -> pd.DataFrame:
    # map skill codes to first two levels
    prob_det["problem_skill_code_mapped"] = prob_det["problem_skill_code"].apply(
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
    no_skill = data.loc[pd.isna(data["problem_skill_code_mapped"])].copy()

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
    no_skill_mask = pd.isna(data["problem_skill_code_mapped"])
    no_skill_2 = data.loc[no_skill_mask].copy()
    # all of them have the same two sequences which themselves have the same skill codes
    sid = no_skill_2.groupby("problem_id")["sequence_id"].unique().to_numpy()[0][0]
    data.loc[no_skill_mask, skill_cols] = seq_det_ns.loc[sid, skill_cols].to_numpy()

    return data


def perform_content_based_recommendation(
    model_params: dict, ut_stud: pd.DataFrame, iu_stud: pd.DataFrame
) -> pd.DataFrame:
    X_train, X_test, y_train, _ = _get_student_train_test_data(  # noqa: N806
        iu_stud, ut_stud, model_params
    )

    if y_train.nunique() == 1:  # noqa: PD101
        return pd.Series([y_train.iloc[0]] * len(X_test), index=X_test.index)
    if model_params["model_type"] == CBModelType.KNN and model_params["k"] > len(
        X_train
    ):
        return np.nan

    # initialize estimator
    estim, pred_type = _initialize_estimator(model_params)

    # fit estimator
    estim_fit = estim.fit(X_train, y_train)

    # make predictions
    if pred_type == "regression":
        # use predict
        y_pred = estim_fit.predict(X_test).clip(0, 1)
    else:  # pred_type == "classification"
        # use predict_proba
        y_pred = estim_fit.predict_proba(X_test)[:, 1]
        # we use predict_proba to get the probability that the record is in the
        # respective class
        # we use the probability to be in class 1, that is to yield a correct answer
        # if it is greater than 0.5, it is classified to class 1, otherwise to class 0

    return pd.Series(y_pred.round(config.ROUND_DECIMALS), index=X_test.index)


def _get_student_train_test_data(
    iu_stud: pd.DataFrame, ut_stud: pd.DataFrame, model_params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # restrict to specified cols
    nec_cols = ["first_answer", *get_used_cols(model_params)]
    iu_stud = iu_stud[nec_cols]
    ut_stud = ut_stud[nec_cols]

    if len(iu_stud) > 1:
        # remove columns that only contain one unique value
        num_diff_vals = iu_stud.nunique()
        drop_cols = [
            col
            for col in num_diff_vals[num_diff_vals == 1].index
            if col != "first_answer"
        ]
        if len(drop_cols) == len(num_diff_vals) - 1:
            # all columns only have one unique value (all rows are duplicates of each
            # other (except for potentially first_answer))
            # keep one column
            drop_cols = drop_cols[1:]
        iu_stud = iu_stud.drop(columns=drop_cols)
        ut_stud = ut_stud.drop(columns=drop_cols)

    # one hot encoding for relevant columns
    iu_stud, ut_stud = _perform_one_hot_encoding(iu_stud, ut_stud)

    # get train and test data
    return (
        iu_stud.drop(columns="first_answer"),
        ut_stud.drop(columns="first_answer"),
        iu_stud["first_answer"],
        ut_stud["first_answer"],
    )


def get_used_cols(model_params: dict) -> list[str]:
    # get used columns from dictionary
    used_cols = config.CB_USED_COLS[model_params["used_columns"]]
    # use extended columns instead of list-column
    if "problem_text_bert_pca" in used_cols:
        used_cols = [col for col in used_cols if col != "problem_text_bert_pca"] + [
            f"bert_pca_{i + 1}" for i in range(32)
        ]
    return used_cols


def _perform_one_hot_encoding(
    iu_stud: pd.DataFrame, ut_stud: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _transform_data(data_matrix: pd.DataFrame) -> pd.DataFrame:
        X = data_matrix[one_hot_cols].copy()  # noqa: N806
        X = pd.DataFrame(  # noqa: N806
            enc_fit.transform(X).toarray(),
            index=X.index,
            columns=enc_fit.get_feature_names_out(),
        )
        return data_matrix.drop(columns=one_hot_cols).merge(
            X, how="left", left_index=True, right_index=True
        )

    # one hot encoding for relevant columns
    one_hot_cols = [
        col for col in iu_stud.columns if col in config.CB_ONE_HOT_COL_CANDS
    ]
    X_iu = iu_stud[one_hot_cols].copy()  # noqa: N806

    # initialize and fit encoder
    enc = OneHotEncoder(handle_unknown="ignore")
    enc_fit = enc.fit(X_iu)
    # transform data
    return _transform_data(iu_stud), _transform_data(ut_stud)


def _initialize_estimator(model_params: dict) -> tuple[Any, str]:  # noqa: C901, PLR0911
    model_type = model_params["model_type"]
    if model_type == CBModelType.DTC:
        return DecisionTreeClassifier(
            random_state=42, max_depth=model_params.get("max_depth")
        ), "classification"
    if model_type == CBModelType.DTR:
        return DecisionTreeRegressor(random_state=42), "regression"
    if model_type == CBModelType.GNB:
        return GaussianNB(), "classification"
    if model_type == CBModelType.KNN:
        return KNeighborsClassifier(n_neighbors=model_params["k"]), "classification"
    if model_type == CBModelType.LINREG:
        return LinearRegression(), "regression"
    if model_type == CBModelType.LOGREG:
        return LogisticRegression(
            random_state=42, max_iter=model_params["max_iter"]
        ), "classification"
    if model_type == CBModelType.RFC:
        return RandomForestClassifier(
            random_state=42,
            n_estimators=model_params["n_estimators"],
            max_depth=model_params.get("max_depth"),
        ), "classification"
    if model_type == CBModelType.RFR:
        return RandomForestRegressor(random_state=42), "regression"
    if model_type == CBModelType.SVC:
        return SVC(random_state=42, probability=True), "classification"
    if model_type == CBModelType.XGB:
        return XGBClassifier(
            random_state=42,
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            learning_rate=model_params["lr"],
        ), "classification"
    raise NotImplementedError
