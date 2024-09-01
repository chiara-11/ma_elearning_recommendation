import pandas as pd


def restrict_details_to_available_assignments(
    act_logs: pd.DataFrame,
    uts: pd.DataFrame,
    ass_det: pd.DataFrame,
    seq_det: pd.DataFrame,
    prob_det: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # restrict assignment_details
    iu_ids = act_logs["assignment_log_id"].unique()
    ut_ids = uts["assignment_log_id"].unique()
    all_ids = list(ut_ids) + list(iu_ids)
    ass_det = ass_det.loc[all_ids].copy()

    # restrict sequence_details
    seq_ids = ass_det["sequence_id"].unique()
    seq_det = seq_det.loc[seq_det["sequence_id"].isin(seq_ids)].copy()

    # restrict problem_details
    # get used problem ids
    prob_ids_al = set(act_logs["problem_id"])
    prob_ids_uts = set(uts["problem_id"])
    prob_ids = prob_ids_al | prob_ids_uts
    # remove problem ids that are not in problem_details
    prob_ids = prob_ids & set(prob_det.index)
    # restrict
    prob_det = prob_det.loc[list(prob_ids)].copy()

    return ass_det, seq_det, prob_det
