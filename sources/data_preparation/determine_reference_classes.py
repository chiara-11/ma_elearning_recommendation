import sys
from itertools import chain
from pathlib import Path

import pandas as pd

sys.path.append(Path("../../sources").resolve())

import utils


def get_reference_classes(data: pd.DataFrame) -> dict[str, dict[str, dict]]:
    # create ass_seq
    ass_seq = create_ass_seq(data)

    # create helping dataframes
    ut_seq_per_class = ass_seq.groupby("ut_class")["ut_seq"].apply(pd.unique)
    iu_seq_per_cid_ts = ass_seq.groupby(["ut_class", "ut_seq"])["iu_seq"].apply(
        pd.unique
    )
    ass_seq = ass_seq.set_index(["ut_seq", "iu_seq"])
    # list of students per class
    stud_per_class = (
        data.loc[data["unit_test"] == 1]
        .groupby("class_id")["student_id"]
        .unique()
        .apply(list)
    )
    # list of problems for each student
    prob_per_stud = data.groupby("student_id")["problem_id"].unique()

    # initialize
    rc_dict = {}

    # determine reference classes
    for cid in ass_seq["ut_class"].unique():
        # initialize details dict
        details_dict = {}
        rc_complete = set()
        # get list of unit test sequences
        test_sequences = ut_seq_per_class.loc[cid]

        for ts in test_sequences:
            # get list of in unit sequence ids belonging to cid and ts
            iu_seq = iu_seq_per_cid_ts[cid, ts]

            # get classes that worked on same ut sequence and on at least one same iu sequence
            reference_classes = ass_seq.loc[ts].loc[iu_seq, "ut_class"].unique()
            if len(reference_classes) > 1:
                reference_classes = [rfc for rfc in reference_classes if rfc != cid]
                rc_complete.update(reference_classes)
            else:
                reference_classes = None
            # fill in dict for test sequence
            details_dict[ts] = {
                "reference_classes": reference_classes,
                "iu_seq": iu_seq,
            }

        # get list of students
        cid_studs = stud_per_class.loc[cid]
        # get list of problems of students in cid
        cid_probs = set(chain.from_iterable(prob_per_stud.loc[cid_studs]))

        # add class dict to complete dict if any reference class exists
        if len(rc_complete) > 0:
            rc_dict[cid] = {
                "students": cid_studs,
                "problems": list(cid_probs),
                "ref_classes_complete": list(rc_complete),
                "details": details_dict,
            }

    return rc_dict


def create_ass_seq(data: pd.DataFrame) -> pd.DataFrame:
    # read assignment_relationships and restrict to ut assignments and iu assignments in df
    ass_rel = _read_and_restrict_ass_rel(data)

    # get info on sequence_id, class_id and student_id per assignment
    ass_info = data.drop_duplicates(subset="assignment_log_id").set_index(
        "assignment_log_id"
    )

    # merge info to assignment_relationships for ut assignments
    ass_seq = ass_rel.merge(
        ass_info[["sequence_id", "class_id", "student_id"]],
        how="left",
        left_on="ut_id",
        right_index=True,
    )

    # merge info to ass_seq for iu assignments
    return ass_seq.merge(
        ass_info[["sequence_id"]],
        how="left",
        left_on="iu_id",
        right_index=True,
        suffixes=["_ut", "_iu"],
    ).rename(
        columns={
            "ut_id": "ut_ass",
            "iu_id": "iu_ass",
            "sequence_id_ut": "ut_seq",
            "sequence_id_iu": "iu_seq",
            "class_id": "ut_class",
        }
    )


def restrict_c2rc_dict(c2rc: dict, rest_file: str) -> dict:
    rc_df = utils.read_data_file(rest_file)
    rc_df = rc_df.groupby(["class_id", "ut_id"])["ref_class"].apply(list).to_frame()
    for cid in rc_df.index.get_level_values("class_id").unique():
        cid_df = rc_df.loc[cid]
        rc_comp = []
        for ts, rc in cid_df.iterrows():
            rc_list = rc["ref_class"]
            c2rc[cid]["details"][ts]["reference_classes"] = rc_list
            rc_comp.extend(rc_list)
        c2rc[cid]["ref_classes_complete"] = list(set(rc_comp))
    return c2rc


def _read_and_restrict_ass_rel(data: pd.DataFrame) -> pd.DataFrame:
    ass_rel = utils.read_assignment_relationships().rename(
        columns={
            "unit_test_assignment_log_id": "ut_id",
            "in_unit_assignment_log_id": "iu_id",
        }
    )
    ut_ass = data.loc[data["unit_test"] == 1, "assignment_log_id"].unique()
    iu_ass = data.loc[data["unit_test"] == 0, "assignment_log_id"].unique()

    return ass_rel.loc[
        (ass_rel["ut_id"].isin(ut_ass)) & (ass_rel["iu_id"].isin(iu_ass))
    ].reset_index(drop=True)
