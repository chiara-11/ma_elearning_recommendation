import pathlib as pl
from enum import Enum

_ROOT_DIR = pl.Path(__file__).parents[1]

DATA_FOLDER = _ROOT_DIR / "data"
RESULTS_FOLDER = _ROOT_DIR / "results"


##################### EVALUATION #######################


class LimType(Enum):
    DYNAMIC = "dynamic"


class RegMetrics(Enum):
    MAE = "mae"
    MSE = "mse"


class ClassMetrics(Enum):
    ACC = "acc"
    F1 = "f1"
    PREC = "precision"
    REC = "recall"


class InfoCols(Enum):
    NUM_UT_PROBS = "num_ut_probs"
    NUM_IU_PROBS = "num_iu_probs"
    MEAN_UT_PERF = "mean_ut_perf"
    MEAN_IU_PERF = "mean_iu_perf"
    NUM_STUD_RC = "num_stud_rc"
    MAX_NUM_IU_PROBS_RC = "max_num_iu_probs_rc"
    MEAN_IU_PERF_RC = "mean_iu_perf_rc"
    MEAN_UT_PERF_RC = "mean_ut_perf_rc"


EVAL_COL_GROUPS = ["info_cols", "reg_metrics", "class_metrics"]


##################### MODELS GENERAL ####################


class RecMethod(Enum):
    CF = "Collaborative Filtering"
    CB = "Content-Based"
    KT = "Knowledge Tracing"
    IRT = "Item Response Theory"


ROUND_DECIMALS = 4


##################### COLLABORATIVE FILTERING ###########


class CFModelType(Enum):
    KNN = "knn"
    KNN_ITEM = "knn_item"


CF_KNN_SIM = ["manhattan"]
CF_KNN_SIM_WEIGHT = ["significance"]
CF_KNN_PRED = ["weightavg", "resnick"]


##################### CONTENT-BASED RECOMMENDATION ######


class CBModelType(Enum):
    DTC = "dtc"
    DTR = "dtr"
    GNB = "gnb"
    KNN = "knn"
    LINREG = "linreg"
    LOGREG = "logreg"
    RFC = "rfc"
    RFR = "rfr"
    SVC = "svc"
    XGB = "xgb"


CB_USED_COLS = {
    "v1": [
        "problem_multipart_position",
        "problem_type",
        "problem_skill_code_1",
        "problem_skill_code_2",
        "problem_contains_image",
        "problem_contains_equation",
        "problem_contains_video",
        "problem_text_bert_pca",
    ],
    "v2": [
        "problem_type",
        "problem_skill_code_2",
        "bert_pca_1",
        "bert_pca_2",
        "bert_pca_3",
        "bert_pca_4",
        "bert_pca_5",
    ],
}

CB_ONE_HOT_COL_CANDS = [
    "problem_multipart_id",
    "problem_type",
    "problem_skill_code",
    "problem_skill_code_mapped",
    "problem_skill_description",
] + [f"problem_skill_code_{i+1}" for i in range(4)]


##################### ITEM RESPONSE THEORY #########


class IRTModelType(Enum):
    WRC = "wrc"
    WORC = "worc"


IRT_ABILITY_METHODS = ["mean", "package", "elo"]
IRT_DIFFICULTY_METHODS_WRC = ["mean", "pc", "1pl", "2pl", "elo"]
IRT_DIFFICULTY_METHODS_WORC = ["constant", "expert"]


##################### KNOWLEDGE TRACING ############


class KTModelType(Enum):
    BKT = "bkt"
    BKT_FORGET = "bkt_forget"
    LFA = "lfa"
    PFA = "pfa"


KT_KNOWLEDGE_PARAM_METHODS = {
    "worc": ["constant", "expert", "ep"],
    "wrc": ["rc", "ep_rc"],
}
KT_PERFORMANCE_PARAM_METHODS = {
    "constant": ["constant"],
    "expert": ["expert_skill", "expert_prob"],
    "rc": ["rc", "expert_skill", "expert_prob"],
    "ep": ["ep"],
    "ep_rc": ["ep_rc"],
}
PFA_BETA = ["item", "skill"]
