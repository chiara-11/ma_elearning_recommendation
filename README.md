# Personalized Exercise Recommendation for Small Group Sizes in e-Learning

Master Thesis by Chiara Purkl  
Submitted 02.09.2024

## Contributing

To contribute, clone the project via the command window:
`git clone git@gitlab.com:chiara-11/elearning-recommender.git`

On Windows the workflow is as follows:  
Ensure that python version `3.11.5` is used by executing `pyenv install 3.11.5` and `pyenv local 3.11.5`.
Initialize the project by creating and activating the virtual environment in the terminal: `python -m venv .venv` and `.venv\Scripts\activate`.
Load the required dependencies using `pip install -r requirements.txt`.

## Data

The data set was introduced as part of the EDM Cup 2023 and can be downloaded here: https://www.kaggle.com/competitions/edm-cup-2023/overview.
It contains data from the ASSISTments online learning platform. On this platform, students from different grades solve math problems in order to complete their assignments.

After downloading the data, all files should be placed in the folder `data`. This is required for executing the code.

## Project Structure

### Data Preparation

All files related to data preparation are stored in the folder `data_preparation`.
The data needed for executing the experiments (additionally to the data provided on https://www.kaggle.com/competitions/edm-cup-2023/data) is already stored in the `data` folder.

The files have been created according to the following procedure:

- The notebook `1_prepare_data_general.ipynb` contains the first and main part of the data preparation process. It generates the data set `data_matrix.csv` and stores it in the `data` folder. This notebook refers to Section 4.4
- Subsequently, the notebook `2_prepare_data_main_approach.ipynb` should be executed to further restrict the data. The file `final_data_main_approach.csv` is created and stored in the `data` folder. This notebook completes Section 4.4
- As some experiments are only performed for a restricted set of reference classes, the notebook `3_restrict_reference_classes.ipynb` should be executed. This produces two data files:
  - `ref_classes_restricted_random.csv` containing only one random reference class per class and UT sequence
  - `ref_classes_restricted` containing a restricted choice of reference classes per class and UT sequence (further details can be found in the notebook)
- Some approaches require the use of expert values. Those are created in the notebook `4_create_expert_values.ipynb` according to the descriptions in Section 5.1.
- There are two more notebooks in the folder `data_preparation` which do not store any data files but illustrate what is done in the corresponding python files.

### Experiments

Predictions are made for each UT assignment. A UT assignment can be uniquely identified by a class, UT sequence, student combination.
The implementation determines the following procedure for the experiments:

- Iterate over all target classes cid.
- Iterate over all UT sequences ts for cid.
- Iterate over all students stud in cid that completed ts.
- If reference classes are used: Iterate over all possible or the restricted reference classes for the stud, the ts, and the cid.

This leads to the individual test cases for which the predictions are performed.

The general structure of the experiments is stored in the following files in the `source` folder:

- `training_general.py`: functions related to both WRC and WORC experiments
- `training_with_rc.py`: functions that are specific to WRC experiments
- `training_without_rc.py`: functions that are specific to WORC experiments

For each method, there is a folder containing the following files:

- A `.py` file containing all functions related to the specific method, which in particular holds the implementation for all approaches described in the respective subsection of Section 5.1 of the report.
- A jupyter notebook (`.ipynb`) to execute the experiments and evaluate the predictions.
  - The experiments are separated into several versions, where each version performs a subset of the experiments.
  - If experiments with reference classes are performed, the experiments for each version are further separated into four parts, each performing the experiments specific to the version for a subset of the target classes.
- Note: For KT, the experiments are separated into BKT and LKT with corresponding notebooks and `.py` files and one file `kt.py` containing functions related to both.

In addition, there is a folder `baseline` containing the computation of the baseline predictions according to the descriptions in Subsection 5.1.1 of the report.

#### Saving Prediction Results

- The predictions are stored in the `results` folder, where there is one file for each version (and part in case of WRC), named after the same.
- The predictions are also evaluated in the respective notebooks. In doing so, one file is created for each approach (and part in case of WRC), named after the same. Those are also stored in the `results` folder.

Although we only used the accuracy and f1 score in the report, for most approaches further metric values are stored:

- MAE, MSE (based on the predicted probabilities)
- Accuracy, F1 Score, Recall and Precision for different thresholds (0.3, 0.5, 0.7, "dynamical") (details can be found in the report)

In the report we only used accuracy and f1 score with a threshold of 0.5.

### Comparison of the Methods

- All files related to the comparison and discussion of the approaches and methods can be found in the folder `evaluation_comparison`.
