# Model Card

## Model Details
This model is a Decision Tree classifier trained to predict whether an individual's income exceeds $50,000 per year based on demographic and employment-related attributes from the U.S. Census dataset. The model was implemented using scikit-learn’s DecisionTreeClassifier and trained using the processed census.csv dataset. The classifier is wrapped in a class called CensusClassifier, which includes preprocessing, training, inference, and evaluation methods.
## Intended Use
The intended use of this model is for educational and research purposes, specifically for demonstrating supervised classification workflows in machine learning, including model training, evaluation, and fairness assessment across population subgroups. It is not intended for production use or real-world decision-making, particularly not in areas such as hiring, lending, or any high-stakes applications involving personal data.
## Training Data
The model was trained on a subset of the U.S. Census dataset, specifically the "Adult" dataset. The dataset includes demographic and employment features such as:

age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, and salary (target).

The categorical features were encoded using a OneHotEncoder, and the target variable (salary) was label-encoded.
## Evaluation Data
The evaluation data consists of 20% of the full dataset, held out from training using a random train-test split. This test set was used to assess the model’s generalization performance.
## Metrics
The model was evaluated using the following metrics:

F1 Score: Measures the harmonic mean of precision and recall.

Precision: Measures the proportion of positive predictions that are correct.

Overall Performance
F1 Score: 0.6228

Precision: 0.625

Performance by Workclass (Selected Slice)
Private: F1 Score: 0.6265, Precision: 0.6356

Self-emp-not-inc: F1 Score: 0.5744, Precision: 0.5533

State-gov: F1 Score: 0.6202, Precision: 0.5797

Federal-gov: F1 Score: 0.5652, Precision: 0.5821

Without-pay: F1 Score: 0.0, Precision: 0.0

Never-worked: F1 Score: 0.0, Precision: 0.0

Significant variance in performance across workclass categories was observed, indicating the importance of evaluating fairness and robustness across subgroups.
## Ethical Considerations
This model is trained on historical census data which may contain biases and imbalances reflective of social and economic inequalities.

Differential performance across demographic groups (e.g., race, sex, native-country) was observed, which could lead to disparate outcomes if used in real-world decision-making.

The dataset may include underrepresented groups, for which the model performance may be poor (e.g., groups with F1 score = 0.0).

The model should not be used for real-world applications without extensive fairness analysis, bias mitigation, and stakeholder input.
## Caveats and Recommendations
The model has limited complexity (Decision Tree) and may not generalize as well as more sophisticated classifiers.

The evaluation is limited to F1 and precision; recall, ROC-AUC, and calibration should be considered for deeper analysis.

The model’s performance varies significantly across data slices, suggesting the need for bias mitigation strategies.

Further hyperparameter tuning and cross-validation could improve overall and subgroup performance.

It is recommended to retrain and reevaluate the model regularly if used beyond academic purposes, especially if the data distribution changes.