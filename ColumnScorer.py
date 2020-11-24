from sklearn import feature_selection
from sklearn import tree


class ColumnScorer:

    def __init__(self, fs_algo, dt_classifier_criterion=None, dt_regressor_criterion=None):
        self._fs_alfo = fs_algo
        self._dt_classifier_criterion = dt_classifier_criterion
        self._dt_regressor_criterion = dt_regressor_criterion
        self._fs_algoritms = {
            'tree': self.decision_tree_feature_importance,
            'mutual_info': self.mutual_info_feature_selection,
            'f_val': self.f_value_feature_selection,
            'chi2': self.chi2_feature_selection,
            'rfe_decision_tree': self.rfe_decision_tree_feature_selection
        }

    def score_columns(self, X_ohe, unq_for_categorical, y, y_enc, y_is_str_dtype, bgu_th):
        return self._fs_algoritms[self._fs_alfo](X_ohe, unq_for_categorical, y, y_enc, y_is_str_dtype, bgu_th)

    def set_ft_algo(self, fs_algo):
        self._fs_alfo = fs_algo

    def set_dt_classifier_criterion(self, dt_classifier_criterion):
        self._dt_classifier_criterion = dt_classifier_criterion

    def set_dt_regressor_criterion(self, dt_regressor_criterion):
        self._dt_regressor_criterion = dt_regressor_criterion

    def rfe_decision_tree_feature_selection(self, X_ohe, unq_for_categorical, y, y_enc, y_is_str_dtype):
        if y_is_str_dtype and y.nunique() < unq_for_categorical:
            clf = tree.DecisionTreeClassifier()
        else:
            clf = tree.DecisionTreeRegressor()
        selector = feature_selection.RFE(clf)
        selector = selector.fit(X_ohe, y_enc)

        return selector.score()

    def mutual_info_feature_selection(self, X_ohe, unq_for_categorical, y, y_enc, y_is_str_dtype):
        if y_is_str_dtype and y.nunique() < unq_for_categorical:
            clf = feature_selection.mutual_info_classif(X_ohe, y_enc)
        else:
            clf = feature_selection.mutual_info_regression(X_ohe, y_enc)
        return clf

    def chi2_feature_selection(self, X_ohe, unq_for_categorical, y, y_enc, y_is_str_dtype):
        return feature_selection.chi2(X_ohe, y_enc)

    def f_value_feature_selection(self, X_ohe, unq_for_categorical, y, y_enc, y_is_str_dtype):
        if y_is_str_dtype and y.nunique() < unq_for_categorical:
            clf = feature_selection.f_classif(X_ohe, y_enc)
        else:
            clf = feature_selection.f_regression(X_ohe, y_enc)
        return clf[0]

    def decision_tree_feature_importance(self, X_ohe, unq_for_categorical, y, y_enc, y_is_str_dtype, bgu_th):
        if y_is_str_dtype and y.nunique() < unq_for_categorical:
            clf = tree.DecisionTreeClassifier(criterion=self._dt_classifier_criterion, min_impurity_decrease=bgu_th)
        else:
            clf = tree.DecisionTreeRegressor(criterion=self._dt_regressor_criterion, min_impurity_decrease=bgu_th)
        clf = clf.fit(X_ohe, y_enc)
        return clf.feature_importances_
