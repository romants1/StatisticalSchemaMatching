from scipy.stats import entropy
import numpy as np
from sklearn import preprocessing
import pandas as pd
import pandas.api.types as ptypes
from configparser import ConfigParser
import sklearn.metrics as metrics
from main_stat_sm import parse_scores
import ColumnScorer
import os
import ast
import time
import math


def calc_modified_dcg(items):
    dcg = 0
    i = 0
    for item in items:
        i += 1
        dcg += (math.pow(2, item) - 1) / math.log(1 + i, 2)
    return dcg


def calc_dcg(items):
    dcg = 0
    i = 0
    for item in items:
        i += 1
        dcg += item / math.log(i + 1, 2)
    return dcg


def filter_columns(df, unq_for_filter):
    for col in df:
        if (ptypes.is_string_dtype(df[col]) and df[col].nunique() > unq_for_filter) or \
                (np.count_nonzero(pd.isnull(df[col])) / len(df[col]) > 0.95):
            del df[col]


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_cols_scores(cols_scores, num_cols, df):
    scores = {}
    for col in df:
        if col in num_cols:
            scores[col] = cols_scores[col]
            del cols_scores[col]
        else:
            scores_sum = 0
            for col_name, score in cols_scores.items():
                if col in col_name:
                    scores_sum = scores_sum + score
            scores[col] = scores_sum
    return scores


def split_str_numeric_cols(df):
    str_cols = []
    num_cols = []
    for col in df:
        if ptypes.is_string_dtype(df[col]):
            str_cols.append(col)
        elif ptypes.is_numeric_dtype(df[col]):
            num_cols.append(col)
    return str_cols, num_cols


def get_table_with_fk(df_1, df_2, column_name1, column_name2):
    col1 = df_1[column_name1].value_counts()
    col2 = df_2[column_name2].value_counts()
    ent1 = entropy(col1)
    ent2 = entropy(col2)
    max_ent1 = entropy(pd.Series(list(range(len(df_1[column_name1])))).value_counts())
    max_ent2 = entropy(pd.Series(list(range(len(df_2[column_name2])))).value_counts())
    if ent1 == max_ent1:
        return df_2, df_1, column_name2, column_name1
    elif ent2 == max_ent2:
        return df_1, df_2, column_name1, column_name2
    elif entropy(col1) > entropy(col2):
        return df_2, df_1, column_name2, column_name1
    return df_1, df_2, column_name1, column_name2


def add_table_name_to_col(df, name):
    new_cols = []
    for col in df:
        new_cols.append(name + '.' + col)
    df.columns = new_cols


def prepare_data(df_1, df_2, fk_pair, table1, table2, sample_size):
    column_name1, column_name2 = fk_pair
    add_table_name_to_col(df_1, table1)
    add_table_name_to_col(df_2, table2)
    df_fk, df_pk, df_fk_column, df_pk_column = get_table_with_fk(df_1, df_2, column_name1, column_name2)
    df_sample = df_fk.sample(frac=sample_size)
    return pd.merge(df_sample, df_pk, how='inner', left_on=[df_fk_column], right_on=[df_pk_column])


def save_results(results_folder, experiment, scores_folder, result, sample_size, ap, fs_algo, end_time, ndcg,
                 base_thresh, bgu_thresh, accuracy, precision, recall, f1, tn, fp, fn, tp, unq_for_filter):
    fs_algo = fs_algo + '_{}_{}'.format(conf['DEFAULT']['DTC_CRITERION'],
                                        conf['DEFAULT']['DTR_CRITERION']) if fs_algo == 'tree' else fs_algo
    path = os.path.join(results_folder, experiment)
    create_folder(path)
    save_scores(path, result, sample_size, fs_algo, base_thresh, bgu_thresh, scores_folder)
    res_df = pd.DataFrame().from_dict(
        {'base_threshold': [base_thresh], 'bgu_threshold': [bgu_thresh], 'sample': [sample_size],
         'feature_selection_algo': [fs_algo], 'time': [end_time], 'avg_precision': [ap], 'ndcg': [ndcg],
         'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': f1, 'tn': tn, 'fp': fp, 'fn': fn,
         'tp': tp})
    if not os.path.isfile(
            os.path.join(path, 'columns_scores', scores_folder, 'results_unique_{}.csv'.format(unq_for_filter))):
        res_df.to_csv(
            os.path.join(path, 'columns_scores', scores_folder, 'results_unique_{}.csv'.format(unq_for_filter)),
            index=False)
    else:
        res_df.to_csv(
            os.path.join(path, 'columns_scores', scores_folder, 'results_unique_{}.csv'.format(unq_for_filter)),
            mode='a', header=False, index=False)


def save_scores(path, result, sample_size, fs_algo, base_thresh, bgu_thresh, scores_folder):
    create_folder(os.path.join(path, 'columns_scores', scores_folder))
    column_scores = pd.DataFrame().from_dict(result).T
    # column_scores = pd.read_csv(path, index_col=0)
    # column_scores = column_scores.div(column_scores.sum(axis=1), axis=0)
    list_dicts = []
    for col in column_scores.columns:
        for idx in column_scores.index:
            pair = {'col1': idx, 'col2': col, 'dt_models_score': column_scores.loc[idx][col]}
            list_dicts.append(pair)
    edge_list_df = pd.DataFrame(list_dicts)
    edge_list_df.to_csv(os.path.join(path, 'columns_scores', scores_folder,
                                     'df_left_predict_score_{}_{}_base_thresh{}_bgu_thresh{}.csv'
                                     .format(sample_size, fs_algo, base_thresh, bgu_thresh)), index=False)


def match_columns(df, unq_for_categorical, fs_algo, unq_for_filter, bgu_th=0):
    res = {}
    filter_columns(df, unq_for_filter)
    for i, column in enumerate(df, start=1):
        y = df[column]
        X = df.drop(columns=[column])
        str_columns, num_columns = split_str_numeric_cols(X)
        X_ohe = pd.get_dummies(X.fillna(-1))
        scores_with_ohe = get_scores(X_ohe, unq_for_categorical, y, fs_algo, bgu_th)
        scores = calc_cols_scores(dict(zip(X_ohe.columns, scores_with_ohe)), num_columns, X)
        res[column] = scores
        print('\rFinished with column {} of {}'.format(i, len(df.columns)), end='')
    print('')
    return res


def get_scores(X_ohe, unq_for_categorical, y, fs_algo, bgu_th):
    le = preprocessing.LabelEncoder()
    y_is_str_dtype = ptypes.is_string_dtype(y)
    le.fit(y.fillna('') if y_is_str_dtype else y.fillna(-1))
    y_enc = le.transform(y.fillna('') if y_is_str_dtype else y.fillna(-1))
    col_scorer = ColumnScorer.ColumnScorer(fs_algo, conf['DEFAULT']['DTC_CRITERION'], conf['DEFAULT']['DTR_CRITERION'])
    return col_scorer.score_columns(X_ohe, unq_for_categorical, y, y_enc, y_is_str_dtype, bgu_th)


def calculate_metrics(base_scores, bg_scores, fk_pair, table1, table2, bgu_th):
    base_scores = {(a, b): c for (a, b, c) in base_scores}
    base_scores[fk_pair] = 1
    base_scores[tuple(reversed(fk_pair))] = 1
    combined_scores = merge_bgu_base_scores(base_scores, bg_scores, table1, table2, bgu_th)
    res_df = pd.DataFrame(combined_scores).fillna(0)
    avg_precision = metrics.average_precision_score(np.array(res_df[5]), np.array(res_df[2]))
    ndcg = calc_dcg(np.array(res_df[4]).astype(float)) / calc_dcg(np.array(res_df[2]).astype(float))
    accuracy = metrics.accuracy_score(np.array(res_df[5]), np.array(res_df[3]))
    precision = metrics.precision_score(np.array(res_df[5]), np.array(res_df[3]))
    recall = metrics.recall_score(np.array(res_df[5]), np.array(res_df[3]))
    f1 = metrics.f1_score(np.array(res_df[5]), np.array(res_df[3]))
    tn, fp, fn, tp = metrics.confusion_matrix(np.array(res_df[5]), np.array(res_df[3])).ravel()
    return avg_precision, ndcg, accuracy, precision, recall, f1, tn, fp, fn, tp


def merge_bgu_base_scores(base_scores, bg_scores, table1, table2, bgu_th):
    combined_scores = []
    col_passed = {}
    for index, row in bg_scores.iterrows():
        for col, val in row.iteritems():
            col1, col2 = (index, col) if index < col else (col, index)
            col_passed[(col1, col2)] = val if (col1, col2) not in col_passed else max(val, col_passed[(col1, col2)])

    for key, val in col_passed.items():
        col1, col2 = key
        bg_score = 0 if ((table1 in col1 and table1 in col2) or (table2 in col1 and table2 in col2)) else val
        base_score_bool = 1 if ((col1, col2) in base_scores and base_scores[(col1, col2)] > 0) or (
                (col2, col1) in base_scores and base_scores[(col2, col1)] > 0) else 0
        base_score_val = base_scores[(col1, col2)] if (col1, col2) in base_scores else 0
        base_score_val = base_scores[(col2, col1)] if (col2, col1) in base_scores else base_score_val
        bg_score_bool = 1 if bg_score > bgu_th else 0
        combined_scores.append((col1, col2, bg_score, bg_score_bool, base_score_val, base_score_bool))
    return combined_scores


def apply_threshold(col_scores, bgu_th):
    res = {}
    for col1, scores_dict in col_scores.items():
        col1_scores = {}
        for col2, val in scores_dict.items():
            col1_scores[col2] = 0.0 if val < bgu_th else val
        res[col1] = col1_scores
    return res


def get_base_scores(data_folder, scores_folder, base_th, exp):
    if scores_folder == 'amdocs_scores':
        f = open(os.path.join(data_folder, scores_folder, '{}.log'.format(round(base_th, 1))), 'r')
        return [(a, b, float(c)) for (a, b, c) in parse_scores(f.read().splitlines(True))]
    else:
        df = pd.read_csv(os.path.join(data_folder, scores_folder, exp + '.csv'))
        df_tuples = list(df.itertuples(index=False, name=None))
        return list(map(lambda x: x if x[2] > base_th else (x[0], x[1], 0), df_tuples))


def main(data_folder, experiments, unq_for_categorical, result_dir, sample_sizes_list, scores_folder, fs_algo,
         thresh_list, out_thresh_type, unq_for_filter):
    for exp in experiments:
        for sample_size in sample_sizes_list:
            start_time = time.time()
            exp = exp.strip()
            fk_pairs = ast.literal_eval(conf['DEFAULT']['FK_PAIRS'])[exp]
            table1, table2 = exp.split('~')
            df1 = pd.read_csv(os.path.join(data_folder, table1 + '.csv'), low_memory=False)
            df2 = pd.read_csv(os.path.join(data_folder, table2 + '.csv'), low_memory=False)
            res_df = prepare_data(df1, df2, fk_pairs[0], table1, table2, sample_size)
            if out_thresh_type == 'out_threshold':
                column_matching(data_folder, exp, fk_pairs, fs_algo, res_df, result_dir, sample_size, scores_folder,
                                start_time, table1, table2, thresh_list, unq_for_categorical, unq_for_filter)
            elif out_thresh_type == 'impurity_threshold':
                column_matching_with_impurity_thresh(data_folder, exp, fk_pairs, fs_algo, res_df, result_dir,
                                                     sample_size, scores_folder, start_time, table1, table2,
                                                     thresh_list, unq_for_categorical, unq_for_filter)


def column_matching(data_folder, exp, fk_pairs, fs_algo, res_df, result_dir, sample_size, scores_folder, start_time,
                    table1, table2, thresh_list, unq_for_categorical, unq_for_filter):
    col_scores = match_columns(res_df, unq_for_categorical, fs_algo, unq_for_filter)
    end_time = time.time() - start_time
    for bgu_th in thresh_list:
        for base_th in np.arange(0.0, 1, 0.1):
            updated_scores = apply_threshold(col_scores, bgu_th)
            base_scores = get_base_scores(data_folder, scores_folder, base_th, exp)
            ap, ndcg, accuracy, precision, recall, f1, tn, fp, fn, tp = calculate_metrics(
                base_scores, pd.DataFrame().from_dict(updated_scores), fk_pairs[0], table1, table2, bgu_th)
            save_results(result_dir, exp, scores_folder, updated_scores, sample_size, ap, fs_algo, end_time,
                         ndcg, base_th, bgu_th, accuracy, precision, recall, f1, tn, fp, fn, tp, unq_for_filter)
    print('Finished sample {} after {} seconds'.format(sample_size, time.time() - start_time))


def column_matching_with_impurity_thresh(data_folder, exp, fk_pairs, fs_algo, res_df, result_dir, sample_size,
                                         scores_folder, start_time, table1, table2, thresh_list, unq_for_categorical,
                                         unq_for_filter):
    for bgu_th in thresh_list:
        col_scores = match_columns(res_df, unq_for_categorical, fs_algo, unq_for_filter, bgu_th)
        end_time = time.time() - start_time
        for base_th in np.arange(0.0, 1, 0.1):
            base_scores = get_base_scores(data_folder, scores_folder, base_th, exp)
            ap, ndcg, accuracy, precision, recall, f1, tn, fp, fn, tp = calculate_metrics(
                base_scores, pd.DataFrame().from_dict(col_scores), fk_pairs[0], table1, table2, bgu_th)
            save_results(result_dir, exp, scores_folder, col_scores, sample_size, ap, fs_algo, end_time,
                         ndcg, base_th, bgu_th, accuracy, precision, recall, f1, tn, fp, fn, tp, unq_for_filter)
    print('Finished sample {} after {} seconds'.format(sample_size, time.time() - start_time))


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')
    main(conf['DEFAULT']['DATA_FOLDER'], conf['DEFAULT']['EXPERIMENTS'].split(','),
         int(conf['DEFAULT']['UNIQUE_VALS_FOR_CATEGORICAL']), conf['DEFAULT']['RESULTS_DIR'],
         list(map(lambda x: float(x), conf['DEFAULT']['RATIO_SAMPLE_SIZES'].split(','))),
         conf['DEFAULT']['SCORES_FOLDER'], conf['DEFAULT']['FEATURE_SELECTION'],
         list(map(lambda x: float(x), conf['DEFAULT']['THRESH_LIST'].split(','))),
         conf['DEFAULT']['OUT_THRESH_TYPE'], eval(conf['DEFAULT']['UNIQUE_VALS_FOR_FILTER']))
