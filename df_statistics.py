import pandas as pd
from collections import Counter
import string
from math import sqrt
from gensim.models import FastText
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.stats import entropy
# from sklearn.metrics import jaccard_score


class ItemsCounter(object):
    """
    ItemsCounter facilitates characters counter for statistical features.
    """
    def __init__(self):
        self._items_dict = dict.fromkeys(string.ascii_letters, 0)
        self._items_dict.update(dict.fromkeys(string.digits, 0))

    def add_key_value(self, key, value):
        self._items_dict[key] = value

    def update_key(self, key, value):
        try:
            self._items_dict[key] = self._items_dict[key] + value
        except KeyError:
            self.add_key_value(key, value)

    def update_with_dict(self, counter_dict):
        for key, value in counter_dict.items():
            self.update_key(key, value)

    def get_dict(self):
        return self._items_dict

    @staticmethod
    def count_chars(df):
        frames = []
        d = pd.DataFrame()
        for col in df:
            d[col] = df[col].apply(lambda x: Counter(str(x)))
        d_sum = d.sum()
        for i in range(0, len(d_sum)):
            row_df = pd.DataFrame.from_dict(d_sum[i], orient='index')
            row_df.columns = [d_sum.index[i]]
            frames.append(row_df)
        final_df = pd.concat(frames, axis=1).fillna(0)
        return final_df

    @staticmethod
    def count_chars_naive(df):
        cols_chars_counter_dict = cols_dict_preprocess(df)
        for row in df.iterrows():
            for item in row[1].iteritems():
                if item:
                    counter_dict = dict(Counter(str(item[1])))
                    cols_chars_counter_dict[item[0]].update_with_dict(counter_dict)
        new_dict = {}
        for dict_el_key, dict_el_val in cols_chars_counter_dict.items():
            new_dict[dict_el_key] = dict_el_val.get_dict()
        cols_chars_counter_df = pd.DataFrame.from_dict(new_dict).fillna(0)
        return cols_chars_counter_df


def cols_dict_preprocess(df):
    cols_dict = {}
    for col in df.columns:
        cols_dict[col] = ItemsCounter()
    return cols_dict


class Statistics:
    @staticmethod
    def ent(data):
        """Calculates entropy of the passed `pd.Series`
        """
        p_data = data.value_counts()  # counts occurrence of each value
        return entropy(p_data, base=2), len(data.value_counts().keys())

    """
    Numeric and textual data statistics functions
    """
    @staticmethod
    def numeric_stats(numeric_df):
        stats_df = pd.concat([numeric_df.max(),
                              numeric_df.min(),
                              numeric_df.mean(),
                              numeric_df.median(),
                              numeric_df.std(),
                              numeric_df.skew(),
                              numeric_df.kurtosis(),
                              # numeric_df.var(),
                    ], axis=1)
        stats_df.columns = ['max', 'min', 'mean', 'median', 'std', 'skew', 'kurt']
        stats_df = stats_df.T
        stats_dict = {}
        for col in numeric_df:
            entropy, unique_vals = Statistics.ent(pd.Series(numeric_df[col]))
            stats_dict[col] = {'entropy': entropy,
                               'unique': unique_vals}
        stats_df = pd.concat([stats_df, pd.DataFrame(stats_dict)])
        return stats_df

    @staticmethod
    def txt_stats(object_df):
        if object_df.empty:
            print("DataFrame is empty!")
            return object_df
        char_count_feats_df = ItemsCounter.count_chars(object_df)
        stats_dict = {}
        for col in object_df:
            entropy, unique_vals = Statistics.ent(pd.Series(object_df[col]))
            col_dict = {'entropy': entropy,

                        'max_len': object_df[col].apply(lambda x: len(str(x))).max(),
                        'min_len': object_df[col].apply(lambda x: len(str(x))).min(),
                        'mean_len': object_df[col].apply(lambda x: len(str(x))).mean(),
                        'unique': unique_vals
                        }
            stats_dict[col] = col_dict
        stats_df = pd.DataFrame(stats_dict)
        final_df = pd.concat([char_count_feats_df, stats_df])
        # final_df = stats_df
        return final_df

    @staticmethod
    def sq_rooted(x):
        return round(sqrt(sum([a * a for a in x])), 3)

    @staticmethod
    def cos_sim(x, y):
        epsilon = 0.000001
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = Statistics.sq_rooted(x) * Statistics.sq_rooted(y)
        try:
            ans = round(numerator / float(denominator), 3)
        except ZeroDivisionError:
            return round(numerator / float(epsilon), 3)  # in case all column values are 0
        return ans

    @staticmethod
    def jaccard_similarity(x, y):
        # intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        # union_cardinality = len(set.union(*[set(x), set(y)]))
        # return intersection_cardinality / float(union_cardinality)
        # return jaccard_score(x, y, average='weighted')
        sum1 = 0
        sum2 = 0
        for (a, b) in zip(x, y):
            sum1 += min(a, b)
            sum2 += max(a, b)
        return sum1/sum2

    @staticmethod
    def similarity(similarity_type, x, y):
        if similarity_type == 'cosine':
            return Statistics.cos_sim(x, y)
        if similarity_type == 'jaccard':
            return Statistics.jaccard_similarity(x, y)
        if similarity_type == 'pearson':
            return pearsonr(x, y)[0]

    @staticmethod
    def eucl_dist(x, y):
        return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    @staticmethod
    def fasttext_embed(df, dim, agg_function):
        if df.empty:
            return
        emb_df = pd.DataFrame()
        df_as_lists = df.T.values.tolist()
        # for col in df.columns:
        #     model = FastText(size=4, window=3, min_count=1)  # instantiate
        #     model.build_vocab(sentences=df[col].tolist())
        #     model.train(sentences=df[col], total_examples=1, epochs=10)
        df_as_lists = list(map(lambda x: list(map(str, x)), df_as_lists))
        fmodel = FastText(size=dim, window=3, min_count=1)  # instantiate
        fmodel.build_vocab(sentences=df_as_lists)
        fmodel.train(sentences=df_as_lists, total_examples=len(df_as_lists), epochs=10)
        for col in df.columns:
            col_embds = fmodel.wv[df[col]]
            if agg_function == 'mean':
                col_mean_embds = np.mean(col_embds, axis=0)
            emb_df[col] = col_mean_embds
        return emb_df






