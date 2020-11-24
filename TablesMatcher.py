import os
import pandas as pd
import random
from df_statistics import Statistics


def find_cat_columns(df, _unq_vals_for_categorical):
    """
    Find the categorical columns of the dataframe
    :param df: The dataframe to check
    :return:
    """
    likely_cat = []
    for var in df.columns:
        if df[var].nunique() <= _unq_vals_for_categorical:
            likely_cat.append(var)
    return likely_cat


def cat_noncat_sort(df, _unq_vals_for_categorical):
    likely_cat = find_cat_columns(df, _unq_vals_for_categorical)
    try:
        df_cat = df[likely_cat]
        df_noncat = df[df.columns[~df.columns.isin(likely_cat)]]
    except KeyError:
        print('Something not ok with categorical filtering, check')
    return df_cat, df_noncat


class Schema:
    """
    Schema definition
    """
    def __init__(self):
        self._table = None
        self._name = None

    def set_from_path(self, path, sample_size, ratio_sampling):
        self._table = pd.read_csv(
            path,
            header=0,
            skiprows=lambda i: i > 0 and random.random() > float(sample_size),
            encoding='latin1', low_memory=False) if ratio_sampling.lower() == 'true'else pd.read_csv(
            path,
            header=0,
            nrows=float(sample_size),
            encoding='latin1', low_memory=False)
        self._name = os.path.basename(os.path.splitext(path)[0])
        self._table.columns = [self._name + '.' + str(col) for col in self._table.columns]
        self._table = self._table.applymap(lambda x: x.lower() if type(x) is str else x)

    def get_table(self):
        return self._table

    def num_of_fields(self):
        return len(self._table.columns)

    def sample(self, frac):
        self._table = self._table.sample(frac=float(frac), random_state=1)

    def copy(self):
        schema_cp = Schema()
        schema_cp._table = self._table.copy()
        schema_cp._name = self._name
        return schema_cp


class Matcher:
    """
    Matcher definition class. Defines the column types (categorical, textual, numeric).
    Finds the columns correlations within the same type according to specific statistics.
    Finds the potential columns for FK detection.
    """
    def __init__(self, *args):
        self._thresh: float = args[0]
        self._similarity_type = args[1]
        self._potential_FK = {}

    def find_match(self, df_t1, df_t2):
        """
        Find the corresponding columns
        :param df_t1: table 1
        :param df_t2: table 2
        :return: potentially corresponding columns
        """
        cos_sim = {}
        for col1 in df_t1:
            cos_sim_col1 = {}
            for col2 in df_t2:
                # 1 - cosine(df_t1[col1], df_t2[col2])
                cos_sim_col1[col2] = Statistics.similarity(self._similarity_type, df_t1[col1], df_t2[col2])
            cos_sim[col1] = cos_sim_col1
        cos_sim_df = pd.DataFrame.from_dict(cos_sim)
        potential_match = cos_sim_df[cos_sim_df >= self._thresh]
        return potential_match

    def stats_match(self, tables1, tables2):
        """

        :return:
        """
        potential_num_match = self.get_potetial_match(tables1[0], tables2[0])
        potential_txt_match = self.get_potetial_match(tables1[1], tables2[1])
        # potential_cat_match = self.get_potetial_match(tables1[2], tables2[2])
        # return potential_num_match, potential_txt_match, potential_cat_match
        return potential_num_match, potential_txt_match

    def get_potetial_match(self, stats_df_t1, stats_df_t2):
        """

        :param data_statistic_fn: the statistics function to apply
        :param schema1: table 1
        :param schema2: tablw 2
        :return: the columns with potential match
        """
        # stats_df_t1 = data_statistic_fn(schema1)
        # stats_df_t2 = data_statistic_fn(schema2)
        stats_intersect = set(stats_df_t1.index).intersection(stats_df_t2.index)
        stats_df_t1 = stats_df_t1.T[list(stats_intersect)].T
        stats_df_t2 = stats_df_t2.T[list(stats_intersect)].T
        potential_txt_match = self.find_match(stats_df_t1, stats_df_t2)
        return potential_txt_match

    # def description_match(self):
    #     self._schema1.describe()
    #     self._schema2.describe()
        #TODO what to do with count, unique top, freq?

    # def intersection_match(self):
    #     for col1 in self._schema1 :
    #         for col2 in self._schema2:
    #             if len(set(self._schema1[col1]).intersection(set(self._schema2[col2]))) \
    #             / self._schema1[col1].nunique() >= self._thresh:
    #                 self._potential_FK['table1'] = col1
    #                 self._potential_FK['table1'] = col2

    # def embedding_match(self, dim=4):
    #     sc1_emb = self.__fasttext_embed(self._schema1, dim, agg_function='mean')
    #     sc2_emb = self.__fasttext_embed(self._schema2, dim, agg_function='mean')
    #     return  sc1_emb, sc2_emb




