import ast
import os
import time
import pandas as pd
import logging
import TablesMatcher
import configparser
from df_statistics import Statistics


def create_dirs(*dirs):
    """
    Create not existing directories
    :param dirs: directories to create
    :return:
    """
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def print_to_file(sample_size, thresh, potent_sim_lists):
    path_ft_sample = os.path.join(path_ft, 'sample{}'.format(sample_size))
    path_ft_thresh = os.path.join(path_ft_sample, 'threshold{}'.format(thresh.strip()))
    path_pm_sample = os.path.join(path_pm, 'sample{}'.format(sample_size))
    create_dirs(path_ft_sample, path_pm_sample, path_ft_thresh)

    name1 = schema1name.split('csv')[0]
    name2 = schema2name.split('csv')[0]
    list_matches = list(zip(*potent_sim_lists))
    list_matches = list(map(lambda x: x.split(name1)[1] if name1 in x else x.split(name2)[1],
                            set(list_matches[0] + list_matches[1])))
    schemapath1 = os.path.join(data_folder, schema1name)
    schemapath2 = os.path.join(data_folder, schema2name)
    df1 = pd.read_csv(schemapath1, usecols=lambda c: c in list_matches, low_memory=False)
    df2 = pd.read_csv(schemapath2, usecols=lambda c: c in list_matches, low_memory=False)
    df1.to_csv(os.path.join(path_ft_thresh, '{}'.format(schema1name)), index=False)
    df2.to_csv(os.path.join(path_ft_thresh, '{}'.format(schema2name)), index=False)

    with open(os.path.join(path_pm_sample, 'potential_matches_{}_{}_{}.csv'.
            format(similarity_type, thresh, exp_name)), 'w') as f2:
        for t in potent_sim_lists:
            f2.write(', '.join(str(s) for s in t) + '\n')


# def generate_samples(sample_size):
#     """
#     Generate sample from the full data. Saved the sampled data sets.
#     :param schema1path: table1
#     :param schema2path: table2
#     :param sample_size: the sample size to generate
#     :return: the sampled tables
#     """
#     schema1 = get_sample(sample_size, dict_tables[schema1name][0], schema1name)
#     schema2 = get_sample(sample_size, dict_tables[schema2name][0], schema2name)
#     return schema1, schema2
#
#
# def get_sample(sample_size, schema, schema_name):
#     # schema = TablesMatcher.Schema()
#     # schema.set_from_path(schema1path)
#     if sample_size == 1:
#         return schema
#     schema.sample(sample_size)
#     path = os.path.join(data_folder, 'samples', 'sample{}'.format(sample_size), schema_name)  # create func
#     if not os.path.exists(os.path.dirname(path)):
#         os.makedirs(os.path.dirname(path))
#     schema.get_table().to_csv(path, index=False)
#     return schema


def config_table():
    """
    Update the configurations according to curent experiment
    :return:
    """
    global schema1name, schema2name, fk_pairs, results_dir, path_ft, path_pm, final_results_path
    schema1name, schema2name = exp_name.split('~')
    schema1name += '.csv'
    schema2name += '.csv'
    try:
        fk_pairs = ast.literal_eval(config['DEFAULT']['FK_PAIRS'])[exp_name]
    except KeyError:
        fk_pairs = []
    results_dir = config['DEFAULT']['RESULTS_DIR'] + '/' + exp_name
    path_ft = os.path.join(results_dir, 'filtered_tables')
    path_pm = os.path.join(results_dir, 'potential_matches')
    final_results_path = os.path.join(results_dir, 'comparisons')

    # Create directories
    create_dirs(path_pm, final_results_path)


def preprocess():
    global experiments
    """
    Preprocess data source tables:
        - Split the tables according to the data type
        - Retrieve statistics for each table
    :param experiment: the pairs of tables to process according to the config file
    :return:
    """
    experiments = list(map(lambda x: x.strip(), experiments))
    # tables_names = []
    # for key in schema_names:
    #     if key in experiments:
    #         tables_names += schema_names[key].values()
    # tabless = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))
    #               and f in tables_names]
    tables = []
    files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    for name in files:
        if any(name.split('.csv')[0] in exp for exp in experiments):
            tables.append(name)
    for table in tables:
        start_time = time.time()
        schemapath = os.path.join(data_folder, table)
        dict_tables[table] = {}
        for sample_size in sample_sizes_list:
            schema = TablesMatcher.Schema()
            schema.set_from_path(schemapath, sample_size, ratio_sampling)

            # Find categorical, numeric and textual columns
            # _schema1_cat, _schema1_noncat = TablesMatcher.cat_noncat_sort(schema.get_table(), unq_for_categorical)
            _schema1_obj = schema.get_table().select_dtypes(include=['object'])
            _schema1_num = schema.get_table()._get_numeric_data()

            # Find statistics for each table data type
            stats_num_df_t1 = Statistics.numeric_stats(_schema1_num)
            stats_text_df_t1 = Statistics.txt_stats(_schema1_obj)
            # stats_cat_df_t1 = Statistics.txt_stats(_schema1_cat)

            if use_fasttext.lower() == 'true':
                fasttext_stat = Statistics.fasttext_embed(_schema1_obj, 3, 'mean')
                stats_text_df_t1 = pd.concat([stats_text_df_t1, fasttext_stat])

                fasttext_len = len(fasttext_stat.index)

                # change row's names
                for i in range(0, fasttext_len):
                    stats_text_df_t1.index.values[len(stats_text_df_t1.index) - (fasttext_len - i)] = 'ft' + str(i + 1)

            # dict_tables[table][sample_size] = [stats_num_df_t1, stats_text_df_t1, stats_cat_df_t1, schema.num_of_fields()]
            prepr_time = time.time() - start_time
            dict_tables[table][sample_size] = [stats_num_df_t1, stats_text_df_t1, prepr_time]
            print("Finished preprocess of {} sample of table {} \n".format(sample_size, table))


def add_existing_scores(potent_sim_lists):
    try:
        with open(os.path.join(data_folder, scores_folder, exp_name + '.log'), 'r') as f:
            df1 = pd.DataFrame(potent_sim_lists)
            content = f.read().splitlines(True)[1:]
            scores = parse_scores(content)
            df2 = pd.DataFrame(scores)
            df3 = pd.merge(df1, df2, on=[0, 1], how='left')
            return list(df3.itertuples(index=False, name=None))
    except:
        return potent_sim_lists


def parse_scores(content):
    scores = []
    filtered_lines = []
    for line in content:
        try:
            l_content = line.split('\t')
            eval(l_content[0])
            filtered_lines.append(line)
        except:
            continue
    for line in filtered_lines:
        l_content = line.split('\t')
        columns = eval(l_content[0])
        scores.append((columns[0], columns[1], l_content[3]))
        scores.append((columns[1], columns[0], l_content[3]))
    return scores


def match():
    # iterate over sample sizes
    for sample_size in sample_sizes_list:
        all_results = []

        # Iterate over thresholds (cosine similarity)
        for thresh in threshs_list:
            start_time = time.time()

            # Set the columns matcher
            sch_matcher = TablesMatcher.Matcher(float(thresh), similarity_type)

            # Match the columns
            potential_matches = sch_matcher.stats_match(dict_tables[schema1name][sample_size]
                                                        , dict_tables[schema2name][sample_size])

            # Print the matching results
            potent_sim_lists = []
            results = {}
            fk_pairs_included = {x: False for x in fk_pairs}
            overall_comparisons = (len(dict_tables[schema1name][sample_size][1].columns) * len(
                dict_tables[schema2name][sample_size][1].columns)) + (len(
                dict_tables[schema1name][sample_size][0].columns) * len(
                dict_tables[schema2name][sample_size][0].columns))
            stats_comparisons = 0

            for tbl_match in potential_matches:
                if tbl_match.empty:  # crashes if tbl_match is empty
                    continue
                pot_list = list(tbl_match[tbl_match > 0].stack().index)
                pot_values = list(tbl_match[tbl_match > 0].stack())
                pot_list = [(t[1], t[0]) for t in pot_list]
                pot_values_list = [(t[1], t[0], x) for t, x in zip(pot_list, pot_values)]
                print('Number of all pairs: {}, number of potential matches: {}'.
                      format(tbl_match.shape[0] * tbl_match.shape[1], len(pot_list)))
                stats_comparisons += len(pot_list)
                print('Potential matches: {}'.format(pot_list))
                for fk_pair in fk_pairs:
                    if fk_pair in pot_list:
                        fk_pairs_included[fk_pair] = True
                # fk_pairs_included.update({fk_pair: True if fk_pair in pot_list else False for fk_pair in fk_pairs})
                potent_sim_lists.extend(pot_values_list)

            potent_sim_lists = add_existing_scores(potent_sim_lists)

            print_to_file(sample_size, thresh, potent_sim_lists)

            end_time = (time.time() - start_time)
            print("--- %s seconds --- {}".format(end_time))

            # Put the results into results dictionary
            results['experiment'] = exp_name
            results['threshold'] = thresh
            results['time'] = end_time
            results['preprocess_time'] = (
                    dict_tables[schema1name][sample_size][2] + dict_tables[schema2name][sample_size][2])
            results['sample'] = sample_size
            results['# all_comparisons'] = overall_comparisons
            results['# stats comparisons'] = stats_comparisons
            results['use_fasttext'] = use_fasttext
            results['similarity_function'] = similarity_type
            for fk_pair in fk_pairs:
                results['{} found'.format(fk_pair)] = fk_pairs_included[fk_pair]
            results['amdocs_time'] = ''
            # results['potential_matches'] = str(potent_sim_lists)
            all_results.append(results)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(final_results_path, 'results_sample_{}_{}.csv'.format(sample_size, exp_name)),
                          index=False)


def main():
    global exp_name
    preprocess()  # get statistics for current experiment
    for exp in experiments:
        exp_name = exp
        config_table()  # update global variables after choosing another exp_name
        match()


# _____ START HERE _____
# Get the program configs

config = configparser.ConfigParser()
config.read('config.ini')
exp_name = None
data_folder = config['DEFAULT']['DATA_FOLDER']
schema1name = None
schema2name = None
similarity_type = config['DEFAULT']['SIMILARITY']
threshs_list = config['DEFAULT']['THRESH_LIST'].split(',')
fk_pairs = None
unq_for_categorical = int(config['DEFAULT']['UNIQUE_VALS_FOR_CATEGORICAL'])
results_dir = None
experiments = config['DEFAULT']['EXPERIMENTS'].split(',')
use_fasttext = config['DEFAULT']['USE_FASTTEXT']
ratio_sampling = config['DEFAULT']['RATIO_SAMPLING']
sample_sizes_list = config['DEFAULT']['RATIO_SAMPLE_SIZES'].split(',') if ratio_sampling.lower() == "true" else \
    config['DEFAULT']['SAMPLE_SIZES'].split(',')
scores_folder = config['DEFAULT']['SCORES_FOLDER']
path_ft = None
path_pm = None
final_results_path = None

# PREPROCESS
dict_tables = {}

# Set the logger

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        # logging.FileHandler('./results/{}/log_{}.txt'.format(exp_name, exp_name)),
        logging.StreamHandler()
    ])

if __name__ == '__main__':
    main()
