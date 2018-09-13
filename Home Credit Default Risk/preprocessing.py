import numpy as np
import pandas as pd
import os

from scipy.stats import skew, kurtosis

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

DATADIR = "input/"
SUBMISSIONS_DIR = "submissions/"

def list_files(input_dir = "input/"):
    return (os.listdir(input_dir))

def read_dataset_csv(input_dir = "input/", filename = "", nrows = None, usecols = None):
    return (pd.read_csv(input_dir + filename, nrows = nrows, usecols = usecols))

def read_train_test(input_dir = "input/", train_file = 'train.csv', test_file = 'test.csv', nrows = None):
    train = pd.read_csv(input_dir + train_file, nrows = nrows)
    test = pd.read_csv(input_dir + test_file, nrows = nrows)
    return train, test

def submit_file(test_id, test_pred, prefix_file_name = "submit", cv_score = None):
    if cv_score != None:
        file_name = "%s_%.6f.csv" % (prefix_file_name, cv_score)
    else:
        file_name = "%s.csv" % (prefix_file_name)
    submit = pd.DataFrame()
    submit['SK_ID_CURR'] = test_id
    submit['TARGET'] = test_pred
    submit[['SK_ID_CURR', 'TARGET']].to_csv(SUBMISSIONS_DIR + file_name, index= False)
    return submit

def get_memory_usage_mb(dataset):
    return (dataset.memory_usage().sum() / 1024 / 1024)

def check_missing(dataset):
    all_data_na_absolute = dataset.isnull().sum()
    all_data_na_percent = (dataset.isnull().sum() / len(dataset)) * 100
    mis_val_table = pd.concat([all_data_na_absolute, all_data_na_percent], axis=1)
    mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'}, inplace = True)
    missing_table = mis_val_table.drop(mis_val_table[mis_val_table.iloc[:, 1] == 0].index).sort_values('% of Total Values', ascending=False).round(2)
    return(missing_table)
    
def check_flag_doc_cols(dataset):
    all_data_absolute = dataset.sum()
    all_data_percent = (dataset.sum() / len(dataset)) * 100
    mis_val_table = pd.concat([all_data_absolute, all_data_percent], axis=1)
    mis_val_table.rename(columns = {0 : 'FLAG 1 Count', 1 : '% of Total'}, inplace = True)
    missing_table = mis_val_table.drop(mis_val_table[mis_val_table.iloc[:, 1] == 0].index).sort_values('% of Total', ascending=False).round(2)
    return(missing_table)
    
def check_categorical_cols_values(dataset, dropna = True, col = "ORGANIZATION_TYPE"):
    all_data_absolute = dataset.loc[:, col].value_counts(dropna = dropna).rename("Count")
    all_data_percent = ((dataset.loc[:, col].value_counts(dropna = dropna) / len(dataset)) * 100).rename("% of Total")
    mis_val_table = pd.concat([all_data_absolute, all_data_percent], axis=1)
    #mis_val_table.rename(columns = {0 : 'Count', 1 : '% of Total'}, inplace = True)
    missing_table = mis_val_table.drop(mis_val_table[mis_val_table.iloc[:, 1] == 0].index).sort_values('% of Total', ascending=False).round(2)
    return(missing_table)
    
def get_feature_groups(dataset):
    num_columns = list(dataset.select_dtypes(exclude=['object', 'category']).columns)
    cat_columns = list(dataset.select_dtypes(include=['object', 'category']).columns)
    return (num_columns, cat_columns)

def get_dtypes_columns(dataset):
    groups = dataset.columns.to_series().groupby(dataset.dtypes).groups
    return (groups)

def get_dtype_columns(dataset, dtypes = None):
    types_cols = get_dtypes_columns(dataset)
    if dtypes == None:
        list_columns = [list(cols) for tipo, cols in types_cols.items()]
    else:
        list_columns = [list(cols) for tipo, cols in types_cols.items() if tipo in dtypes]
    columns = []
    for cols in list_columns:
        columns += [c for c in cols if 'SK_ID' not in c]
    return(columns)

def get_missing_cols(dataset, dtypes = None):
    cols = get_dtype_columns(dataset, dtypes = dtypes)
    cols_null = [col for col in cols if dataset[col].isnull().any()]
    return (cols_null)

def get_categorical_missing_cols(dataset):
    cat_cols_null = get_missing_cols(dataset, dtypes = [np.dtype(object)])
    return (cat_cols_null)

def get_numerical_missing_cols(dataset):
    num_cols_null = get_missing_cols(dataset, dtypes = [np.dtype(np.int64), np.dtype(np.float64)])
    return (num_cols_null)

def add_columns_was_missing(X):
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    new_columns = []
    for col in cols_with_missing:
        new_col = col + '_WAS_MISSING'
        new_columns.append(new_col)
        X[new_col] = X[col].isnull()
    return X

def handle_missing_mode(X, mode_cols, group_by_cols = None):
        
    if group_by_cols == None:
        #X_ = X.copy()[mode_catcols]
        X_ = X.copy()
        for col in mode_cols:
            if X_[col].isnull().any():
                X_[col] = X_[col].transform(lambda x: x.fillna(x.mode()[0]))
    else:
        #X_ = X.copy()[mode_catcols + group_by_cols]
        X_ = X.copy()
        for col in mode_cols:
            if X_[col].isnull().any():
                X_[col] = X_.groupby(group_by_cols)[col].transform(lambda x: x.fillna(x.mode()[0]))
        
    return(X_)

def handle_missing_median(X, median_cols, group_by_cols = None):
        
    if group_by_cols == None:
        #X_ = X.copy()[mode_catcols]
        X_ = X.copy()
        for col in median_cols:
            if X_[col].isnull().any():
                X_[col] = X_[col].transform(lambda x: x.fillna(x.median()))
    else:
        #X_ = X.copy()[mode_catcols + group_by_cols]
        X_ = X.copy()
        for col in median_cols:
            if X_[col].isnull().any():
                X_[col] = X_.groupby(group_by_cols)[col].transform(lambda x: x.fillna(x.median()))
        
    return(X_)

class HandleMissingModeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return handle_missing_mode(X, mode_catcols = self.cols)

    def fit(self, X, y=None):
        self.cols = get_categorical_missing_cols(X)
        return self
    
class HandleMissingMedianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return handle_missing_median(X, median_cols = self.cols)

    def fit(self, X, y=None):
        self.cols = get_numerical_missing_cols(X)
        return self

def hot_encode(X, columns = None):
    if columns == None:
        return pd.get_dummies(X)
    else:
        return pd.get_dummies(X, columns = columns)
    
def correlation_target(dataset, target = "TARGET"):
    corr = dataset.corr()[target].sort_values(ascending = False)
    return(corr)
    
def correlation_matrix(dataset, target = 'TARGET', nvar = 10):
    corrmat = dataset.corr()
    cols = corrmat.nlargest(nvar + 1, target)[target].index
    cm = np.corrcoef(dataset[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                     annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return list(cols[1:])

def get_domain_knowledge_features(X):
    X_domain = X.copy()
    exc_cols = ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE','WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
    flag_doc_cols = [c for c in X_domain.columns if c.startswith("FLAG_DOCUMENT_")]
    client_living_avg_cols = [c for c in X_domain.columns if c.endswith("_AVG")]
    client_living_mode_cols = [c for c in X_domain.columns if c.endswith("_MODE") and c not in exc_cols]
    client_living_median_cols = [c for c in X_domain.columns if c.endswith("_MEDI")]
    
    X_domain['CREDIT_INCOME_PERCENT'] = X_domain['AMT_CREDIT'] / X_domain['AMT_INCOME_TOTAL']
    X_domain['CREDIT_GOODS_PRICE_PERCENT'] = X_domain['AMT_CREDIT'] / X_domain['AMT_GOODS_PRICE']
    X_domain['ANNUITY_INCOME_PERCENT'] = X_domain['AMT_ANNUITY'] / X_domain['AMT_INCOME_TOTAL']
    X_domain['CREDIT_TERM'] = X_domain['AMT_ANNUITY'] / X_domain['AMT_CREDIT']
    X_domain['DAYS_EMPLOYED_PERCENT'] = X_domain['DAYS_EMPLOYED'] / X_domain['DAYS_BIRTH']
    X_domain['INCOME_CREDIT_PERC'] = X_domain['AMT_INCOME_TOTAL'] / X_domain['AMT_CREDIT']
    X_domain['INCOME_PER_PERSON'] = X_domain['AMT_INCOME_TOTAL'] / X_domain['CNT_FAM_MEMBERS']
    X_domain['CHILDREN_RATIO'] = X_domain['CNT_CHILDREN'] / X_domain['CNT_FAM_MEMBERS']
    X_domain["HOW_MANY_DOCUMENTS"] = X_domain.loc[:, flag_doc_cols].sum(axis=1)
    X_domain["EXT_SOURCE_SUM"] = X_domain.loc[:, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    X_domain["EXT_SOURCE_AVG"] = X_domain.loc[:, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    
    cols_flag_del = ['FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_13',
'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_17',
'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12']
    
    return (X_domain.drop(cols_flag_del, axis = 1))

def agg_numeric(df, group_var, df_name, agg_funcs = ['count', 'mean', 'max', 'min', 'sum', 'std', 'var']):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop([col], axis = 1)
            
    group_ids = df.loc[:, group_var].values
    numeric_df = df.select_dtypes(include = ['number'])
    numeric_df.loc[:, group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(agg_funcs).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg

def count_categorical(df, group_var, df_name, agg_funcs = ['sum', 'mean'], cols_alias = ['count', 'count_norm']):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns
    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes(include = ['object', 'category']))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(agg_funcs)
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in cols_alias:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical

# Function to calculate correlations with the target for a dataframe
def target_corrs(df):

    # List of correlations
    corrs = []

    # Iterate through the columns 
    for col in df.columns:
        print(col)
        # Skip the target column
        if col != 'TARGET':
            # Calculate correlation with the target
            corr = df['TARGET'].corr(df[col])

            # Append the list as a tuple
            corrs.append((col, corr))
            
    # Sort by absolute magnitude of correlations
    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)
    
    return corrs

# Plots the disribution of a variable colored by value of the target
def kde_target(var_name, df):
    
    # Calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)

def get_counts_features(df, group_var, count_var, df_name):
    counts = df.groupby(group_var)[count_var].agg('count')
    counts.name = df_name + '_ROWCOUNT'
    return (counts.reset_index())
    
def get_engineered_features(df, group_var, df_name, num_agg_funcs = ['mean', 'max', 'min', 'sum', 'std', 'var'], cat_agg_funcs = ['sum', 'mean'], cols_alias = ['count', 'count_norm']):
    numerical_agg = agg_numeric(df, group_var = group_var, df_name = df_name, agg_funcs = num_agg_funcs)
    if (any(df.dtypes == 'object') or any(df.dtypes == 'category')):
        categorical_agg = count_categorical(df, group_var = group_var, df_name = df_name, agg_funcs = cat_agg_funcs, cols_alias = cols_alias).reset_index()
        return numerical_agg.merge(categorical_agg, on = group_var, how = 'inner')
    else:
        return(numerical_agg)

def get_engineered_features_from_file(filename, group_var, df_name, drop_cols = None):
    if drop_cols == None:
        df = read_dataset_csv(filename = filename)
    else:
        df = read_dataset_csv(filename = filename).drop(drop_cols, axis=1)
    df_agg = get_engineered_features(df, group_var, df_name)
    return df_agg

def aggregate_client(df, parent_df, group_vars, df_names):
    """Aggregate a dataframe with data at the loan level 
    at the client level
    
    Args:
        df (dataframe): data at the loan level
        group_vars (list of two strings): grouping variables for the loan 
        and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
        names (list of two strings): names to call the resulting columns
        (example ['cash', 'client'])
        
    Returns:
        df_client (dataframe): aggregated numeric stats at the client level. 
        Each client will have a single row with all the numeric data aggregated
    """
    
    df_agg = get_engineered_features(df, group_var = group_vars[0], df_name = df_names[0])

    # Merge to include the SK_ID_CURR
    #bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_balance_agg, on = 'SK_ID_BUREAU', how = 'left')
    df_by_loan = df_agg.merge(parent_df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')
    df_by_loan = df_by_loan.drop([group_vars[0]], axis=1)
    # Aggregate the stats for each client
    df_by_client = agg_numeric(df_by_loan, group_var = group_vars[1], df_name = df_names[1])
    
    return df_by_client

def aggregate_client_2(df, group_vars, df_names):
    """Aggregate a dataframe with data at the loan level 
    at the client level
    
    Args:
        df (dataframe): data at the loan level
        group_vars (list of two strings): grouping variables for the loan 
        and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
        names (list of two strings): names to call the resulting columns
        (example ['cash', 'client'])
        
    Returns:
        df_client (dataframe): aggregated numeric stats at the client level. 
        Each client will have a single row with all the numeric data aggregated
    """
    
    df_agg = get_engineered_features(df, group_var = group_vars[0], df_name = df_names[0])

    # Merge to include the SK_ID_CURR
    #bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_balance_agg, on = 'SK_ID_BUREAU', how = 'left')
    df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')
    df_by_loan = df_by_loan.drop([group_vars[0]], axis = 1)
    # Aggregate the stats for each client
    df_by_client = agg_numeric(df_by_loan, group_var = group_vars[1], df_name = df_names[1])
    
    return df_by_client

def equal_columns(col_a, col_b):
    return np.all(col_a == col_b)

from tqdm import tqdm
def duplicate_columns(df, return_dataframe = False, verbose = False, progress = True):
    '''
        a function to detect and possibly remove duplicated columns for a pandas dataframe
    '''
    # group columns by dtypes, only the columns of the same dtypes can be duplicate of each other
    groups = df.columns.to_series().groupby(df.dtypes).groups
    duplicated_columns = {}
    

        
    for dtype, col_names in groups.items():
        column_values = df[col_names]
        num_columns = len(col_names)
        
        if progress == True:
            it = tqdm(range(num_columns))
        else:
            it = range(num_columns)
        # find duplicated columns by checking pairs of columns, store first column name if duplicate exist 
        for i in it:
            column_i = column_values.iloc[:,i]
            for j in range(i + 1, num_columns):
                column_j = column_values.iloc[:,j]
                if equal_columns(column_i, column_j):
                    if verbose: 
                        print("column {} is a duplicate of column {}".format(col_names[i], col_names[j]))
                    duplicated_columns[col_names[j]] = col_names[i]
                    break
    if not return_dataframe:
        # return the column names of those duplicated exists
        return duplicated_columns
    else:
        # return a dataframe with duplicated columns dropped 
        return df.drop(labels = list(duplicated_columns.keys()), axis = 1)
    
import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df



def agg_categorical_numeric(df, df_name, group_var = ['SK_ID_CURR', 'CREDIT_ACTIVE'], funcs = ['sum', 'mean'], target_numvar = ['DAYS_CREDIT', 'AMT_ANNUITY']):
    
    #counts = pd.get_dummies(df.loc[:, group_var])
    #counts = counts.groupby(group_var[0]).agg('sum')
    #counts.columns = [c[c.find(group_var[1]+'_')+len(group_var[1]+'_'):].upper()+'_COUNT' for c in list(counts.columns)]
    
    df_1 = df.loc[:, group_var + target_numvar].groupby(group_var).agg(funcs)
    
    column_names = []
    for var in df_1.columns.levels[0]:
        for stat in list(df_1.columns.levels[1].get_values()):
            column_names.append('%s_%s' % (var, stat.upper()))
    df_1.columns = column_names
    
    df_2 = df_1.pivot_table(columns=[group_var[1]], values= column_names, index= [group_var[0]], fill_value=0)
    
    column_names = []
    for var in df_2.columns.levels[0]:
        field_name = var[:var.rfind("_")]
        sufix = var[var.rfind("_")+1:]
        for stat in df_2.columns.levels[1]:
            column_names.append('%s_%s_%s_%s_%s' % (df_name.upper(), group_var[1], stat.upper(), field_name, sufix))
        
    df_2.columns = column_names
    df_2 = df_2.reset_index()
    
    return (df_2)


def agg_numeric_2(df, group_var, df_name, agg_funcs = ['count', 'mean', 'max', 'min', 'sum', 'std', 'var']):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    id_vars = [col for col in df.columns if 'SK_ID' in col and col not in group_var]
    if len(id_vars) > 0:
        df = df.drop(id_vars, axis = 1)
            
    #group_ids = df.loc[:, group_var].values
    numeric_df = df.select_dtypes(include = ['number'])
    #numeric_df.loc[:, group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(agg_funcs).reset_index()

    # Need to create new column names
    columns = group_var[:]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var not in group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat.upper()))

    agg.columns = columns
    return agg

def count_categorical_2(df, group_var, df_name, agg_funcs = ['sum', 'mean'], cols_alias = ['COUNT', 'COUNT_NORM']):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns
    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    cat_columns = list(df.select_dtypes(include=['object', 'category']).columns)
    categorical = pd.get_dummies(df.loc[:, group_var + cat_columns])
    
    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(agg_funcs).reset_index()
    
    column_names = group_var[:]
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Skip the grouping variable
        if var not in group_var:
            # Iterate through the stats in level 1
            for stat in cols_alias:
                # Make a new column name
                column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names

    return categorical

def get_engineered_features_2(df, group_var, df_name):
    num_agg = agg_numeric_2(df, group_var, df_name)
    if (any(df.dtypes == 'object') or any(df.dtypes == 'category')):
        cat_agg = count_categorical_2(df, group_var, df_name)
        return num_agg.merge(cat_agg, on = group_var, how = "inner")
    else:
        return num_agg
