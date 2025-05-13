# --- Visualization Setup ---
# Uncomment and run the following line in your environment (e.g., Kaggle Notebook) if needed
# !pip install matplotlib seaborn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Make sure numpy is imported

# Set plot style for better aesthetics
sns.set_theme(style="whitegrid")

# --- Control Flag for Visualizations ---
PLOT_VISUALIZATIONS = True # Set to False to disable plotting for faster runs / less memory
# --- End Visualization Setup ---

import pandas as pd
import warnings
import gc # Garbage Collector
from tqdm import tqdm
from datetime import datetime
# import lightgbm as lgb # Removed as ML part is deleted
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier # Removed
# from sklearn.metrics import roc_auc_score # Removed
# from sklearn.model_selection import train_test_split # Removed
warnings.filterwarnings('ignore')
tqdm.pandas()

# --- Utility function for memory reduction ---
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df
# --- End Utility function ---


print("Loading transactions...")
# Load only absolutely necessary columns initially
transactions_train = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
                                 usecols = ['t_dat', 'customer_id', 'article_id'],
                                 dtype={'article_id': 'int32', 'customer_id': 'str'}) # Keep article_id as int for now for memory, customer_id needs hashing for category
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'], format='%Y-%m-%d')
# Optimize article_id if cardinality allows, or keep as int32/str
# transactions_train['article_id'] = transactions_train['article_id'].astype(str) # Convert back to str if needed later, do it just-in-time
print(f"Transactions shape: {transactions_train.shape}")
transactions_train = reduce_mem_usage(transactions_train)
gc.collect()


lastday = transactions_train['t_dat'].max()
transactions_train['dow'] = transactions_train['t_dat'].dt.dayofweek.astype(np.int8) # Downcast
transactions_train['ldbw'] = transactions_train['t_dat'] - pd.to_timedelta(transactions_train['dow'] - 1, unit='D')
# The following operation can be memory intensive if the slice is large.
# Consider optimizing if it causes issues, but it might be acceptable.
mask_dow_ge_2 = transactions_train['dow'] >= 2
transactions_train.loc[mask_dow_ge_2 , 'ldbw'] = \
                transactions_train.loc[mask_dow_ge_2 , 'ldbw'] + \
                pd.Timedelta(days=7) # Simplified timedelta creation
del transactions_train['dow'], mask_dow_ge_2
gc.collect()

print("Calculating weekly sales...")
# Use size() for efficiency
weekly_sales = transactions_train.groupby(['ldbw', 'article_id'], observed=True).size().reset_index(name='count')
weekly_sales = reduce_mem_usage(weekly_sales)
gc.collect()


print("Merging weekly sales...")
# Ensure article_id type matches before merge if converted back to str earlier
transactions_train['article_id'] = transactions_train['article_id'].astype('int32') # Ensure type match
weekly_sales['article_id'] = weekly_sales['article_id'].astype('int32') # Ensure type match
transactions_train = pd.merge(transactions_train, weekly_sales, on=['ldbw', 'article_id'], how='left')
del weekly_sales # Delete immediately
gc.collect()
transactions_train['count'] = transactions_train['count'].astype('float32') # Prepare for division, might contain NaN


print("Calculating target week sales...")
weekly_sales_agg = transactions_train.loc[transactions_train['ldbw']==lastday, ['article_id', 'count']].rename(columns={'count':'count_targ'})
# Aggregate before merge to reduce memory
weekly_sales_agg = weekly_sales_agg.groupby('article_id', observed=True)['count_targ'].first().reset_index() # Use first() as count should be the same for that day/article
weekly_sales_agg = reduce_mem_usage(weekly_sales_agg)
gc.collect()

print("Merging target week sales...")
transactions_train = pd.merge(transactions_train, weekly_sales_agg, on='article_id', how='left', suffixes=('', '_y'))
# Drop duplicated columns if any (shouldn't happen with suffixes)
# transactions_train = transactions_train.loc[:,~transactions_train.columns.duplicated()]
del weekly_sales_agg
gc.collect()


transactions_train['count_targ'].fillna(0, inplace=True)
transactions_train['count'].fillna(1, inplace=True) # Avoid division by zero/NaN; if count is NaN, implies article only sold in last week? Needs check. Assume 1.
# Ensure types before division
transactions_train['count_targ'] = transactions_train['count_targ'].astype('float32')
transactions_train['count'] = transactions_train['count'].astype('float32')
transactions_train['quotient'] = (transactions_train['count_targ'] / transactions_train['count']).astype('float32')

# --- Create train_fo and calculate value ---
print("Calculating 'value' score...")
# Select only necessary columns for train_fo
train_fo = transactions_train[['customer_id', 'article_id', 'ldbw', 'quotient']].copy()
del transactions_train # Delete original transactions_train now if not needed elsewhere
gc.collect()

train_fo['week_x'] = ((lastday - train_fo['ldbw']) / np.timedelta64(7, 'D')).astype(int).astype(np.int16) # Downcast
train_fo['sup_1'] = 0.1
# Use numpy maximum for potential speedup on simple operations
train_fo['week_x'] = np.maximum(train_fo['week_x'].to_numpy(), train_fo['sup_1'].to_numpy()).astype('float32') # week_x becomes float now
del train_fo['sup_1'] # Remove helper column
gc.collect()

train_fo['y'] = (1.2e3 / train_fo['week_x']).astype('float32')
train_fo['sup_25'] = 25.0 # Use float
train_fo['y'] = np.maximum(train_fo['y'].to_numpy(), train_fo['sup_25'].to_numpy()).astype('float32')
del train_fo['sup_25']
gc.collect()

# Calculate value per row first
train_fo['value'] = (train_fo['quotient'] * train_fo['y']).astype('float32')

# Aggregate value per customer-article
# Keep only relevant columns before groupby
train_fo = train_fo[['customer_id', 'article_id', 'value']]
value_agg = train_fo.groupby(['customer_id', 'article_id'])['value'].sum().reset_index()
value_agg = reduce_mem_usage(value_agg)
del train_fo # Delete the per-row value DataFrame
gc.collect()

# --- Continue with Strategy 1 ---
print("Processing Strategy 1: High Potential...")
# Rename train_fo to avoid confusion, it's now aggregated value
train_fo = value_agg[value_agg['value'] > 180].copy() # Filter first
# del value_agg # Keep value_agg for Strategy 3 if possible!
gc.collect()

# Calculate rank on the filtered data
train_fo['rank'] = train_fo.groupby("customer_id")["value"].rank("dense", ascending=False).astype(np.int8) # Downcast rank
train_fo = train_fo.loc[train_fo['rank'] <= 12]
# train_fo now contains customer_id, article_id, value, rank for Strat 1 candidates

# Sort for potential later use, but groupby aggregation doesn't need sorted input
# purchase_records = train_fo.sort_values(['customer_id', 'value'], ascending=False)#.reset_index(drop=True)
# Instead of sorting the whole DF, sort within group during aggregation if needed

# Aggregate predictions for Strategy 1
# --- FIX: Format article_id with leading zeros before aggregation ---
train_fo['article_id_str'] = train_fo['article_id'].apply(lambda x: f'{x:010d}')
purchase_records = train_fo.sort_values(['customer_id', 'value'], ascending=False)\
                           .groupby('customer_id')['article_id_str']\
                           .apply(list)\
                           .reset_index(name='prediction_a') # Rename directly to avoid clash later
del train_fo
gc.collect()
print(f"Strategy 1 records shape: {purchase_records.shape}")


# --- Strategy 2: Pairs ---
print("Processing Strategy 2: Pairs...")
# Reload transactions if deleted, only necessary cols
print("Reloading transactions for pairs...")
transactions_train = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
                                 usecols = ['t_dat', 'customer_id', 'article_id'],
                                 dtype={'article_id': 'int32', 'customer_id': 'str'})
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'], format='%Y-%m-%d')
transactions_train = reduce_mem_usage(transactions_train)
gc.collect()

# Filter data for pair calculation
transactions_pairs = transactions_train[transactions_train['t_dat'] > datetime(2020, 6, 22)].copy()
del transactions_train # Delete the reloaded full transactions
gc.collect()

def calc_pairs(train):
    print("Calculating pairs...")
    # Calculate all articles purchased together
    # Use size() and avoid intermediate merge if possible, but current logic needs merge
    # Consider optimizing this function if it's a bottleneck
    train['article_id'] = train['article_id'].astype(str) # Needs string for list aggregation maybe? Ensure consistency.
    dt = train.groupby(['customer_id','t_dat'])['article_id'].agg(list).rename('pair').reset_index()
    df = pd.merge(train[['customer_id', 't_dat', 'article_id']], dt, on=['customer_id', 't_dat'], how='left')
    del dt
    gc.collect()

    # base_count - use value_counts for potential speedup?
    base_count = df['article_id'].value_counts().reset_index()
    base_count.columns = ['article_id', 'base_count']
    base_count = reduce_mem_usage(base_count)
    gc.collect()

    # Explode the rows vs list of articles
    # This is memory intensive!
    print("Exploding pairs...")
    df = df[['article_id', 'pair']].explode(column='pair')
    print(f"Shape after explode: {df.shape}")
    gc.collect()

    # Discard duplicates
    df = df.loc[df['article_id'] != df['pair']].reset_index(drop=True)
    gc.collect()

    # Count how many times each pair combination happens
    print("Counting pair occurrences...")
    df = df.groupby(['article_id', 'pair'], observed=True).size().rename('pair_count').reset_index()
    gc.collect()

    # Sort by frequency
    # df = df.sort_values(['article_id' ,'pair_count'], ascending=[False, False]).reset_index(drop=True) # Sorting large DF is slow

    print("Merging base count...")
    df = pd.merge(df, base_count, on='article_id', how='left')
    del base_count
    gc.collect()
    df['base_count'].fillna(1, inplace=True) # Avoid division by zero
    df['pair_count'] = df['pair_count'].astype('float32')
    df['base_count'] = df['base_count'].astype('float32')
    df['ratio'] = (df['pair_count'] / df['base_count']).astype('float32')
    gc.collect()

    # Pick only top N most frequent pair - use nlargest after sorting within group
    print("Selecting top pairs per article...")
    df = df.sort_values(['article_id', 'pair_count'], ascending=[True, False]) # Sort for groupby().head()
    df = df.groupby('article_id').head(2).reset_index(drop=True)
    gc.collect()

    return df

articles_pair = calc_pairs(transactions_pairs)
del transactions_pairs
gc.collect()

articles_pair_base = articles_pair[articles_pair['ratio'] > 0.12].copy() # Filter first
del articles_pair
gc.collect()
# articles_pair_base.sort_values('ratio', ascending=False, inplace=True) # Sorting might not be needed
# Aggregate pairs per article
articles_pair_base = articles_pair_base.groupby('article_id')['pair'].agg(list).reset_index()
articles_pair_base = reduce_mem_usage(articles_pair_base)
articles_pair_base.rename(columns={'pair':'pair_list'}, inplace=True) # Rename for clarity
gc.collect()
print(f"Base pairs shape: {articles_pair_base.shape}")


def pair_zh(strat1_preds, pair_data):
    print("Merging Strategy 1 and Strategy 2...")
    # Explode Strat 1 preds
    train = strat1_preds.explode('prediction_a') # Use renamed column
    train.rename(columns={'prediction_a':'article_id'}, inplace=True) # Rename for merge

    # Merge pairs
    # Ensure article_id types match before merge
    train['article_id'] = train['article_id'].astype(str)
    pair_data['article_id'] = pair_data['article_id'].astype(str)
    train = pd.merge(train, pair_data, on='article_id', how='left')
    # No fillna('0') needed if pair_list is list or NaN

    # Aggregate back, applying falt logic implicitly
    def combine_preds_pairs(group):
        preds = group['article_id'].tolist() # Original preds from strat 1
        pairs = group['pair_list'].dropna().tolist() # List of lists of pairs

        # Flatten pairs
        flat_pairs = []
        for sublist in pairs:
            flat_pairs.extend(sublist)

        # Combine and deduplicate keeping order
        combined = preds + flat_pairs
        unique_list = []
        seen = set()
        for item in combined:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)
        return unique_list[:12] # Apply falt logic (dedup, keep order, top 12)

    final_predictions = train.groupby('customer_id').apply(combine_preds_pairs).reset_index(name='prediction_a') # Result back in prediction_a
    del train # Delete intermediate exploded dataframe
    gc.collect()
    return final_predictions

purchase_records = pair_zh(purchase_records, articles_pair_base)
del articles_pair_base # Delete after use
gc.collect()
print(f"Shape after Strat 1+2 merge: {purchase_records.shape}")


# --- Customers Data ---
print("Loading and processing customers data...")
customers = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/customers.csv',
                        usecols=['customer_id', 'age'],
                        dtype={'customer_id': 'str'}) # Keep customer_id as string
customers = reduce_mem_usage(customers) # Downcast age if possible (float32)

bins = list(range(16, 81, 4)) + [100]
names = list(range(len(bins) - 1))
customers['age_group'] = pd.cut(customers['age'], bins=bins, labels=names, right=False)
# Convert age_group to smallest possible int, handle NaN during cut
customers['age_group'] = pd.to_numeric(customers['age_group'], errors='coerce') # Coerce errors during conversion
customers['age_group'] = customers['age_group'].fillna(len(names)).astype(np.int8) # Fill NaN with a special code (e.g., 16 for 99) and downcast
# Assuming 99 means unknown/NaN, let's map our special code back if needed, or use 16 directly
customers.loc[customers['age'].isna(), 'age_group'] = 99 # Assign 99 if original age was NaN
customers['age_group'] = customers['age_group'].astype(np.int8)


del customers['age'] # Delete original age column
gc.collect()


# --- Strategy 4: Age Group Popular --- (Calculate earlier as needed by Strat 3)
print("Processing Strategy 4: Age Group Popular...")
# Reload transactions if deleted, only necessary cols
print("Reloading transactions for age popular...")
transactions_train = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
                                 usecols = ['t_dat', 'customer_id', 'article_id'],
                                 dtype={'article_id': 'int32', 'customer_id': 'str'})
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'], format='%Y-%m-%d')
transactions_train = reduce_mem_usage(transactions_train)
gc.collect()

agegroup = transactions_train[transactions_train['t_dat'] > datetime(2020, 9, 15)][['customer_id', 'article_id']].copy()
del transactions_train # Delete full transactions again
gc.collect()
# Ensure customer_id types match for merge
agegroup = pd.merge(agegroup, customers[['customer_id', 'age_group']], on='customer_id', how='left')
# del agegroup['customer_id'] # Keep for potential future use?
agegroup = agegroup[agegroup['age_group'] != 99] # Exclude unknown age for this calculation
agegroup['age_group'] = agegroup['age_group'].astype(np.int8) # Ensure type
agegroup['article_id'] = agegroup['article_id'].astype(str) # Need str for join later

def generate_age_top(age_data, num11, num22):
    print(f"Generating Top {num22-num11} per age group...")
    def hope(x, num1, num2):
        # Use value_counts().index directly which is faster
        # Get the top N article IDs as strings from the index
        ids_str = x.value_counts().index.tolist()[num1:num2]
        # --- FIX: Convert string ID back to int before formatting ---
        ids_str_formatted = []
        for i_str in ids_str:
            try:
                # Convert string to int, then format with leading zeros
                ids_str_formatted.append(f'{int(i_str):010d}')
            except ValueError:
                # Handle cases where the string might not be a valid integer
                print(f"Warning: Could not format article_id '{i_str}' as integer in hope function. Skipping.")
                pass # Skip this problematic ID
        # --- END FIX ---
        return ' '.join(ids_str_formatted)

    # Groupby might be faster with observed=True if age_group is category
    age_articles = age_data.groupby('age_group')['article_id']\
                           .progress_apply(lambda x: hope(x, num11, num22)).reset_index()

    # Calculate overall top for age_group 99
    print("Calculating overall top for age 99...")
    overall_top_articles = ' '.join(age_data['article_id'].value_counts().index.tolist()[num11:num22])
    # Check if 99 already exists, if so update, otherwise append/concat
    if 99 in age_articles['age_group'].values:
        age_articles.loc[age_articles['age_group'] == 99, 'article_id'] = overall_top_articles
    else:
        # Use concat instead of loc[index] which might be unstable if index changes
        age_99_df = pd.DataFrame({'age_group': [np.int8(99)], 'article_id': [overall_top_articles]})
        age_articles = pd.concat([age_articles, age_99_df], ignore_index=True)


    age_articles['article_id_list'] = age_articles['article_id'].map(lambda x:x.split(' ')) # Keep list separate
    age_articles = reduce_mem_usage(age_articles)
    return age_articles

age_articles = generate_age_top(agegroup, 0, 12)
# Create the string version for final merge later
age_articles_str = age_articles[['age_group', 'article_id']].copy()
age_articles_str.rename(columns={'article_id':'age_popular_str'}, inplace=True)

# Rename list column for clarity before merge
age_articles.rename(columns={'article_id_list':'age_popular_list'}, inplace=True)
del agegroup # Delete intermediate DF
gc.collect()
print(f"Age popular shape: {age_articles.shape}")


# --- Articles Data ---
print("Loading and processing articles data...")
articles = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/articles.csv',
                       usecols = ['article_id', 'detail_desc', 'index_group_no'],
                       dtype={'article_id': 'str', 'detail_desc': 'str', 'index_group_no': 'int8'}) # Specify types
articles = reduce_mem_usage(articles)

# Factorize 'detail_desc' - potentially high cardinality!
articles['detail_feature_no'], _ = pd.factorize(articles['detail_desc'])
articles['detail_feature_no'] = articles['detail_feature_no'].astype(np.int16) # Or int32 if > 32k unique descriptions
del articles['detail_desc']
gc.collect()

# articles.rename(columns={'detail_desc':'detail_feature_no', 'index_group_no':'population_no'}, inplace=True) # Do renaming before factorize if needed

def articles_table(data):
    # Convert feature numbers to string labels directly
    data['label_A'] = data['detail_feature_no'].astype(str)
    data['label_B'] = data['index_group_no'].astype(str) # Already int8, convert to str

    # Use category type if cardinality is suitable AFTER conversion to string
    # data['label_A'] = data['label_A'].astype('category')
    # data['label_B'] = data['label_B'].astype('category')

    data = data[['article_id', 'label_A', 'label_B']] # Keep only needed cols
    gc.collect()
    return data

articles = articles_table(articles)
articles = reduce_mem_usage(articles)
gc.collect()
print(f"Articles processed shape: {articles.shape}")

# --- Strategy 3: Similarity Recs ---
# This part involves the age_rank_rec function which is complex and loops.
# Ensure memory is managed inside the loop.

# Reload transactions_train_pt equivalent data if needed, or reuse components
print("Reloading/Preparing data for Strategy 3...")
# We need user history with purchase times and aggregated value
# Let's reconstruct transactions_train_pt essentials carefully to save memory
transactions_train = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
                                 usecols=['t_dat', 'customer_id', 'article_id'],
                                 dtype={'article_id': 'int32', 'customer_id': 'str'})
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'], format='%Y-%m-%d')
transactions_train = reduce_mem_usage(transactions_train)

# Calculate purchase_times
purchase_times = transactions_train.groupby(['customer_id', 'article_id'], observed=True).size()\
                                   .reset_index(name='purchase_times')
purchase_times = reduce_mem_usage(purchase_times)
gc.collect()

# Get last purchase day info (needed for sorting/filtering, maybe?)
# transactions_train['purchaseday_to_last'] = (lastday - transactions_train['t_dat']).dt.days
# last_purchase_info = transactions_train.loc[transactions_train.groupby(['customer_id', 'article_id'])['t_dat'].idxmax()][['customer_id', 'article_id', 'purchaseday_to_last']]
# last_purchase_info = reduce_mem_usage(last_purchase_info)
# gc.collect()

# Reload aggregated value if needed (or ensure value_agg still exists)
print("Reusing aggregated value for Strategy 3...")
# value_agg was kept after Strat 1 filtering
# Ensure it's available or handle error
if 'value_agg' not in locals():
    print("Error: value_agg dataframe not found. Recalculation or loading needed.")
    # Add code here to load from file or recalculate value_agg
    # For now, exit or raise error
    raise NameError("value_agg not defined for Strategy 3")


# Create the base for Strategy 3: transactions_train_pt equivalent
print("Creating base data for Strategy 3 (transactions_train_pt)...")
transactions_train_pt = transactions_train.sort_values('t_dat')\
                                        .drop_duplicates(['customer_id', 'article_id'], keep='last')
transactions_train_pt = transactions_train_pt[['customer_id', 'article_id', 't_dat']] # Keep only needed base cols
transactions_train_pt = pd.merge(transactions_train_pt, purchase_times, on=['customer_id', 'article_id'], how='left')
# Ensure merge keys types match for value_agg
transactions_train_pt['article_id'] = transactions_train_pt['article_id'].astype('int32') # Match type if needed
value_agg['article_id'] = value_agg['article_id'].astype('int32') # Match type if needed
transactions_train_pt = pd.merge(transactions_train_pt, value_agg, on=['customer_id', 'article_id'], how='left')
# Ensure merge keys types match for articles
transactions_train_pt['article_id'] = transactions_train_pt['article_id'].astype(str) # Back to str for articles merge
articles['article_id'] = articles['article_id'].astype(str)
transactions_train_pt = pd.merge(transactions_train_pt, articles, on='article_id', how='left') # Add labels A/B
# Merge age_group from customers
print("Merging customer age group into transactions_train_pt...")
customers['customer_id'] = customers['customer_id'].astype(str) # Ensure type match
transactions_train_pt = pd.merge(transactions_train_pt, customers[['customer_id', 'age_group']], on='customer_id', how='left')
# --- End Merge age_group

transactions_train_pt['purchase_times'].fillna(0, inplace=True)
transactions_train_pt['value'].fillna(0, inplace=True)
transactions_train_pt = reduce_mem_usage(transactions_train_pt)
del transactions_train, purchase_times, value_agg # Clean up source data
gc.collect()
print(f"Base data for Strat 3 shape: {transactions_train_pt.shape}")


def recommand(train, articles_subset, label, A_num=5, B_num=60, C_num=12):
    # Function relies on 'rank' column existing in articles_subset
    # It also modifies train inplace, which might be okay but be aware
    print(f"Running recommand for label {label}...")
    if articles_subset.empty or label not in articles_subset.columns or label not in train.columns:
        print(f"Skipping recommand for {label}, empty data or missing label column.")
        # Return empty dataframe with expected columns?
        return pd.DataFrame(columns=['customer_id', 'prediction'])


    # Optimize groupby apply if possible
    print("Mapping recommend groups...")
    # Pre-calculate map for efficiency
    recommend_A_mapgroup = articles_subset.groupby(label)['article_id'].apply(lambda x: x.tolist()[:A_num]) # Pre-compute the map

    train['article_id_recommend'] = train[label].map(recommend_A_mapgroup)
    # Handle cases where label might not be in map (fillna with empty list)
    train['article_id_recommend'] = train['article_id_recommend'].apply(lambda x: x if isinstance(x, list) else [])
    # train.fillna('0', inplace=True) # Avoid filling with string '0', use empty list or handle NaN

    # Remove labels only if they are not needed later? Be careful.
    # del train['label_A'], train['label_B'] # Only delete if not needed

    def flat(x, lnum): # Helper inside recommand
        res = []
        for i in x: # x is series of lists
           if isinstance(i, list): # Check if item is a list before extending
                res.extend(i)
        # Deduplicate
        seen = set()
        unique_list = []
        for item in res:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)
        return unique_list[:lnum]

    print("Aggregating recommendations...")
    # Use agg instead of groupby().apply() if possible, but complex function makes it hard
    temp = train.groupby('customer_id')['article_id_recommend'].apply(lambda x: flat(x.tolist(), lnum=B_num))
    # Create DataFrame from temp series directly
    train_rec = temp.reset_index()
    train_rec.columns = ['customer_id', 'prediction'] # prediction contains list now
    del temp # Delete intermediate series
    gc.collect()

    # Remove original columns from 'train' is tricky as we need customer_id
    # Instead, work with train_rec which has the result
    # train_rec['count'] = train_rec['prediction'].apply(len)
    # train_rec = train_rec[train_rec['count']>0]
    # del train_rec['count']

    print("Ranking recommendations...")
    train_rec = train_rec.explode('prediction')
    # Merge with ranks, ensure article_id types match
    articles_subset['article_id'] = articles_subset['article_id'].astype(str)
    train_rec['prediction'] = train_rec['prediction'].astype(str) # Ensure prediction (article_id) is string
    train_rec = pd.merge(train_rec, articles_subset[['article_id', 'rank']], left_on='prediction', right_on='article_id', how='left')
    train_rec['rank'].fillna(9999, inplace=True) # Handle missing ranks (articles not in subset)
    train_rec['rank'] = train_rec['rank'].astype(np.int32) # Downcast rank
    train_rec.sort_values(['customer_id', 'rank'], ascending=[True, True], inplace=True) # Sort by rank

    # Aggregate final top N per customer
    # --- FIX: Format prediction (article_id) with leading zeros ---
    temp = train_rec.groupby('customer_id')['prediction'].apply(
        lambda x: [f'{int(i):010d}' for i in x.tolist()[:C_num]]
    )
    # --- END FIX ---
    final_rec = temp.reset_index()
    del temp, train_rec # Delete intermediate dataframes
    gc.collect()
    # final_rec['prediction'] is the final list of article IDs
    gc.collect()
    print("Recommand complete.")
    return final_rec


def age_rank_rec(train_base, train_history, customer_data, articles_data, label_rec ,rank_limit, num1=5, num2=70, num3=12):
    print(f"\nStarting age_rank_rec for label: {label_rec}...")
    # --- FIX: Initialize res with correct columns ---
    res = pd.DataFrame(columns=['customer_id', 'prediction'])
    # --- END FIX ---

    # Merge customer data once outside loop if possible
    # Check if age_group already exists in train_base before merging
    if 'age_group' not in train_base.columns:
         print("Merging age_group into train_base inside age_rank_rec...") # Should not happen ideally if called correctly
         train_base = pd.merge(train_base, customer_data[['customer_id', 'age_group']], on='customer_id', how='left')

    # Check if age_group already exists in train_history before merging
    if 'age_group' not in train_history.columns:
         print("Merging age_group into train_history inside age_rank_rec...") # Should not happen ideally if called correctly
         train_history = pd.merge(train_history, customer_data[['customer_id', 'age_group']], on='customer_id', how='left') # History needs age too

    # Use unique age groups present in data for looping
    # Add safety check for column existence
    if 'age_group' not in train_base.columns:
        print(f"FATAL ERROR in age_rank_rec: train_base is missing 'age_group' column after potential merge!")
        return res # Return the empty DF with columns defined

    age_groups_to_process = sorted(train_base['age_group'].dropna().unique()) # Drop NA before unique
    print(f"Processing {len(age_groups_to_process)} age groups...")

    for i in tqdm(age_groups_to_process):
        print(f"\nProcessing age group {i}...")
        # Filter base and history for the current age group
        train_age_history_temp = train_history[train_history['age_group'] == i]
        train_base_temp = train_base[train_base['age_group'] == i]

        if train_base_temp.empty:
            print(f"Skipping age group {i}, no base data.")
            continue

        print("Calculating article sales for age group...")
        # Filter history data for time window
        history_recent_week = train_age_history_temp[train_age_history_temp['t_dat'] > datetime(2020, 9, 15)]
        history_recent_month = train_age_history_temp[train_age_history_temp['t_dat'] > datetime(2020, 8, 22)]

        # Calculate sales counts
        articles_sales_week_temp = history_recent_week.groupby('article_id', observed=True).size().reset_index(name='sale_week_temp')
        articles_sales_month_temp = history_recent_month.groupby('article_id', observed=True).size().reset_index(name='sale_month_temp')

        # Prepare articles subset for this age group
        print("Preparing articles subset with ranks...")
        articles_temp = articles_data.copy() # Operate on a copy
        # Ensure article_id types match for merge
        articles_temp['article_id'] = articles_temp['article_id'].astype(str)
        articles_sales_week_temp['article_id'] = articles_sales_week_temp['article_id'].astype(str)
        articles_sales_month_temp['article_id'] = articles_sales_month_temp['article_id'].astype(str)

        articles_temp = pd.merge(articles_temp, articles_sales_week_temp, on='article_id', how='left')
        articles_temp = pd.merge(articles_temp, articles_sales_month_temp, on='article_id', how='left')
        articles_temp['sale_week_temp'].fillna(0, inplace=True)
        articles_temp['sale_month_temp'].fillna(0, inplace=True)
        # Downcast sales counts
        articles_temp['sale_week_temp'] = articles_temp['sale_week_temp'].astype(np.int32)
        articles_temp['sale_month_temp'] = articles_temp['sale_month_temp'].astype(np.int32)


        articles_temp.sort_values(by=['sale_week_temp', 'sale_month_temp'], inplace=True, ascending=[False, False])
        # Filter by sales > 0 before ranking
        articles_temp = articles_temp[articles_temp['sale_week_temp'] > 0].copy() # Important to copy after filter
        if articles_temp.empty:
             print(f"Skipping age group {i}, no articles with sales>0.")
             del train_age_history_temp, train_base_temp, history_recent_week, history_recent_month
             del articles_sales_week_temp, articles_sales_month_temp, articles_temp
             gc.collect()
             continue

        articles_temp['rank'] = articles_temp['sale_week_temp'].rank(method='first', ascending=False).astype(np.int32)
        articles_temp = articles_temp[articles_temp['rank'] <= rank_limit]
        articles_temp = articles_temp[['article_id', label_rec, 'rank']].copy() # Select needed columns
        articles_temp = reduce_mem_usage(articles_temp)


        # Run recommand
        rec = recommand(train_base_temp, articles_temp, label=label_rec, A_num=num1, B_num=num2, C_num=num3)
        res = pd.concat([res, rec], ignore_index=True)

        # --- Memory Cleanup Inside Loop ---
        del train_age_history_temp, train_base_temp, history_recent_week, history_recent_month
        del articles_sales_week_temp, articles_sales_month_temp, articles_temp, rec
        gc.collect()
        print(f"Finished age group {i}. Result shape: {res.shape}")
        # --- End Memory Cleanup ---

    del train_base, train_history # Delete inputs to function after loop
    gc.collect()
    print(f"Finished age_rank_rec for label: {label_rec}. Final shape: {res.shape}")
    return res


# --- Execute Strategy 3 ---
print("\nExecuting Strategy 3A (Label A)...")
# Prepare input for rec A: head(5) of user history based on value & times
train_head5 = transactions_train_pt.sort_values(by=['value', 'purchase_times'], ascending=[False, False])
train_head5 = train_head5.groupby('customer_id').head(5)
# --- Debug Print ---
print("Columns in train_head5 before calling rec_A:", train_head5.columns)
# --- End Debug Print ---
# No need to merge customers again if age_group is already in train_head5 from transactions_train_pt merge
# Ensure age_group is passed to the function in train_base
rec_A = age_rank_rec(train_head5[['customer_id', 'label_A', 'age_group']], # Include age_group
                     transactions_train_pt[['customer_id', 'article_id', 't_dat', 'age_group']], # Pass history with age
                     customers[['customer_id', 'age_group']], # Pass customer mapping
                     articles[['article_id', 'label_A']], # Pass articles with label A
                     label_rec='label_A', rank_limit=1000, num1=5, num2=20, num3=8)
del train_head5 # Delete intermediate
gc.collect()
print(f"Rec A shape: {rec_A.shape}")


print("\nExecuting Strategy 3B (Label B)...")
# Prepare input for rec B: Top 1 label B based on head(18) of user history
train_B_base = transactions_train_pt.sort_values(by=['value', 'purchase_times'], ascending=[False, False])
train_B_base = train_B_base.groupby('customer_id').head(18)
# train_B_base = pd.merge(train_B_base, customers[['customer_id', 'age_group']], on='customer_id', how='left') # Already has age_group
# Find top 1 label B per customer efficiently
top_label_b = train_B_base.groupby('customer_id')['label_B'].apply(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
top_label_b.columns = ['customer_id', 'top_label_B']

# Prepare train_B for recommand: just customer_id and their top label_B
train_B_lov1 = top_label_b.dropna().copy() # Use only customers where a top label was found, copy to avoid SettingWithCopyWarning
train_B_lov1.rename(columns={'top_label_B':'label_B'}, inplace=True)
# Merge age_group into train_B_lov1 before passing to function
print("Merging age_group into train_B_lov1...")
train_B_lov1 = pd.merge(train_B_lov1, customers[['customer_id', 'age_group']], on='customer_id', how='left')
train_B_lov1 = train_B_lov1.dropna(subset=['age_group']) # Drop users if age_group couldn't be merged (shouldn't happen)
train_B_lov1['age_group'] = train_B_lov1['age_group'].astype(np.int8)

rec_B = age_rank_rec(train_B_lov1[['customer_id', 'label_B', 'age_group']], # Include age_group
                     transactions_train_pt[['customer_id', 'article_id', 't_dat', 'age_group']], # Pass history
                     customers[['customer_id', 'age_group']], # Pass customer mapping
                     articles[['article_id', 'label_B']], # Pass articles with label B
                     label_rec='label_B', rank_limit=800, num1=12, num2=30, num3=12)

del train_B_base, top_label_b, train_B_lov1, transactions_train_pt # Delete intermediates
gc.collect()
print(f"Rec B shape: {rec_B.shape}")


# --- Final Submission Generation ---
print("\nGenerating final submission...")
# Start with all customers
sub = customers[['customer_id', 'age_group']].copy() # Keep age_group for final popular merge

# Merge Strategy 1+2
sub = pd.merge(sub, purchase_records, on='customer_id', how='left')
# Rename prediction_a to avoid confusion
sub.rename(columns={'prediction_a':'pred_strat12'}, inplace=True)
# Fill missing predictions with empty list []
sub['pred_strat12'] = sub['pred_strat12'].apply(lambda x: x if isinstance(x, list) else [])
del purchase_records
gc.collect()

# Merge Strategy 3A
sub = pd.merge(sub, rec_A, on='customer_id', how='left')
sub.rename(columns={'prediction':'pred_strat3A'}, inplace=True)
sub['pred_strat3A'] = sub['pred_strat3A'].apply(lambda x: x if isinstance(x, list) else [])
del rec_A
gc.collect()

# Merge Strategy 3B
sub = pd.merge(sub, rec_B, on='customer_id', how='left')
sub.rename(columns={'prediction':'pred_strat3B'}, inplace=True)
sub['pred_strat3B'] = sub['pred_strat3B'].apply(lambda x: x if isinstance(x, list) else [])
del rec_B
gc.collect()

# Apply wash function (using lists directly is better than string conversion)
def wash_final(row):
    # Combine lists in desired priority order
    combined = row['pred_strat12'] + row['pred_strat3A'] + row['pred_strat3B']
    # Deduplicate keeping order
    seen = set()
    unique_list = []
    for item in combined:
        # --- FIX: Ensure item is a valid ID string and not just '0' ---
        if isinstance(item, str) and item != '0' and item not in seen:
            unique_list.append(item)
            seen.add(item)
        # Optional: Add check for non-string types if necessary
        # elif isinstance(item, (int, np.integer)) and item != 0 and item not in seen:
        #    unique_list.append(str(item)) # Convert valid integers to string
        #    seen.add(item)
        # --- END FIX ---
    return unique_list # Return the list of unique items

print("Applying wash function...")
sub['prediction_combined'] = sub.apply(wash_final, axis=1)
del sub['pred_strat12'], sub['pred_strat3A'], sub['pred_strat3B'] # Delete intermediate lists
gc.collect()

# Handle cold start users (where prediction_combined is empty)
# Merge age popular list for all users first
sub = pd.merge(sub, age_articles[['age_group', 'age_popular_list']], on='age_group', how='left')
sub['age_popular_list'] = sub['age_popular_list'].apply(lambda x: x if isinstance(x, list) else [])
del age_articles # Delete after merge

# Define final prediction based on combined and age popular
def final_prediction_logic(row):
    combined = row['prediction_combined']
    age_popular = row['age_popular_list']

    if not combined: # If combined list is empty (cold start or no recs from 1,2,3)
        final_list = age_popular
    else:
        # Append age popular items only if they are not already in combined
        seen = set(combined)
        additional_popular = []
        for item in age_popular:
            if item not in seen:
                additional_popular.append(item)
                seen.add(item) # Add to seen to avoid adding duplicates from age_popular itself if any
        final_list = combined + additional_popular

    return ' '.join(final_list[:12]) # Join and take top 12

print("Generating final prediction string...")
sub['prediction'] = sub.apply(final_prediction_logic, axis=1)

# Select final columns and save
sub = sub[['customer_id', 'prediction']]
print(f"Final submission shape: {sub.shape}")
sub.to_csv('submission.csv', index=False)
print("Submission file created.")
gc.collect()


# --- Visualizations (Conditional) ---
if PLOT_VISUALIZATIONS:
    print("\n--- Generating Visualizations (using sampled data) ---")
    VIZ_SAMPLE_SIZE = 1000000 # Sample size for large DF plots

    # Reload transactions minimally for plots if deleted
    print("Reloading transactions (sample) for visualization...")
    transactions_train_viz = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
                                 usecols = ['customer_id', 'article_id'],
                                 dtype={'article_id': 'int32', 'customer_id': 'str'})
    if len(transactions_train_viz) > VIZ_SAMPLE_SIZE:
        transactions_train_viz = transactions_train_viz.sample(VIZ_SAMPLE_SIZE, random_state=42)
    transactions_train_viz = reduce_mem_usage(transactions_train_viz)
    gc.collect()


    # --- Visualization: User Purchase Frequency ---
    print("Generating User Purchase Frequency plot...")
    user_purchase_counts = transactions_train_viz.groupby('customer_id').size() # Use size
    plt.figure(figsize=(12, 6))
    sns.histplot(user_purchase_counts, bins=50, kde=False, log_scale=(False, True))
    plt.title('Distribution of Number of Purchases per Customer (Sampled, Log Scale Y-axis)', fontsize=16)
    plt.xlabel('Number of Purchases', fontsize=12)
    plt.ylabel('Number of Customers (Log Scale)', fontsize=12)
    plt.xlim(0, max(1, user_purchase_counts.quantile(0.99))) # Ensure xlim lower bound is >= 0
    plt.show()
    print("Plotting complete.")
    # --- End Visualization ---

    # --- Visualization: Article Sales Popularity ---
    print("Generating Article Sales Popularity plot...")
    article_purchase_counts = transactions_train_viz.groupby('article_id').size() # Use size
    plt.figure(figsize=(12, 6))
    sns.histplot(article_purchase_counts, bins=50, kde=False, log_scale=(True, True))
    plt.title('Distribution of Number of Sales per Article (Sampled, Log Scale Axes)', fontsize=16)
    plt.xlabel('Number of Sales per Article (Log Scale)', fontsize=12)
    plt.ylabel('Number of Articles (Log Scale)', fontsize=12)
    plt.show()
    print("Plotting complete.")
    del transactions_train_viz # Delete viz data
    gc.collect()
    # --- End Visualization ---

    # --- Visualization: Customer Age Distribution ---
    print("Generating Customer Age Distribution plot...")
    # Reload customers if needed, or assume 'customers' df still exists
    customers_viz = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/customers.csv',
                            usecols=['customer_id', 'age'],
                            dtype={'customer_id': 'str'})
    customers_viz = reduce_mem_usage(customers_viz)

    plt.figure(figsize=(12, 6))
    if 'age' in customers_viz.columns:
        sns.histplot(customers_viz['age'].dropna(), bins=range(16, 81, 4), kde=False)
        plt.title('Distribution of Customer Ages', fontsize=16)
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Number of Customers', fontsize=12)
        plt.show()
        print("Plotting complete.")
    else:
        print("Skipping age distribution plot: 'age' column not found.")
    del customers_viz
    gc.collect()
    # --- End Visualization ---

    # --- Visualization: Distribution of Calculated 'value' Score ---
    # This requires recalculating or loading 'value_agg' again, might be slow
    # Or plot from the filtered 'train_fo' if it was kept before deletion in Strat 1 part
    print("Generating 'value' Score Distribution plot (requires value_agg)...")
    # Placeholder: Assume value_agg is loaded/available for plotting
    # value_agg_viz = pd.read_csv('value_agg.csv') # Placeholder load
    # plt.figure(figsize=(12, 6))
    # sns.histplot(value_agg_viz['value'], bins=50, kde=False, log_scale=(True, False))
    # plt.title("Distribution of Aggregated 'value' Score (Log Scale X-axis)", fontsize=16)
    # plt.xlabel("Aggregated 'value' Score (Log Scale)", fontsize=12)
    # plt.ylabel("Frequency", fontsize=12)
    # if not value_agg_viz.empty and 180 < value_agg_viz['value'].max():
    #      plt.axvline(180, color='r', linestyle='--', label='Threshold (180)')
    #      plt.legend()
    # plt.show()
    # print("Plotting complete.")
    # del value_agg_viz
    # gc.collect()
    print("Skipping 'value' plot due to complexity of regenerating data just for viz.")
    # --- End Visualization ---

    # --- Visualization: Time Decay Factor 'y' Curve ---
    print("Generating Time Decay Factor 'y' Curve plot...")
    weeks_ago = np.linspace(0.1, 52, 100)
    y_decay = np.maximum(1200 / weeks_ago, 25).astype(np.float32) # Use float32
    plt.figure(figsize=(10, 5))
    plt.plot(weeks_ago, y_decay)
    plt.title('Time Decay Factor "y" vs. Weeks Since Purchase', fontsize=16)
    plt.xlabel('Weeks Since Purchase', fontsize=12)
    plt.ylabel('Decay Factor "y"', fontsize=12)
    plt.grid(True)
    plt.show()
    print("Plotting complete.")
    # --- End Visualization ---

    # --- Visualization Aid: Show Top Articles for Contrasting Age Groups ---
    print("Showing Top 12 Article examples for contrasting age groups...")
    # Requires 'age_articles_str' to be available
    if 'age_articles_str' in locals() and 'names' in locals(): # Check if needed vars exist
        try:
            young_group_label = names[1]
            young_articles_str = age_articles_str.loc[age_articles_str['age_group'] == young_group_label, 'age_popular_str'].iloc[0]
            print(f"\nTop 12 articles for Age Group ~{16 + young_group_label*4}-{16 + (young_group_label+1)*4 -1}:")
            print(young_articles_str)

            middle_age_group_label = names[9]
            middle_articles_str = age_articles_str.loc[age_articles_str['age_group'] == middle_age_group_label, 'age_popular_str'].iloc[0]
            print(f"\nTop 12 articles for Age Group ~{16 + middle_age_group_label*4}-{16 + (middle_age_group_label+1)*4 -1}:")
            print(middle_articles_str)
            print("\n(Shows difference in popular items across age groups)")
        except Exception as e:
            print(f"Could not display age group examples: {e}")
    else:
        print("Skipping age group examples: 'age_articles_str' or 'names' not found at this point.")
    # print("Skipping age group examples display in code modification for simplicity.") # Simpler than robust check
    # --- End Visualization Aid ---

else:
    print("PLOT_VISUALIZATIONS flag is set to False. Skipping visualizations.")

print("\nScript finished.")