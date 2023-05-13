'''
Set of transformation functions for the final Workflow
'''

# Imports
import pandas as pd
import re
import numpy as np
from math import log
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from itertools import groupby
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def tweetFilter(df, languages = ['en']):
    """   
    Takes as an argument a DataFrame of Tweets and applies the propsed filters:
        - Filter out non-English tweets: The NLP model used is trained only on English tweets, to preserve the quality of the analysis;
        - Remove mentions: Mentions in Tweets are removed, to reduce noise in the text;
        - Remove URLs: URLs are also removed, for the same reasons as above;
        - Remove undefined characters: Some tweets contain undefined non-parsed characters that were removed to preserve the quality of the data

    Returns a filtered DataFrame of Tweets
    """

    # Filter languages
    filtered_df = df[df.lang.isin(languages)]
    # Remove URLs
    filtered_df.loc[:,'text'] = filtered_df.text.apply(lambda x: re.sub("http\S+", "", x))
    # Remove mentions
    filtered_df.loc[:,'text'] = filtered_df.text.apply(lambda x: re.sub("\@(\w)*", "@user", x))
    # Correct abbreviations
    abbreviation_dict = {
        '\&amp;': '&',
        '\&gt;' : '>',
        '\&lt': '<'
        }
    for abbreviation, correct in abbreviation_dict.items():
        filtered_df.loc[:,'text'] = filtered_df.text.apply(lambda x: re.sub(abbreviation, correct, x))

    return filtered_df
    

def edgeListMaker(user_info, follows, tweets, save_edgelist=False, edgelist_path="./edgelist.csv"):
    '''
    Takes as an argument a DataFrame of users, follows, and tweets.
    Creates an undirected graph with users as nodes and an edge connecting those users
    that follow each other. The edge weight is given by the strength of their relationship (see documentation for details).
    Saves the edgelist as a CSV file apt for use in both NetworkX and Gephi.
    Returns an edgelist as dataframe.
    '''
    # Read users info (to get # following)
    users = user_info[["id", "following_count"]]

    # Extract ids
    edgelist = follows[["mainUserID", "id"]]

    # Filter only follows between members of the community
    edgelist = edgelist[edgelist.id.isin(edgelist.mainUserID)]

    # Get number of tweets per user (for the period)
    tweets = tweets.groupby("author_id").size().reset_index().rename(columns={0: "tweets"})

    # Create a unique key to unify edges
    edgelist["edge"] = edgelist[["mainUserID", "id"]].min(axis=1).astype(str) + "-" + edgelist[["mainUserID", "id"]].max(axis=1).astype(str) # Order edges
    edge_count = edgelist.groupby("edge").size().reset_index() # Get reciprocate edges
    edgelist.drop_duplicates("edge", inplace=True)
    edgelist = edgelist.merge(edge_count, on="edge", how="left")
    edgelist = edgelist[edgelist[0] == 2]

    # Drop auxiliary "edge" column
    edgelist.drop("edge", axis=1, inplace=True)
    edgelist.drop(0, axis=1, inplace=True)

    # Rename columns
    edgelist.columns = ["source", "target"]

    # Keep only largest connected component
    G = nx.from_pandas_edgelist(edgelist)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        edgelist = nx.to_pandas_edgelist(G)

    ## Calculate average weight for edges
    # Get following counts
    edgelist = edgelist.merge(users, left_on="source", right_on="id", how="left")
    edgelist.drop("id", axis=1, inplace=True)
    edgelist = edgelist.merge(users, left_on="target", right_on="id", how="left")
    edgelist.drop("id", axis=1, inplace=True)

    # Get tweet count
    edgelist = edgelist.merge(tweets, left_on="source", right_on="author_id", how="left")
    edgelist.drop("author_id", axis=1, inplace=True)
    edgelist = edgelist.merge(tweets, left_on="target", right_on="author_id", how="left")
    edgelist.drop("author_id", axis=1, inplace=True)
    
    # Calculate 1/probability (to avoid small floating points)
    # Take care of users with no Tweets
    edgelist.loc[:,["tweets_x", "tweets_y"]] = edgelist.loc[:,["tweets_x", "tweets_y"]].fillna(0)
    # Multiply the tweets over the following counts   
    edgelist["weight"] = np.sqrt((edgelist["tweets_x"] / edgelist["following_count_x"]) * (edgelist["tweets_y"] / edgelist["following_count_y"]))

    # Cleanup
    edgelist.drop(["following_count_x", "following_count_y", "tweets_x", "tweets_y"], axis=1, inplace=True)

    # Change column names
    edgelist.columns = ["source", "target", "weight"]

    if save_edgelist:
        edgelist.to_csv(edgelist_path, encoding="utf-8", index=False)

    return edgelist


def _add_start_end_dates(df, id, start_date, end_date):
    """Aligns the activity range by "filling" missing dates for a user with zero counts.
    Ranges may vary if a user has not tweeted on the exact start or end date. 

    Args: 
        df (pd.DataFrame): DataFrame for unique user
        id (int): User author_id
    Returns:
        df (pandas.DataFrame): DataFrame for unique user with full date range
    """
    # get range of active dates for specific user
    active_range = df[df['author_id'] == id]['created_at'].unique()

    # add start and end dates if not there
    if start_date not in active_range:
        start_row = {'author_id': id, 'created_at': start_date, 'count':0}
        df = df.append(start_row, ignore_index=True)

    if end_date not in active_range:
        end_row = {'author_id': id, 'created_at': end_date, 'count':0}
        df = df.append(end_row, ignore_index=True)

    return df


def _get_tweet_counts(df, timestep, cutoff=False):
    """Calculates tweet activity levels (# of tweets) for all users by specified timestep.
    
    Args: 
        df (pandas.DataFrame): DataFrame with tweets, author_id, etc.
        timestep (str): size of timestep as accepted by the pandas.Grouper "freq" parameter.
    
    Returns: 
        all_counts (pands.DataFrame): DataFrame with author_id as rows and timestep counts as columns.
    """
    # get daily counts for all users
    user_freq = df.groupby(['author_id', 'created_at']).size()
    user_freq = pd.DataFrame(user_freq).reset_index()
    user_freq = user_freq.rename(columns={0: "count"})

    # get list of unique user ID's
    user_list = df.author_id.unique()

    # get date range for entire network
    start_date = pd.to_datetime(df['created_at'].min())
    if not cutoff:
        end_date = pd.to_datetime(df['created_at'].max())
    else: 
        end_date = cutoff

    first = True
    for i in user_list:
        # get list of active days 
        unique_user = user_freq[user_freq['author_id'] == i]
        # add start and end dates for complete range
        unique_user = _add_start_end_dates(unique_user, i, start_date, end_date)
        # group by frequency
        timestep_counts = unique_user.groupby([pd.Grouper(key='created_at', freq=timestep)])['count'].sum()

        # add frequency counts to new DF
        timestep_counts = pd.DataFrame(timestep_counts).reset_index()
        timestep_counts = timestep_counts.set_index('created_at')
        timestep_counts = timestep_counts.rename(columns={"count": i}) 
        
        if first:
            all_counts = timestep_counts.T
            first = False
        else: 
            all_counts = all_counts.append(timestep_counts.T)
    
    return all_counts


def _get_user_activity(df, max_gap, min_activity):
    """Finds .
    
    Args: 
        df (pandas.DataFrame): DataFrame with author_id as rows and timestep counts as columns.
        max_gap (int): Maximum number of consecutive gaps.
        min_activity (int): Minimum tweet activity per timestep. 
    
    Returns: 
        users_within_limit (list): list of all user ID's within specified limits.
    """
    df_trans = df.T
    user_list = df_trans.columns

    users_exceeding_limit = []
    users_within_limit = []
    
    # calculate consecutive gaps in user activity
    for i in user_list:
        unique_user = pd.DataFrame(df_trans[i])
        unique_user['above_threshold'] = unique_user >= min_activity
        consecutive_gaps = [len(list(gaps)) for is_above, gaps in groupby(unique_user['above_threshold']) if is_above == False]

        # register users according to activity level
        exceeds_threshold = False
        for gap in consecutive_gaps:
            if gap > max_gap:
                users_exceeding_limit.append(i)
                exceeds_threshold = True
                break
        if not exceeds_threshold:
            users_within_limit.append(i)

    return users_within_limit


def getActiveUsers(df, timestep, max_gap, min_activity):
    """Retrieves list of users that have activity level within specified limits.

    Args: 
        df (pandas.DataFrame): pandas dataframe with all tweets, author_id, etc.
        timestep (str): size of timestep in format accepted by pandas.Grouper "freq" parameter.
        max_gap (int): maximum length of consecutive gaps below minimum activity level.
        min_activity (int): minimum number of tweets within timestep.
    
    Returns:
        users_within_limit (list): list of all user ID's within specified limits.
    """
    all_user_counts = _get_tweet_counts(df, timestep, cutoff=False)
    users_within_limit = _get_user_activity(all_user_counts, max_gap, min_activity)

    return users_within_limit


def probabilites2entropy(probs):
    '''
    Takes the dataframe resulted from TweetNLP classifier, a column containing a dictionary with emotion probabilities. 
    It reshapes the dataframe to have emotion probabilities as individual columns, calculates the normalized entropy and
    appends it to a new column.
    '''
    # Extract the predictions and calculate the entropy
    joy = []
    optimism = []
    anger = []
    sadness = []

    for ix, row in probs.iterrows():
        joy.append(row["probability"]["joy"])
        optimism.append(row["probability"]["optimism"])
        anger.append(row["probability"]["anger"])
        sadness.append(row["probability"]["sadness"])

    probs["joy"] = joy
    probs["optimism"] = optimism
    probs["anger"] = anger
    probs["sadness"] = sadness
    probs.drop("probability", axis=1, inplace=True)

    # Calculate the Normalized Entropy
    probs["nEntropy"] = probs[["joy", "optimism", "anger", "sadness"]].apply(stats.entropy, axis=1)
    probs["nEntropy"] = probs.nEntropy / (np.log(4))

    return probs


def undefine(tweet_emotions, entropy):
    """
    Takes a Tweets Dataframe and an entropy threshold. Sets all emotion probabilities to zero for those above the entropy
    threshold.
    Returns a new DF.
    """
    df = tweet_emotions.copy()

    df.loc[df.nEntropy > entropy, ["joy", "optimism", "anger", "sadness"]] = 0

    return df


def remove_noise(tweet_emotions):
    """
    Keep highest probable emotion and set remaining to zero.
    """
    # Work with a copy
    df = tweet_emotions.copy()

    # Emotion columns
    cols = ["joy", "optimism", "anger", "sadness"]
    
    # Max probability emotion
    df["max"] = df[cols].max(axis=1)
    
    # Filter emotions
    for c in cols:
        df.loc[:,c] = df[c].where(df[c] == df["max"], 0)

    # Remove aux max colum
    df.drop("max", axis=1, inplace=True)
    
    return df


def find_weekly_values(df, half_max_retweets = True, retweet_values = 0.5, 
                       index_order = ['author_id', 'week']):
    if not 'week' in df.columns:
        df = add_weeks(df)
    means = find_means(df.copy(), retweet_values)
    max_value = find_max(df, half_max_retweets, retweet_values)
    weekly_value = aggregate(means, max_value)
    weekly_value = fill_empty_weeks(weekly_value)
    weekly_value = weekly_value.reset_index().sort_values(["author_id", "week"], ignore_index=True)
    return weekly_value


def aggregate(means, maxs):
    value_columns = ['optimism', 'joy', 'anger', 'sadness']
    for col in value_columns:
        means['agg_' + col] = means['mean' + col] * maxs['max_' + col]
    # Return only the relevant columns
    return means[['agg_' + col for col in value_columns] + ['weekly_tweet_values']]
    

def find_means(df, retweet_values):
    value_columns = ['optimism', 'joy', 'anger', 'sadness']
    # Setting the value for each tweet
    df['tweet_value'] = [retweet_values if retweet else 1 for retweet in df.retweeted]
    
    for column in value_columns:
        df[column] = df[column] * df['tweet_value']
    # Finding the sum for each week
    weekly_sum = df.groupby(['week', 'author_id'])[value_columns + ['tweet_value']].agg('sum')
    # Adjusting for the sum of tweets for each week
    for col in value_columns:
        weekly_sum[col] = weekly_sum[col] / weekly_sum['tweet_value']
    
    name_change_dict = {}
    for name in value_columns:
        name_change_dict[name] = 'mean' + name
        
    name_change_dict['tweet_value'] = 'weekly_tweet_values'
    return weekly_sum.rename(name_change_dict, axis = 'columns')
    
    
def find_max(df, half_max_retweets, retweet_values):
    value_columns = ['optimism', 'joy', 'anger', 'sadness']
    if half_max_retweets:
        df[value_columns][df.retweeted] *= retweet_values
    max_values = df.groupby(['week', 'author_id'])[value_columns].agg('max')
    
    # Change the name to max-value:
    name_change_dict = {}
    for name in value_columns:
        name_change_dict[name] = 'max_' + name 
    return max_values.rename(name_change_dict, axis='columns')

    
def add_weeks(df):
    df['week'] = ((pd.to_datetime(df['created_at']).dt.dayofyear) / 7).astype('int')
    return df


def fill_empty_weeks(df):    
    # Create the dataframe where all weeks are filled
    weeks = df.index.get_level_values('week').unique()
    authors = df.index.get_level_values('author_id').unique()
    
    all_values = [weeks, authors]
    
    all_combinations = pd.MultiIndex.from_product(all_values, names=["author", "week"])
    
    
    difference = all_combinations.difference(df.index, sort=None)
    columns = df.columns
    diff_df = pd.DataFrame(np.zeros((len(difference), len(columns))), columns = columns, index = difference)
    return pd.concat([df, diff_df])


def interpolate_values(df, activity_threshold, emotion_idx, act_idx):
    # handle first row
    if df.iloc[0,act_idx] < activity_threshold:
        for i in emotion_idx:
            frac = df.iloc[0,act_idx] / activity_threshold
            df.iloc[0,i] = (df.iloc[1,i] * (1-frac)) + (df.iloc[0,i] * frac)

    # handle last row
    if df.iloc[(df.shape[0]-1),act_idx] < activity_threshold:
        for i in emotion_idx:    
            frac = df.iloc[(df.shape[0]-1),act_idx] / activity_threshold
            df.iloc[(df.shape[0]-1),i] = (df.iloc[df.shape[0]-2,i] * (1-frac)) + (df.iloc[df.shape[0]-1,i] * frac)

    # loop through all remaining rows:
    for i in range(1, df.shape[0]-1):
        if df.iloc[i,act_idx] < activity_threshold:
            for emo in emotion_idx:
                # weigh interpolation value by potential activity < threshold
                frac = df.iloc[i,act_idx] / activity_threshold
                interpolated_value = (df.iloc[i-1,emo] + df.iloc[i+1,emo]) / 2
                interpolated_frac =  interpolated_value * (1-frac)
                existing_value_frac = df.iloc[i,emo] * frac
                final_value = interpolated_frac + existing_value_frac
                df.iloc[i,emo] = final_value
    
    return df


def GraphProperties(G, communities=[]):

    avg_degree = np.mean([v for k,v in list(nx.degree(G))])
    avg_weighted_degree = np.mean([v for k,v in list(nx.degree(G, weight="weight"))])

    print("-"*24)
    print(f"# Of Nodes: {len(G.nodes)}")
    print(f"# Of Edges: {len(G.edges)}")
    print(f"Avg. Degree: {avg_degree:.2f}")
    print(f"Avg. Weighted Degree: {avg_weighted_degree:.2f}")
    print(f"Diameter: {nx.diameter(G)}")
    print(f"Avg. Path Length: {nx.average_shortest_path_length(G, weight='weight'):.3f}")
    print(f"Density: {nx.density(G):.4f}")
    print(f"Avg. Clustering Coeff.: {nx.average_clustering(G, weight='weight', count_zeros=True):.3f}")
    if not len(communities) == 0:
        print(f"Modularity.: {nx.community.modularity(G, communities, weight='weight'):.3f}")
    print("-"*24)


def summaryStats(df, palette, title):

    stats = pd.DataFrame()
    stats = df.loc[:,["joy", "optimism", "anger", "sadness"]].describe().T
    stats.drop("count", axis=1, inplace=True)
    stats.loc[:,stats.columns[1:]] = stats.astype(float).applymap('{:.3f}'.format)

    print(stats)
    _ = pd.melt(df, value_vars=["optimism", "joy", "anger", "sadness"])
    sns.set(rc={'figure.figsize':(12,4)})
    #sns.kdeplot(data=_, x="value", hue="variable", fill=True, common_norm=True, palette=palette, alpha=.3, linewidth=0)
    bp = sns.boxplot(data=_, x="value", y="variable", palette=palette)
    bp.set(ylabel=None)
    bp.set(title=title)
    plt.show()


def visualize_mean_sd(o_df, palette, net):
    """
    Function to vizualise the mean and standard deveation for all authors in a dataframe, for each of the emotions
     """
    # Saving only the emotions and the author, as they are the only relevant info
    df = o_df[['author_id', 'joy', 'anger', 'sadness', 'optimism']]
    # Finding the means for each author, and saving it as a frame with the columns ['id', 'feeling', 'mean']
    means = df.groupby('author_id').mean()
    means = pd.melt(means.reset_index(), id_vars = 'author_id', 
                   value_vars = ['joy', 'anger', 'sadness', 'optimism'], 
                   var_name = 'feeling',
                   value_name = 'mean',
                  )

    # Finding the standard distribution, and saving it as a frame with the columns ['id', 'feeling', 'mean']
    sd = df.groupby('author_id').std()
    sd = pd.melt(sd.reset_index(), id_vars = 'author_id', 
                   value_vars = ['joy', 'anger', 'sadness', 'optimism'], 
                   var_name = 'feeling',
                   value_name = 'standard deviation',
                  )
    
    # Combining the frames 
    sd_and_mean_df = pd.merge(means, sd, left_on = ['author_id', 'feeling'], right_on =  ['author_id', 'feeling'])

    #Making the plot
    fig = sns.scatterplot(data = sd_and_mean_df, x = 'mean', y = 'standard deviation', hue = 'feeling', 
        palette = palette, size = 0.3, alpha = 0.7)
    fig.set(title = f"Standard deviation and mean for {net}")
    plt.show()


def cramers_v(contingency_table):
    """
    Takes a contingency table with networks as rows and emotions as columns.
    Returns Chi square statistics, expected table and Cramer's V test.
    """
    dataset = contingency_table.values
    # Chi-squared test statistic, sample size, and minimum of rows and columns
    X2, pvalue, dof, expected_freq = stats.chi2_contingency(dataset, correction=False)
    N = np.sum(dataset)
    minimum_dimension = min(dataset.shape)-1
    
    # Calculate Cramer's V
    result = np.sqrt((X2/N) / minimum_dimension)
    
    # Print the result
    print(f"Chi Square Test")
    print(f"T-Statistic: {X2}")
    print(f"Degress of Freedom: {dof}")
    print(f"P-Value: {pvalue}")
    print(f"Observed Frequencies:")
    print(contingency_table)
    print(f"Expected Frequencies:")
    print(pd.DataFrame(expected_freq.astype(int), index=contingency_table.index, columns=contingency_table.columns))
    print()
    print(f"Cramer's V Test: {result}")


def load_annotations(path):
    """
    Given a path, loads annotation batches, merges and returns a dataframe with samples and labels.
    """
    samples_list = []
    labels_list = []

    # Read text and labels from annotation's batch summaries
    for f in os.listdir(path):

        if f.endswith(" - Samples.csv"):
            t = pd.read_csv(f"{path}/{f}", encoding="utf-8")
            samples_list.append(t)
        elif f.endswith(" - Summary.xlsx"):
            l = pd.read_excel(f"{path}/{f}", sheet_name="Annotations")
            labels_list.append(l)

    # Concatenate together
    samples_df = pd.concat(samples_list).reset_index(drop=True).text
    labels_df = pd.concat(labels_list).reset_index(drop=True).GOLDEN

    return samples_df, labels_df


def mean_change(odf, emotions):
    """Find the mean change across the network each week for each feeling given a sub-network."""
    df = odf.copy()
    df = df.sort_values(['author_id', 'week'])

    for emotion in emotions:

        df[emotion + '_change'] = df[emotion].diff().abs()

    return df[df.week != 0]


def absolute_change(net, emotion_order):
    """Find the absolute change across the network each week for each feeling, given a sub-nework"""
    overlap = [i % 51 != 0 for i in range(int(len(net)/4))][:-1]
    overlap[0] = True

    values = net.set_index(['author_id', 'emotion', 'week'])
    values.sort_index()


    moved = values.diff().abs().reset_index()
    moved = moved[moved['week'] != 50]
    final = moved.groupby(['emotion', 'week']).value.sum()
    final = final.reset_index()
    emotions = []
    for emotion in emotion_order:
        emotions.extend(final[final.emotion == emotion].value)

    return emotions
    
def see_sub_grup_weekly_change(odf: pd.DataFrame, week : int, emotion : str, window : int, net : str, show_absolute_emotion = False, show_relative_emotion = False):
    """Function to show how an emotion changes within subgroups within a window from a spicific week."""
    df = odf.copy()
    # Looking only at the emotion and the window
    df = df[df['variable'] == emotion]
    df = df[(df.week > week-window) & (df.week < week + window)]
    nr_groups = len(df.Group.unique())
    # Adding the sum of the emotion for the week, to calculate the percentage each subgroup contribiutes
    rolling_sum = []
    for i in range((window-1)*2+1):
        for _ in range(nr_groups):
            rolling_sum.append(df.value[i*nr_groups:i*nr_groups+nr_groups].sum())

    df['weekly_sum'] = rolling_sum
    df[f'Percentage {emotion}'] = df['value'] / df['weekly_sum']

    # Finding the absolute change for each group per week
    df  = df.sort_values(by = ['Group', 'week'])
    df['abs_change'] = df.value.diff()

    # Show to better understand
    if show_absolute_emotion:
        sns.lineplot(data = df, x = 'week', y = 'value', hue = 'Group', palette = 'rocket').set(title = f'total expression of {emotion} for {net}')
        plt.show()
    if show_relative_emotion:
        sns.lineplot(data = df, x = 'week', y = f'Percentage {emotion}', hue = 'Group', palette = 'rocket').set(title = f'percentage expression of {emotion} for {net}')
        plt.show()

    # Display the DF with the information
    print(f"Looking for change in the {net} for changes in {emotion} in week {week}")
    display(df)
    # Adding a clear divider between entries
    print("-"*100)