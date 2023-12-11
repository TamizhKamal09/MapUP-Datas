#!/usr/bin/env python
# coding: utf-8

# # Question 1: Distance Matrix Calculation

# In[9]:


import pandas as pd


# In[10]:


get_ipython().system('pip install networkx')


# In[11]:


df = pd.read_csv("dataset-3.csv")
df.head()


# In[12]:


import pandas as pd
import networkx as nx

def calculate_distance_matrix(df) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges and distances to the graph
    for index, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])
        G.add_edge(row['id_end'], row['id_start'], distance=row['distance'])  # Ensure bidirectional distances

    # Calculate the distance matrix
    distance_matrix = nx.floyd_warshall_numpy(G, weight='distance', nodelist=sorted(G.nodes))

    # Convert the distance matrix to a DataFrame
    distance_df = pd.DataFrame(distance_matrix, index=sorted(G.nodes), columns=sorted(G.nodes))

    return distance_df

# Assuming df is your dataset DataFrame
df = pd.read_csv("dataset-3.csv")
distance_result = calculate_distance_matrix(df)
print(distance_result)



# # Question 2: Unroll Distance Matrix

# In[13]:


def unroll_distance_matrix(df) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create an empty DataFrame to store unrolled data
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Iterate through each row of the distance matrix DataFrame
    for index, row in df.iterrows():
        id_start = row.name  # Get the id_start from the row index
        # Iterate through each column of the distance matrix
        for id_end, distance in row.items():
            # Exclude same id_start to id_end combinations
            if id_start != id_end:
                unrolled_df = unrolled_df.append({'id_start': id_start, 'id_end': id_end, 'distance': distance}, ignore_index=True)

    return unrolled_df

# Assuming distance_result is the DataFrame from Question 1
unrolled_result = unroll_distance_matrix(distance_result)
print(unrolled_result)


# # Question 3: Finding IDs within Percentage Threshold

# In[20]:


def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter rows where id_start is equal to the reference_id
    reference_rows = df[df['id_start'] == reference_id]

    if reference_rows.empty:
        # If no rows found for the reference_id, return an empty DataFrame
        return pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Calculate the average distance for the reference_id
    reference_average_distance = reference_rows['distance'].mean()

    # Calculate the percentage threshold (10%)
    percentage_threshold = 0.10

    # Calculate the threshold range for acceptable distances
    lower_threshold = reference_average_distance - (reference_average_distance * percentage_threshold)
    upper_threshold = reference_average_distance + (reference_average_distance * percentage_threshold)

    # Filter rows where the distance is within the threshold range
    filtered_rows = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    return filtered_rows.sort_values(by=['id_start', 'id_end'])


# # Question 4: Calculate Toll Rate

# In[21]:


def calculate_toll_rate(df) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Create columns for each vehicle type with their respective rate coefficients
    df['moto'] = 0.8 * df['distance']
    df['car'] = 1.2 * df['distance']
    df['rv'] = 1.5 * df['distance']
    df['bus'] = 2.2 * df['distance']
    df['truck'] = 3.6 * df['distance']

    return df


# # Question 5: Calculate Time-Based Toll Rates

# In[22]:


def calculate_time_based_toll_rates(df) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Convert 'start_time' and 'end_time' to datetime.time
    df['start_time'] = pd.to_datetime(df['start_time']).dt.time
    df['end_time'] = pd.to_datetime(df['end_time']).dt.time

    # Define time ranges and discount factors
    time_ranges = [(datetime.time(0, 0, 0), datetime.time(10, 0, 0)),
                   (datetime.time(10, 0, 0), datetime.time(18, 0, 0)),
                   (datetime.time(18, 0, 0), datetime.time(23, 59, 59))]

    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Apply discount factors based on time intervals
    for _, row in df.iterrows():
        for time_range, discount_factor in zip(time_ranges, weekday_discount_factors):
            if (row['start_time'] >= time_range[0] and row['start_time'] <= time_range[1]) or \
                    (row['end_time'] >= time_range[0] and row['end_time'] <= time_range[1]):
                df.loc[_, 'moto':'truck'] *= discount_factor

        # Apply weekend discount factor
        if row['start_day'] in ['Saturday', 'Sunday'] and row['end_day'] in ['Saturday', 'Sunday']:
            df.loc[_, 'moto':'truck'] *= weekend_discount_factor

    return df


# In[ ]:




