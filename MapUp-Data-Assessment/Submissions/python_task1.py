#!/usr/bin/env python
# coding: utf-8

# # Question 1 - Car Matrix Generation

# In[40]:


import pandas as pd

df = pd.read_csv("dataset-1.csv")
df.head()


# In[39]:


def generate_car_matrix(df) -> pd.DataFrame:
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'id_1', 'id_2', and 'car' columns.

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values,
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Pivot the DataFrame to get the desired matrix
    matrix_df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    matrix_df.values[[range(len(matrix_df))]*2] = 0

    return matrix_df


# # Question 2 - Car Type Count Calculation

# In[38]:


def get_type_count(df) -> dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'car' column.

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Add a new categorical column 'car_type'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

# Assuming df is your dataset DataFrame
result_type_counts = get_type_count(df)
print(result_type_counts)



# # Question 3 - Bus Count Index Retrieval

# In[41]:


def get_bus_indexes(df) -> list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'bus' column.

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Calculate the mean of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the list in ascending order
    bus_indexes.sort()

    return bus_indexes


# Assuming df is your dataset DataFrame
result_bus_indexes = get_bus_indexes(df)
print(result_bus_indexes)



# # Question 4- Route Filtering

# In[42]:


def filter_routes(df) -> list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'route' and 'truck' columns.

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Group by 'route' and calculate the mean of 'truck' values for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' value is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of routes in ascending order
    selected_routes.sort()

    return selected_routes


# Assuming df is your dataset DataFrame
result_filtered_routes = filter_routes(df)
print(result_filtered_routes)



# # Question 5- Matrix Value Modification

# In[44]:


def generate_car_matrix(df) -> pd.DataFrame:
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'id_1', 'id_2', and 'car' columns.

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values,
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Pivot the DataFrame to get the desired matrix
    matrix_df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    matrix_df.values[[range(len(matrix_df))]*2] = 0

    return matrix_df

def multiply_matrix(matrix) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame): Input DataFrame representing the matrix.

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Apply custom conditions to multiply matrix values
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix


# Assuming df is your dataset DataFrame
result_matrix = generate_car_matrix(df)
print("Original Matrix:")
print(result_matrix)

# Modify the matrix using the multiply_matrix function
modified_result_matrix = multiply_matrix(result_matrix)
print("\nModified Matrix:")
print(modified_result_matrix)



# # Question 6- Time Check

# In[59]:


df=pd.read_csv("dataset-2.csv")
df.head()


# In[64]:


import pandas as pd

def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'id', 'name', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime', 'able2Hov2', 'able2Hov3', 'able3Hov2', 'able3Hov3', 'able5Hov2', 'able5Hov3', 'able4Hov2', 'able4Hov3'.

    Returns:
        pd.Series: Boolean series indicating if each (`id`, `id_2`) pair has incorrect timestamps.
    """
    try:
        # Combine 'startDay' and 'startTime', and 'endDay' and 'endTime' to create 'timestamp' column
        df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
        df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

        # Extract day and time components
        df['day'] = df['start_timestamp'].dt.day_name()
        df['hour'] = df['start_timestamp'].dt.hour

        # Group by ('id', 'id_2') and check if each pair covers a full 24-hour period and all 7 days
        completeness_check = df.groupby(['id', 'id_2']).apply(lambda x: (len(x['day'].unique()) == 7) and (x['hour'].nunique() == 24))

        return completeness_check
    except Exception as e:
        print(f"Error: {e}")
        problematic_rows = df[df.applymap(lambda x: isinstance(x, str) and not x.isdigit() if isinstance(x, str) else False)].dropna(how='all')

        print("Problematic Rows:")
        print(problematic_rows)
        return pd.Series(False, index=df.index)

# Assuming df is your dataset DataFrame
df = pd.read_csv("dataset-2.csv")
completeness_result = time_check(df)
print(completeness_result)





