

import pandas as pd

def calculate_distance_matrix(df):
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): DataFrame with columns 'A', 'B', and 'Distance'.

    Returns:
        pandas.DataFrame: Distance matrix
    """

    df_copy = df.copy()

    pivot_table = df_copy.pivot(index='id_start', columns='id_end', values='distance')


    pivot_table = pivot_table.fillna(0)


    distance_matrix = pivot_table + pivot_table.T


    rows = range(distance_matrix.shape[0])
    distance_matrix.values[rows, rows] = 0

    return distance_matrix

df = pd.read_csv('dataset-3.csv') 
result = calculate_distance_matrix(df)
print(result)











def unroll_distance_matrix(distance_matrix, nan_placeholder=-1):
    """
    Unroll a distance matrix into a DataFrame.

    Args:
        distance_matrix (pandas.DataFrame): Input distance matrix.
        nan_placeholder (float): Placeholder for NaN values.

    Returns:
        pandas.DataFrame: Unrolled DataFrame with columns 'id_start', 'id_end', and 'distance'.
    """
    # Extract unique IDs from the distance matrix
    unique_ids = distance_matrix.index

    # Initialize an empty list to store unrolled data
    unrolled_data = []

    # Iterate over unique combinations of IDs
    for start in unique_ids:
        for end in unique_ids:
            # Exclude same start and end combinations
            if start != end:
                distance = distance_matrix.at[start, end]

                # Replace NaN with a placeholder
                if pd.isna(distance):
                    distance = nan_placeholder

                # Append data to the list
                unrolled_data.append({'id_start': start, 'id_end': end, 'distance': distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Assuming result_matrix is the distance matrix from Question 1
# If not, you can replace it with the actual distance matrix.
result = calculate_distance_matrix(df)

# Call the unroll_distance_matrix function
unrolled_df = unroll_distance_matrix(result)

# Print or further use the unrolled_df
print(unrolled_df)












def find_ids_within_ten_percentage_threshold(df, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame with columns 'id_start', 'id_end', and 'distance'.
        reference_id (int): Reference ID for calculating the average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """

    reference_data = df[df['id_start'] == reference_id]


    reference_avg_distance = reference_data['distance'].mean()


    threshold_lower = reference_avg_distance - 0.1 * reference_avg_distance
    threshold_upper = reference_avg_distance + 0.1 * reference_avg_distance

    print(f"Reference ID: {reference_id}")
    print(f"Reference Average Distance: {reference_avg_distance}")
    print(f"Threshold Lower: {threshold_lower}")
    print(f"Threshold Upper: {threshold_upper}")

    result_df = df[(df['distance'] >= threshold_lower) & (df['distance'] <= threshold_upper)]

    return result_df


unrolled_df = unroll_distance_matrix(result_matrix)

reference_id = 1001400
result_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

print(result_within_threshold)












def calculate_toll_rate(df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): Unrolled DataFrame with columns 'id_start', 'id_end', 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with added columns 'moto', 'car', 'rv', 'bus', and 'truck'
                          representing toll rates for each vehicle type.
    """
   
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Add columns for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df


unrolled_df = unroll_distance_matrix(result_matrix)

result_with_toll_rate = calculate_toll_rate(unrolled_df)

print(result_with_toll_rate)












import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df):
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): DataFrame with columns 'id_start', 'id_end', 'distance', 'moto', 'car', 'rv', 'bus', 'truck'.

    Returns:
        pandas.DataFrame: DataFrame with added columns 'start_day', 'start_time', 'end_day', 'end_time'
                          representing time intervals and modified toll rates based on the specified time ranges.
    """

    required_columns = ['id_start', 'id_end', 'distance', 'moto', 'car', 'rv', 'bus', 'truck']
    if not set(required_columns).issubset(df.columns):
        raise ValueError("Input DataFrame is missing required columns.")

    time_ranges = {
        'weekday_morning': (time(0, 0, 0), time(10, 0, 0)),
        'weekday_afternoon': (time(10, 0, 0), time(18, 0, 0)),
        'weekday_evening': (time(18, 0, 0), time(23, 59, 59)),
        'weekend': (time(0, 0, 0), time(23, 59, 59))
    }

    discount_factors = {
        'weekday_morning': 0.8,
        'weekday_afternoon': 1.2,
        'weekday_evening': 0.8,
        'weekend': 0.7
    }


    start_days, start_times, end_days, end_times, rates = [], [], [], [], []


    unique_pairs = df[['id_start', 'id_end']].drop_duplicates()
    for index, row in unique_pairs.iterrows():
        id_start, id_end = row['id_start'], row['id_end']

        for time_range, (start_time, end_time) in time_ranges.items():
            start_days.extend(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            start_times.extend([start_time] * 7)
            end_days.extend(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            end_times.extend([end_time] * 7)

            if time_range == 'weekend':
                rate = df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'distance'].mean() * discount_factors[time_range]
            else:
                rate = df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'distance'].sum() * discount_factors[time_range]

            rates.extend([rate] * 7)

 
    result_df = pd.DataFrame({
        'start_day': start_days,
        'start_time': start_times,
        'end_day': end_days,
        'end_time': end_times,
        'rate': rates
    })

    return result_df


result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rate)

print(result_with_time_based_toll_rates)

