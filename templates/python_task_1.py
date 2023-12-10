import pandas as pd
def generate_car_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """

    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    for idx in car_matrix.index:
        if idx in car_matrix.columns:
            car_matrix.loc[idx, idx] = 0
    return car_matrix

data = pd.read_csv('dataset-1.csv')
result_matrix = generate_car_matrix(data)
print(result_matrix)







def get_type_count(df):
    """
    Categorizes 'car' values into types, adds a new column 'car_type', 
    and returns a dictionary of counts for each car_type category.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'car' column.

    Returns:
        dict: A dictionary with car_type categories as keys and their counts as values.
    """
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    car_type_counts = df['car_type'].value_counts().to_dict()
    car_type_counts = dict(sorted(car_type_counts.items()))
    return car_type_counts

encodings = ['utf-8', 'latin-1', 'ISO-8859-1']

for encoding in encodings:
    try:
        df = pd.read_csv('dataset-1.csv', encoding=encoding)
        break  
    except UnicodeDecodeError:
        continue  

result = get_type_count(df)
print(result)







def get_bus_indexes(df) -> list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    
    if 'bus' not in df.columns:
        raise ValueError("The DataFrame must contain a 'bus' column.")

    bus_mean = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    return bus_indexes

def main():
    
    file_path = ('dataset-1.csv')
    df = pd.read_csv(file_path)
    bus_indexes = get_bus_indexes(df)
    bus_indexes.sort()
    print("Indices where 'bus' values are greater than twice the mean:", bus_indexes)

if __name__ == "__main__":
    main()







def filter_routes(df: pd.DataFrame) -> list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """

    route_avg_truck = df.groupby('route')['truck'].mean()

    
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

  
    filtered_routes.sort()

    return filtered_routes


filtered_routes_list = filter_routes(df)
print(filtered_routes_list)







def multiply_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame): Input matrix DataFrame.

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """

    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    modified_matrix = modified_matrix.round(1)

    return modified_matrix


modified_matrix_df = multiply_matrix(result_matrix)
print(modified_matrix_df)










import pandas as pd
from datetime import timedelta

def time_check(df):
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: Return a boolean series
    """
    try:
       
        df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
        df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')
    
        df = df.dropna(subset=['start_datetime', 'end_datetime'])   
        df['duration'] = df['end_datetime'] - df['start_datetime']
        
    
        start_of_day = pd.to_datetime('00:00:00')
        end_of_day = pd.to_datetime('23:59:59')
        
      
        completeness_check = (
            (df['start_datetime'].dt.time == start_of_day.time()) &
            (df['end_datetime'].dt.time == end_of_day.time()) &
            (df['duration'] == timedelta(days=7))
        )
        
        return completeness_check
    
    except Exception as e:
        print(f"Error: {e}")
        print(df[df['start_datetime'].isnull() | df['end_datetime'].isnull()])
        raise


dataset_path = 'dataset-2.csv'
df = pd.read_csv(dataset_path)


result = time_check(df)

print(result)
