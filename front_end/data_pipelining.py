# This file is going to require changes to the way it takes dates and formats them
# Might also have to paginate the responses since it will get a shit ton of data

import pandas as pd
import requests
from datetime import datetime


def run_pipeline(early_time_bound, late_time_bound):

    # Dates

    print(early_time_bound)
    print(late_time_bound)
    # early_time_bound = '2024-9-02'
    # late_time_bound = '2024-10-02'

    # Define the URL to query
    url = f"https://services3.arcgis.com/dty2kHktVXHrqO8i/arcgis/rest/services/Crime_Incidents/FeatureServer/0/query?where=ReportedDate%20%3E=%20DATE%20%27{early_time_bound}%27%20AND%20ReportedDate%20%3C=%20DATE%20%27{late_time_bound}%27&outFields=*&returnIdsOnly=true&outSR=4326&f=json"

    # Function to split a list into chunks of a specified size

    def chunk_list(data, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    # Send a GET request to fetch the data
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the response JSON data to a Python dictionary
        data = response.json()

        # Extract the relevant part of the data (we want object ids)
        ids = data['objectIds']

        df = pd.DataFrame(ids, columns=['ObjectID'])

        print(f"Number of records: {len(df)}")
        '''
        print(df.describe())

        print(df.head())
        '''

        chunk_size = 250
        object_id_chunks = list(chunk_list(ids, chunk_size))

        all_features = []

        for object_id_chunk in object_id_chunks:
            # Convert the chunk of IDs to a comma-separated string
            object_id_list = ','.join(map(str, object_id_chunk))

            # Define the URL to query the features for the given ObjectIDs
            url_features = f"https://services3.arcgis.com/dty2kHktVXHrqO8i/arcgis/rest/services/Crime_Incidents/FeatureServer/0/query?objectIds={object_id_list}&outFields=*&f=json"

            # print(f"url feature: {url_features}")
            # Send a GET request to fetch the data for the ObjectIDs in this chunk
            response_features = requests.get(url_features)

            if response_features.status_code == 200:
                # Convert the response JSON data to a Python dictionary
                data_features = response_features.json()

                # Extract the relevant part of the data (the 'features' field contains the crime data)
                features = data_features.get('features', [])

                # Append the features from this chunk to the overall list
                all_features.extend(features)
            else:
                print(
                    f"Failed to retrieve features for chunk: {response_features.status_code}")

        # Once all features are collected, normalize into a DataFrame
        if all_features:
            # Normalize the 'attributes' part of each feature into a DataFrame
            df = pd.json_normalize([feature['attributes']
                                    for feature in all_features])

            print(f"Number of records: {len(df)}")
            print(df.describe())

            # Display the first few rows of the DataFrame
            print(df.head())
        else:
            print("No features found in the responses.")

    else:
        print(f"Failed to retrieve data: {response.status_code}")

    return df
