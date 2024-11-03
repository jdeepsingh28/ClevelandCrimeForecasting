import pandas as pd
from sklearn.preprocessing import LabelEncoder

crime = pd.read_csv(
    "/Users/nharms/Documents/College/CS/Senior_Project/crime_with_weather.csv")

# drop unwanted columns from the dataset
cols = ['census_tract_geoid', 'census_block_group', 'census_bg_geoid', 'census_block_geoid', 'lat', 'lon', 'std_parcelpin', 'city',
        'zip', 'address_public', 'ward', 'object_id', 'primary_key', 'case_number', 'district', 'reported_date', 'dow_name', 'statute',
        'stat_desc', 'time_group', 'date', 'days_ago', 'geoid']

crime = crime.drop(columns=cols)


cols = ['census_tract', 'census_block']

# Loop through each column and count 'Not Located' occurrences
for col in cols:
    count_not_located = (crime[col] == 'Not Located').sum()
    print(f"'{col}' has {count_not_located} 'Not Located' entries")

    # Function to create extracted month and day columns


def add_extracted_date_columns(crime):
    crime['offense_date'] = pd.to_datetime(
        crime['offense_date'], errors='coerce')
    # Drop rows where 'offense_date' is NaT
    crime.dropna(subset=['offense_date'], inplace=True)
    crime['extracted_month'] = crime['offense_date'].dt.month
    crime['extracted_day'] = crime['offense_date'].dt.day
    return crime

# Function to check for discrepancies


def date_discrepancy(crime, month, day):
    inconsistent_records = crime[(crime[month] != crime['extracted_month']) | (
        crime[day] != crime['extracted_day'])]
    return inconsistent_records.shape[0]


# Add extracted month and day columns first
crime = add_extracted_date_columns(crime)

# Checking discrepancies before dropping 'offense_month' and 'offense_day'
initial_discrepancies = date_discrepancy(crime, 'offense_month', 'offense_day')
print(f"Initial discrepancies: {initial_discrepancies}")

# Dropping the original columns now
crime = crime.drop(columns=['offense_month', 'offense_day'])

# Perform a final discrepancy check to confirm consistency (should be zero)
final_discrepancies = date_discrepancy(
    crime, 'extracted_month', 'extracted_day')
print(f"Final discrepancies: {final_discrepancies}")

# Only use entries from years between 2018-2022 (Census data is for those years)
# Years before do not have enough values
crime = crime[(crime['offense_year'] > 2017) & (crime['offense_year'] < 2023)]
crime.shape[0]

# Look at rows with null entries
rows_with_null = crime[crime.isnull().any(axis=1)]

duplicate_rows = crime[crime.duplicated(keep=False)]
print(f"Total exact duplicates found: {duplicate_rows.shape[0]}")

# Group duplicates and check for value differences across columns
duplicate_groups = duplicate_rows.groupby('offense_date')
for name, group in duplicate_groups:
    print(f"\nGroup: {name}")

# Drop exact duplicates and keep only the first occurrence
crime = crime.drop_duplicates()
duplicate_rows = crime[crime.duplicated(keep=False)]
print(f"Total exact duplicates found: {duplicate_rows.shape[0]}")

mappings = {}

cols = ['ucr_desc', 'offense_year',
        'time_block', 'census_tract', 'census_block']
for col in cols:
    le = LabelEncoder()
    crime[col + '_numeric'] = le.fit_transform(crime[col])

    mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

for col, mapping in mappings.items():
    print(f"{col}: {mapping}")
    print("\n")

# Drop last remaining unneeded columns
# Only remaining column that may be unnessary is OffenseDate
# It is useful to have a column in datetime format

cols = ['ucr_desc', 'offense_year',
        'time_block', 'census_tract', 'census_block']

crime = crime.drop(columns=cols)

# Important to note that we do not perform any feature scaling here
# This is because random forests are scale-invariant

crime.to_csv("crime_weather_preprocessed", index=False)
