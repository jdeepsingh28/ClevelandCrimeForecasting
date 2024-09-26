import pandas as pd

# (1) Load both datasets
#crime data
crime_population = pd.read_csv("/Users/eviemiller/Documents/Capstone/crime-population-merged")
#census data
race = pd.read_csv("/Users/eviemiller/Documents/Capstone/race_census_cleveland.csv")

# convert FIPS in race to string for merge
race['Geographic Identifier - FIPS Code'] = race['Geographic Identifier - FIPS Code'].astype(str)

# convert FIPs in crime_population to string for merge
crime_population['FIPS'] = crime_population['FIPS'].astype(str)

# merge crime_population with race on FIPs
crime_population_crime_merged = pd.merge(crime_population, race, left_on='FIPS', right_on='Geographic Identifier - FIPS Code', how='left')

crime_population_crime_merged.to_csv("/Users/eviemiller/Documents/Capstone/crime-population-race-merged.csv", index=False)
print(crime_population_crime_merged.head())