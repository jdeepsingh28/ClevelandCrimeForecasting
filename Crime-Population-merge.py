import pandas as pd

# (1) Load both datasets
#crime data
crime = pd.read_csv("/Users/eviemiller/Documents/Capstone/violent_crime_multiple")
#census data
population = pd.read_csv("/Users/eviemiller/Documents/Capstone/population_census_cleveland.csv")

# (2) Convert census tracts to FIPs codes
def census_tract_to_fips(census_tract):
    #delete the numeric part and remove "Census Tract" prefix
    tract = census_tract.replace("Census Tract ", "").replace(".", "")
    #add Cuyahoga county code "39035" to match the FIPS code format for county
    return f"39035{tract}"
#create a FIPS column in crime data with function
crime['FIPS'] = crime['CENSUS_TRACT'].apply(census_tract_to_fips)

# (3) convert FIPs column in population to a string to match format for crime for merge
population['Geographic Identifier - FIPS Code'] = population['Geographic Identifier - FIPS Code'].astype(str)

# (4) left merge (crime is left) on FIPs to make sure all rows from crime data stay
merged_data = pd.merge(crime, population, left_on='FIPS', right_on='Geographic Identifier - FIPS Code', how='left')

#convert
merged_data.to_csv("crime-population-merged", index=False)

print(merged_data.head())
print(merged_data.columns)

