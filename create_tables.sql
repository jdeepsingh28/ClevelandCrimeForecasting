CREATE TABLE crimes (
	object_id	VARCHAR, 
	primary_key	VARCHAR, 
	case_number	VARCHAR, 
	district	VARCHAR, 
	ucr_desc	VARCHAR, 
	time_group	VARCHAR, 
	reported_date	TIMESTAMP, 
	offense_month	INT, 
	offense_day	INT, 
	time_block	VARCHAR, 
	dow_name	VARCHAR, 
	dow	INT, 
	hour_of_day	INT, 
	days_ago	INT, 
	offense_date	TIMESTAMP, 
	statute	VARCHAR, 
	city	VARCHAR, 
	zip	INT, 
	stat_desc	VARCHAR, 
	address_public	VARCHAR, 
	std_parcelpin	VARCHAR, 
	ward	VARCHAR, 
	census_tract	VARCHAR, 
	census_tract_geoid	VARCHAR, 
	census_block_group	VARCHAR, 
	census_bg_geoid	VARCHAR, 
	census_block	VARCHAR, 
	census_block_geoid	VARCHAR, 
	lat	FLOAT, 
	lon	FLOAT, 
	PRIMARY KEY (primary_key)
); 

CREATE TABLE census_blocks (
	geoid	VARCHAR, 
	geometry	VARCHAR, 
	lat_center	FLOAT, 
	lon_center	FLOAT, 
	PRIMARY KEY (geoid)
); 

CREATE TABLE historical_weather (
	lat	FLOAT, 
	lon	FLOAT, 
	date	DATE, 
	temp_max	FLOAT, 
	temp_min	FLOAT, 
	daylight_duration	FLOAT, 
	precipitation_sum	FLOAT, 
	precipitation_hours	FLOAT, 
	PRIMARY KEY (lat, lon, date)
); 



CREATE TABLE forecasted_weather (
	lat	FLOAT, 
	lon	FLOAT, 
	date	DATE, 
	temp_max	FLOAT, 
	temp_min	FLOAT, 
	daylight_duration	FLOAT, 
	precipitation_sum	FLOAT, 
	precipitation_hours	FLOAT, 
	PRIMARY KEY (lat, lon, date)
); 