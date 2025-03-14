# imports
import csv

# open file
gdp_growth_data_usa_official = open("csvFiles/gdp_growth_data_usa_official.csv", "r")

# create reader for file
gdp_growth_data_usa_official = csv.reader(gdp_growth_data_usa_official)

# create list from file data
data_list = list(gdp_growth_data_usa_official)

# create dictionaries for file data storage
gdp_growth_data_usa_official_dictionary_sum = {}
gdp_growth_data_usa_official_dictionary_average = {}
gdp_growth_data_usa_official_dictionary_count = {}
gdp_growth_data_usa_official_dictionary_average_1954_to_2017 = {}

# populate dictionaries
for i in range(len(data_list)):
    if (i != 0):
        if data_list[i][0].split("-")[0] not in gdp_growth_data_usa_official_dictionary_sum:
            gdp_growth_data_usa_official_dictionary_sum[data_list[i][0].split("-")[0]] = float(data_list[i][1])
            gdp_growth_data_usa_official_dictionary_count[data_list[i][0].split("-")[0]] = 1
        else:
            gdp_growth_data_usa_official_dictionary_sum[data_list[i][0].split("-")[0]] += float(data_list[i][1])
            gdp_growth_data_usa_official_dictionary_count[data_list[i][0].split("-")[0]] += 1
        # data_list[i][0] = data_list[i][0].split("-")[0]
        # data_list[i][1] = float(data_list[i][1])

# populate average dictionary
for key in gdp_growth_data_usa_official_dictionary_count:
    gdp_growth_data_usa_official_dictionary_average[key] = gdp_growth_data_usa_official_dictionary_sum.get(key) / gdp_growth_data_usa_official_dictionary_count.get(key)

# select desired years
years_to_keep = list(range(1953, 2018))

# change data type of desired years to string
for i in range(len(years_to_keep)):
    years_to_keep[i] = str(years_to_keep[i])

# populate dictionary average from 1954 to 2017
for i in range(1, len(years_to_keep)):
    gdp_growth_data_usa_official_dictionary_average_1954_to_2017[years_to_keep[i]] = gdp_growth_data_usa_official_dictionary_average.get(years_to_keep[i])

# return cleaned gdp data
def get_cleaned_gdp_data():
    return gdp_growth_data_usa_official_dictionary_average_1954_to_2017

# print(years_to_keep)

# print(gdp_growth_data_usa_official_dictionary_average_1954_to_2017)