import csv

gdp_growth_data_usa_official = open("csvFiles/gdp_growth_data_usa_official.csv", "r")

gdp_growth_data_usa_official = csv.reader(gdp_growth_data_usa_official)

data_list = list(gdp_growth_data_usa_official)

gdp_growth_data_usa_official_dictionary_sum = {}
gdp_growth_data_usa_official_dictionary_average = {}
gdp_growth_data_usa_official_dictionary_count = {}
gdp_growth_data_usa_official_dictionary_average_2015_to_2017 = {}

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

for key in gdp_growth_data_usa_official_dictionary_count:
    gdp_growth_data_usa_official_dictionary_average[key] = gdp_growth_data_usa_official_dictionary_sum.get(key) / gdp_growth_data_usa_official_dictionary_count.get(key)

years_to_keep = list(range(1953, 2018))

for i in range(len(years_to_keep)):
    years_to_keep[i] = str(years_to_keep[i])

for i in range(1, len(years_to_keep)):
    gdp_growth_data_usa_official_dictionary_average_2015_to_2017[years_to_keep[i]] = gdp_growth_data_usa_official_dictionary_average.get(years_to_keep[i])

def get_cleaned_gdp_data():
    return gdp_growth_data_usa_official_dictionary_average_2015_to_2017

# print(years_to_keep)

# print(gdp_growth_data_usa_official_dictionary_average_2015_to_2017)