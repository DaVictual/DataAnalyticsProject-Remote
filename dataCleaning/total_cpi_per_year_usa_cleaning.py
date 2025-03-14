#general imports
import csv

#open csv file
total_cpi_per_year_usa = open("csvFiles/total_cpi_per_year_usa.csv", "r")

#create a reader object for the csv file
total_cpi_per_year_usa = csv.reader(total_cpi_per_year_usa)

#convert csv to a list
data_list = list(total_cpi_per_year_usa)

# create dictionary for the sum of each monthly reported cpi for any given year
cpi_dictionary_sum = {}
# create dictionary for number of months where the cpi was reported in any given year
cpi_dictionary_count = {}
# create a dictionary for the average cpi for any given year
cpi_dictionary_average = {}
# create a dictionary for the average cpi for years 1954-2017
cpi_dictionary_average_1954_to_2017 = {}
# create a dictionary for the inflation year over year for years 1954-2017
cpi_dictionary_inflation_year_over_year_1954_to_2017 = {}

# populate cpi dictionary sum and count variants
for row in range(1, len(data_list)):
    year = 1

    for month in range(2, len(data_list[row])):
        if(data_list[row][month] != ''):
            if data_list[row][year] not in cpi_dictionary_sum:
                cpi_dictionary_sum[data_list[row][year]] = float(data_list[row][month])
                cpi_dictionary_count[data_list[row][year]] = 1
            else:
                cpi_dictionary_sum[data_list[row][year]] += float(data_list[row][month])
                cpi_dictionary_count[data_list[row][year]] += 1

# populate cpi dictionary average variant
for key in cpi_dictionary_count:
    cpi_dictionary_average[key] = cpi_dictionary_sum.get(key) / cpi_dictionary_count.get(key)

# create list of years to keep (integer)
years_to_keep = list(range(1953,2018))

# make list of years to keep string
for i in range(len(years_to_keep)):
    years_to_keep[i] = str(float((years_to_keep[i])))

# populate cpi dictionary average 1954 to 2017 variant
for key in years_to_keep:
    cpi_dictionary_average_1954_to_2017[key] = cpi_dictionary_average[key]

# populate cpi dictionary inflation year over year 1954 to 2017 variant
for i in range(1, len(years_to_keep)):
    cpi_dictionary_inflation_year_over_year_1954_to_2017[str(int(float(years_to_keep[i])))] = (((cpi_dictionary_average_1954_to_2017.get(years_to_keep[i]) / cpi_dictionary_average_1954_to_2017.get(years_to_keep[i-1])) - 1) * 100)

cpi_dictionary_inflation_year_over_year_1955_to_2018 = {}

for key in cpi_dictionary_inflation_year_over_year_1954_to_2017:
    if int(key)+1 < 2018:
        cpi_dictionary_inflation_year_over_year_1954_to_2017[key] = cpi_dictionary_inflation_year_over_year_1954_to_2017.get(str(int(key)+1))
    else:
        cpi_dictionary_inflation_year_over_year_1954_to_2017[key] = 2.43899951453

def get_cleaned_cpi_data():
    return cpi_dictionary_inflation_year_over_year_1954_to_2017

def get_clean_cpi_data_next_year_prediction():
    return cpi_dictionary_inflation_year_over_year_1954_to_2017


