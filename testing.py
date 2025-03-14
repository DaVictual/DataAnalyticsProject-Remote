# general imports
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# open csv file
interest_rate_data_usa_csv_file = open("interest_rate_data_usa.csv", "r")

# create reader object for the csv file
interest_rate_data_usa_csv_reader = csv.reader(interest_rate_data_usa_csv_file)

# convert csv to a list
data_list = list(interest_rate_data_usa_csv_reader)

# create dictionary for the sum of each monthly reported federal funds rate for any given year
interest_rate_dictionary_sum = {}
# create dictionary for number of months where the effective federal funds rate was reported in any given year
interest_rate_dictionary_count = {}
# create a dictionary for the average federal funds rate for any given year
interest_rate_dictionary_average = {}

# populate interest rate dictionary sum and count variants
for row in range(1, len(data_list)):
    effective_federal_funds_rate = 6
    year = 0

    if (data_list[row][effective_federal_funds_rate] != ''):
        if data_list[row][year] not in interest_rate_dictionary_sum:
            interest_rate_dictionary_sum[data_list[row][year]] = float(data_list[row][effective_federal_funds_rate])
            interest_rate_dictionary_count[data_list[row][year]] = 1
        else:
            interest_rate_dictionary_sum[data_list[row][year]] += float(data_list[row][effective_federal_funds_rate])
            interest_rate_dictionary_count[data_list[row][year]] += 1

# populate interest rate dictionary average variant
for key in interest_rate_dictionary_count:
    interest_rate_dictionary_average[key] = interest_rate_dictionary_sum.get(key) / interest_rate_dictionary_count.get(key)

def get_cleaned_interest_rate_data():
    return interest_rate_dictionary_average

# testing
# print(interest_rate_dictionary_sum)
# print("Separator")
# print(interest_rate_dictionary_count)
# print("Separator")
# print(interest_rate_dictionary_average)

# visualization in google colab
# names = list(interest_rate_dictionary_average.keys())
# values = list(interest_rate_dictionary_average.values())

# fig, ax = plt.subplots()

# plt.title("Average effective federal funds rate by year")
# plt.ylabel("Average effective federal funds rate")
# plt.xlabel("Year")
# ax.bar(names, values)
# ax.tick_params(axis="x", labelrotation = 90)
# fig.set_figwidth(15)
# plt.show()

#open csv file
total_cpi_per_year_usa = open("total_cpi_per_year_usa.csv", "r")

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


def get_cleaned_cpi_data():
    return cpi_dictionary_inflation_year_over_year_1954_to_2017
# testing
# print(cpi_dictionary_average)
# print("separator")
# print(cpi_dictionary_average_1954_to_2017)
# print(cpi_dictionary_inflation_year_over_year_1954_to_2017)

gdp_growth_data_usa_official = open("gdp_growth_data_usa_official.csv", "r")

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

# cleaned cpi and interest rate data (dictionaries) from 1954 to 2017
cleaned_interest_rate_data_1954_to_2017 = get_cleaned_interest_rate_data()
cleaned_inflation_year_over_year_data_1954_to_2017 = get_cleaned_cpi_data()
cleaned_gdp_data_1954_to_2017 = get_cleaned_gdp_data()

# create a dictionary for the combined clean interest rate and cpi data from 1954 to 2017
combined_cleaned_interest_inflation_data_1954_to_2017 = {}

cleaned_inflation_year_over_year_data_1954_to_2017_categories = {}

# assign category to cleaned inflation year over year data
for key in cleaned_inflation_year_over_year_data_1954_to_2017:
    if cleaned_inflation_year_over_year_data_1954_to_2017.get(key) < 2:
        inflation_category = "low"
        cleaned_inflation_year_over_year_data_1954_to_2017_categories[key] = inflation_category
    elif (2 <= cleaned_inflation_year_over_year_data_1954_to_2017.get(key) <= 4):
        inflation_category = "moderate"
        cleaned_inflation_year_over_year_data_1954_to_2017_categories[key] = inflation_category
    else:
        inflation_category = "high"
        cleaned_inflation_year_over_year_data_1954_to_2017_categories[key] = inflation_category

# print(cleaned_inflation_year_over_year_data_1954_to_2017_categories)

# populate combined clean interest and cpi data dictionary from 1954 to 2017
for key in cleaned_inflation_year_over_year_data_1954_to_2017:
    combined_cleaned_interest_inflation_data_1954_to_2017[key] = [cleaned_interest_rate_data_1954_to_2017[key], cleaned_gdp_data_1954_to_2017[key], cleaned_inflation_year_over_year_data_1954_to_2017[key], cleaned_inflation_year_over_year_data_1954_to_2017_categories[key]]

df = pd.DataFrame.from_dict(data = combined_cleaned_interest_inflation_data_1954_to_2017, orient="index", columns=["interest_rate", "gdp", "inflation", "inflation_category"])
df.reset_index(inplace=True)
df.rename(columns={"index":"year"}, inplace=True)
df["inflation_category_encoded"] = df["inflation_category"].map({"low":0, "moderate":1, "high":2})

X = df.drop(["year", "inflation_category", "inflation", "inflation_category_encoded"], axis=1)
y = df["inflation_category_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced",max_iter=1000)
)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

classification_label_count_dictionary = {}
classification_label_count_dictionary['moderate'] = df['inflation_category'].value_counts().get('moderate')
classification_label_count_dictionary['low'] = df['inflation_category'].value_counts().get('low')
classification_label_count_dictionary['high'] = df['inflation_category'].value_counts().get('high')

highest_count_classification_label = -1 * math.inf
total_count_classification_labels = df['inflation_category'].value_counts().sum()

for key in classification_label_count_dictionary:
  if classification_label_count_dictionary[key] > highest_count_classification_label:
    highest_count_classification_label = classification_label_count_dictionary[key]


# print(classification_label_count_dictionary)

# print("test set prediction: ", y_pred_test)
# print("training set prediction: ", y_pred_train)

# df.to_csv("interest_rate_and_inflation_data_1954_to_2017.csv", index=False)

# testing
# print(cleaned_inflation_year_over_year_data_1954_to_2017)
# print(f"\nSeparator\n")
# print(cleaned_interest_rate_data_1954_to_2017)
# print(combined_cleaned_interest_inflation_data_1954_to_2017)

# # assign attributes to X and classification labels to y
# y = df["inflation"]
# X = df.drop(["inflation", "year"], axis=1)

# # create train_testsplit of 80-20, 80% training, 20% test for the X and y variables
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# # create linear regression model and fit it to the training data (X_train and y_train)
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# # make predictions for the X_train and X_test sets using the fitted linear regression model
# y_lr_train_pred = lr.predict(X_train)
# y_lr_test_pred = lr.predict(X_test)

# # compute mean squared error and r-squared for the training set
# lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
# lr_train_r2 = r2_score(y_train, y_lr_train_pred)

# # compute mean squared error and r-squared for the test set
# lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
# lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# # place the results of the training and test set mean squared error and r-squared into a pandas dataframe
# lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
# lr_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]

# # create linear regression line
# z = np.polyfit(y_train, y_lr_train_pred, 1)
# p = np.poly1d(z)

# # construct scatter plot to show the linear regression line along with the experimental inflation vs predicted inflation
# plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.8)
# plt.ylabel("Predict inflation")
# plt.xlabel("Experimental inflation")
# plt.plot(y_train, p(y_train), "#F8766D")
# plt.show()