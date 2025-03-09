# general imports
import interest_rate_data_usa_cleaning as interest
import total_cpi_per_year_usa_cleaning as cpi
import gdp_growth_data_usa_official_cleaning as gdp

# google colab imports
import pandas as pd

# cleaned cpi and interest rate data (dictionaries) from 1954 to 2017
cleaned_interest_rate_data_1954_to_2017 = interest.get_cleaned_interest_rate_data()
cleaned_inflation_year_over_year_data_1954_to_2017 = cpi.get_cleaned_cpi_data()
cleaned_gdp_data_1954_to_2017 = gdp.get_cleaned_gdp_data()

# create a dictionary for the combined clean interest rate and cpi data from 1954 to 2017
combined_cleaned_interest_inflation_data_1954_to_2017 = {}

# populate combined clean interest and cpi data dictionary from 1954 to 2017
for key in cleaned_inflation_year_over_year_data_1954_to_2017:
    combined_cleaned_interest_inflation_data_1954_to_2017[key] = [cleaned_interest_rate_data_1954_to_2017[key], cleaned_inflation_year_over_year_data_1954_to_2017[key], cleaned_gdp_data_1954_to_2017[key]]

# google colab dataframe visualization
df = pd.DataFrame.from_dict(data = combined_cleaned_interest_inflation_data_1954_to_2017, orient="index", columns=["interest_rate", "inflation", "gdp"])
df.reset_index(inplace=True)
df.rename(columns={"index":"year"}, inplace=True)
print(df)

# df.to_csv("interest_rate_and_inflation_data_1954_to_2017.csv", index=False)

# testing
# print(cleaned_inflation_year_over_year_data_1954_to_2017)
# print(f"\nSeparator\n")
# print(cleaned_interest_rate_data_1954_to_2017)
# print(combined_cleaned_interest_inflation_data_1954_to_2017)