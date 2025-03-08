# general imports
import interest_rate_data_usa_cleaning as interest
import total_cpi_per_year_usa_cleaning as cpi
import pandas as pd

# cleaned cpi and interest rate data (dictionaries) from 1954 to 2017
cleaned_interest_rate_data_1954_to_2017 = interest.get_cleaned_interest_rate_data()
cleaned_inflation_year_over_year_data_1954_to_2017 = cpi.get_cleaned_cpi_data()

# create a dictionary for the combined clean interest rate and cpi data from 1954 to 2017
combined_cleaned_interest_cpi_data_1954_to_2017 = {}

# populate combined clean interest and cpi data dictionary from 1954 to 2017
for key in cleaned_inflation_year_over_year_data_1954_to_2017:
    combined_cleaned_interest_cpi_data_1954_to_2017[key] = [cleaned_interest_rate_data_1954_to_2017[key], cleaned_inflation_year_over_year_data_1954_to_2017[key]]

df = pd.DataFrame.from_dict(data = combined_cleaned_interest_cpi_data_1954_to_2017, orient="index", columns=["interest_rate", "inflation"])
print(df)

# testing
# print(cleaned_inflation_year_over_year_data_1954_to_2017)
# print(f"\nSeparator\n")
# print(cleaned_interest_rate_data_1954_to_2017)
# print(combined_cleaned_interest_cpi_data_1954_to_2017)