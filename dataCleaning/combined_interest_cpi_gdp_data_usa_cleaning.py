# general imports
import interest_rate_data_usa_cleaning as interest
import total_cpi_per_year_usa_cleaning as cpi
import gdp_growth_data_usa_official_cleaning as gdp

# cleaned cpi and interest rate data (dictionaries) from 1954 to 2017
cleaned_interest_rate_data_1954_to_2017 = interest.get_cleaned_interest_rate_data()
cleaned_inflation_year_over_year_data_1954_to_2017 = cpi.get_cleaned_cpi_data()
cleaned_gdp_data_1954_to_2017 = gdp.get_cleaned_gdp_data()

# create a dictionary for the combined clean interest rate and cpi data from 1954 to 2017
combined_cleaned_interest_inflation_gdp_data_1954_to_2017 = {}

# populate combined clean interest and cpi data dictionary from 1954 to 2017
for key in cleaned_inflation_year_over_year_data_1954_to_2017:
    combined_cleaned_interest_inflation_gdp_data_1954_to_2017[key] = [cleaned_interest_rate_data_1954_to_2017[key], cleaned_inflation_year_over_year_data_1954_to_2017[key], cleaned_gdp_data_1954_to_2017[key]]