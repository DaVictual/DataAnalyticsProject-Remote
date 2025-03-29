# Import file
import csv
import pandas as pd

# Create a DataFrame for the CSV file
df = pd.read_csv("csvFiles/tax_rate_data_usa.csv")

# Clean data in the Bottom Bracket Rate % and Top Bracket Rate % columns
df["Bottom Bracket Rate %"] = df["Bottom Bracket Rate %"].str.strip("Â ")
df["Top Bracket Rate %"] = df["Top Bracket Rate %"].str.rstrip("Â ")

# Function for returning the cleaned data
def get_cleaned_tax_rate_data():
    return df.to_dict()