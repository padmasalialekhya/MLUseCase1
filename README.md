# Donor Feature Pipeline
Combines Salesforce donations, NetSuite payments, and marketing engagement
data into one feature table to ML model for analysing "will this donor donate again?"


# Requirements
- Python 3.9+
- Install dependencies:
  uv add -r requirements.txt
  (installs `pandas`)

# How to run
python3 feature_pipeline.py


# Requires - 
`salesforce_data.csv`, `netsuite_data.csv`, and `marketing_data.csv`in the working directory. 

# Output -
Produces output file `feature_table.csv` in the directory

# What it does
1. Clean & validate each source (drop missing/negative/duplicate rows).
2. Engineer features per donor:
   - RFM (recency, frequency, monetary value)
   - Behavioral (recurring/lapsed/trend)
   - Financial (settlement time, refund rate)
   - Marketing engagement (open/click rate)
   - Campaign/channel diversity
3. Join all features into one table (one row per donor), validate it (row
   counts, nulls, value ranges), and save to CSV.


