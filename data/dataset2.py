import requests
import pandas as pd

# API URL for Fear and Greed Index (limit=365 gives up to 365 records)
url = 'https://api.alternative.me/fng/?limit=365'

# Make the request to Alternative.me API
response = requests.get(url)
data = response.json()

# Extract the data
fng_data = data['data']

# Print the number of records and the earliest date
print(f"Number of records: {len(fng_data)}")
print(f"Oldest record date: {fng_data[-1]['timestamp']}")

# Convert to DataFrame
df_fng = pd.DataFrame(fng_data)

# Save to CSV
df_fng.to_csv('fear_greed_index_data.csv', index=False)
print("Fear and Greed Index data saved to 'fear_greed_index_data.csv'")
