import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.foreignexchange import ForeignExchange

#APIs to use - Key & URL - Only need 1 usually
api_key = 'M83XDO2TK8UTLUDB'
API_URL = "https://www.alphavantage.co/query"

#Retrieving data
cc = ForeignExchange(key=api_key, output_format='pandas')
data, meta_data = cc.get_currency_exchange_intraday(from_symbol='EUR', to_symbol='GBP')

#Save data to CSV file
data.to_csv('EUR-GBP.csv')
#Give the console something to display
print(data)

#Plotting data to graph
data['4. close'].plot()
plt.title('Intraday EUR/GBP')
plt.show()

#Very cool, can dump to JSON file, works fine, cannot convert to pandas
# i = 1
# while i == 1:
#     data = {
#         "function": "FX_INTRADAY",
#         "from_symbol": "EUR",
#         "to_symbol": "GBP",
#         "interval": "5min",
#         "datatype": "json",
#         "apikey": "M83XDO2TK8UTLUDB",
#         }
#
#     response = requests.get(API_URL, params=data)
#     response_json = response.json()
#     # with open('output.json', 'w') as fp:
#     #     json.dump(data, fp)
#     # time.sleep(60)
#
#     data = pd.DataFrame.from_dict(response_json['Time Series FX (5min)'], orient= 'index').sort_index(axis=1)
#     data = data.rename(columns={ '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close'})
#     data = data[[ 'Open', 'High', 'Low', 'Close']]
#     data.tail() # check OK or not
