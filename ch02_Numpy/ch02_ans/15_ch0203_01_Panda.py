import pandas as pd # pandas-df.py

names = pd.Series(['SF', 'San Jose', 'Sacramento'])
sizes = pd.Series([852469, 1015785, 485199])

#df = pd.DataFrame({ 'Cities': names, 'Size': sizes })

df = pd.DataFrame({ 'City name': names, 'sizes': sizes })

print('df:', df)
