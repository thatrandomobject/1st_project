import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)

df = pd.read_csv('csv_files_git/Auto Sales data.csv')
sales_mean = df['SALES'].mean().round(2)
# print(f'Pardavimų vidurkis: ${sales_mean}')
count_orders_by_status = df['STATUS'].value_counts()
# print(count_orders_by_status)
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], dayfirst=True)
sorted_df = df.sort_values('ORDERDATE', ascending=True)
# sorted_df_g = sorted_df.groupby('ORDERNUMBER')['SALES'].sum()
# print(sorted_df['ORDERDATE'])
# print(sorted_df_g)
sorted_df['DAYS_SINCE_LASTORDER'] = sorted_df['ORDERDATE'].diff().dt.days
df2 = sorted_df[['DAYS_SINCE_LASTORDER', 'ORDERDATE', 'ORDERNUMBER']].head(10)

monthly_sales = df.groupby(pd.Grouper(key='ORDERDATE', freq='M')).sum()
# plt.figure(figsize=(10, 8))
# monthly_sales['SALES'].plot()
# plt.title('Monthly sales over-time')
# plt.xlabel('Month')
# plt.ylabel('Total Sales')
# plt.show()

decomposed = seasonal_decompose(monthly_sales['SALES'], model='additive')
# plt.figure(figsize=(12, 8))
# decomposed.plot()
# plt.show()

grouped_df = df.groupby('PRODUCTLINE').agg({'QUANTITYORDERED':'sum', 'SALES':'sum'})
grouped_df = grouped_df.sort_values('SALES', ascending=False)
grouped_df['SALES'].plot(kind='bar')
# plt.title('Product-line performance based on total sales')
# plt.xlabel('Product-line')
# plt.xticks(rotation=17)
# plt.ylabel('Total sales')
# plt.show()

geo_segmentation = df.groupby('COUNTRY').agg({'SALES':'sum'}).sort_values(by='SALES', ascending=False)
geo_segmentation.plot(kind='bar')
plt.title('Sales by country')
plt.xlabel('Country')
plt.xticks(rotation=35, fontsize=7, horizontalalignment='right')
plt.ylabel('Total sales')
plt.show()

