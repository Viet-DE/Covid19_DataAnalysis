import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load data
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

""" CLEAN """
# Bỏ các cột kh cần thiết
confirmed_df.drop(['Province/State', 'Lat', 'Long'], axis = 1, inplace = True)
deaths_df.drop(['Province/State', 'Lat', 'Long'], axis = 1, inplace = True)
recovered_df.drop(['Province/State', 'Lat', 'Long'], axis = 1, inplace = True)
# Tổng hợp dữ liệu theo các quốc gia
confirmed_df = confirmed_df.groupby('Country/Region').sum()
deaths_df = deaths_df.groupby('Country/Region').sum()
recovered_df = recovered_df.groupby('Country/Region').sum()


""" PROCESSING """
# Tính toán tổng số ca nhiễm, tử vong và hồi phục
total_infections = confirmed_df.iloc[:, -1]
total_deaths = deaths_df.iloc[:, -1]
total_recoveries = recovered_df.iloc[:, -1]
# Tính toán số ca nhiễm và tử vong tăng hàng ngày
daily_infections = confirmed_df.iloc[:, -1] - confirmed_df.iloc[:, -2]
daily_deaths = deaths_df.iloc[:, -1] - deaths_df.iloc[:, -2]


""" VISUALIATION """
# Tạo Bar Chart của 10 quốc gia hàng đầu có nhiều trường hợp được Confirmed nhất
top_10_confirmed = confirmed_df.iloc[:, -1].sort_values(ascending = False)[:10]
plt.figure(figsize = (10, 5))
sns.barplot(x = top_10_confirmed.values, y = top_10_confirmed.index)
plt.xlabel('Các TH được Confirmed')
plt.ylabel('Các nước')
plt.title('10 quốc gia hàng đầu có nhiều trường hợp được Confirmed nhất')
plt.show()


# Chuẩn hóa dữ liệu
X = StandardScaler().fit_transform(confirmed_df)
# Dùng PCA để giảm kích thước
pca = PCA(n_components = 2)
principal_components = pca.fit_transform(X)


""" CLUSERING """
# Thực hiện phân cụm K-Means
kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(principal_components)
# Trực quan hóa các cụm
plt.scatter(principal_components[kmeans.labels_ == 0, 0], principal_components[kmeans.labels_ == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(principal_components[kmeans.labels_ == 1, 0], principal_components[kmeans.labels_ == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(principal_components[kmeans.labels_ == 2, 0], principal_components[kmeans.labels_ == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Phân cụm K-Means')
plt.legend()
plt.show()


""" ĐÁNH GIÁ MÔ HÌNH """
# Chuẩn bị dữ liệu cho hồi quy
X = np.arange(len(total_infections)).reshape(-1, 1)
y = total_infections.values
# Hồi quy TT & Dự đoán
reg = LinearRegression().fit(X, y)
future_dates = np.arange(len(total_infections), len(total_infections) + 30).reshape(-1, 1)
future_predictions = reg.predict(future_dates)
# Đánh giá mô hình
mse = mean_squared_error(y, reg.predict(X))
r2 = r2_score(y, reg.predict(X))

# Trực quan hóa dự đoán
plt.plot(X, y, label = 'Actual Infections')
plt.plot(future_dates, future_predictions, label = 'Predicted Infections')
plt.xlabel('Days since start of pandemic')
plt.ylabel('Total Infections')
plt.title('Linear Regression Predictions of Future Infections')
plt.legend()
plt.show()

print('Mean Squared Error:', mse)
print('R^2 Score:', r2)
