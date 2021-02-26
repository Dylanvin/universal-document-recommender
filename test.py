from sklearn.datasets import fetch_20newsgroups
import pandas as pd

dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
target_names = dataset.target_names
news_df = pd.DataFrame({'Category': dataset.target, 'Text': dataset.data})


for index, row in news_df.iterrows():
    news_df.loc[index, "Category"] = target_names[news_df.loc[index, "Category"]]

print(news_df.head(5))