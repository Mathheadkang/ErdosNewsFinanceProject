# News topics and future stock price movement


# Team members:
[Xiangwei Peng](https://github.com/xpeng-26)

[Xiaokang Wang](https://github.com/Mathheadkang)
# Introduction
This project is for the [Erdös Institude](https://www.erdosinstitute.org) 2024 Fall data science boot camp.

This project aims to find the hidden relationship between the occuracy of different news topics and its impact on the stock price. There are many projects try to connect the fiancial news or financial-related social medias to the stock price. This project interest in the non-financial news. For example, how will the political news influence the stock price. Our project provide with a general method to deal with all kinds of topics.

# Dataset
There are two news dataset:
- [Headline](https://www.kaggle.com/datasets/rmisra/news-category-dataset): This data set from [Kaggle](https://www.kaggle.com) contains around 210k news headlines with labeled category from 2012 to 2022. This dataset is served to train the category classifier.
- [All_news](https://components.one/datasets/all-the-news-2-news-articles-dataset): This dataset contains uncategorized 2,688,878 news articles and essays from 27 American publications, spanning January 1, 2016 to April 2, 2020. This dataset is served to cluster the topics and the features to predict the future stock price.

The stock price dataset: [Yahoo API](https://developer.yahoo.com/api/), we use the history stock price of 20 companies from 2016-01-01 to 2019-12-31.

# Environment

# News model
## Preprocessing
For both news dataset, we run the almost identical preprocessing pipeline:
- Null removing and columns dropping
- Tokenization
- Stop words removing
- Lemmatization

For the [All_news](https://components.one/datasets/all-the-news-2-news-articles-dataset) data set, we keep the 100 tokens as the headline.

For the *Explainatory Data Analysis*, you can find in [EDA_headline](EDA_headline.ipynb), [EDA_Example_All_News](EDA_Example_All_News.ipynb), [All_News_EDA](All_News_EDA.ipynb).

## Classification

Using [Headline](https://www.kaggle.com/datasets/rmisra/news-category-dataset) as the training set, we build a classifier model. This model is *solft-voting* ensemble classifier using:
- Multi-Logistic Classifer
- Random Forest
- XGBoost Classifier
- CNN
  
This model, we choose the *Term frequency-Inverse document frequency (Tf-idf)* embedding to emphasize the importance of words in the headline. A more detailed discussion of the classification can be found in [Classification](Classification.ipynb).

Using the classifer, we can label [All_news](https://components.one/datasets/all-the-news-2-news-articles-dataset).

## Clustering
By running the content of [All_news](https://components.one/datasets/all-the-news-2-news-articles-dataset), we can cluster different topics. The model is based on:
- Latent Dirichlet Allocation (LDA)
- Hierarchical Dirichlet Process (HDP)

We choose around 500 clusters. More exploration can be found in [explore_hdp](explore_hdp.ipynb). Later, we call these clusters as topics.

# Stock model

Our model for stock price is 

$$r(t) = \hat{r}(t, \text{market}) + f(t, \text{news}) + \epsilon$$,

where $r$ is the daily return, $t$ is the time, $\epsilon$ is the residual. For $\hat{r}$ part, we use the French-Famma 5 factor models. 

$$ \hat{r}(t) = \beta_0 + \beta_1 M E R_t+\beta_2 S M B_t+\beta_3 H M L_t+ \beta_4 R M W_t+\beta_5 C M A_t $$

All the factors are global to the market. We get the factors from: [](). A detailed discussion can be found in [FF5](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html).

The news model $f$ is obtained by the regression with the residual $r(t)-\hat{r}(t,\text{market})$. The features we chooose are the occurance of each significant topics of news. We try different regressors:
- Ridge
- Lasso
- Random Forest Regressor
- XGBoost Regressor

We chooose XGBoost with penalty among others for the least mean square error in the test set and contains more trading information. A detailed discussion can be found in [Price_predicting](Price_predicting.ipynb).

# Get started
Go the the configuration file [predict_stock_w_news.toml](predict_stock_w_news.toml), change to your own local path of the dataset and check all the models you want to run. Then go the [predict_stock_w_news.py](predict_stock_w_news.py).
