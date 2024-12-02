# 1 - Introduction

In this project we will be exploring machine learning and building a predictive model for estimating apartment prices in Poland based on various features, including apartment specifications (such as size, number of rooms, etc) and location. The data has been recorded in three cities, Poznań, Warszawa and Kraków. This also includes coordinates as the location in the city may affect the prices of the apartment. The main motivation of using this dataset was firstly, as the author of the project is of Polish origin, and secondly the Polish real estate market has seen significant growth in recent years, so predicting prices is an essential tool for both buyers and sellers. The goal of the machine learning model is to provide a data-driven solution for forecasting apartment prices based off factors that can influence the value of these apartments. For example, factors which can influence the prices include size of the apartment in square feet, number of rooms, the floor the apartment is located, and the geographic location. These can drastically influence the price of the apartments, for eaxmple apartments higher up may cost significantly more, or those who are in close proximity to  city or public transportation/landmarks and location etc.

By using historical data on previous transaction, our machine learning model will learn to capture the patterns in the features from the data so it can allow us to make accurate predictions on future property listings. This could be very useful for potential buyers in determining fair market prices and investors who need a reliable tool for market data analysis. In this analysis we will go through the process of applying the machine learning model and looking at the relationships between various features providing insights into which factors influence the price the most. 

### 2 - Problem Statement

Like every other market in the world, the real estate market in Poland is complex and dynamic with apartment prices influenced by many factors as explained earlier. As a result, predicting the price of an apartment is a challenging task for a human to do for buyers and sellers, so we can employ machine learning models and techniques to help us with this. The primary goal of this project is to build a machine learning model which is capable of predicting the price of an apartment based off the features and columns provided by the data in which we can aim to develop a model that can reliably estimate these apartment prices. 

We can summarize te problem as follow:

- **Input**: Dataset containing all the information about apartments in Poland. This includes the following columns: Apartment Size (in square meters), number of rooms, floor number, location, year bought and other relevant details we will cover later.

- **Output**: A predicted price for each apartment based off input. This can be changed to anything depending on the needs of the project but it is optimised for price prediction. 

This model will provide value by:

1.  Help potential buyers estimate and negotiate for a fair price for an apartment
2.  Assist sellers in pricing their properties competitvely and accurately
3.  Enable real estate professionals and investors make more informed decisions

We will solve this probem by employing supervised learning techniques where historical apartment prices will be predicted based on features provided in the dataset. Our approach will involve data pre-processing, feature engineering, model selection, model training and model evaluation using a range of different techniques and models.

*_More detailed version of this can be found in the jupyter notebook_*