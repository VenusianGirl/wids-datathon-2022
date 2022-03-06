# WiDS Datathon 2022

This repository contains my source code as it relates to my solution for the 2022 Women in Data Science ("WiDS") kaggle competition.  My final solution ended up in the top 10% of all solutions.

For a better viewing experience, I would recommend viewing my notebooks on Kaggle rather than here on GitHub.  You can find the notebooks at the below links:

- Machine learning notebook
- Exploratory data analysis notebook (link to be included soon, currently cleaning this notebook up)

## Solution Abstract

An abstract of my approach to the competition is included below:

>  *The 2022 WiDS Kaggle competition involved predicting a buildings Site Energy Usage Intensity metric (regression).  The dataset included 100k rows of data for sixty different types of buildings (e.g. residential, schools, hotels, data centers, grocery stores and so on).  To solve the problem, I first separated the dataset into twelve individual datasets based on buildings with similar energy usage patterns and other characteristics. I then engineer features, perform leave one group out cross validation, and finally train an ensemble model (XGBoost, LightGBM, and CatBoost regressors) using Kaggle's GPUs for each dataset on the most powerful features.*
