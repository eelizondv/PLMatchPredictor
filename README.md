# English Premier League Match Outcome Prediction

## Introduction

### Problem Contextualization

Predicting the outcome of sports matches has always been a challenging problem in the field of artificial intelligence and machine learning. There are many variables that go into making a prediction, such as team performance, player conditions, and external factors.

This project focuses on predicting the outcome of specific matches in the English Premier League (PL). The PL is arguably the most competitive football league in the world, attracting a massive pool of talented players. The season runs from August until May, where 20 teams compete in a league format to accumulate the most points throughout their 38-game campaign and lift the prestigious title.

### Motivation for AI Use

The primary motivation for employing AI in predicting match outcomes is to harness the power of data-driven decision-making. Traditional methods of forecasting match results, often based on expert opinion or simple statistics, can be subjective and limited in scope. Machine learning models, particularly ensemble methods which we'll use in this project, can leverage vast amounts of historical data to identify complex patterns and interactions between variables, providing more accurate and reliable predictions. These models can be particularly useful for fans, analysts, and sport bettors by increasing their capacity to make informed predictions.


### Dataset

The dataset used in this project is comprised of the historical data of PL matches from the 1993 season up until the 2023 season, stored in a CSV file named `premier-league-matches.csv`. 

This dataset was downloaded from Kaggle and can be found [here](https://www.kaggle.com/datasets/evangower/premier-league-matches-19922022?select=premier-league-matches.csv).

The dataset includes the following attributes:

- `index`: The unique ID for the match
- `Season_End_Year`: The season in which the match was played
- `Wk`: The matchweek in which the match was played
- `Date`: The date on which the match was played
- `Home`: The home team
- `Away`: The away team
- `HomeGoals`: The number of goals scored by the home team
- `AwayGoals`: The number of goals scored by the away team
- `FTR`: Full-time result (H for home win, D for draw, A for away win)

## Usage Instructions

### Requirements
To run the code, you need to have the following libraries installed:

- **pandas**: For data manipulation and analysis
- **numpy**: For mathematical functions
- **scikit-learn (sklearn)**: For machine learning models and utilities
- **xgboost**: For the eXtreme Gradient Boosting algorithm
- **joblib**: For saving and loading trained models and encoders
- **matplotlib**: For plotting data
- **seaborn**: For plot styling

## Thanks!

Thanks for taking the time to check this project out! If you have any improvements that could be made please feel free to reach out!