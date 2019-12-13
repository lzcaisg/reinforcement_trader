# FYP: Reinforcement Learning for Time Sequence Data

## Fancy Ideas
1. Given a **whole universe of** stock/bond/securities, decide on a portfolio
that diversify the risks (Can use covariance and dendrogram approach)
2. Given the history data of selected portfolio containing several assets,
develop a trading agent that determines the weight of capital allocating
on each asset
3. Given the history data of one asset, and its current (latest) position,
determine its future performance

## Programmable Tasks
1. Given 180, 90, 60, 30 (calendar) days of normalized history data (or minmax data),
determine the **percentile** of today's price in FUTURE 7, 14, 30, 60, 90 days
- 30 days history -> 7, 14 days future
- 60 days history -> 7, 14, 30 days future
- 90 days history -> 7, 14, 30, 60 days future
- 180 days history -> 7, 14, 30, 60, 90 days future
2. Given the prediction from the four models, decide on the weight adjusted
and calculate the return after 7 days
3. Using the previous model, decide the best holding date, and trained against
the actual value
4. Calculate and store all the actual value beforehand, so that we can load
the calculated result for training

## Risks and Potential Issues
1. The result is not significant. RL requires millions of training data
2. The environment is not well-defined and the model found some shortcut
3. The model is overfitting some pattern (e.g. S&P 500 is monotonously
growing for 10 years)
