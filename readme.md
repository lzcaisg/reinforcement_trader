
# Final Year Project (NTU SCSE AY1920): Reinforcement Trading for Multi-Market Portfolio with Crisis Avoidance

CAI Lingzhi, NTU SCSE AY1920

This documentation aims to briefly introduce the project and the source code. 

As a quick guide, the main program to train the reinforcement learning model is in `./from github/Stock-Trading-Environment/main.py` and `./from github/Stock-Trading-Environment/env/MonthlyRebalancingEnv.py`.

The massive testing program (generating the heatmap matrix in the last section of the report) is written in `./from github/Stock-Trading-Environment/start_0418.py` and `./from github/Stock-Trading-Environment/env/FinalEnv.py`.

If you are reading this documentation from the DropBox, you can visit the github repo of the project to check the full version, or if necessary, to check the editing history (although I highly not recommand to do so). The github repo is at [the address](https://github.com/lzcaisg/reinforcement_trader).

The default branch is multiple-market-predictor.

The reports are in `Report and Paper` folder.

The opinions, conclusions and recommendations contained therein are for reference only and do not constitute investment advice of any kind at any time. Performance figures only stands for past performance. Past performance should not be used as a predictor of future returns. The author will not bear any legal responsibility for any consequences caused by following or using this project.

## Overview

This project aims to build a reinforcement learning for rebalancing portfolio in multiple stock market indexes. We use Brazil, Taiwan and NASDAQ as the high-, medium- and low-risk market. Statistically, higher risk results in higher yield, and therefore we invest in the riskier market when the condision is good, and return to lower risk market when it turns bearish.

## Lessons Learned

Some lessons learned during this project:
1. **DO NOT build the RL framework and baseline by yourself**. Use the wheels available. Rebuilding those stuff (e.g. [Tianshou 天授](https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88) by THU Undergrads) may already take one whole year before you step to the finance problem. If that's what you actually looking for, I would suggest you to communicate with the prof ASAP and see how he or his PhD can help. You would be wasting time if you are playing it by yourself. 
2. **DO NOT start with raw price and all-in/all-out strageties.** Many hello-world projects use such strategy but it is just useless. You will be wasting time trying to optimize it. 
3. **Implementation of multiple-asset portfolio is different from single-asset portfolio.** The transaction phase is not the same. If you are only holding one asset, you just need to cash-out during rebalancing. However, if you are holding multiple assets with no cash on hand, you need to sell some in order to buy in others. It would be a bit harder to calculate the transaction fee. You should also consider the exchange rates if the prices are in different currencies. Details are in my report if you are interested.
4. **Talk to Prof at least once per week.** Prof may or may not be able help you with the detailed implementation and codings, however, it would still be helpful to elaborate your ideas and check your progress. Go and ask for help immediately if you find yourself stuck in somewhere for more than 3 days (Learned during my internship, and the original threshold is "half a day" during work).
5. **The best practice is to use virtual environment before everything if you have multiple devices/servers/computers.** If you are only doing this on your laptop, it would be fine not setting those environment anyways. However, if you are developing on multiple computers (e.g. PC+Laptop, Mac+Windows) or on servers, it would be good to synchronize the environment. I never did that and now my laptop and PC is in a mess, but anyways I can reformat them after graduation lol.

## Online Tutorials Resources
The tutorial for beginners can be found here:

 - [Trade and Invest Smarter — The Reinforcement Learning Way: Adam King](https://towardsdatascience.com/trade-smarter-w-reinforcement-learning-a5e91163f315): This is an introduction of TensorTrade, a similar project started in 2019. The framework predefined many types of action spaces, observation spaces and reward functions. At first it seems to support only raw price and all-in/all-out strategy, which is kind of dumb and useless. Therefore I choose to define my own framework. HOWEVER, this framework has been actively updated by the community during the past year, and I realize that it is now easier to customize the above RL settings in this work than in my code, sincerely.

- [Create custom gym environments from scratch — A stock market example: Adam King](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e). As the previous work is less useful during my FYP, I chose to define my own environment and agent, which is kind of painful.

You can do a DFS from the above two articles and see what resources (especially technical documentations) they are referencing to. But before that, make sure you have a clear goal and don't be lost during researching. Talk to your prof regularly.

If the full artical is not available due to the view count limit, try to open it in the incognito mode of your browser.

## Introduction of My Codes

### Dependencies

I didn't summarize all the dependencies, but generally the project requires:

- `Jupyter Notebook`
- `Numpy`
- `Pandas`
- `Pickle` (If you are working in multiple machines where the numpy version is different, there may be problems in I/O.)
- `Tensorflow` (v1.\*, I never use 2.\*+ as it was just released during that time and was super buggy, probably it is better now. And I was recycling all the settings of my GPU that I set two years ago for all the projects, so really I hate the updated version lol)
- `matplotlib`
- `seaborn`

### Main Projects

The repo is quite messy as it is used for digging answers to the finance questions but not meant to nicely design a software system (I hate software engineering). Instead of encapsulating everything as the SE course teaches, I am coding the project in a straightforward manner so that it can be understood and modified a lot more easily. 

The main program to train the reinforcement learning model is in `./from github/Stock-Trading-Environment/main.py` and `./from github/Stock-Trading-Environment/env/MonthlyRebalancingEnv.py`.

The massive testing program (generating the heatmap matrix in the last section of the report) is written in `./from github/Stock-Trading-Environment/start_0418.py` and `./from github/Stock-Trading-Environment/env/FinalEnv.py`.

The main program, as the foler name `from github` suggests, is initially forked from Adam King's github. It would be easier to use his code to test your settings of dependencies. However, I forget to restructure the repo before it becomes too big to be reformat. Anyway it is just a one-year project and it is able to run during this year.

The data are .csv downloaded from yahoo finance and investing.com. You probably need to clean the data before feeding it to the model. This is done in `CSVUtils.py`.

The training result and output is saved by `pickle` in binary. 

The visualization part are all written in the `Jupyter Notebook` in the main folder. The names is quite straightforward and there are some sample outputs stored in the files as well. However, I would suggest you to rewrite the visualization for your own analysis.

I think I have put sufficient comments in the codes as I also forget what I was coding from time to time. I won't be updating this project and the codes are provided "AS IS". All the best.
