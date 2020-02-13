1: ALL DATA, Train 20k, Test 2k, NOT USED
2: ALL DATA, Train 20k, Test 2k,
3: ALL DATA, Train 50k, Test 2k,
4: ALL DATA, Train 100k, Test 2k,

5. Train: 1990.01-1999.12 50k, Test: 2015.01-2019.12,
6. Train: 2004.01-2013.12 50k, Test: 2015.01-2019.12,

7. Use Actual Profit (Net Value - BuyAndHold Value) as reward, 
   Train: 2004.01-2013.12 50k, Test: 2015.01-2019.12,

8. Use backward window, Use Actual Profit as reward, 
   Train: 2004.01-2013.12 50k, Test: 2015.01-2019.12,

9. Use backward window, Use Actual Profit as reward, 
   Train: 2004.01-2013.12 100k, Test: 2015.01-2019.12,

10. Same as 9, add Detailed Record Output for each day in each test

11. Same as 10, Use 3-dim Action Space

12. Same as 11, Filter out the small amount transaction