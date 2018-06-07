## Stats, Econometrics and ML Methods 
A side project exploring estimation methods based on **random forests** and comparing them with traditional econometric methods (e.g. linear regression, logistic regression). 

### Fema Flood Insurance 
Low adoption rates of catastrophic insurance for floods or mudslides have been in the news recently. In addition to storm related flooding, there are also increasing concerns about _sunny-day floods_. These are (increasingly frequent) situations where sea water is driven up coastal sewer systems by strong tides (particularly in the absence of storms or hurricanes). 

This project estimates the impact of prices and geographic (long/lat) location on insurance adoption rates. Estimates are obtained using regression and also using a blend of random forest based estimation with linear regression. 

* [Initial mapping of premium rates and adoption levels](./femaData.ipynb) 

### ML and econometrics
My approach to blending machine learning tools with traditional econometrics framework is to think about the **anatomy of a regression** formula. The formula states that the **partial effect** of a variable X on Y can be estimated as *beta = Cov(Resid.Y, Resid.X)/Var(Resid.X)*, where these are residuals obtained by regressing Y and X on *all other covariates*. These regressions, in turn, can be based on linear regression, or any other statistical learning method such as random forests or neural networks. Standard errors can be obtained using resampling methods e.g. bootsrap, jackknife etc.  

* [Multiple regression and generalized random forest](./genRandForest.ipynb) : simple "text book" example of a wage regression to compare linear models with random forest based estimation.

* [Intro to mixed logit regression with random coefficients](./mixedLogit.ipynb)  Todo: extend example to cover random forest approach using multinomial logit and classification trees with CRAN package 'ranger'.

### Reference texts/papers
* Kenneth Train, **Discrete choice methods with simulation**, Cambridge Univ. Press (pdf version on Train's website)
* Stefan Wager, Trevor Hastie, and Bradley Efron, **Confidence intervals for random forests: the Jackknife and the infinitesimal jackknife**, Journal of Machine Learning Research (2014).
* Susan Athey and Guido Imbens, **Recursive partitioning for heterogeneous causal effects**, PNAS (2016)
* Susan Athey, Julie Tibshirani and Stefan Wager, **Generalized Random Forests**, arXiv:1610.01271v3 (2017)
