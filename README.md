## Stats, Econometrics and ML 
* Exploring **tree-based** ML methods such as  *random forests*, *boosting* (e.g xgboost or Microsoft's Light GBM), comparing or combining  them with traditional econometric methods (e.g. linear regression, logistic regression). 
* Using *natural language processing* tools to visualize and analyze text data, such as regulatory filings.

### FCC regulatory filings and comments
Competition, or the lack thereof, among service providers not only determines economic efficiency but also the rate of adoption of new technologies. In the broadband connectivity space many other factors also influence technology adoption. These include local governments, national regulators, utilities, railroads, owners of cellular towers, workers (in all these companies), contractors and, of course, end users. This analysis is an attempt to use natural language processing tools to quantify the market and institutional complexities in this sphere.
* [Jupyter nb 1](./FCC17_84.ipynb): Using *R, tm, tidytext* a couple of lines of Bash. Has more detailed information.
* [Jupyter nb 2](./fccPyClusters.ipynb): Using python with *nltk*, plots are a bit different from R version. Clustering is done  (separately) both before and after dimensionality reduction (with singular values decomp. and princal comps.). Unlike the R version, which seoarates the unigrams and bigram visualization, this version combines both unigrams and bigrams in one token-document matrix.
* [D3 visualization](https://bl.ocks.org/petecarkeek/c7da7590422d55e0b1dde588d9835df1) (Jypter notebook uses ggplot2) on Mike Bostock's bl.ocks.org

### Fema Flood Insurance 
"_Floods are the most common and costly natural hazard in the nation._" ([source](https://www.fema.gov/wildfires-you-need-flood-insurance)). 

Low adoption rates of catastrophic insurance for floods or mudslides have been in the news recently. In addition to storm related flooding, there are also increasing concerns about _sunny-day floods_. These are (increasingly frequent) situations where sea water is driven up coastal sewer systems by strong tides (particularly in the absence of storms or hurricanes). 

This project estimates the impact of prices and geographic (long/lat) location on insurance adoption rates. Estimates are obtained using regression and also using a blend of random forest based estimation with linear regression. 

* [mapping and estimating elasticity of insurance adoption](./femaData.ipynb) 

### ML and econometrics
My approach to blending machine learning tools with traditional econometrics framework is to think about the **anatomy of a regression** formula. The formula states that the **partial effect** of a variable X on Y can be estimated as *beta = Cov(Resid.Y, Resid.X)/Var(Resid.X)*, where these are residuals obtained by regressing Y and X on *all other covariates*. These regressions, in turn, can be based on linear regression, or any other statistical learning method such as random forests or neural networks. Standard errors can be obtained using resampling methods e.g. bootsrap, jackknife etc.  

* [Multiple regression and generalized random forest](./genRandForest.ipynb) : simple "text book" example of a wage regression to compare linear models with random forest based estimation.

* [Mixed logit regression with random coefficients](./mixedLogit.ipynb)  Todo: extend example to cover random forest approach using multinomial logit and classification trees with CRAN package 'ranger'.

### Reference texts/papers
* Kenneth Train, **Discrete choice methods with simulation**, Cambridge Univ. Press (pdf version on Train's website)
* Stefan Wager, Trevor Hastie, and Bradley Efron, **Confidence intervals for random forests: the Jackknife and the infinitesimal jackknife**, Journal of Machine Learning Research (2014).
* Susan Athey and Guido Imbens, **Recursive partitioning for heterogeneous causal effects**, PNAS (2016)
* Susan Athey, Julie Tibshirani and Stefan Wager, **Generalized Random Forests**, arXiv:1610.01271v3 (2017)
