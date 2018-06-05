## Stats, Econometrics and ML Methods 
Some side projects exploring estimation methods from econometrics (linear regression, multinomial logistic regression, count models) and **tree-based machine learning** methods, random forests in particular. 

One approach to think about blending Machine Learning tools into the traditional econometrics framework is to use the **anatomy of a regression** formula. The formula states that the **partial effect** of a variable X on Y can be estimated as *beta = Cov(Resid.Y, Resid.X)/Var(Resid.X)*, where these are residuals obtained by regressing Y and X on *all other covariates*. These regressions, in turn, can be based on linear regression, or any other statistical learning method such as random forests or neural networks. Standard errors can be obtained using resampling methods e.g. bootsrap, jackknife etc.  

* [Multiple regression and generalized random forest](./genRandForest.ipynb)
* [Intro to mixed logit regression with random coefficients](./mixedLogit.ipynb)
* Multinomial logit and classification trees (with CRAN package 'ranger')
* Instrumental variables regression with generalized random forests  

### Fema Flood Insurance 
* [Initial mapping of premium rates and adoption levels](./femaData.ipynb) 

### Reference texts/papers
* Kenneth Train, **Discrete choice methods with simulation**, Cambridge Univ. Press (pdf version on Train's website)
* Stefan Wager, Trevor Hastie, and Bradley Efron, **Confidence intervals for random forests: the Jjackknife and the infinitesimal jackknife**, Journal of Machine Learning Research (2014).
* Susan Athey and Guido Imbens, **Recursive partitioning for heterogeneous causal effects**, PNAS (2016)
* Susan Athey, Julie Tibshirani and Stefan Wager, **Generalized Random Forests**, arXiv:1610.01271v3 (2017)

