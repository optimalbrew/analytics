## Machine learning tools in economic analysis

### Feed forward neural network to study early repayment and default in personal loan markets
Investors in fixed income securities (like p2p lending) must account for the possibility of default. This is made easier by credit ratings. However, early debt repayment can also hurt investors by altering their expected cash flow streams and leaving them with excess liquidity. This is especially a problem for those who do not have automated reinvestment strategies (most investors in peer to peer personal loans markets?). Multinomial logistic regression can be used to predict default, or early repayment and linear regression (or survival analysis) can be used to predict the time of default or early repayment. I use a feed forward neural network to do both in one step. 

[Link to repository](https://github.com/petecarkeek/repayDefault/) with some illustrative maps, bash script to download the data, pySpark code for ETL and to run linear regressions, and the pyTorch code for implementing the feed forward NN (for multivariate regressions/predictions). 

### A pyTorch implementation of two-sided marketplace pricing
**Motivation:** Consider a marketplace that connects pet owners with pet-care service providers. The platform charges fees of 5% to pet owners while it charges 25% to service providers. For any given transaction price, say $20 for a dog walk, the buyer spends $20 + $1 = $21 net, while the service provider receives $20 - $5 = $15 net. The platform earns $1+$5 = $6 (i.e. 30% of $20). Will the platform's revenue be the same with other fee-combinations that still add up to the same 30% (e.g. 15% to each side)? Of course not, as the new fee structure will alter the willingness to use the service (e.g. increase owners' out of pocket costs in our example to $20+$3=$23). In two-sided markets, changes in the price struture can lead to large changes in user participation, transaction volume and profitability. In some cases, it is beneficial to subsidize one side of the market by pricing the service below the platform's cost (and making it up on the other side of the market).

I implement a pricing model for two-sided markets using pyTorch [Jupyter Notebook](./Marketplace.ipynb). A simple illustration of how deep learning frameworks (TensorFlow, pyTorch) with automatic gradient computations can be used for general mathematical modeling and some optimization (not just for neural networks). 

### FCC regulatory filings and comments
Competition, or the lack thereof, among service providers not only determines economic efficiency but also the rate of adoption of new technologies. In the broadband connectivity space many other factors also influence technology adoption. These include local governments, national regulators, utilities, railroads, owners of cellular towers, workers (in all these companies), contractors and, of course, end users. This analysis uses natural language processing tools to quantify the market and institutional complexities in this sphere.
* [Jupyter nb 1](./FCC17_84.ipynb): Using *R, tm, tidytext* a couple of lines of Bash. Has more detailed information.
* [Jupyter nb 2](./fccPyClusters.ipynb): Using python with *nltk*, plots are a bit different from R version. Clustering is done  (separately) both before and after dimensionality reduction (with singular values decomp. and princal comps.). Unlike the R version, which seoarates the unigrams and bigram visualization, this version combines both unigrams and bigrams in one token-document matrix.
* [D3 visualization](https://bl.ocks.org/petecarkeek/c7da7590422d55e0b1dde588d9835df1) (Jypter notebook uses ggplot2) on Mike Bostock's bl.ocks.org

### Mapping Fema Flood Insurance Adoption Rates and Sensitivity 
"_Floods are the most common and costly natural hazard in the nation._" ([source: fema.gov](https://www.fema.gov/wildfires-you-need-flood-insurance)). 

Low adoption rates of catastrophic insurance for floods or mudslides often show up in news reports following wildfires. In addition to storm related flooding, there are also increasing concerns about _sunny-day floods_. These are (increasingly frequent) situations where sea water is driven up coastal sewer systems by strong tides (particularly in the absence of storms or hurricanes). 

This project [maps and estimates](https://github.com/petecarkeek/floodInsurance) the impact of prices and geographic (long/lat) location on insurance adoption rates. Estimates are obtained using regression and using a blend of random forest based estimation with linear regression.

### Spectrum Sharing in Licensed Wireless Communication Bands
Pricing models to study the sharing of radio frequency resources between existing (e.g. cellular networks) and emerging technologies (e.g. robotics, sensor networks). Link to [repo](https://github.com/petecarkeek/Spectrum_Sharing) to programs used to generate plots for [IEEE paper on dynamic spectrum allocation in licensed wireless bands](https://ieeexplore.ieee.org/abstract/document/8301016/). 

### ML and econometrics
My approach to blending machine learning tools with traditional econometrics framework is to think about the **anatomy of a regression** formula (e.g. see 'Mostly harmless econometrics' by Angrist and Pischke). The formula states that the **partial effect** of a variable X on Y can be estimated as *beta = Cov(Resid.Y, Resid.X)/Var(Resid.X)*, where these are residuals obtained by regressing Y and X on *all other covariates*. These regressions, in turn, can be based on linear regression, or any other statistical learning method such as random forests or neural networks. Standard errors can be obtained using resampling methods e.g. bootsrap, jackknife etc.  

* [Multiple regression and generalized random forest](./genRandForest.ipynb) : simple "text book" example of a wage regression to compare linear models with random forest based estimation.

* [Mixed logit regression with random coefficients](./mixedLogit.ipynb).
