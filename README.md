## Machine learning tools in economic analysis

### [Early repayment and default in personal loan markets](https://github.com/petecarkeek/consumerFinance/)
Investors in fixed income securities (like p2p lending) must account for the possibility of default. This is made easier by credit ratings. However, early debt repayment can also hurt investors by altering their expected cash flow streams and leaving them with excess liquidity. This is especially a problem for those who do not have automated reinvestment strategies (most investors in peer to peer personal loans markets?). Multinomial logistic regression can be used to predict default, or early repayment and linear regression (or survival analysis) can be used to predict the time of default or early repayment. I use a feed forward neural network to do both in one step. 

[Link to repository](https://github.com/petecarkeek/consumerFinance/) with some illustrative maps, bash script to download the data, pySpark code for ETL and to run linear regressions, and the pyTorch code for implementing the feed forward NN (for multivariate regressions/predictions). 

### A pyTorch implementation of two-sided marketplace pricing
In two-sided markets, changes in the "price struture" (the ratio of fees charged to both sides) can lead to large changes in user participation, transaction volume and profitability. In some cases, it is beneficial to subsidize one side of the market and making it up on the other side of the market. This [Jupyter Notebook](./Marketplace.ipynb) illustrates of how deep learning frameworks (e.g. pyTorch) with automatic gradient computations can be used for general math modeling and optimization in economics (and not just to implement neural networks). Such models of optimization can then be linked with user experience or demand analytics.

**Example:** Consider a marketplace that connects pet owners with pet-care service providers. The platform charges fees of 5% to pet owners while it charges 25% to service providers. For any given transaction price, say $20 for a dog walk, the buyer spends $20 + $1 = $21 net, while the service provider receives $20 - $5 = $15 net. The platform earns $1+$5 = $6 (i.e. 30% of $20). Will the platform's revenue be the same with other fee-combinations that still add up to the same 30% (e.g. 15% to each side)? Of course not, as the new fee structure will alter the willingness to use the service (e.g. increase owners' out of pocket costs in our example to $20+$3=$23). 

 

### Visualization of filings on FCC policy proposals

This project uses simple natural language processing tools such as FastText for word2vec embeddings, and count-based methods such as Tf-Idf to visualize differnet positions taken by distinct stakeholders in the communication policy space. The visualization is based on standard dimensionality reduction techniques (singular value decomposition and t-SNE). [Link to repo](https://github.com/petecarkeek/FCC/).

One issue concerns restrictions placed on access to utility poles. Utility poles not only carry power lines, but also support communication network gear such as fiber optic cables and related hardware. The poles thus have a *power space* (transmission lines, usually higher up the pole) and also a *communication space* (usually lower on the pole). Network gear may be attached directly to the poles (pole mounted), or attached to steel cables running along poles (strand mounted). Poles may be owned by local governments or utility companies. There are regional variations on how quickly and easily communication companies can obtain permissions to access poles to attach new equipment or modify existing equipment. The FCC recently moved to relax these restrictions. Naturally different parties have starkly different views and these are represented in filings by individuals and organizations (including lobbying groups).

### Mapping Fema Flood Insurance Adoption Rates and Sensitivity 
"_Floods are the most common and costly natural hazard in the nation._" ([source: fema.gov](https://www.fema.gov/wildfires-you-need-flood-insurance)). 

Low adoption rates of catastrophic insurance for floods or mudslides often show up in news reports following wildfires. In addition to storm related flooding, there are also increasing concerns about _sunny-day floods_. These are (increasingly frequent) situations where sea water is driven up coastal sewer systems by strong tides (particularly in the absence of storms or hurricanes). 

This [project (repo) maps and estimates](https://github.com/petecarkeek/floodInsurance) the impact of prices and geographic (long/lat) location on insurance adoption rates. Estimates are obtained using regression and using a blend of random forest based estimation with linear regression.

### Spectrum Sharing in Licensed Wireless Communication Bands
Pricing models to study the sharing of radio frequency resources between existing (e.g. cellular networks) and emerging technologies (e.g. robotics, sensor networks). Link to [repo](https://github.com/petecarkeek/Spectrum_Sharing) with programs used for numerical examples and plots for [IEEE paper](https://ieeexplore.ieee.org/abstract/document/8301016/) on dynamic spectrum allocation in licensed wireless bands. 

### ML and econometrics
One approach to blending machine learning tools with traditional econometrics framework is to think about the **anatomy of a regression** formula (e.g. see 'Mostly harmless econometrics' by Angrist and Pischke). The formula states that the **partial effect** of a variable X on Y can be estimated as *beta = Cov(Resid.Y, Resid.X)/Var(Resid.X)*, where these are residuals obtained by regressing Y and X on *all other covariates*. These regressions, in turn, can be based on linear regression, or any other statistical learning method such as random forests or neural networks. Standard errors can be obtained using resampling methods e.g. bootsrap, jackknife etc.  

* [Multiple regression and generalized random forest](./genRandForest.ipynb) : simple "text book" example of a wage regression to compare linear models with random forest based estimation.

* [Treatment Effects Estimation](./avgTreatmentEffect.ipynb): regression and propensity score based methods for treatment effects estimation. The logistic regression in the example can be substituted with non-parametric ML tools such as random forest regressions. 
