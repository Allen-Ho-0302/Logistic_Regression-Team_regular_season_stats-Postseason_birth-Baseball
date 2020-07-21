Using data from Baseball Reference

Using SQL Server and Python(Spyder)

Gathering MLB regular season team stats from 2012-2019, including stats like PA, AB, H, 2B, 3B, HR, RBI, SB, BA, OBP, SLG, etc 

See how these stats influence postseason birth

Take OBP's result for example:

                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -46.1098      6.456     -7.142      0.000     -58.764     -33.456
OBP          141.2667     19.986      7.068      0.000     102.095     180.439


Intercept = -46.10975255466771
Slope = 141.2666813157292 ===> beta coefficient is where the likelihood takes on maximum value

Confidence Interval of Intercept and Slope
Intercept  -58.763962  -33.455543
OBP        102.094524  180.438839

Multiplicative effect on the odds(how much the odds of breaking into postseason will multiply per unit increase in OBP)
OBP          2.245640e+61

Estimated probability at x = 0.333:  0.7175  ---->imply that if your team has a season avg obp of 0.333, you have over 70% chance of breaking into postseason

Estimated rate of change in probability at x = 0.333:  28.6344

covariance matrix:
            Intercept         OBP
Intercept   41.684431 -128.994834
OBP       -128.994834  399.446666

SE:  19.9862(squareroot of 399.446666)

Wald statistic:  7.0682 ====>when Wald statistics>2, imply the variable(obp here) is statistically significant

Confidence interval of the odds: imply that with a unit increase in obp, the odds of breaking into postseason is multiplied by minimum of 2.183174e+44 to maximum of 2.309894e+78
Intercept  3.013946e-26  2.954217e-15
OBP        2.183174e+44  2.309894e+78

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

multivariate logistic regression results, take BA+OBP+SLG+OPS for example:

Generalized Linear Model Regression Results                  

Dep. Variable:                   POST   No. Observations:                  240
Model:                            GLM   Df Residuals:                      235
Model Family:                Binomial   Df Model:                            4
Link Function:                  logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -107.74
Date:                Mon, 20 Jul 2020   Deviance:                       215.49
Time:                        21:39:18   Pearson chi2:                     213.
No. Iterations:                     5                                         
Covariance Type:            nonrobust                                         

                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -45.3516      6.671     -6.799      0.000     -58.426     -32.277
BA           -40.6929     24.142     -1.686      0.092     -88.009       6.624
OBP         -379.9535    337.463     -1.126      0.260   -1041.369     281.462
SLG         -551.9089    340.083     -1.623      0.105   -1218.460     114.642
OPS          551.5402    339.361      1.625      0.104    -113.596    1216.676


VIF: variance inflation factor, if the VIF is above 2.5 should consider there is effect of multicollinearity on fitted model, result is quite accurate because OBP, SLG and OPS are highly correlated to each other

   variables          VIF
0         BA     2.466888
1        OBP   584.537475
2        SLG  3010.441751
3        OPS  5539.163303
4  Intercept   805.919300

Compare deviance of null and residual model(the model of BA+OBP+SLG+OPS)
difference in deviance: 90.03965842095988

If adding one variable to the model at a time can decrease deviance more than 1, we can say the addition is valuable
Adding BA to the null model reduces deviance by:  31.766
Adding OBP to the BA model reduces deviance by:  -52.756
Adding SLG to the BA model reduces deviance by: -10.567
Adding OPS to the BA model reduces deviance by:  -29.213
=====> imply that adding BA to the null model, it is valuable, however not valuable for adding OBP, SLG, OPS later.

model matrix, dmatrix(): y=x1+x2+x3... respresent as y=X

model matrix with OBP and OPS:
   Intercept    OBP    OPS
0        1.0  0.336  0.789
1        1.0  0.352  0.848
2        1.0  0.338  0.810
3        1.0  0.329  0.767
4        1.0  0.338  0.832

model matrix for OBP with log transformation

                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept      51.0160      7.303      6.985      0.000      36.702      65.330
np.log(OBP)    45.5547      6.460      7.052      0.000      32.894      58.215



Overall One Line Conclusion: if your team has a season avg obp of 0.333, you have over 70% chance of breaking into postseason 







