# Introduction

This project uses an anonymised dataset, consisting of 712 records (out of approximately 2224) of passengers on the RMS Titanic, which tragically sank on 15 April 1912. The variables are as so:

• `Survived`: 1 = yes, 0 = no - the response variable

• `Pclass`: Passenger class - 1 (1st), 2 (2nd), 3 (3rd)

• `Sex`: ‘male’, ‘female’

• `Age`: 1 = child (under 18), 2 = adult (18 to 60), 3 = senior (over 60)

• `Parch`: Number of parents and/or children on board for a passenger

• `Embarked`: Port of embarkment - C = Cherbourg, S = Southampton, Q = Queenstown

All variables are as factors, apart from Parch. The aim of the project is to analyse the dataset using logistic regression modelling. 

Note that this dataset is public, although reduced for the purposes of this analysis. The extended dataset can be found at: https://www.kaggle.com/c/titanic/data?select=test.csv, where there is an ongoing competition to predict the odds of survival using machine learning techniques.

# Part I

First, we fit a logistic regression model with a logit link function, including all covariates. The model we are fitting is of the form:

![Screenshot 2024-07-29 at 10 32 20](https://github.com/user-attachments/assets/e3da07fb-ebc7-46da-bbdb-9f0b0708b7f0)

Where:

- **$I_{⟨PClass=2⟩}$**: Indicator variable, 1 for an individual in 2nd class, 0 otherwise. Similarly for **$I_{⟨PClass=3⟩}$**.
- **$I_{⟨Sex=male⟩}$**: Indicator variable, 1 for a male passenger, 0 otherwise.
- **$I_{⟨Age=2⟩}$**: Indicator variable, 1 for an adult passenger, 0 otherwise. Similarly for **$I_{⟨Age=3⟩}$***.
- **$x_{Parch}$**: Number of parents/children with the passenger.
- **$I_{⟨Embarked=Q⟩}$**: Indicator variable, 1 if embarked in Queenstown, 0 otherwise. Similarly for **$I_{⟨Embarked=S⟩}$**.

The reference categories are first class, child, female, and Cherbourg embarkment, so their coefficients are not visible in the model.
In this case, we get the following R summary:
```r
    Call:
glm(formula = Survived ~ Pclass + Sex + Age + Parch + Embarked,
    family = binomial(link = "logit"), data = titanic.df)

Deviance Residuals:
    Min       1Q   Median       3Q      Max
-2.3646  -0.7084  -0.4173   0.6936   2.3847

Coefficients:
               Estimate Std. Error z value Pr(>|z|)    
(Intercept)     3.5307     0.4009   8.806  < 2e-16 ***
Pclass2        -0.9303     0.2775  -3.352  0.000802 ***
Pclass3        -2.0729     0.2649  -7.827  5.01e-15 ***
Sexmale        -2.5566     0.2154 -11.868  < 2e-16 ***
Age2           -0.7980     0.2592  -3.079  0.002077 **
Age3           -1.8475     0.6687  -2.763  0.005732 **
Parch          -0.1520     0.1148  -1.324  0.185661    
EmbarkedQ      -0.8867     0.5729  -1.548  0.121647    
EmbarkedS      -0.5003     0.2655  -1.884  0.059533 .
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 960.90  on 711  degrees of freedom
Residual deviance: 652.31  on 703  degrees of freedom
AIC: 670.31

Number of Fisher Scoring iterations: 4
`````

Using the R summary, we have the fitted model:

![Screenshot 2024-07-29 at 10 47 47](https://github.com/user-attachments/assets/352296dc-84f7-4514-a4a1-73d2388e5179)

Looking closer at the summary, the larger the absolute values of z-values, the more likely that the true coefficient isn’t zero, and the p-values determine the statistical significance. 
They indicate a rejection of the null hypothesis that the true coefficients are zero if the p-value is (typically) < 0.05. 
We see that the `Sexmale` z-value is the largest, hence the p-value is the smallest. `Parch`, `EmbarkedQ`, and `EmbarkedS` show p-values > 0.05, 
indicating not enough evidence to reject the null hypothesis for these. These variables may not be strong predictors of survival.

# Part II

Next, we fit a reduced model which contains only the covariates `PClass`, `Sex` and `Age`. To test whether the missing covariates don’t make a significant contribution to the fit, 
we apply an analysis of deviance, using the Chi-squared test. The result is as follows:

```r
Model 1: Survived ~ Pclass + Sex + Age + Parch + Embarked
Model 2: Survived ~ Pclass + Sex + Age
  Resid. Df Resid. Dev Df Deviance Pr(>Chi)
1       703     652.31
2       706     658.40 -3  -6.0839   0.1076
````

The p-value of 0.1076 (> 0.05) indicates that there’s not enough evidence to reject the null hypothesis that the reduced model (Model 2) fits the data as well as the more complex one (Model 1). So, excluding `Parch` and `Embarked` does not significantly reduce the model’s ability to fit the data.
Calling the summary for the reduced model, it shows that all covariates and their p-values are less than 0.05, showing that they are all statistically significant. Further investigation with reducing the model further and performing
analysis of deviance against all combinations of models which contain the three covariates. Looking at all p-values against our original reduced model, we see that for all models they have a p-value lower than the significance level of 0.05, 
thus we conclude that the model cannot be reduced any further.

# Part III

We create a binary variable within titanic.df, called pred.surv which maps any estimated probability of survival > 0.5 to 1, and if ≤ 0.5, maps to 0. With this, we create a confusion matrix of predicted vs actual using this new variable. Table 1 shows the results. The rate of correct classification, i.e the accuracy is 0.782/78.2% (to 3 s.f). 
This is an acceptable rate, but improvement could possibly be made, for example by adding interaction terms. Additionally, the total number of survived passengers in the dataset is 288 (out of 712). 
This means that this is a class imbalance with majority class being 0 - not survived, which can greatly affect the model. One way to combat this could be adding class weights.


|  Actual/Predicted |      0      |      1      |
|-------------------|-------------|-------------|
|        0          | 359         | 65          |
|        1          | 90          | 198         |

# Part IV

Now let's estimate the odds of survival for an adult female in 2nd class. From the reduced model, we utilise the intercept, the coefficient for `Pclass` = 2, and `Age` = 2, so we have the log odds being:

![Screenshot 2024-07-29 at 11 01 23](https://github.com/user-attachments/assets/9d815262-bb14-4452-bbf3-570050878595)

which we take the exponent of, to find the odds being 3.45 (to 2dp). In this context, an adult female in 2nd class is 3.45 times more likely to survive the sinking than to not survive. These relatively high odds indicate a strong chance of survival for this specific group.


Now, we wish to find the odds ratio between an adult female in 1st class to a senior male travelling 3rd class. 
We can do this by calculating the odds for both groups individually, then dividing the odds of a 1st class adult female surviving by the odds of a 3rd class senior male surviving. 
Applying all relevant coefficients, we get the following log-odds:

![Screenshot 2024-07-29 at 11 02 18](https://github.com/user-attachments/assets/e65ca7e3-2b21-4ca7-909f-46eb9b63ce6f)

Giving us the log odds ratio:

![Screenshot 2024-07-29 at 11 02 30](https://github.com/user-attachments/assets/36ec8b9b-d694-407c-ba06-367373c0c5c1)

to two decimal places. This means that an adult female in 1st class is approximately 339 times more likely to survive than a senior male in 3rd class. 
Contextually this makes sense, albeit an extremely high odds ratio, as the evacuation process during the Titanic sinking was that “women and children first” 
boarded the evacuation boats, and 1st class passengers had better access to the boats than other classes [1].

# Part V

Finally, we estimate the probability of survival for an adult female travelling 2nd class and find an approximate 95% confidence interval for the true value. Going back to Part IV, we have already found the odds for this, 
so to get the probability of survival we use the predict function on the reduced model; this gives us a value of 0.775. 
Constructing the 95% confidence intervals gives [ 0.6894816 , 0.8429738 ] . This is a fairly wide range, giing uncertainty around an exact probability of survival,
but the predicted probability falls slightly above the CI’s midpoint, suggesting the model’s estimate as reliable within the statistical certainty defined by the interval.

# References 

[1] Marie Look. Who were the titanic survivors?, Oct 2017. Available at https://history.howstuffworks.com/history-vs-myth/colossal-conspiracies-about-why-titanic-sank.htm (Accessed: 19/03/2024).
