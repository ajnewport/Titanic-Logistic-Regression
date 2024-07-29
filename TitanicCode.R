

df <- load("/Users/milly/Downloads/titanic.rdata")
head(titanic.df)
summary(titanic.df)
sum(titanic.df$Survived==1)

class(titanic.df$Parch)

titanic.df$Pclass<-factor(titanic.df$Pclass)
titanic.df$Sex<-factor(titanic.df$Sex)
titanic.df$Age<-factor(titanic.df$Age)
titanic.df$Embarked<-factor(titanic.df$Embarked)

# 1. Fitting the logistic reg model with logit link 
model1 <- glm(Survived ~ Pclass + Sex + Age + Parch + Embarked, 
              family=binomial(link="logit"), data=titanic.df) # logit is default but I put it here for completeness

summary(model1)

exp(model1$coeff)

model2 <- glm(Survived ~ Pclass + Sex + Age, family=binomial(link="logit"), data = titanic.df)
summary(model2)

anova(model1, test="Chisq")
anova(model2)
anova(model1, model2, test = "Chisq") #use this - p value prob - accept null hypothesis that the two vars have no stat significance
# model cannot be reduced any further apparently. CHECK

model3 <- glm(Survived ~ Sex + Age, family=binomial(link="logit"), data = titanic.df)
anova(model2, model3, test = "Chisq")

model4 <- glm(Survived ~ Pclass + Age, family=binomial(link="logit"), data = titanic.df)
anova(model2, model4, test = "Chisq")

model5 <- glm(Survived ~ Pclass + Sex, family=binomial(link="logit"), data = titanic.df)
anova(model2, model5, test = "Chisq")

# 3. Predicting survival 
probabilities <- predict(model2, type = "response")
pred.surv <- ifelse(probabilities > 0.5, 1, 0)
titanic.df <- cbind(titanic.df, pred.surv)

comparison_table <- table(titanic.df$Survived, pred.surv)
print(comparison_table)

#    pred.surv
#     0   1
# 0 359  65
# 1  90 198

# Same but for model 1 (baseline?)
prob1 <- predict(model1, type = "response")
pred.surv <- ifelse(prob1 > 0.5, 1, 0)
titanic.df <- cbind(titanic.df, pred.surv)

comparison_table <- table(titanic.df$Survived, pred.surv)
print(comparison_table)



correctly_classified <- sum(diag(comparison_table))
total_cases <- sum(comparison_table)
proportion_correct <- correctly_classified / total_cases
print(proportion_correct) # Accuracy: 0.7823034

# 4. Probabilities 
# Define the coefficients
intercept <- 3.0672
beta_pclass2 <- -1.0673
beta_age_adult <- -0.7608 

# Calculate the log odds for an adult female in 2nd class
log_odds <- intercept + beta_pclass2 + beta_age_adult

# Calculate the odds
odds <- exp(log_odds)

# Calculate the probability of survival
probability <- exp(log_odds) / (1 + exp(log_odds))

# Print the results
cat("Odds of Survival: ", odds, "\n")
# Odds of Survival:  41.26439 
# odds of surviving 41 times greater than not surviving
cat("Probability of Survival: ", probability, "\n")
# Probability of Survival:  0.9763394


# Coefficients for the comparison categories (assuming coefficients for 2nd and 3rd class are negative and male is negative)
beta_pclass1_to_3rd <- 0 # Implicitly, since 1st class is the reference and has no coefficient, and 3rd class is the comparison
beta_adult_to_senior <- -0.7608 # For an adult, assuming senior is the reference category and its impact is negative


# Calculate the difference in log odds (log odds ratio) for an adult female in 1st class vs. a senior male in 3rd class
log_odds_ratio <-  beta_pclass1_to_3rd + beta_adult_to_senior

# Calculate the odds ratio
odds_ratio <- exp(log_odds_ratio)

# Print the odds ratio
cat("Odds Ratio for survival (Adult Female in 1st class vs. Senior Male in 3rd class): ", odds_ratio, "\n")
# Odds Ratio for survival (Adult Female in 1st class vs. Senior Male in 3rd class):  5.585087 
# Adult female travelling first class is around 5.6 times more likely to survive compared to senior male, 3rd class.



# 4ii REDUX

# Calculating the odds for a female in 1st class - both are the leading factors so we just use the intercept 

log_odds_female_1st_class <- intercept + beta_age_adult

# Calculate the odds of survival for an adult female in first class
odds_female_1st_class <- exp(log_odds_female_1st_class) # 10.03822...


# The coefficients for a senior male in 3rd class:
beta_male <- -2.4809
beta_3rd_class <- -2.2334
beta_senior <- -1.8734

# Now, calculate the odds of survival for a senior male in 3rd class
log_odds_oldmale_3rd_class <- intercept + beta_male + beta_3rd_class + beta_senior

# Calculate the odds of survival for an senior male in third class
odds_oldmale_3rd_class <- exp(log_odds_oldmale_3rd_class)  # 0.02958....

# Odds ratio:
odds_ratio <- odds_female_1st_class / odds_oldmale_3rd_class


# Odds ratio : 339.3052009.....

#5. confidence intervals 

# Creating a new data frame for the scenario: adult female in 2nd class
new_data <- data.frame(Pclass = factor("2", levels = c("1", "2", "3")),
                       Sex = factor("female", levels = c("male", "female")),
                       Age = factor("2", levels = c("1", "2", "3")))

# Predicting the probability of survival
predicted_probability <- predict(model2, newdata = new_data, type = "response")

# Calculating confidence intervals
# Note: For logistic regression, confidence intervals on the probability scale can be complex due to the non-linearity of the logit transformation.
# A common approach is to calculate CIs for the log odds and then convert them to probabilities.

# Predicting log odds with standard errors to get confidence interval
predicted_log_odds <- predict(model2, newdata = new_data, type = "link", se.fit = TRUE)
log_odds <- predicted_log_odds$fit
se_log_odds <- predicted_log_odds$se.fit

# Z value for 95% confidence interval
z_value <- qnorm(0.975)

# Confidence interval on the log odds scale
ci_lower_log_odds <- log_odds - z_value * se_log_odds
ci_upper_log_odds <- log_odds + z_value * se_log_odds

# Convert log odds CI to probability CI
ci_lower_prob <- exp(ci_lower_log_odds) / (1 + exp(ci_lower_log_odds))
ci_upper_prob <- exp(ci_upper_log_odds) / (1 + exp(ci_upper_log_odds))

# Print results
cat("Predicted Probability of Survival: ", predicted_probability, "\n")
# Predicted Probability of Survival:  0.7754091 
cat("95% CI for Probability of Survival: [", ci_lower_prob, ", ", ci_upper_prob, "]", "\n")
# 95% CI for Probability of Survival: [ 0.6894816 ,  0.8429738 ]


