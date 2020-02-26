
#--------------------------------------------------------------
# Importing Dataset:

dataset = read.csv('Salary_Data.csv')
dataset = dataset[,2:3]

# -------------------------------------------------------------
# Splitting Dataset into Training & Test Set:

# Install caTools if not yet installed onto computer
# install.packages('caTools')
library(caTools)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# -------------------------------------------------------------
# Feature Scaling:

training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])

# -------------------------------------------------------------
# Fitting Linear Regression to Training Data:

regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# -------------------------------------------------------------
# Predicting Test Set Results: 

y_pred = predict(regressor, newdata = test_set)

# -------------------------------------------------------------
# Visualizing Training Set Results:

# Install ggplot2 if not yet installed onto computer:
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = ' blue') +
  ggtitle('Salary vs Years Experience (Training Set)') + 
  xlab('Years Experience') + 
  ylab('Salary')

# -------------------------------------------------------------
# Visualizing Test Set Results:

ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = test_set)),
            colour = ' blue') +
  ggtitle('Salary vs Years Experience (Test Set)') + 
  xlab('Years Experience') + 
  ylab('Salary')
