# Data Preprocessing Template

# Importing Dataset
dataset = read.csv('Data.csv')

# Taking Care of Missing Data
# Choose method to replace missing data, usual = mean
dataset$Var1 = ifelse(is.na(dataset$Age),
                     ave(dataset$Var1, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Var1)
dataset$Var2 = ifelse(is.na(dataset$Var2),
                      ave(dataset$Var2, FUN = function(x) mean(x, na.rm = TRUE)),
                      dataset$Var2)

#Encoding Categorical Data
#Assign categorical values numerical values 
dataset$CatVar1 = factor(dataset$CatVar1,
                        levels = c('value1', 'value2', 'value3'),
                        labels = c(1, 2, 3))
#No or yes
dataset$CatVar2 = factor(dataset$CatVar2,
                         levels = c('No', 'Yes'),
                         labels = c(0, 1))

# Splitting the dataset into the Training and Test set
# Install package
library(caTool)
set.seed(123)
split = sample.split(dataset$CatVar2, SplitRatio = 0.8)
training_set = subset(dataset, split ==TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Reaplce num with select columns indexes for scaling
training_set[:, num] = scale(training_set[:, num])
test_set[:, num] = scale(test_set[:, num])

#



