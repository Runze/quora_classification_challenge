library(stringr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(caret)
library(coefplot)
options(stringsAsFactors = F)
setwd('/Users/Runze/Google Drive/quora_classification_challenge')

# read in training data (the first 4500 rows)
train = read.table('input00.txt', sep = ' ', skip = 1, nrows = 4500)
str(train)

# remove answer id (the first column) and rename others
train = dplyr::select(train, -V1)
names(train)[1] = 'rating'
names(train)[2:ncol(train)] = paste0('v_', seq_len(ncol(train)-1))
train = data.frame(lapply(train, function(x) gsub('[0-9]+:', '', x)))
str(train)

# select and transform variables
# check value frequencies for each variable
sapply(train, function(x) length(table(x)))

# v_1 has a different value for each observation and
# v_22 and v_23 have the same value for all observations
# hence, remove all 3 of them
train = dplyr::select(train, -v_1, -v_22, -v_23)

# identify and convert factor and numerical variables
# based on the value frequencies calculated above
fac_vars = c('rating', 'v_3', 'v_9', 'v_10', 'v_13', 'v_14', 
             'v_15', 'v_16', 'v_17', 'v_19')
num_vars = names(train)[!(names(train) %in% fac_vars)]

train[, fac_vars] = data.frame(lapply(train[, fac_vars], as.factor))
train[, num_vars] = data.frame(lapply(train[, num_vars], as.numeric))
levels(train$rating) = list('down' = -1, 'up' = 1)
str(train)

# split into training and test set
set.seed(2014)
train_ind = createDataPartition(train$rating, p = .75, list = F)
train_tr = train[train_ind, ]
train_te = train[-train_ind, ]

# check for missing value
table(sapply(train_tr, function(x) sum(is.na(x))))

# analyze the training set
# visualize relationships with the response variable

# analyze factor variables
# visualize level distributions
train_fac = train_tr[, fac_vars] %>%
  melt(id = 'rating') %>%
  group_by(variable, value) %>%
  summarise(freq = n())
train_fac %>% head(10)
  
ggplot(train_fac, aes(x = value, y = freq)) +
  geom_bar(stat = 'identity', fill = 'light blue') +
  facet_wrap(~ variable, scales = 'free')

# v_19 is not helpful
train_tr = dplyr::select(train_tr, -v_19)
fac_vars = fac_vars[fac_vars != 'v_19']

# v_9's higher levels can be binned together
levels(train_tr$v_9) = c('0', rep('1', 6))

# visualize relationship with the response variable
train_fac = train_tr[, fac_vars] %>%
  melt(id = 'rating') %>%
  group_by(variable, value, rating) %>%
  summarise(freq = n()) %>%
  mutate(success_rate  = freq / sum(freq)) %>%
  filter(rating == 'up')
train_fac %>% head(10)

ggplot(train_fac, aes(x = value, y = success_rate)) +
  geom_bar(fill = 'light pink', stat = 'identity') + 
  facet_wrap(~ variable, scales = 'free')

# although some variables look very indicative,
# they are actuall due to the infrequencies of certain levels

# analyze numeric variables
# visualize distribution
train_num = train_tr[, c('rating', num_vars)] %>%
  melt(id = 'rating')

hist = ggplot(train_num, aes(x = value)) + 
  geom_histogram(fill = 'light green') +
  facet_wrap(~ variable, scales = 'free')
suppressMessages(print(hist))

# many variables are very skewed
# try boxcox transformation
# first add 1 to variables
train_tr[, num_vars] = data.frame(lapply(train_tr[, num_vars],
                                         function(x) x + 1))

# boxcox transformation
bc = preProcess(train_tr[, num_vars], method = 'BoxCox')
train_tr[, num_vars] = predict(bc, train_tr[, num_vars])

# plot again
train_num_2 = train_tr[, c('rating', num_vars)] %>%
  melt(id = 'rating')

hist_2 = ggplot(train_num_2, aes(x = value)) + 
  geom_histogram(fill = 'light green') +
  facet_wrap(~ variable, scales = 'free')
suppressMessages(print(hist_2))

# better but still not good enough
# visualize relationship with the response variable
ggplot(train_num_2, aes(x = rating, y = value)) +
  geom_boxplot(fill = 'light yellow') +
  facet_wrap(~ variable, scales = 'free')

# train models
# first convert factor variables to dummy variables
rating_tr = train_tr$rating
train_tr_m = model.matrix(rating ~ ., data = train_tr)[, -1]

# check for near-0 variance variables
nz_var = nearZeroVar(train_tr_m)
colnames(train_tr_m[, nz_var])
train_tr_m = train_tr_m[, -nz_var]

# check for highly correlated variables
high_corr = findCorrelation(cor(train_tr_m)) # none

# final variables (used to subset test tests)
final_vars = colnames(train_tr_m)

# 10-fold cv
ctrl = trainControl(method = 'cv', number = 10, classProbs = T,
                    summaryFunction = twoClassSummary)

# logistic regression
set.seed(2014)
logit_m = train(x = train_tr_m, y = rating_tr,
                method = 'glm', family = 'binomial',
                metric = 'ROC', trControl = ctrl)
logit_m
coefplot(logit_m)

# svm
set.seed(2014)
svm_g = expand.grid(sigma = .0638, C = seq(1, 5, 1))
svm_m = train(x = train_tr_m, y = rating_tr,
              method = 'svmRadial', preProcess = c('center', 'scale'),
              metric = 'ROC', trControl = ctrl, tuneGrid = svm_g)
svm_m

# rf
set.seed(2014)
rf_m = train(x = train_tr_m, y = rating_tr,
              method = 'rf', ntree = 1000,
              metric = 'ROC', trControl = ctrl)
rf_m

# gbm
set.seed(2014)
gbm_g = expand.grid(n.trees = seq(100, 300, 50), interaction.depth = 1:3, 
                    shrinkage = c(.01, .1))
gbm_m = train(x = train_tr_m, y = rating_tr,
              method = 'gbm', preProcess = c('center', 'scale'),
              metric = 'ROC', trControl = ctrl, tuneGrid = gbm_g, verbose = F)
gbm_m

resamps = resamples(list(logistic = logit_m, svm = svm_m, random_forest = rf_m, gbm = gbm_m))
parallelplot(resamps)

save(logit_m, file = 'logit_m.RData')
save(svm_m, file = 'svm_m.RData')
save(rf_m, file = 'rf_m.RData')
save(gbm_m, file = 'gbm_m.RData')

# apply to test set
# bin v_9's higher levels
levels(train_te$v_9) = c('0', rep('1', 6))

# add 1 to numeric variables
train_te[, num_vars] = data.frame(lapply(train_te[, num_vars],
                                         function(x) x + 1))

# boxcox transformation
train_te[, num_vars] = predict(bc, train_te[, num_vars])

# convert to model matrix
rating_te = train_te$rating
train_te_m = model.matrix(rating ~ ., data = train_te)[, -1]
train_te_m = train_te_m[, final_vars]

# only apply random forest and gbm
rf_f = predict(rf_m, train_te_m, type = 'prob')
gbm_f = predict(gbm_m, train_te_m, type = 'prob')

mean_f = (rf_f$up + gbm_f$up) / 2
roc_te = roc(rating_te, mean_f)
plot(roc_te) # .9017

# apply to the real test set (the remaining rows in the file)
test = read.table('input00.txt', sep = ' ', skip = 4502)
test = dplyr::select(test, -V1)
names(test) = paste0('v_', seq_len(ncol(test)))
test = data.frame(lapply(test, function(x) gsub('[0-9]+:', '', x)))

fac_vars = fac_vars[fac_vars != 'rating']
test = test[, c(fac_vars, num_vars)]

test[, fac_vars] = data.frame(lapply(test[, fac_vars], as.factor))
test[, num_vars] = data.frame(lapply(test[, num_vars], as.numeric))
str(test)

# apply models
# bin v_9's higher levels
levels(test$v_9) = c('0', rep('1', 5))

# add 1 to numeric variables
test[, num_vars] = data.frame(lapply(test[, num_vars],
                                     function(x) x + 1))

# boxcox transformation
test[, num_vars] = predict(bc, test[, num_vars])

# convert to model matrix
test_m = model.matrix(~ ., data = test)[, -1]
test_m = test_m[, final_vars]

# only apply random forest and gbm
rf_f_test = predict(rf_m, test_m, type = 'prob')
gbm_f_test = predict(gbm_m, test_m, type = 'prob')

mean_f_test = (rf_f_test$up + gbm_f_test$up) / 2

# compare with test outcome
test_outcome = read.table('output00.txt', sep = ' ')
test_rating = as.factor(test_outcome$V2)
levels(test_rating) = list('down' = -1, 'up' = 1)

# roc
roc_test = roc(test_rating, mean_f_test)
plot(roc_test) # .8972
