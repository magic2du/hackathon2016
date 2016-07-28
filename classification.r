# Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set
# Classification
install.packages("MASS")
install.packages("randomForest")
#install.packages("caret",repos = "http://cran.r-project.org", 
#                 dependencies = c("Depends", "Imports", "Suggests"))
library(MASS)
library(randomForest)
library(gbm)
library(caret)

# get training dataset
filename <- "E:/mycode/R/human activity-Hackathon2016 Oneweek/Data/Train/X_train.txt"
training_data <- read.table(filename,header=F,na.strings=c(""))

filename <- "E:/mycode/R/human activity-Hackathon2016 Oneweek/Data/Train/y_train.txt"
training_label <- read.table(filename,header=F, na.strings=c(""),col.names = "label", colClasses = "factor")

training_data_labeled <- cbind(training_data, training_label)

# get test dataset
filename <- "E:/mycode/R/human activity-Hackathon2016 Oneweek/Data/Test/X_test.txt"
test_data <- read.table(filename,header=F,na.strings=c(""))

filename <- "E:/mycode/R/human activity-Hackathon2016 Oneweek/Data/Test/y_test.txt"
test_label <- read.table(filename,header=F, na.strings=c(""),col.names = "label", colClasses = "factor")

test_data_labeled <- cbind(test_data, test_label)

# check missing values
missingvalue_list <- sapply(training_data_labeled,function(x) sum(is.na(x)))
any(missingvalue_list > 0)

######
# Method1: fit linear discriminant analysis (LDA)
lda_fit <- lda(formula=label~., data=training_data_labeled)
print(lda_fit)

# evaluate the fitted model
lda_predict <- predict(lda_fit, test_data_labeled)

# build confusion matrix
cm <- table(test_data_labeled$label, lda_predict$class)
diag(prop.table(cm,1))
sum(diag(prop.table(cm)))   #0.9487666  == sum(diag(ct)/sum(ct))

######
# Method2: fit Random Forest
rf_fit <- randomForest(formula=label~., data=training_data_labeled, mtry=24, ntree=100, importance=TRUE)
print(rf_fit)
#plot(rf_fit)
#varImpPlot(rf_fit)
rf_predict <- predict(rf_fit, test_data_labeled, type="response")
cm <- table(test_data_labeled$label, rf_predict)
accuracy <- sum(diag(cm)/sum(cm))      #0.9095509 when ntree=100


######
# Method3: fit gbm
gbm_fit <- gbm(formula=label~., data=training_data_labeled, distribution = "multinomial", n.trees = 500)
print(gbm_fit)
gbm_predict <- predict(gbm_fit, test_data_labeled, n.tree=500, type="response")
gbm_predictclass <- colnames(gbm_predict)[apply(gbm_predict, 1, which.max)]   # convert probability to category
cm <- table(test_data_labeled$label, gbm_predictclass)
accuracy <- sum(diag(cm)/sum(cm))    #0.7602783 when n.trees=500
