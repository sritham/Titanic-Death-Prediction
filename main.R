#-----------------------------------LOAD DATASET----------------------------
setwd("C:\\Sritham\\ML_project\\titanic")

#loading data with removal of NA values

data_test <- read.csv("test.csv", na.strings = c(""))
data_train <- read.csv("train.csv", na.strings = c(""))

#-----------------------------------DATA STR----------------------------
str(data_train)
str(data_test)

#-----------------------------------CLEANING DATASET----------------------------

data_train$PassengerId <- as.factor(data_train$PassengerId)
data_train$Survived <- as.factor(data_train$Survived)
data_train$Pclass <- as.factor(data_train$Pclass)
data_train$SibSp <- as.factor(data_train$SibSp)
data_train$Parch <- as.factor(data_train$Parch)

which(is.na(data_train))

data_train[!complete.cases(data_train$Age),'Age'] <- mean(data_train$Age, na.rm = T)
data_train[!complete.cases(data_train$Cabin),'Cabin']
data_train[!complete.cases(data_train$Embarked),'Embarked'] <- 'Q'

#-----------------------------------ADDITIONAL ANALYSIS----------------------------

cor(x = data_train$Age, y = data_train$Fare)

# --------------------------------- DEALING WITH CATEGORICAL VARIABLES ------------
#Gender Ticket EMbarked Parch SibSp Pclass
#DUMMY VARIABLES - Sex,Pclass, SibSp, Parch, Embarked
#TURN TO CHAR - Ticket
data_train$Embarked <- as.character(data_train$Embarked)

data_train$Male <- as.logical(0)
data_train$Female <- as.logical(0)

for(i in 1:NROW(data_train$Sex)){
  if(data_train$Sex[i] == 'male'){
    data_train$Male[i] <- as.logical(1)
  }
  if(data_train$Sex[i] == 'female'){
    data_train$Female[i] <- as.logical(1)
  }
}

data_train$Pclass <- as.numeric(data_train$Pclass)
data_train$SibSp <- as.numeric(data_train$SibSp)
data_train$Parch <- as.numeric(data_train$Parch)

data_train$Q <- as.logical(0)
data_train$C <- as.logical(0)
data_train$S <- as.logical(0)

for(i in 1:NROW(data_train$Embarked)){
  if(data_train$Embarked[i] == 'S'){
    data_train$S[i] <- as.logical(1)
  }
  if(data_train$Embarked[i] == 'C'){
    data_train$C[i] <- as.logical(1)
  }
  if(data_train$Embarked[i] == 'Q'){
    data_train$Q[i] <- as.logical(1)
  }
}

data_train$Ticket <- as.character(data_train$Ticket)

#----------------------------------- SPLIT ----------------------------
set.seed(123)
index <- sample(1:nrow(data_train),0.7 * nrow(data_train))
data_train_Train <- data_train[index,]
data_train_test <- data_train[-index,]

#----------------------------------- ENSEMBLE ----------------------------
barplot(table(data_train_Train$Survived))

#----------------------------------- GBM ----------------------------
library(caret)
library(dplyr)
colnames(data_train_Train)
columnNames <- select(data_train_Train,S,
                      C,
                      Q,
                      Fare,
                      Parch,
                      Pclass,
                      SibSp,
                      Age,
                      Male,
                      Female,
                      Pclass)

outcomeName <- 'Survived'
predictors <- names(columnNames)[names(columnNames) != outcomeName]

modelLookup('gbm')

?trainControl

fitControl <- trainControl(method = "none") 

model_gbm <- train(data_train_Train[,predictors],data_train_Train[,outcomeName],
                  method = 'gbm', 
                  trControl = fitControl)

#SUMMARIZING THE MODEL
#summarizes the important variables
gbmImp <- summary(model_gbm)
gbmImp
plot(gbmImp)

#-------------------------------- TO MEASURE THE ACCURACY TEST IT ON THE TEST DATA AFTER SPLIT --------------
gbm.predict <- predict(model_gbm, data_train_test[,predictors], type = 'raw')
confusionMatrix(gbm.predict,data_train_test[,outcomeName])

#-------------------------------- DRAW ROC CURVE --------------------
library(pROC)
gbm.probs <- predict(model_gbm, data_train_test[,predictors],type="prob") 
gbm.plot<-plot(roc(data_train_test$Survived,gbm.probs[,2]))

#area under the curve
auc(data_train_test$Survived,gbm.probs[,2])

#-------------------------------- TUNING --------------
fitControl2 <- trainControl(method = "repeatedcv",
                               number = 50,
                              repeats = 10,
                               sampling = "up")   # control parameters for training
# see help(trainControl) for details

gbm.tuned1<-train(data_train_Train[,predictors],data_train_Train[,outcomeName],   #model retraining
                 method='gbm',
                 trControl=fitControl2)
#SUMMARIZING THE MODEL
#summarizes the important variables
gbmImp_tuned1 <- summary(gbm.tuned1)
gbmImp_tuned1
plot(gbmImp)

#-------------------------------- AFTER TUNING 1 - TO MEASURE THE ACCURACY TEST IT ON THE TEST DATA AFTER SPLIT --------------
gbm.predict_tuning1 <- predict(gbm.tuned1, data_train_test[,predictors], type = 'raw')
confusionMatrix(gbm.predict_tuning1,data_train_test[,outcomeName])

gbm.plot<-plot(roc(data_train_test$Survived,gbm.probs[,2]))
gbm.probs_tuning1 <- predict(gbm.tuned1, data_train_test[,predictors],type="prob") 
gbm.plot_tuning1 <- plot(roc(data_train_test$Survived,gbm.probs_tuning1[,2]))

#-------------------------------- TESTING THE TEST DATA ------------------------------
gbm.predict_test <- predict(model_gbm, data_test[,predictors], type = 'raw')
f <- gbm.predict_test


#--------------------------------- KNN -------------------------------------
data_train_Train$Ticket <- as.character(data_train_Train$Ticket)
data_train_test$Ticket <- as.character(data_train_test$Ticket)

fitControl <- trainControl(method = "none") 

model_knn <- train(data_train_Train[,predictors],data_train_Train[,outcomeName],
                   method = 'rf', 
                   trControl = fitControl)

knnImp <- varImp(model_knn)
knnImp

knn_predict1 <- predict(model_knn, data_train_test[,predictors], type = 'raw')
#-------------------------------- EXTRACTING THE DATA TO CSV---------------------------
final <- cbind.data.frame('PassengerId' = data_test$PassengerId, 'Survived' = f)

library("data.table")
fwrite(final, file = "submission_sri.csv")


#------------------------------- 
