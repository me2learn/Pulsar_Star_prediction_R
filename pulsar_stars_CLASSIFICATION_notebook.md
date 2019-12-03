``` r
# Predicting PULSAR Star

getwd()
```

    ## [1] "/home/myubu/R_projects_Github/Pulsar_Star_prediction_R"

``` r
setwd("/home/myubu/R_projects_Github/Pulsar_Star_prediction_R")

pulsar <- read.csv("pulsar_stars.csv")

str(pulsar)
```

    ## 'data.frame':    17898 obs. of  9 variables:
    ##  $ Mean.of.the.integrated.profile              : num  140.6 102.5 103 136.8 88.7 ...
    ##  $ Standard.deviation.of.the.integrated.profile: num  55.7 58.9 39.3 57.2 40.7 ...
    ##  $ Excess.kurtosis.of.the.integrated.profile   : num  -0.2346 0.4653 0.3233 -0.0684 0.6009 ...
    ##  $ Skewness.of.the.integrated.profile          : num  -0.7 -0.515 1.051 -0.636 1.123 ...
    ##  $ Mean.of.the.DM.SNR.curve                    : num  3.2 1.68 3.12 3.64 1.18 ...
    ##  $ Standard.deviation.of.the.DM.SNR.curve      : num  19.1 14.9 21.7 21 11.5 ...
    ##  $ Excess.kurtosis.of.the.DM.SNR.curve         : num  7.98 10.58 7.74 6.9 14.27 ...
    ##  $ Skewness.of.the.DM.SNR.curve                : num  74.2 127.4 63.2 53.6 252.6 ...
    ##  $ target_class                                : int  0 0 0 0 0 0 0 0 0 0 ...

``` r
head(pulsar)
```

    ##   Mean.of.the.integrated.profile Standard.deviation.of.the.integrated.profile
    ## 1                      140.56250                                     55.68378
    ## 2                      102.50781                                     58.88243
    ## 3                      103.01562                                     39.34165
    ## 4                      136.75000                                     57.17845
    ## 5                       88.72656                                     40.67223
    ## 6                       93.57031                                     46.69811
    ##   Excess.kurtosis.of.the.integrated.profile Skewness.of.the.integrated.profile
    ## 1                               -0.23457141                         -0.6996484
    ## 2                                0.46531815                         -0.5150879
    ## 3                                0.32332837                          1.0511644
    ## 4                               -0.06841464                         -0.6362384
    ## 5                                0.60086608                          1.1234917
    ## 6                                0.53190485                          0.4167211
    ##   Mean.of.the.DM.SNR.curve Standard.deviation.of.the.DM.SNR.curve
    ## 1                 3.199833                               19.11043
    ## 2                 1.677258                               14.86015
    ## 3                 3.121237                               21.74467
    ## 4                 3.642977                               20.95928
    ## 5                 1.178930                               11.46872
    ## 6                 1.636288                               14.54507
    ##   Excess.kurtosis.of.the.DM.SNR.curve Skewness.of.the.DM.SNR.curve target_class
    ## 1                            7.975532                     74.24222            0
    ## 2                           10.576487                    127.39358            0
    ## 3                            7.735822                     63.17191            0
    ## 4                            6.896499                     53.59366            0
    ## 5                           14.269573                    252.56731            0
    ## 6                           10.621748                    131.39400            0

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(caretEnsemble)
```

    ## 
    ## Attaching package: 'caretEnsemble'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     autoplot

``` r
library(mlbench)
library(ggplot2)
library(GGally)
```

``` r
# Distribution of Target Variable
prop.table(table(pulsar$target_class))
```

    ## 
    ##          0          1 
    ## 0.90842552 0.09157448

``` r
## There is big class imbalance with 90.8% predictions for "NOT PULSAR STAR"
## & 9.1% for "PULSAR STAR".

options(repr.plot.width=4, repr.plot.height=8)
ggplot(data = pulsar, aes(x=target_class)) + 
geom_bar(width = 0.1, fill = "steelblue") +
geom_text(stat='count', aes(label=..count..), vjust=-0.5)
```

![](pulsar_stars_CLASSIFICATION_notebook_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
options(repr.plot.width=10, repr.plot.height=15)
ggpairs(pulsar, aes(colour=as.factor(target_class), alpha=0.4))
```

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

    ## Warning in cor(x, y, method = method, use = use): the standard deviation is zero

![](pulsar_stars_CLASSIFICATION_notebook_files/figure-markdown_github/unnamed-chunk-2-2.png)

``` r
dim(pulsar)
```

    ## [1] 17898     9

``` r
# splitting data into train and test
seed <- 7
set.seed(seed)
split = 0.8
trainindex <- createDataPartition(y = pulsar$target_class,
                                 p = split,
                                 list = F )

train_data <- pulsar[trainindex,]
test_data <- pulsar[-trainindex,]

# Checking the dimension of train & test data
dim(train_data)
```

    ## [1] 14319     9

``` r
dim(test_data)
```

    ## [1] 3579    9

``` r
# Plotting the distribution of Target Variable in Test Data
options(repr.plot.width=4, repr.plot.height=8)
ggplot(data = test_data, aes(x=target_class)) + 
geom_bar(width = 0.1, fill = "steelblue") +
geom_text(stat='count', aes(label=..count..), vjust=-0.5)
```

![](pulsar_stars_CLASSIFICATION_notebook_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r
seed <- 7
metric <- "Accuracy"
control <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 3)
```

``` r
set.seed(seed)
# Using Logistic Regerssion classifier, to evaluate performance of sampling techniques.
# Here we are using data as is, without applying any sampling technique.
fit.glm <- train(as.factor(target_class)~.,
                data = train_data,
                method = "glm",
                preProc = c("center", "scale"),
                trControl = control,
                metric = metric)

# Making Predictions
test_features <- test_data[,1:8]
test_target <- test_data[,9]

glm_predictions <- predict(fit.glm,test_features)

length(test_target)
```

    ## [1] 3579

``` r
# summarize results
confusionMatrix(glm_predictions, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3237   61
    ##          1   21  260
    ##                                           
    ##                Accuracy : 0.9771          
    ##                  95% CI : (0.9716, 0.9817)
    ##     No Information Rate : 0.9103          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.8513          
    ##                                           
    ##  Mcnemar's Test P-Value : 1.656e-05       
    ##                                           
    ##             Sensitivity : 0.80997         
    ##             Specificity : 0.99355         
    ##          Pos Pred Value : 0.92527         
    ##          Neg Pred Value : 0.98150         
    ##              Prevalence : 0.08969         
    ##          Detection Rate : 0.07265         
    ##    Detection Prevalence : 0.07851         
    ##       Balanced Accuracy : 0.90176         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
# Without any Sampling to take care of class imbalance, we have got an accuracy of  260 / 321 (80.99%)
```

``` r
#### Our objective for this exercise is to correctly classify as many Pulsar stars as possible. So we will use different classification models and sampling techniques to achieve highest correct classification (1).

set.seed(seed)
# Using Logistic Regerssion classifier, to evaluate performance of sampling techniques.
# Here we are "upsampling the data".
control$sampling <- "up"
fit.glm_up <- train(as.factor(target_class)~.,
                data = train_data,
                method = "glm",
                preProc = c("center", "scale"),
                trControl = control,
                metric = metric)

glm_predictions_up <- predict(fit.glm_up, test_features)
confusionMatrix(glm_predictions_up, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3169   27
    ##          1   89  294
    ##                                           
    ##                Accuracy : 0.9676          
    ##                  95% CI : (0.9613, 0.9731)
    ##     No Information Rate : 0.9103          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.8174          
    ##                                           
    ##  Mcnemar's Test P-Value : 1.481e-08       
    ##                                           
    ##             Sensitivity : 0.91589         
    ##             Specificity : 0.97268         
    ##          Pos Pred Value : 0.76762         
    ##          Neg Pred Value : 0.99155         
    ##              Prevalence : 0.08969         
    ##          Detection Rate : 0.08215         
    ##    Detection Prevalence : 0.10701         
    ##       Balanced Accuracy : 0.94429         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
# By upSampling to take care of class imbalance, we have got an accuracy of  294 / 321 (91.58%)
```

``` r
set.seed(seed)
# Using Logistic Regerssion classifier, to evaluate performance of sampling techniques.
# Here we are "SMOTEsampling the data".
control$sampling <- "smote"
fit.glm_smote <- train(as.factor(target_class)~.,
                data = train_data,
                method = "glm",
                preProc = c("center", "scale"),
                trControl = control,
                metric = metric)
```

    ## Loading required package: grid

``` r
glm_predictions_smote <- predict(fit.glm_smote, test_features)
confusionMatrix(glm_predictions_smote, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3196   29
    ##          1   62  292
    ##                                           
    ##                Accuracy : 0.9746          
    ##                  95% CI : (0.9689, 0.9795)
    ##     No Information Rate : 0.9103          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.8512          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.0007951       
    ##                                           
    ##             Sensitivity : 0.90966         
    ##             Specificity : 0.98097         
    ##          Pos Pred Value : 0.82486         
    ##          Neg Pred Value : 0.99101         
    ##              Prevalence : 0.08969         
    ##          Detection Rate : 0.08159         
    ##    Detection Prevalence : 0.09891         
    ##       Balanced Accuracy : 0.94531         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
# By SMOTE Sampling to take care of class imbalance, we have got an accuracy of  292 / 321 (90.97%)
```

We will use UPSAMPLING with different classifiers, to check if we can
achieve better score.

We will use the below classifiers:

LDA, SVM, KNN, NaiveBayes, CART, Bagged CART, Random Forest, Gradient
Boosting.

``` r
#LDA
control$sampling = "up"
set.seed(seed)
fit.pulsar.lda_up = train(as.factor(target_class)~., 
                          data = train_data, 
                          method = "lda", 
                          preProc = c("center", "scale"), 
                          trControl = control, 
                          metric = metric)

# LDA
pulsar_lda_up_predictions <- predict(fit.pulsar.lda_up,test_features)
confusionMatrix(pulsar_lda_up_predictions, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3214   36
    ##          1   44  285
    ##                                           
    ##                Accuracy : 0.9776          
    ##                  95% CI : (0.9723, 0.9822)
    ##     No Information Rate : 0.9103          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.8646          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.4338          
    ##                                           
    ##             Sensitivity : 0.88785         
    ##             Specificity : 0.98649         
    ##          Pos Pred Value : 0.86626         
    ##          Neg Pred Value : 0.98892         
    ##              Prevalence : 0.08969         
    ##          Detection Rate : 0.07963         
    ##    Detection Prevalence : 0.09193         
    ##       Balanced Accuracy : 0.93717         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
# LDA predicted 285 / 321 (88.79%) class 1's accurately

#SVMRADIAL
control$sampling = "up"
set.seed(seed)
fit.pulsar.svmradial_up = train(as.factor(target_class)~., 
                                data = train_data, 
                                method = "svmRadial", 
                                preProc = c("center", "scale"), 
                                trControl = control, 
                                metric = metric)

# SVM
pulsar_svm_up_predictions <- predict(fit.pulsar.svmradial_up,test_features)
confusionMatrix(pulsar_svm_up_predictions, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3193   28
    ##          1   65  293
    ##                                          
    ##                Accuracy : 0.974          
    ##                  95% CI : (0.9683, 0.979)
    ##     No Information Rate : 0.9103         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.8487         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.0001892      
    ##                                          
    ##             Sensitivity : 0.91277        
    ##             Specificity : 0.98005        
    ##          Pos Pred Value : 0.81844        
    ##          Neg Pred Value : 0.99131        
    ##              Prevalence : 0.08969        
    ##          Detection Rate : 0.08187        
    ##    Detection Prevalence : 0.10003        
    ##       Balanced Accuracy : 0.94641        
    ##                                          
    ##        'Positive' Class : 1              
    ## 

``` r
# SVM predicted 293 / 321 (91.28%) class 1's accurately

#KNN
control$sampling = "up"
set.seed(seed)
fit.pulsar.knn_up = train(as.factor(target_class)~., 
                          data = train_data, 
                          method = "knn", 
                          preProc = c("center", "scale"), 
                          trControl = control, 
                          metric = metric)

# KNN
pulsar_knn_up_predictions <- predict(fit.pulsar.knn_up,test_features)
confusionMatrix(pulsar_knn_up_predictions, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3062   28
    ##          1  196  293
    ##                                          
    ##                Accuracy : 0.9374         
    ##                  95% CI : (0.929, 0.9451)
    ##     No Information Rate : 0.9103         
    ##     P-Value [Acc > NIR] : 1.502e-09      
    ##                                          
    ##                   Kappa : 0.6899         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ##                                          
    ##             Sensitivity : 0.91277        
    ##             Specificity : 0.93984        
    ##          Pos Pred Value : 0.59918        
    ##          Neg Pred Value : 0.99094        
    ##              Prevalence : 0.08969        
    ##          Detection Rate : 0.08187        
    ##    Detection Prevalence : 0.13663        
    ##       Balanced Accuracy : 0.92631        
    ##                                          
    ##        'Positive' Class : 1              
    ## 

``` r
# KNN predicted 293 / 321 (91.28%) class 1's accurately

#NaiveBayes
control$sampling = "up"
set.seed(seed)
fit.pulsar.nb_up = train(as.factor(target_class)~., 
                         data = train_data, 
                         method = "nb", 
                         preProc = c("center", "scale"), 
                         trControl = control, 
                         metric = metric)
```

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 25

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 133

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 214

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 520

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 563

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 745

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1165

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1370

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 25

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 38

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 56

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 86

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 133

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 214

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 277

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 284

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 379

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 407

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 478

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 510

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 520

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 534

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 563

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 734

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 744

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 745

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 751

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 922

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1017

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1072

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1145

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1165

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1199

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1334

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1370

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1387

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1397

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1406

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 158

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 281

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 300

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 358

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 506

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 636

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 648

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 728

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 845

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 851

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 866

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 981

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1113

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1256

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 92

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 133

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 139

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 158

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 193

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 281

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 300

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 358

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 360

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 394

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 506

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 636

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 648

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 670

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 728

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 765

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 845

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 851

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 866

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 895

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 981

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 989

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1009

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1113

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1170

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1215

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1230

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1256

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1399

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1401

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 133

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 288

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 360

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 379

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 563

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 640

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 828

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 886

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 945

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1064

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1081

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1138

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1196

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1339

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1343

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 16

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 78

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 133

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 254

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 288

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 302

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 360

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 379

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 424

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 440

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 441

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 446

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 509

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 563

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 582

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 640

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 706

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 807

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 828

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 886

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 929

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 933

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 945

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 967

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 971

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1064

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1081

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1086

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1138

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1196

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1207

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1339

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1343

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1397

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 173

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 264

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 333

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 466

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 475

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 499

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1025

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1054

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1253

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1368

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1415

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 59

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 72

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 151

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 173

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 245

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 258

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 264

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 333

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 431

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 466

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 475

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 499

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 567

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 578

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 657

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 699

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 711

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 749

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 820

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 855

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 862

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 916

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 967

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 993

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1025

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1027

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1054

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1058

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1071

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1092

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1119

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1218

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1253

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1368

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1415

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 96

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 310

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 315

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 477

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 513

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 526

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 708

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 780

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1173

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1257

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1340

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1406

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 96

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 221

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 296

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 310

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 315

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 374

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 398

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 466

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 477

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 500

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 513

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 526

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 587

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 590

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 708

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 780

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 808

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1008

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1044

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1069

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1168

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1173

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1257

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1340

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1371

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1386

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1401

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1406

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1419

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 76

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 325

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 349

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 376

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 543

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 584

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 633

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 680

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 761

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 944

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 975

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1151

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 76

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 125

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 245

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 260

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 310

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 325

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 349

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 376

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 412

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 415

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 447

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 469

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 496

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 513

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 533

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 543

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 584

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 620

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 633

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 680

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 761

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 768

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 854

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 944

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 975

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 994

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1008

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1068

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1075

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1373

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1389

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1406

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1431

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 58

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 67

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 199

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 286

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 496

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 508

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 596

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 670

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 784

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 802

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 835

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1325

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 14

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 58

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 67

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 77

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 199

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 217

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 286

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 366

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 408

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 416

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 480

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 496

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 503

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 508

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 596

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 597

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 670

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 714

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 733

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 750

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 784

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 802

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 835

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 840

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 843

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 888

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 955

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 991

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1033

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1072

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1140

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1180

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1215

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1325

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1372

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1416

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 278

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 410

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 416

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 506

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 993

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1034

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1154

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 90

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 158

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 172

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 237

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 278

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 357

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 363

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 410

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 416

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 454

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 506

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 509

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 636

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 681

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 783

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 907

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 959

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 993

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1034

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1041

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1108

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1145

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1402

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 66

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 89

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 211

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 410

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 715

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 749

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 859

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 955

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1006

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1152

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1313

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1325

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 66

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 71

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 89

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 93

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 106

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 211

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 274

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 292

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 338

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 410

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 447

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 451

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 492

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 522

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 530

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 553

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 715

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 749

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 810

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 856

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 859

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 901

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 955

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 967

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 990

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1006

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1101

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1152

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1226

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1270

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1313

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1325

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1337

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 272

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 335

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 843

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 949

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1095

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1147

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1173

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1183

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1260

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1269

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 46

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 176

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 205

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 219

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 272

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 335

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 342

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 527

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 550

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 556

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 617

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 733

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 752

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 788

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 843

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 879

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 949

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1001

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1002

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1069

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1095

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1135

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1147

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1173

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1183

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1218

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1260

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1269

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1281

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1342

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1425

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1430

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1431

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 229

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 330

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 740

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 782

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 882

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 915

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1156

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1320

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 20

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 90

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 191

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 229

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 260

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 289

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 330

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 391

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 450

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 603

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 626

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 638

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 740

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 782

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 861

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 882

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 915

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 975

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 991

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1082

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1141

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1179

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1320

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1432

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 87

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 299

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 523

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 541

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 87

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 96

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 97

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 118

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 157

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 170

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 203

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 243

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 299

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 400

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 426

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 498

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 523

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 541

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 552

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 618

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 632

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 733

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 776

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 812

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 852

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 898

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 991

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1048

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1149

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1187

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1408

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1423

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 299

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 521

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 701

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 750

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 846

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 930

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 994

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1120

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1240

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1241

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1243

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1341

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 64

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 100

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 158

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 263

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 299

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 429

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 488

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 491

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 521

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 701

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 729

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 750

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 846

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 847

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 930

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 966

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 994

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 995

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1056

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1071

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1087

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1090

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1120

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1240

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1241

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1243

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1272

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1341

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1372

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 62

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 65

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 301

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 346

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 383

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 397

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 442

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 484

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 495

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 522

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 584

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 696

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1124

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1171

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1220

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1352

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1405

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 35

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 62

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 65

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 82

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 144

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 145

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 266

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 301

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 309

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 346

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 366

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 383

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 388

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 397

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 402

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 432

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 442

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 484

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 488

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 489

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 495

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 522

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 525

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 584

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 696

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 751

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 761

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 783

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 811

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 856

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 914

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1019

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1076

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1124

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1171

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1201

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1220

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1352

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1353

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1405

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 22

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 44

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 185

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 221

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 290

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 331

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 347

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 356

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 797

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 843

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1136

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1147

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1251

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1404

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 22

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 44

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 51

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 88

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 185

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 221

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 290

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 331

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 347

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 356

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 464

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 476

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 669

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 753

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 797

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 843

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1136

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1147

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1210

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1251

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1365

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1390

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1396

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1404

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1425

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 361

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 440

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 562

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 672

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 853

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 998

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1078

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1325

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1363

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 22

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 95

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 236

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 255

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 276

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 286

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 361

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 369

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 440

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 441

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 444

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 538

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 562

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 672

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 806

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 826

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 853

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 906

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 954

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 968

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 998

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1009

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1035

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1078

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1217

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1325

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1337

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1363

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1380

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1393

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 153

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 284

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 302

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 559

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 625

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 633

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 643

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 658

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 696

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 742

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 820

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 868

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 964

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1071

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1177

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1303

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1365

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1432

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 75

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 153

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 179

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 284

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 294

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 302

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 364

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 382

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 406

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 418

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 454

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 458

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 507

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 509

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 520

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 546

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 559

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 598

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 625

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 633

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 643

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 658

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 696

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 697

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 740

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 742

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 820

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 868

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 899

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 964

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 987

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 995

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1020

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1062

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1071

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1075

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1081

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1177

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1234

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1303

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1327

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1333

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1365

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1432

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 115

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 262

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 274

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 343

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 492

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 500

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 506

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 593

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1079

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1182

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1254

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 115

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 199

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 214

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 235

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 262

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 274

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 343

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 345

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 352

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 371

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 412

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 415

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 450

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 492

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 500

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 506

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 524

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 593

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 624

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 737

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 739

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 841

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 845

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 940

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1046

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1076

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1079

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1134

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1182

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1214

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1229

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1254

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1386

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1400

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1408

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1432

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 113

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 218

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 286

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 727

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 899

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 951

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 967

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 982

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1087

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1147

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 56

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 218

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 286

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 357

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 527

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 562

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 594

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 611

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 713

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 727

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 790

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 899

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 914

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 951

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 967

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 982

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1003

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1033

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1071

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1087

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1125

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1147

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1202

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1235

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1377

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1405

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1414

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1419

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 54

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 89

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 291

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 295

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 495

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 502

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 803

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 818

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 899

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1082

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1344

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 54

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 68

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 89

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 140

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 202

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 235

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 291

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 295

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 476

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 495

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 502

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 655

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 705

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 759

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 803

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 818

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 918

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 924

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 947

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 969

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1009

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1019

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1061

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1082

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1088

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1129

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1139

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1209

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1344

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1393

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1403

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 67

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 75

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 296

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 371

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 447

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 601

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 960

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1181

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1257

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1298

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 17

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 67

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 75

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 94

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 168

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 222

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 246

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 254

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 269

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 296

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 371

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 391

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 447

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 458

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 500

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 549

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 559

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 601

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 633

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 766

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 960

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 991

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1002

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1122

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1133

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1144

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1181

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1257

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1298

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1396

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1415

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 92

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 324

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 420

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 465

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 506

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 510

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 516

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 535

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 681

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 689

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 839

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 864

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1249

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 22

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 92

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 257

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 324

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 332

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 420

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 427

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 465

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 498

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 506

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 510

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 516

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 535

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 681

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 689

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 733

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 780

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 836

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 839

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 840

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 851

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 864

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 896

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 906

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 952

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 970

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1072

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1157

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1170

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1249

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1305

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1327

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1391

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1398

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1433

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 73

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 80

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 292

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 323

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 390

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 725

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 776

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 885

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 979

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1077

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1147

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1180

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1321

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 73

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 80

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 292

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 323

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 365

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 390

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 395

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 423

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 541

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 570

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 590

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 712

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 725

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 776

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 819

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 851

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 885

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 945

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 979

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1010

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1022

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1072

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1077

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1080

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1100

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1111

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1147

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1180

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1193

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1224

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1321

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 272

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 318

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 477

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 816

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 852

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 949

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1143

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1346

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 39

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 75

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 110

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 166

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 245

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 272

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 279

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 283

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 318

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 331

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 393

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 411

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 442

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 477

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 501

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 674

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 738

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 816

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 852

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 887

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 918

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 949

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1047

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1063

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1076

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1129

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1346

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1400

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1413

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 183

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 223

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 541

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 624

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 640

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 721

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 899

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1001

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1068

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1084

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1125

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1269

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 62

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 92

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 137

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 140

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 183

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 192

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 196

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 223

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 300

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 479

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 541

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 576

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 616

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 624

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 640

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 653

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 721

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 734

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 866

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 899

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 924

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1001

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1006

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1068

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1084

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1125

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1229

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1269

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 28

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 92

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 173

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 213

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 302

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 319

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 368

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 379

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 553

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 642

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 702

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 826

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1160

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1251

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1406

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 28

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 37

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 68

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 92

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 173

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 184

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 213

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 300

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 302

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 319

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 368

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 379

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 461

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 512

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 518

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 544

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 551

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 553

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 642

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 670

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 702

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 747

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 757

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 826

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 868

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 913

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1076

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1081

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1087

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1089

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1160

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1223

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1227

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1251

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1332

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1374

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1406

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 211

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 515

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 526

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 553

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 839

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 961

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1310

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1344

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 149

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 211

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 271

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 354

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 449

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 505

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 515

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 526

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 553

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 568

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 635

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 709

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 800

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 839

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 899

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 961

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1020

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1191

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1220

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1278

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1310

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1344

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1381

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1401

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1419

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 85

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 342

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 488

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 513

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 535

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 803

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 839

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 934

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 962

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1065

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1364

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1372

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1387

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1407

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1410

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 68

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 86

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 196

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 260

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 340

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 341

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 342

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 366

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 367

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 399

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 488

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 513

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 535

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 674

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 803

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 839

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 934

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 962

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 975

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 988

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1065

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1344

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1364

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1372

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1387

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1407

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1410

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 70

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 283

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 324

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 704

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 747

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1175

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1180

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1222

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1261

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 70

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 92

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 152

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 242

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 251

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 283

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 320

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 324

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 495

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 531

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 543

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 704

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 747

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 769

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 790

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 822

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 882

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 921

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1007

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1032

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1067

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1099

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1149

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1175

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1180

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1214

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1222

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1245

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1261

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1369

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1399

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1425

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1431

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 114

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 271

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 359

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 495

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 595

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 609

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 756

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 854

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1061

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1099

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1338

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1432

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 64

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 79

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 114

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 271

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 290

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 359

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 383

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 393

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 419

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 421

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 430

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 436

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 457

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 495

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 499

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 559

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 596

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 609

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 732

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 746

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 756

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 764

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 854

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 919

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 940

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 958

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 973

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 979

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1013

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1019

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1061

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1076

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1099

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1131

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1338

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1372

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1396

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1432

``` r
# NB
pulsar_nb_up_predictions <- predict(fit.pulsar.nb_up,test_features)
```

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 71

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 152

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 154

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 185

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 543

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 544

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 725

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1364

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1374

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1467

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1492

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1583

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1604

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1717

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1892

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1901

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1946

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2002

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2124

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2148

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2346

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2353

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2520

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2570

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2618

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2971

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 3242

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 3294

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 3470

``` r
confusionMatrix(pulsar_nb_up_predictions, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3109   39
    ##          1  149  282
    ##                                           
    ##                Accuracy : 0.9475          
    ##                  95% CI : (0.9396, 0.9546)
    ##     No Information Rate : 0.9103          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.7214          
    ##                                           
    ##  Mcnemar's Test P-Value : 1.871e-15       
    ##                                           
    ##             Sensitivity : 0.87850         
    ##             Specificity : 0.95427         
    ##          Pos Pred Value : 0.65429         
    ##          Neg Pred Value : 0.98761         
    ##              Prevalence : 0.08969         
    ##          Detection Rate : 0.07879         
    ##    Detection Prevalence : 0.12042         
    ##       Balanced Accuracy : 0.91639         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
# NB predicted 282 / 321 (87.85%) class 1's accurately

#CART
control$sampling = "up"
set.seed(seed)
fit.pulsar.cart_up = train(as.factor(target_class)~., 
                           data = train_data, 
                           method = "rpart", 
                           preProc = c("center", "scale"), 
                           trControl = control, 
                           metric = metric)

# CART
pulsar_cart_up_predictions <- predict(fit.pulsar.cart_up,test_features)
confusionMatrix(pulsar_cart_up_predictions, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3203   49
    ##          1   55  272
    ##                                           
    ##                Accuracy : 0.9709          
    ##                  95% CI : (0.9649, 0.9762)
    ##     No Information Rate : 0.9103          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.8235          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.6239          
    ##                                           
    ##             Sensitivity : 0.84735         
    ##             Specificity : 0.98312         
    ##          Pos Pred Value : 0.83180         
    ##          Neg Pred Value : 0.98493         
    ##              Prevalence : 0.08969         
    ##          Detection Rate : 0.07600         
    ##    Detection Prevalence : 0.09137         
    ##       Balanced Accuracy : 0.91524         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
# CART predicted 272 / 321 (84.74%) class 1's accurately

#Bagged CART
control$sampling = "up"
set.seed(seed)
fit.pulsar.treebag_up = train(as.factor(target_class)~., 
                              data = train_data, 
                              method = "treebag", 
                              preProc = c("center", "scale"), 
                              trControl = control, 
                              metric = metric)

# Bagging
pulsar_bagging_up_predictions <- predict(fit.pulsar.treebag_up,test_features)
confusionMatrix(pulsar_bagging_up_predictions, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3223   47
    ##          1   35  274
    ##                                           
    ##                Accuracy : 0.9771          
    ##                  95% CI : (0.9716, 0.9817)
    ##     No Information Rate : 0.9103          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.8573          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.2245          
    ##                                           
    ##             Sensitivity : 0.85358         
    ##             Specificity : 0.98926         
    ##          Pos Pred Value : 0.88673         
    ##          Neg Pred Value : 0.98563         
    ##              Prevalence : 0.08969         
    ##          Detection Rate : 0.07656         
    ##    Detection Prevalence : 0.08634         
    ##       Balanced Accuracy : 0.92142         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
# Bagging predicted 274 / 321 (85.36%) class 1's accurately

#Random Forest
control$sampling = "up"
set.seed(seed)
fit.pulsar.rf_up = train(as.factor(target_class)~., 
                         data = train_data, 
                         method = "rf", 
                         preProc = c("center", "scale"), 
                         trControl = control, 
                         metric = metric)

# Random Forest Score
pulsar_rf_up_predictions <- predict(fit.pulsar.rf_up,test_features)
confusionMatrix(pulsar_rf_up_predictions, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3229   46
    ##          1   29  275
    ##                                           
    ##                Accuracy : 0.979           
    ##                  95% CI : (0.9738, 0.9835)
    ##     No Information Rate : 0.9103          
    ##     P-Value [Acc > NIR] : < 2e-16         
    ##                                           
    ##                   Kappa : 0.8685          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.06467         
    ##                                           
    ##             Sensitivity : 0.85670         
    ##             Specificity : 0.99110         
    ##          Pos Pred Value : 0.90461         
    ##          Neg Pred Value : 0.98595         
    ##              Prevalence : 0.08969         
    ##          Detection Rate : 0.07684         
    ##    Detection Prevalence : 0.08494         
    ##       Balanced Accuracy : 0.92390         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
# RF predicted 275 / 321 (85.67%) class 1's accurately

#Stochastic Gradient Boosting
control$sampling = "up"
set.seed(seed)
fit.pulsar.sgb_up = train(as.factor(target_class)~., 
                          data = train_data, 
                          method = "gbm", 
                          preProc = c("center", "scale"), 
                          trControl = control, 
                          metric = metric)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2460            -nan     0.1000    0.0699
    ##      2        1.1318            -nan     0.1000    0.0570
    ##      3        1.0369            -nan     0.1000    0.0473
    ##      4        0.9575            -nan     0.1000    0.0398
    ##      5        0.8904            -nan     0.1000    0.0335
    ##      6        0.8325            -nan     0.1000    0.0289
    ##      7        0.7835            -nan     0.1000    0.0245
    ##      8        0.7412            -nan     0.1000    0.0210
    ##      9        0.7047            -nan     0.1000    0.0182
    ##     10        0.6719            -nan     0.1000    0.0162
    ##     20        0.4904            -nan     0.1000    0.0058
    ##     40        0.3848            -nan     0.1000    0.0012
    ##     60        0.3538            -nan     0.1000    0.0002
    ##     80        0.3445            -nan     0.1000    0.0001
    ##    100        0.3381            -nan     0.1000    0.0003
    ##    120        0.3322            -nan     0.1000    0.0000
    ##    140        0.3284            -nan     0.1000    0.0000
    ##    150        0.3263            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2410            -nan     0.1000    0.0720
    ##      2        1.1231            -nan     0.1000    0.0591
    ##      3        1.0252            -nan     0.1000    0.0489
    ##      4        0.9423            -nan     0.1000    0.0411
    ##      5        0.8712            -nan     0.1000    0.0356
    ##      6        0.8103            -nan     0.1000    0.0302
    ##      7        0.7588            -nan     0.1000    0.0254
    ##      8        0.7132            -nan     0.1000    0.0225
    ##      9        0.6729            -nan     0.1000    0.0197
    ##     10        0.6391            -nan     0.1000    0.0169
    ##     20        0.4401            -nan     0.1000    0.0066
    ##     40        0.3446            -nan     0.1000    0.0008
    ##     60        0.3165            -nan     0.1000    0.0003
    ##     80        0.3024            -nan     0.1000    0.0003
    ##    100        0.2916            -nan     0.1000    0.0001
    ##    120        0.2822            -nan     0.1000    0.0000
    ##    140        0.2733            -nan     0.1000    0.0002
    ##    150        0.2683            -nan     0.1000    0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2366            -nan     0.1000    0.0746
    ##      2        1.1143            -nan     0.1000    0.0611
    ##      3        1.0124            -nan     0.1000    0.0508
    ##      4        0.9270            -nan     0.1000    0.0431
    ##      5        0.8544            -nan     0.1000    0.0365
    ##      6        0.7905            -nan     0.1000    0.0318
    ##      7        0.7368            -nan     0.1000    0.0266
    ##      8        0.6908            -nan     0.1000    0.0230
    ##      9        0.6491            -nan     0.1000    0.0208
    ##     10        0.6127            -nan     0.1000    0.0182
    ##     20        0.4209            -nan     0.1000    0.0057
    ##     40        0.3301            -nan     0.1000    0.0011
    ##     60        0.3013            -nan     0.1000    0.0003
    ##     80        0.2839            -nan     0.1000    0.0000
    ##    100        0.2690            -nan     0.1000    0.0002
    ##    120        0.2578            -nan     0.1000    0.0003
    ##    140        0.2477            -nan     0.1000    0.0002
    ##    150        0.2437            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2498            -nan     0.1000    0.0688
    ##      2        1.1373            -nan     0.1000    0.0561
    ##      3        1.0428            -nan     0.1000    0.0468
    ##      4        0.9649            -nan     0.1000    0.0391
    ##      5        0.8991            -nan     0.1000    0.0330
    ##      6        0.8426            -nan     0.1000    0.0283
    ##      7        0.7939            -nan     0.1000    0.0241
    ##      8        0.7520            -nan     0.1000    0.0211
    ##      9        0.7155            -nan     0.1000    0.0180
    ##     10        0.6837            -nan     0.1000    0.0159
    ##     20        0.5040            -nan     0.1000    0.0052
    ##     40        0.3956            -nan     0.1000    0.0012
    ##     60        0.3654            -nan     0.1000    0.0005
    ##     80        0.3551            -nan     0.1000    0.0005
    ##    100        0.3488            -nan     0.1000    0.0001
    ##    120        0.3417            -nan     0.1000    0.0002
    ##    140        0.3374            -nan     0.1000    0.0001
    ##    150        0.3358            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2430            -nan     0.1000    0.0715
    ##      2        1.1263            -nan     0.1000    0.0582
    ##      3        1.0290            -nan     0.1000    0.0488
    ##      4        0.9467            -nan     0.1000    0.0410
    ##      5        0.8773            -nan     0.1000    0.0346
    ##      6        0.8171            -nan     0.1000    0.0298
    ##      7        0.7659            -nan     0.1000    0.0256
    ##      8        0.7196            -nan     0.1000    0.0231
    ##      9        0.6801            -nan     0.1000    0.0196
    ##     10        0.6454            -nan     0.1000    0.0173
    ##     20        0.4459            -nan     0.1000    0.0067
    ##     40        0.3453            -nan     0.1000    0.0006
    ##     60        0.3198            -nan     0.1000    0.0004
    ##     80        0.3061            -nan     0.1000    0.0002
    ##    100        0.2949            -nan     0.1000    0.0002
    ##    120        0.2842            -nan     0.1000    0.0003
    ##    140        0.2733            -nan     0.1000    0.0002
    ##    150        0.2702            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2380            -nan     0.1000    0.0738
    ##      2        1.1166            -nan     0.1000    0.0608
    ##      3        1.0163            -nan     0.1000    0.0499
    ##      4        0.9297            -nan     0.1000    0.0431
    ##      5        0.8567            -nan     0.1000    0.0365
    ##      6        0.7942            -nan     0.1000    0.0313
    ##      7        0.7396            -nan     0.1000    0.0270
    ##      8        0.6919            -nan     0.1000    0.0239
    ##      9        0.6499            -nan     0.1000    0.0208
    ##     10        0.6152            -nan     0.1000    0.0173
    ##     20        0.4204            -nan     0.1000    0.0057
    ##     40        0.3292            -nan     0.1000    0.0006
    ##     60        0.2966            -nan     0.1000    0.0002
    ##     80        0.2754            -nan     0.1000    0.0001
    ##    100        0.2634            -nan     0.1000    0.0000
    ##    120        0.2523            -nan     0.1000    0.0002
    ##    140        0.2408            -nan     0.1000    0.0002
    ##    150        0.2357            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2462            -nan     0.1000    0.0698
    ##      2        1.1309            -nan     0.1000    0.0574
    ##      3        1.0355            -nan     0.1000    0.0477
    ##      4        0.9554            -nan     0.1000    0.0398
    ##      5        0.8877            -nan     0.1000    0.0337
    ##      6        0.8308            -nan     0.1000    0.0286
    ##      7        0.7815            -nan     0.1000    0.0248
    ##      8        0.7388            -nan     0.1000    0.0213
    ##      9        0.7023            -nan     0.1000    0.0183
    ##     10        0.6696            -nan     0.1000    0.0161
    ##     20        0.4880            -nan     0.1000    0.0055
    ##     40        0.3800            -nan     0.1000    0.0010
    ##     60        0.3500            -nan     0.1000    0.0002
    ##     80        0.3385            -nan     0.1000    0.0002
    ##    100        0.3306            -nan     0.1000    0.0001
    ##    120        0.3245            -nan     0.1000    0.0000
    ##    140        0.3209            -nan     0.1000    0.0000
    ##    150        0.3193            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2422            -nan     0.1000    0.0722
    ##      2        1.1240            -nan     0.1000    0.0591
    ##      3        1.0253            -nan     0.1000    0.0494
    ##      4        0.9417            -nan     0.1000    0.0415
    ##      5        0.8714            -nan     0.1000    0.0354
    ##      6        0.8110            -nan     0.1000    0.0299
    ##      7        0.7590            -nan     0.1000    0.0257
    ##      8        0.7133            -nan     0.1000    0.0228
    ##      9        0.6747            -nan     0.1000    0.0194
    ##     10        0.6399            -nan     0.1000    0.0173
    ##     20        0.4402            -nan     0.1000    0.0053
    ##     40        0.3409            -nan     0.1000    0.0008
    ##     60        0.3125            -nan     0.1000    0.0003
    ##     80        0.2976            -nan     0.1000   -0.0000
    ##    100        0.2878            -nan     0.1000    0.0003
    ##    120        0.2788            -nan     0.1000   -0.0000
    ##    140        0.2703            -nan     0.1000    0.0002
    ##    150        0.2675            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2361            -nan     0.1000    0.0748
    ##      2        1.1131            -nan     0.1000    0.0616
    ##      3        1.0122            -nan     0.1000    0.0502
    ##      4        0.9248            -nan     0.1000    0.0436
    ##      5        0.8509            -nan     0.1000    0.0368
    ##      6        0.7870            -nan     0.1000    0.0318
    ##      7        0.7315            -nan     0.1000    0.0276
    ##      8        0.6854            -nan     0.1000    0.0228
    ##      9        0.6438            -nan     0.1000    0.0206
    ##     10        0.6062            -nan     0.1000    0.0186
    ##     20        0.4090            -nan     0.1000    0.0058
    ##     40        0.3151            -nan     0.1000    0.0007
    ##     60        0.2844            -nan     0.1000    0.0003
    ##     80        0.2687            -nan     0.1000    0.0002
    ##    100        0.2544            -nan     0.1000    0.0002
    ##    120        0.2428            -nan     0.1000    0.0003
    ##    140        0.2342            -nan     0.1000    0.0000
    ##    150        0.2305            -nan     0.1000    0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2433            -nan     0.1000    0.0708
    ##      2        1.1279            -nan     0.1000    0.0579
    ##      3        1.0317            -nan     0.1000    0.0480
    ##      4        0.9507            -nan     0.1000    0.0405
    ##      5        0.8824            -nan     0.1000    0.0341
    ##      6        0.8245            -nan     0.1000    0.0290
    ##      7        0.7745            -nan     0.1000    0.0248
    ##      8        0.7318            -nan     0.1000    0.0213
    ##      9        0.6942            -nan     0.1000    0.0190
    ##     10        0.6621            -nan     0.1000    0.0161
    ##     20        0.4786            -nan     0.1000    0.0055
    ##     40        0.3706            -nan     0.1000    0.0011
    ##     60        0.3409            -nan     0.1000    0.0003
    ##     80        0.3304            -nan     0.1000    0.0001
    ##    100        0.3253            -nan     0.1000    0.0001
    ##    120        0.3210            -nan     0.1000    0.0000
    ##    140        0.3166            -nan     0.1000    0.0001
    ##    150        0.3148            -nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2400            -nan     0.1000    0.0728
    ##      2        1.1218            -nan     0.1000    0.0591
    ##      3        1.0234            -nan     0.1000    0.0489
    ##      4        0.9396            -nan     0.1000    0.0420
    ##      5        0.8684            -nan     0.1000    0.0355
    ##      6        0.8079            -nan     0.1000    0.0300
    ##      7        0.7559            -nan     0.1000    0.0258
    ##      8        0.7094            -nan     0.1000    0.0227
    ##      9        0.6697            -nan     0.1000    0.0197
    ##     10        0.6337            -nan     0.1000    0.0180
    ##     20        0.4312            -nan     0.1000    0.0047
    ##     40        0.3330            -nan     0.1000    0.0010
    ##     60        0.3069            -nan     0.1000    0.0003
    ##     80        0.2944            -nan     0.1000    0.0001
    ##    100        0.2797            -nan     0.1000    0.0003
    ##    120        0.2702            -nan     0.1000    0.0002
    ##    140        0.2640            -nan     0.1000    0.0001
    ##    150        0.2604            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2348            -nan     0.1000    0.0754
    ##      2        1.1109            -nan     0.1000    0.0615
    ##      3        1.0082            -nan     0.1000    0.0512
    ##      4        0.9214            -nan     0.1000    0.0431
    ##      5        0.8486            -nan     0.1000    0.0364
    ##      6        0.7869            -nan     0.1000    0.0311
    ##      7        0.7328            -nan     0.1000    0.0268
    ##      8        0.6844            -nan     0.1000    0.0243
    ##      9        0.6424            -nan     0.1000    0.0206
    ##     10        0.6050            -nan     0.1000    0.0187
    ##     20        0.4093            -nan     0.1000    0.0046
    ##     40        0.3158            -nan     0.1000    0.0006
    ##     60        0.2891            -nan     0.1000    0.0005
    ##     80        0.2729            -nan     0.1000    0.0003
    ##    100        0.2589            -nan     0.1000    0.0002
    ##    120        0.2481            -nan     0.1000    0.0002
    ##    140        0.2378            -nan     0.1000    0.0002
    ##    150        0.2338            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2462            -nan     0.1000    0.0705
    ##      2        1.1313            -nan     0.1000    0.0579
    ##      3        1.0355            -nan     0.1000    0.0478
    ##      4        0.9543            -nan     0.1000    0.0403
    ##      5        0.8850            -nan     0.1000    0.0340
    ##      6        0.8271            -nan     0.1000    0.0291
    ##      7        0.7772            -nan     0.1000    0.0246
    ##      8        0.7339            -nan     0.1000    0.0214
    ##      9        0.6970            -nan     0.1000    0.0185
    ##     10        0.6637            -nan     0.1000    0.0167
    ##     20        0.4832            -nan     0.1000    0.0052
    ##     40        0.3793            -nan     0.1000    0.0017
    ##     60        0.3512            -nan     0.1000    0.0002
    ##     80        0.3413            -nan     0.1000    0.0001
    ##    100        0.3342            -nan     0.1000    0.0001
    ##    120        0.3279            -nan     0.1000    0.0001
    ##    140        0.3233            -nan     0.1000    0.0001
    ##    150        0.3213            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2414            -nan     0.1000    0.0722
    ##      2        1.1217            -nan     0.1000    0.0601
    ##      3        1.0216            -nan     0.1000    0.0499
    ##      4        0.9383            -nan     0.1000    0.0418
    ##      5        0.8671            -nan     0.1000    0.0356
    ##      6        0.8063            -nan     0.1000    0.0302
    ##      7        0.7529            -nan     0.1000    0.0266
    ##      8        0.7079            -nan     0.1000    0.0225
    ##      9        0.6683            -nan     0.1000    0.0197
    ##     10        0.6339            -nan     0.1000    0.0170
    ##     20        0.4327            -nan     0.1000    0.0049
    ##     40        0.3292            -nan     0.1000    0.0007
    ##     60        0.3032            -nan     0.1000    0.0007
    ##     80        0.2876            -nan     0.1000    0.0003
    ##    100        0.2762            -nan     0.1000    0.0002
    ##    120        0.2630            -nan     0.1000    0.0002
    ##    140        0.2540            -nan     0.1000    0.0000
    ##    150        0.2500            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2339            -nan     0.1000    0.0759
    ##      2        1.1125            -nan     0.1000    0.0607
    ##      3        1.0084            -nan     0.1000    0.0522
    ##      4        0.9204            -nan     0.1000    0.0438
    ##      5        0.8463            -nan     0.1000    0.0367
    ##      6        0.7835            -nan     0.1000    0.0313
    ##      7        0.7299            -nan     0.1000    0.0267
    ##      8        0.6832            -nan     0.1000    0.0231
    ##      9        0.6411            -nan     0.1000    0.0209
    ##     10        0.6031            -nan     0.1000    0.0187
    ##     20        0.4046            -nan     0.1000    0.0060
    ##     40        0.3116            -nan     0.1000    0.0005
    ##     60        0.2805            -nan     0.1000    0.0004
    ##     80        0.2606            -nan     0.1000    0.0002
    ##    100        0.2468            -nan     0.1000    0.0003
    ##    120        0.2362            -nan     0.1000    0.0001
    ##    140        0.2265            -nan     0.1000    0.0001
    ##    150        0.2207            -nan     0.1000    0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2444            -nan     0.1000    0.0706
    ##      2        1.1274            -nan     0.1000    0.0582
    ##      3        1.0304            -nan     0.1000    0.0482
    ##      4        0.9497            -nan     0.1000    0.0402
    ##      5        0.8804            -nan     0.1000    0.0345
    ##      6        0.8223            -nan     0.1000    0.0292
    ##      7        0.7719            -nan     0.1000    0.0252
    ##      8        0.7286            -nan     0.1000    0.0216
    ##      9        0.6908            -nan     0.1000    0.0188
    ##     10        0.6585            -nan     0.1000    0.0163
    ##     20        0.4737            -nan     0.1000    0.0053
    ##     40        0.3617            -nan     0.1000    0.0011
    ##     60        0.3299            -nan     0.1000    0.0006
    ##     80        0.3199            -nan     0.1000    0.0002
    ##    100        0.3138            -nan     0.1000   -0.0000
    ##    120        0.3087            -nan     0.1000    0.0002
    ##    140        0.3039            -nan     0.1000    0.0001
    ##    150        0.3015            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2407            -nan     0.1000    0.0728
    ##      2        1.1208            -nan     0.1000    0.0599
    ##      3        1.0219            -nan     0.1000    0.0498
    ##      4        0.9374            -nan     0.1000    0.0420
    ##      5        0.8666            -nan     0.1000    0.0353
    ##      6        0.8057            -nan     0.1000    0.0303
    ##      7        0.7525            -nan     0.1000    0.0265
    ##      8        0.7061            -nan     0.1000    0.0233
    ##      9        0.6670            -nan     0.1000    0.0195
    ##     10        0.6334            -nan     0.1000    0.0167
    ##     20        0.4289            -nan     0.1000    0.0069
    ##     40        0.3288            -nan     0.1000    0.0007
    ##     60        0.3029            -nan     0.1000    0.0002
    ##     80        0.2868            -nan     0.1000    0.0002
    ##    100        0.2770            -nan     0.1000    0.0001
    ##    120        0.2680            -nan     0.1000    0.0001
    ##    140        0.2618            -nan     0.1000    0.0002
    ##    150        0.2519            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2359            -nan     0.1000    0.0747
    ##      2        1.1115            -nan     0.1000    0.0623
    ##      3        1.0071            -nan     0.1000    0.0523
    ##      4        0.9193            -nan     0.1000    0.0440
    ##      5        0.8460            -nan     0.1000    0.0367
    ##      6        0.7821            -nan     0.1000    0.0320
    ##      7        0.7276            -nan     0.1000    0.0272
    ##      8        0.6789            -nan     0.1000    0.0246
    ##      9        0.6368            -nan     0.1000    0.0211
    ##     10        0.5997            -nan     0.1000    0.0185
    ##     20        0.3975            -nan     0.1000    0.0059
    ##     40        0.3033            -nan     0.1000    0.0011
    ##     60        0.2697            -nan     0.1000    0.0003
    ##     80        0.2510            -nan     0.1000    0.0002
    ##    100        0.2364            -nan     0.1000    0.0001
    ##    120        0.2250            -nan     0.1000    0.0002
    ##    140        0.2151            -nan     0.1000    0.0001
    ##    150        0.2117            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2463            -nan     0.1000    0.0696
    ##      2        1.1315            -nan     0.1000    0.0573
    ##      3        1.0366            -nan     0.1000    0.0474
    ##      4        0.9566            -nan     0.1000    0.0398
    ##      5        0.8887            -nan     0.1000    0.0336
    ##      6        0.8312            -nan     0.1000    0.0286
    ##      7        0.7827            -nan     0.1000    0.0241
    ##      8        0.7393            -nan     0.1000    0.0216
    ##      9        0.7025            -nan     0.1000    0.0186
    ##     10        0.6712            -nan     0.1000    0.0154
    ##     20        0.4838            -nan     0.1000    0.0059
    ##     40        0.3754            -nan     0.1000    0.0011
    ##     60        0.3445            -nan     0.1000    0.0006
    ##     80        0.3337            -nan     0.1000    0.0003
    ##    100        0.3269            -nan     0.1000    0.0001
    ##    120        0.3206            -nan     0.1000    0.0001
    ##    140        0.3169            -nan     0.1000    0.0001
    ##    150        0.3153            -nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2423            -nan     0.1000    0.0720
    ##      2        1.1254            -nan     0.1000    0.0583
    ##      3        1.0276            -nan     0.1000    0.0484
    ##      4        0.9442            -nan     0.1000    0.0417
    ##      5        0.8750            -nan     0.1000    0.0347
    ##      6        0.8146            -nan     0.1000    0.0302
    ##      7        0.7619            -nan     0.1000    0.0263
    ##      8        0.7174            -nan     0.1000    0.0221
    ##      9        0.6771            -nan     0.1000    0.0201
    ##     10        0.6426            -nan     0.1000    0.0171
    ##     20        0.4375            -nan     0.1000    0.0064
    ##     40        0.3400            -nan     0.1000    0.0009
    ##     60        0.3164            -nan     0.1000    0.0001
    ##     80        0.2998            -nan     0.1000    0.0002
    ##    100        0.2883            -nan     0.1000    0.0002
    ##    120        0.2774            -nan     0.1000    0.0002
    ##    140        0.2702            -nan     0.1000    0.0001
    ##    150        0.2672            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2341            -nan     0.1000    0.0759
    ##      2        1.1100            -nan     0.1000    0.0619
    ##      3        1.0082            -nan     0.1000    0.0506
    ##      4        0.9208            -nan     0.1000    0.0436
    ##      5        0.8483            -nan     0.1000    0.0362
    ##      6        0.7867            -nan     0.1000    0.0310
    ##      7        0.7313            -nan     0.1000    0.0276
    ##      8        0.6827            -nan     0.1000    0.0240
    ##      9        0.6409            -nan     0.1000    0.0209
    ##     10        0.6052            -nan     0.1000    0.0178
    ##     20        0.4059            -nan     0.1000    0.0058
    ##     40        0.3130            -nan     0.1000    0.0008
    ##     60        0.2848            -nan     0.1000    0.0005
    ##     80        0.2661            -nan     0.1000    0.0001
    ##    100        0.2514            -nan     0.1000    0.0002
    ##    120        0.2358            -nan     0.1000    0.0005
    ##    140        0.2257            -nan     0.1000   -0.0000
    ##    150        0.2223            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2483            -nan     0.1000    0.0692
    ##      2        1.1352            -nan     0.1000    0.0568
    ##      3        1.0410            -nan     0.1000    0.0470
    ##      4        0.9621            -nan     0.1000    0.0396
    ##      5        0.8950            -nan     0.1000    0.0333
    ##      6        0.8373            -nan     0.1000    0.0286
    ##      7        0.7881            -nan     0.1000    0.0246
    ##      8        0.7459            -nan     0.1000    0.0210
    ##      9        0.7087            -nan     0.1000    0.0186
    ##     10        0.6768            -nan     0.1000    0.0159
    ##     20        0.4929            -nan     0.1000    0.0058
    ##     40        0.3819            -nan     0.1000    0.0010
    ##     60        0.3501            -nan     0.1000    0.0006
    ##     80        0.3404            -nan     0.1000    0.0001
    ##    100        0.3319            -nan     0.1000    0.0002
    ##    120        0.3285            -nan     0.1000    0.0001
    ##    140        0.3244            -nan     0.1000    0.0000
    ##    150        0.3225            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2448            -nan     0.1000    0.0705
    ##      2        1.1291            -nan     0.1000    0.0576
    ##      3        1.0323            -nan     0.1000    0.0483
    ##      4        0.9511            -nan     0.1000    0.0406
    ##      5        0.8817            -nan     0.1000    0.0344
    ##      6        0.8230            -nan     0.1000    0.0292
    ##      7        0.7718            -nan     0.1000    0.0258
    ##      8        0.7279            -nan     0.1000    0.0218
    ##      9        0.6891            -nan     0.1000    0.0194
    ##     10        0.6559            -nan     0.1000    0.0164
    ##     20        0.4529            -nan     0.1000    0.0054
    ##     40        0.3543            -nan     0.1000    0.0008
    ##     60        0.3295            -nan     0.1000    0.0002
    ##     80        0.3157            -nan     0.1000    0.0004
    ##    100        0.3041            -nan     0.1000    0.0002
    ##    120        0.2951            -nan     0.1000    0.0002
    ##    140        0.2832            -nan     0.1000    0.0002
    ##    150        0.2804            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2393            -nan     0.1000    0.0734
    ##      2        1.1155            -nan     0.1000    0.0617
    ##      3        1.0129            -nan     0.1000    0.0511
    ##      4        0.9260            -nan     0.1000    0.0433
    ##      5        0.8520            -nan     0.1000    0.0367
    ##      6        0.7890            -nan     0.1000    0.0315
    ##      7        0.7337            -nan     0.1000    0.0276
    ##      8        0.6875            -nan     0.1000    0.0231
    ##      9        0.6451            -nan     0.1000    0.0212
    ##     10        0.6105            -nan     0.1000    0.0172
    ##     20        0.4137            -nan     0.1000    0.0050
    ##     40        0.3210            -nan     0.1000    0.0005
    ##     60        0.2938            -nan     0.1000    0.0003
    ##     80        0.2769            -nan     0.1000    0.0004
    ##    100        0.2652            -nan     0.1000    0.0001
    ##    120        0.2536            -nan     0.1000    0.0001
    ##    140        0.2450            -nan     0.1000    0.0001
    ##    150        0.2408            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2461            -nan     0.1000    0.0701
    ##      2        1.1309            -nan     0.1000    0.0573
    ##      3        1.0355            -nan     0.1000    0.0475
    ##      4        0.9556            -nan     0.1000    0.0401
    ##      5        0.8882            -nan     0.1000    0.0336
    ##      6        0.8298            -nan     0.1000    0.0292
    ##      7        0.7799            -nan     0.1000    0.0246
    ##      8        0.7370            -nan     0.1000    0.0215
    ##      9        0.7008            -nan     0.1000    0.0179
    ##     10        0.6682            -nan     0.1000    0.0165
    ##     20        0.4860            -nan     0.1000    0.0054
    ##     40        0.3790            -nan     0.1000    0.0013
    ##     60        0.3482            -nan     0.1000    0.0002
    ##     80        0.3380            -nan     0.1000    0.0002
    ##    100        0.3322            -nan     0.1000    0.0002
    ##    120        0.3267            -nan     0.1000    0.0000
    ##    140        0.3223            -nan     0.1000    0.0000
    ##    150        0.3205            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2400            -nan     0.1000    0.0724
    ##      2        1.1226            -nan     0.1000    0.0589
    ##      3        1.0238            -nan     0.1000    0.0495
    ##      4        0.9402            -nan     0.1000    0.0415
    ##      5        0.8695            -nan     0.1000    0.0354
    ##      6        0.8084            -nan     0.1000    0.0303
    ##      7        0.7555            -nan     0.1000    0.0265
    ##      8        0.7089            -nan     0.1000    0.0232
    ##      9        0.6699            -nan     0.1000    0.0194
    ##     10        0.6344            -nan     0.1000    0.0177
    ##     20        0.4280            -nan     0.1000    0.0065
    ##     40        0.3284            -nan     0.1000    0.0010
    ##     60        0.3039            -nan     0.1000    0.0004
    ##     80        0.2893            -nan     0.1000    0.0007
    ##    100        0.2752            -nan     0.1000    0.0001
    ##    120        0.2633            -nan     0.1000    0.0001
    ##    140        0.2556            -nan     0.1000    0.0001
    ##    150        0.2517            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2333            -nan     0.1000    0.0762
    ##      2        1.1105            -nan     0.1000    0.0611
    ##      3        1.0075            -nan     0.1000    0.0514
    ##      4        0.9198            -nan     0.1000    0.0438
    ##      5        0.8470            -nan     0.1000    0.0362
    ##      6        0.7835            -nan     0.1000    0.0318
    ##      7        0.7295            -nan     0.1000    0.0268
    ##      8        0.6804            -nan     0.1000    0.0244
    ##      9        0.6372            -nan     0.1000    0.0217
    ##     10        0.6016            -nan     0.1000    0.0178
    ##     20        0.4013            -nan     0.1000    0.0056
    ##     40        0.3115            -nan     0.1000    0.0005
    ##     60        0.2833            -nan     0.1000    0.0003
    ##     80        0.2672            -nan     0.1000    0.0001
    ##    100        0.2495            -nan     0.1000    0.0002
    ##    120        0.2390            -nan     0.1000    0.0002
    ##    140        0.2279            -nan     0.1000    0.0001
    ##    150        0.2229            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2434            -nan     0.1000    0.0711
    ##      2        1.1253            -nan     0.1000    0.0585
    ##      3        1.0282            -nan     0.1000    0.0483
    ##      4        0.9461            -nan     0.1000    0.0409
    ##      5        0.8764            -nan     0.1000    0.0344
    ##      6        0.8177            -nan     0.1000    0.0292
    ##      7        0.7674            -nan     0.1000    0.0253
    ##      8        0.7243            -nan     0.1000    0.0217
    ##      9        0.6870            -nan     0.1000    0.0188
    ##     10        0.6536            -nan     0.1000    0.0166
    ##     20        0.4694            -nan     0.1000    0.0055
    ##     40        0.3627            -nan     0.1000    0.0014
    ##     60        0.3352            -nan     0.1000    0.0005
    ##     80        0.3245            -nan     0.1000    0.0001
    ##    100        0.3185            -nan     0.1000    0.0001
    ##    120        0.3122            -nan     0.1000    0.0001
    ##    140        0.3083            -nan     0.1000    0.0001
    ##    150        0.3065            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2411            -nan     0.1000    0.0725
    ##      2        1.1229            -nan     0.1000    0.0591
    ##      3        1.0242            -nan     0.1000    0.0491
    ##      4        0.9402            -nan     0.1000    0.0419
    ##      5        0.8703            -nan     0.1000    0.0347
    ##      6        0.8095            -nan     0.1000    0.0304
    ##      7        0.7574            -nan     0.1000    0.0258
    ##      8        0.7126            -nan     0.1000    0.0223
    ##      9        0.6728            -nan     0.1000    0.0200
    ##     10        0.6374            -nan     0.1000    0.0179
    ##     20        0.4366            -nan     0.1000    0.0061
    ##     40        0.3389            -nan     0.1000    0.0011
    ##     60        0.3121            -nan     0.1000    0.0002
    ##     80        0.2969            -nan     0.1000    0.0002
    ##    100        0.2862            -nan     0.1000    0.0001
    ##    120        0.2769            -nan     0.1000    0.0001
    ##    140        0.2678            -nan     0.1000    0.0002
    ##    150        0.2645            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2339            -nan     0.1000    0.0760
    ##      2        1.1099            -nan     0.1000    0.0617
    ##      3        1.0078            -nan     0.1000    0.0506
    ##      4        0.9222            -nan     0.1000    0.0427
    ##      5        0.8494            -nan     0.1000    0.0359
    ##      6        0.7844            -nan     0.1000    0.0323
    ##      7        0.7292            -nan     0.1000    0.0276
    ##      8        0.6829            -nan     0.1000    0.0231
    ##      9        0.6409            -nan     0.1000    0.0209
    ##     10        0.6030            -nan     0.1000    0.0190
    ##     20        0.4056            -nan     0.1000    0.0058
    ##     40        0.3130            -nan     0.1000    0.0003
    ##     60        0.2827            -nan     0.1000    0.0005
    ##     80        0.2670            -nan     0.1000    0.0003
    ##    100        0.2550            -nan     0.1000    0.0002
    ##    120        0.2455            -nan     0.1000    0.0002
    ##    140        0.2306            -nan     0.1000    0.0001
    ##    150        0.2244            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2471            -nan     0.1000    0.0696
    ##      2        1.1323            -nan     0.1000    0.0569
    ##      3        1.0372            -nan     0.1000    0.0472
    ##      4        0.9585            -nan     0.1000    0.0396
    ##      5        0.8914            -nan     0.1000    0.0337
    ##      6        0.8338            -nan     0.1000    0.0286
    ##      7        0.7845            -nan     0.1000    0.0245
    ##      8        0.7424            -nan     0.1000    0.0210
    ##      9        0.7055            -nan     0.1000    0.0183
    ##     10        0.6730            -nan     0.1000    0.0161
    ##     20        0.4876            -nan     0.1000    0.0053
    ##     40        0.3759            -nan     0.1000    0.0013
    ##     60        0.3465            -nan     0.1000    0.0001
    ##     80        0.3349            -nan     0.1000    0.0002
    ##    100        0.3283            -nan     0.1000    0.0001
    ##    120        0.3213            -nan     0.1000    0.0001
    ##    140        0.3170            -nan     0.1000    0.0000
    ##    150        0.3148            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2419            -nan     0.1000    0.0726
    ##      2        1.1238            -nan     0.1000    0.0589
    ##      3        1.0254            -nan     0.1000    0.0492
    ##      4        0.9416            -nan     0.1000    0.0419
    ##      5        0.8709            -nan     0.1000    0.0353
    ##      6        0.8086            -nan     0.1000    0.0308
    ##      7        0.7564            -nan     0.1000    0.0259
    ##      8        0.7105            -nan     0.1000    0.0230
    ##      9        0.6712            -nan     0.1000    0.0197
    ##     10        0.6367            -nan     0.1000    0.0171
    ##     20        0.4275            -nan     0.1000    0.0065
    ##     40        0.3283            -nan     0.1000    0.0013
    ##     60        0.3030            -nan     0.1000    0.0003
    ##     80        0.2880            -nan     0.1000   -0.0000
    ##    100        0.2770            -nan     0.1000    0.0001
    ##    120        0.2698            -nan     0.1000    0.0000
    ##    140        0.2617            -nan     0.1000    0.0001
    ##    150        0.2575            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2378            -nan     0.1000    0.0736
    ##      2        1.1143            -nan     0.1000    0.0620
    ##      3        1.0113            -nan     0.1000    0.0513
    ##      4        0.9247            -nan     0.1000    0.0432
    ##      5        0.8525            -nan     0.1000    0.0360
    ##      6        0.7890            -nan     0.1000    0.0313
    ##      7        0.7360            -nan     0.1000    0.0264
    ##      8        0.6897            -nan     0.1000    0.0228
    ##      9        0.6475            -nan     0.1000    0.0211
    ##     10        0.6098            -nan     0.1000    0.0188
    ##     20        0.4161            -nan     0.1000    0.0047
    ##     40        0.3198            -nan     0.1000    0.0010
    ##     60        0.2930            -nan     0.1000    0.0009
    ##     80        0.2769            -nan     0.1000    0.0002
    ##    100        0.2634            -nan     0.1000    0.0003
    ##    120        0.2520            -nan     0.1000    0.0002
    ##    140        0.2422            -nan     0.1000    0.0001
    ##    150        0.2361            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2447            -nan     0.1000    0.0708
    ##      2        1.1292            -nan     0.1000    0.0578
    ##      3        1.0325            -nan     0.1000    0.0483
    ##      4        0.9519            -nan     0.1000    0.0406
    ##      5        0.8838            -nan     0.1000    0.0342
    ##      6        0.8240            -nan     0.1000    0.0294
    ##      7        0.7740            -nan     0.1000    0.0251
    ##      8        0.7310            -nan     0.1000    0.0214
    ##      9        0.6936            -nan     0.1000    0.0187
    ##     10        0.6604            -nan     0.1000    0.0166
    ##     20        0.4795            -nan     0.1000    0.0053
    ##     40        0.3742            -nan     0.1000    0.0016
    ##     60        0.3452            -nan     0.1000    0.0003
    ##     80        0.3347            -nan     0.1000    0.0001
    ##    100        0.3288            -nan     0.1000   -0.0000
    ##    120        0.3236            -nan     0.1000    0.0001
    ##    140        0.3205            -nan     0.1000    0.0000
    ##    150        0.3187            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2421            -nan     0.1000    0.0722
    ##      2        1.1235            -nan     0.1000    0.0593
    ##      3        1.0235            -nan     0.1000    0.0498
    ##      4        0.9388            -nan     0.1000    0.0421
    ##      5        0.8673            -nan     0.1000    0.0354
    ##      6        0.8055            -nan     0.1000    0.0305
    ##      7        0.7527            -nan     0.1000    0.0263
    ##      8        0.7079            -nan     0.1000    0.0225
    ##      9        0.6690            -nan     0.1000    0.0195
    ##     10        0.6328            -nan     0.1000    0.0179
    ##     20        0.4319            -nan     0.1000    0.0066
    ##     40        0.3319            -nan     0.1000    0.0005
    ##     60        0.3072            -nan     0.1000    0.0005
    ##     80        0.2926            -nan     0.1000    0.0004
    ##    100        0.2831            -nan     0.1000    0.0003
    ##    120        0.2751            -nan     0.1000    0.0000
    ##    140        0.2671            -nan     0.1000    0.0000
    ##    150        0.2598            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2368            -nan     0.1000    0.0742
    ##      2        1.1136            -nan     0.1000    0.0614
    ##      3        1.0134            -nan     0.1000    0.0501
    ##      4        0.9289            -nan     0.1000    0.0421
    ##      5        0.8548            -nan     0.1000    0.0367
    ##      6        0.7918            -nan     0.1000    0.0315
    ##      7        0.7383            -nan     0.1000    0.0267
    ##      8        0.6921            -nan     0.1000    0.0229
    ##      9        0.6490            -nan     0.1000    0.0214
    ##     10        0.6117            -nan     0.1000    0.0184
    ##     20        0.4185            -nan     0.1000    0.0048
    ##     40        0.3268            -nan     0.1000    0.0011
    ##     60        0.2939            -nan     0.1000    0.0004
    ##     80        0.2753            -nan     0.1000    0.0002
    ##    100        0.2610            -nan     0.1000    0.0002
    ##    120        0.2452            -nan     0.1000    0.0002
    ##    140        0.2384            -nan     0.1000    0.0001
    ##    150        0.2348            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2451            -nan     0.1000    0.0704
    ##      2        1.1295            -nan     0.1000    0.0575
    ##      3        1.0335            -nan     0.1000    0.0476
    ##      4        0.9538            -nan     0.1000    0.0401
    ##      5        0.8862            -nan     0.1000    0.0337
    ##      6        0.8282            -nan     0.1000    0.0289
    ##      7        0.7784            -nan     0.1000    0.0245
    ##      8        0.7364            -nan     0.1000    0.0208
    ##      9        0.6987            -nan     0.1000    0.0189
    ##     10        0.6658            -nan     0.1000    0.0163
    ##     20        0.4866            -nan     0.1000    0.0052
    ##     40        0.3784            -nan     0.1000    0.0016
    ##     60        0.3509            -nan     0.1000    0.0001
    ##     80        0.3397            -nan     0.1000    0.0001
    ##    100        0.3319            -nan     0.1000    0.0002
    ##    120        0.3267            -nan     0.1000    0.0001
    ##    140        0.3217            -nan     0.1000    0.0000
    ##    150        0.3196            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2416            -nan     0.1000    0.0725
    ##      2        1.1231            -nan     0.1000    0.0590
    ##      3        1.0247            -nan     0.1000    0.0491
    ##      4        0.9414            -nan     0.1000    0.0416
    ##      5        0.8704            -nan     0.1000    0.0353
    ##      6        0.8104            -nan     0.1000    0.0300
    ##      7        0.7591            -nan     0.1000    0.0257
    ##      8        0.7130            -nan     0.1000    0.0230
    ##      9        0.6735            -nan     0.1000    0.0197
    ##     10        0.6402            -nan     0.1000    0.0166
    ##     20        0.4377            -nan     0.1000    0.0069
    ##     40        0.3378            -nan     0.1000    0.0008
    ##     60        0.3115            -nan     0.1000    0.0004
    ##     80        0.2971            -nan     0.1000    0.0001
    ##    100        0.2856            -nan     0.1000    0.0001
    ##    120        0.2756            -nan     0.1000    0.0001
    ##    140        0.2665            -nan     0.1000    0.0001
    ##    150        0.2595            -nan     0.1000    0.0006
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2355            -nan     0.1000    0.0751
    ##      2        1.1128            -nan     0.1000    0.0612
    ##      3        1.0127            -nan     0.1000    0.0503
    ##      4        0.9245            -nan     0.1000    0.0436
    ##      5        0.8507            -nan     0.1000    0.0367
    ##      6        0.7888            -nan     0.1000    0.0312
    ##      7        0.7339            -nan     0.1000    0.0273
    ##      8        0.6884            -nan     0.1000    0.0225
    ##      9        0.6456            -nan     0.1000    0.0211
    ##     10        0.6106            -nan     0.1000    0.0172
    ##     20        0.4136            -nan     0.1000    0.0053
    ##     40        0.3232            -nan     0.1000    0.0006
    ##     60        0.2902            -nan     0.1000    0.0003
    ##     80        0.2735            -nan     0.1000    0.0002
    ##    100        0.2587            -nan     0.1000    0.0001
    ##    120        0.2452            -nan     0.1000    0.0004
    ##    140        0.2363            -nan     0.1000    0.0002
    ##    150        0.2311            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2462            -nan     0.1000    0.0698
    ##      2        1.1322            -nan     0.1000    0.0569
    ##      3        1.0371            -nan     0.1000    0.0476
    ##      4        0.9581            -nan     0.1000    0.0397
    ##      5        0.8911            -nan     0.1000    0.0337
    ##      6        0.8328            -nan     0.1000    0.0289
    ##      7        0.7832            -nan     0.1000    0.0246
    ##      8        0.7406            -nan     0.1000    0.0213
    ##      9        0.7039            -nan     0.1000    0.0183
    ##     10        0.6714            -nan     0.1000    0.0162
    ##     20        0.4892            -nan     0.1000    0.0051
    ##     40        0.3790            -nan     0.1000    0.0010
    ##     60        0.3465            -nan     0.1000    0.0005
    ##     80        0.3363            -nan     0.1000    0.0001
    ##    100        0.3290            -nan     0.1000    0.0000
    ##    120        0.3232            -nan     0.1000   -0.0000
    ##    140        0.3188            -nan     0.1000    0.0000
    ##    150        0.3164            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2415            -nan     0.1000    0.0728
    ##      2        1.1220            -nan     0.1000    0.0597
    ##      3        1.0227            -nan     0.1000    0.0495
    ##      4        0.9396            -nan     0.1000    0.0416
    ##      5        0.8687            -nan     0.1000    0.0355
    ##      6        0.8084            -nan     0.1000    0.0303
    ##      7        0.7570            -nan     0.1000    0.0257
    ##      8        0.7106            -nan     0.1000    0.0233
    ##      9        0.6694            -nan     0.1000    0.0206
    ##     10        0.6351            -nan     0.1000    0.0172
    ##     20        0.4311            -nan     0.1000    0.0066
    ##     40        0.3312            -nan     0.1000    0.0010
    ##     60        0.3056            -nan     0.1000    0.0001
    ##     80        0.2932            -nan     0.1000    0.0004
    ##    100        0.2822            -nan     0.1000    0.0002
    ##    120        0.2729            -nan     0.1000    0.0000
    ##    140        0.2651            -nan     0.1000    0.0003
    ##    150        0.2611            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2373            -nan     0.1000    0.0746
    ##      2        1.1159            -nan     0.1000    0.0605
    ##      3        1.0129            -nan     0.1000    0.0512
    ##      4        0.9276            -nan     0.1000    0.0426
    ##      5        0.8528            -nan     0.1000    0.0372
    ##      6        0.7895            -nan     0.1000    0.0312
    ##      7        0.7359            -nan     0.1000    0.0270
    ##      8        0.6874            -nan     0.1000    0.0240
    ##      9        0.6467            -nan     0.1000    0.0202
    ##     10        0.6100            -nan     0.1000    0.0182
    ##     20        0.4128            -nan     0.1000    0.0049
    ##     40        0.3212            -nan     0.1000    0.0008
    ##     60        0.2931            -nan     0.1000    0.0004
    ##     80        0.2739            -nan     0.1000    0.0004
    ##    100        0.2622            -nan     0.1000    0.0001
    ##    120        0.2499            -nan     0.1000    0.0001
    ##    140        0.2409            -nan     0.1000    0.0000
    ##    150        0.2355            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2486            -nan     0.1000    0.0689
    ##      2        1.1352            -nan     0.1000    0.0564
    ##      3        1.0410            -nan     0.1000    0.0468
    ##      4        0.9627            -nan     0.1000    0.0392
    ##      5        0.8960            -nan     0.1000    0.0331
    ##      6        0.8398            -nan     0.1000    0.0282
    ##      7        0.7917            -nan     0.1000    0.0241
    ##      8        0.7502            -nan     0.1000    0.0207
    ##      9        0.7127            -nan     0.1000    0.0186
    ##     10        0.6816            -nan     0.1000    0.0155
    ##     20        0.5009            -nan     0.1000    0.0052
    ##     40        0.3933            -nan     0.1000    0.0010
    ##     60        0.3634            -nan     0.1000    0.0003
    ##     80        0.3533            -nan     0.1000    0.0001
    ##    100        0.3470            -nan     0.1000    0.0000
    ##    120        0.3420            -nan     0.1000    0.0001
    ##    140        0.3382            -nan     0.1000    0.0001
    ##    150        0.3364            -nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2436            -nan     0.1000    0.0713
    ##      2        1.1262            -nan     0.1000    0.0587
    ##      3        1.0291            -nan     0.1000    0.0481
    ##      4        0.9463            -nan     0.1000    0.0411
    ##      5        0.8768            -nan     0.1000    0.0345
    ##      6        0.8175            -nan     0.1000    0.0295
    ##      7        0.7657            -nan     0.1000    0.0258
    ##      8        0.7205            -nan     0.1000    0.0222
    ##      9        0.6821            -nan     0.1000    0.0190
    ##     10        0.6484            -nan     0.1000    0.0166
    ##     20        0.4439            -nan     0.1000    0.0059
    ##     40        0.3419            -nan     0.1000    0.0005
    ##     60        0.3174            -nan     0.1000    0.0002
    ##     80        0.3026            -nan     0.1000    0.0002
    ##    100        0.2930            -nan     0.1000    0.0001
    ##    120        0.2834            -nan     0.1000    0.0001
    ##    140        0.2759            -nan     0.1000    0.0000
    ##    150        0.2720            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2388            -nan     0.1000    0.0738
    ##      2        1.1155            -nan     0.1000    0.0612
    ##      3        1.0134            -nan     0.1000    0.0510
    ##      4        0.9277            -nan     0.1000    0.0427
    ##      5        0.8559            -nan     0.1000    0.0358
    ##      6        0.7947            -nan     0.1000    0.0306
    ##      7        0.7406            -nan     0.1000    0.0272
    ##      8        0.6929            -nan     0.1000    0.0237
    ##      9        0.6539            -nan     0.1000    0.0195
    ##     10        0.6166            -nan     0.1000    0.0184
    ##     20        0.4171            -nan     0.1000    0.0058
    ##     40        0.3278            -nan     0.1000    0.0014
    ##     60        0.2969            -nan     0.1000    0.0008
    ##     80        0.2775            -nan     0.1000    0.0002
    ##    100        0.2610            -nan     0.1000    0.0001
    ##    120        0.2482            -nan     0.1000    0.0002
    ##    140        0.2407            -nan     0.1000    0.0000
    ##    150        0.2360            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2460            -nan     0.1000    0.0701
    ##      2        1.1312            -nan     0.1000    0.0572
    ##      3        1.0354            -nan     0.1000    0.0477
    ##      4        0.9550            -nan     0.1000    0.0398
    ##      5        0.8869            -nan     0.1000    0.0339
    ##      6        0.8295            -nan     0.1000    0.0283
    ##      7        0.7793            -nan     0.1000    0.0248
    ##      8        0.7363            -nan     0.1000    0.0214
    ##      9        0.6996            -nan     0.1000    0.0184
    ##     10        0.6679            -nan     0.1000    0.0157
    ##     20        0.4839            -nan     0.1000    0.0062
    ##     40        0.3756            -nan     0.1000    0.0011
    ##     60        0.3442            -nan     0.1000    0.0005
    ##     80        0.3337            -nan     0.1000    0.0000
    ##    100        0.3268            -nan     0.1000    0.0001
    ##    120        0.3215            -nan     0.1000    0.0000
    ##    140        0.3177            -nan     0.1000    0.0001
    ##    150        0.3154            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2426            -nan     0.1000    0.0717
    ##      2        1.1231            -nan     0.1000    0.0595
    ##      3        1.0250            -nan     0.1000    0.0490
    ##      4        0.9425            -nan     0.1000    0.0412
    ##      5        0.8734            -nan     0.1000    0.0347
    ##      6        0.8132            -nan     0.1000    0.0302
    ##      7        0.7605            -nan     0.1000    0.0262
    ##      8        0.7160            -nan     0.1000    0.0220
    ##      9        0.6762            -nan     0.1000    0.0201
    ##     10        0.6404            -nan     0.1000    0.0179
    ##     20        0.4357            -nan     0.1000    0.0067
    ##     40        0.3332            -nan     0.1000    0.0009
    ##     60        0.3082            -nan     0.1000    0.0003
    ##     80        0.2921            -nan     0.1000    0.0002
    ##    100        0.2775            -nan     0.1000    0.0002
    ##    120        0.2684            -nan     0.1000    0.0001
    ##    140        0.2603            -nan     0.1000    0.0002
    ##    150        0.2567            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2383            -nan     0.1000    0.0736
    ##      2        1.1151            -nan     0.1000    0.0616
    ##      3        1.0119            -nan     0.1000    0.0513
    ##      4        0.9257            -nan     0.1000    0.0429
    ##      5        0.8511            -nan     0.1000    0.0369
    ##      6        0.7865            -nan     0.1000    0.0323
    ##      7        0.7310            -nan     0.1000    0.0279
    ##      8        0.6834            -nan     0.1000    0.0236
    ##      9        0.6412            -nan     0.1000    0.0208
    ##     10        0.6043            -nan     0.1000    0.0182
    ##     20        0.4093            -nan     0.1000    0.0054
    ##     40        0.3189            -nan     0.1000    0.0011
    ##     60        0.2903            -nan     0.1000    0.0004
    ##     80        0.2728            -nan     0.1000    0.0007
    ##    100        0.2583            -nan     0.1000    0.0001
    ##    120        0.2432            -nan     0.1000    0.0002
    ##    140        0.2340            -nan     0.1000   -0.0000
    ##    150        0.2284            -nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2444            -nan     0.1000    0.0704
    ##      2        1.1287            -nan     0.1000    0.0580
    ##      3        1.0320            -nan     0.1000    0.0480
    ##      4        0.9508            -nan     0.1000    0.0405
    ##      5        0.8819            -nan     0.1000    0.0342
    ##      6        0.8238            -nan     0.1000    0.0293
    ##      7        0.7742            -nan     0.1000    0.0249
    ##      8        0.7302            -nan     0.1000    0.0217
    ##      9        0.6928            -nan     0.1000    0.0186
    ##     10        0.6605            -nan     0.1000    0.0160
    ##     20        0.4800            -nan     0.1000    0.0054
    ##     40        0.3774            -nan     0.1000    0.0012
    ##     60        0.3483            -nan     0.1000    0.0002
    ##     80        0.3377            -nan     0.1000    0.0003
    ##    100        0.3310            -nan     0.1000    0.0001
    ##    120        0.3252            -nan     0.1000    0.0001
    ##    140        0.3209            -nan     0.1000    0.0001
    ##    150        0.3186            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2412            -nan     0.1000    0.0729
    ##      2        1.1244            -nan     0.1000    0.0588
    ##      3        1.0249            -nan     0.1000    0.0495
    ##      4        0.9412            -nan     0.1000    0.0417
    ##      5        0.8703            -nan     0.1000    0.0353
    ##      6        0.8097            -nan     0.1000    0.0304
    ##      7        0.7578            -nan     0.1000    0.0260
    ##      8        0.7113            -nan     0.1000    0.0229
    ##      9        0.6724            -nan     0.1000    0.0195
    ##     10        0.6369            -nan     0.1000    0.0174
    ##     20        0.4373            -nan     0.1000    0.0050
    ##     40        0.3364            -nan     0.1000    0.0009
    ##     60        0.3105            -nan     0.1000    0.0004
    ##     80        0.2966            -nan     0.1000    0.0003
    ##    100        0.2851            -nan     0.1000    0.0002
    ##    120        0.2763            -nan     0.1000    0.0000
    ##    140        0.2689            -nan     0.1000    0.0001
    ##    150        0.2654            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2356            -nan     0.1000    0.0753
    ##      2        1.1152            -nan     0.1000    0.0602
    ##      3        1.0124            -nan     0.1000    0.0514
    ##      4        0.9258            -nan     0.1000    0.0434
    ##      5        0.8535            -nan     0.1000    0.0358
    ##      6        0.7919            -nan     0.1000    0.0306
    ##      7        0.7371            -nan     0.1000    0.0273
    ##      8        0.6892            -nan     0.1000    0.0236
    ##      9        0.6479            -nan     0.1000    0.0208
    ##     10        0.6106            -nan     0.1000    0.0186
    ##     20        0.4174            -nan     0.1000    0.0051
    ##     40        0.3221            -nan     0.1000    0.0007
    ##     60        0.2916            -nan     0.1000    0.0002
    ##     80        0.2746            -nan     0.1000    0.0002
    ##    100        0.2622            -nan     0.1000    0.0000
    ##    120        0.2500            -nan     0.1000    0.0002
    ##    140        0.2391            -nan     0.1000    0.0002
    ##    150        0.2349            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2469            -nan     0.1000    0.0697
    ##      2        1.1329            -nan     0.1000    0.0570
    ##      3        1.0378            -nan     0.1000    0.0474
    ##      4        0.9581            -nan     0.1000    0.0397
    ##      5        0.8907            -nan     0.1000    0.0337
    ##      6        0.8333            -nan     0.1000    0.0287
    ##      7        0.7842            -nan     0.1000    0.0247
    ##      8        0.7415            -nan     0.1000    0.0212
    ##      9        0.7051            -nan     0.1000    0.0181
    ##     10        0.6721            -nan     0.1000    0.0165
    ##     20        0.4900            -nan     0.1000    0.0053
    ##     40        0.3822            -nan     0.1000    0.0011
    ##     60        0.3530            -nan     0.1000    0.0003
    ##     80        0.3421            -nan     0.1000    0.0001
    ##    100        0.3349            -nan     0.1000   -0.0000
    ##    120        0.3297            -nan     0.1000    0.0000
    ##    140        0.3269            -nan     0.1000    0.0000
    ##    150        0.3248            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2403            -nan     0.1000    0.0726
    ##      2        1.1214            -nan     0.1000    0.0595
    ##      3        1.0225            -nan     0.1000    0.0497
    ##      4        0.9392            -nan     0.1000    0.0414
    ##      5        0.8679            -nan     0.1000    0.0357
    ##      6        0.8069            -nan     0.1000    0.0304
    ##      7        0.7540            -nan     0.1000    0.0260
    ##      8        0.7071            -nan     0.1000    0.0234
    ##      9        0.6683            -nan     0.1000    0.0196
    ##     10        0.6339            -nan     0.1000    0.0172
    ##     20        0.4287            -nan     0.1000    0.0048
    ##     40        0.3274            -nan     0.1000    0.0006
    ##     60        0.3019            -nan     0.1000    0.0004
    ##     80        0.2900            -nan     0.1000    0.0001
    ##    100        0.2795            -nan     0.1000    0.0002
    ##    120        0.2688            -nan     0.1000    0.0001
    ##    140        0.2585            -nan     0.1000    0.0003
    ##    150        0.2554            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2379            -nan     0.1000    0.0737
    ##      2        1.1143            -nan     0.1000    0.0621
    ##      3        1.0110            -nan     0.1000    0.0513
    ##      4        0.9240            -nan     0.1000    0.0434
    ##      5        0.8500            -nan     0.1000    0.0366
    ##      6        0.7850            -nan     0.1000    0.0321
    ##      7        0.7315            -nan     0.1000    0.0267
    ##      8        0.6850            -nan     0.1000    0.0232
    ##      9        0.6446            -nan     0.1000    0.0202
    ##     10        0.6068            -nan     0.1000    0.0187
    ##     20        0.4094            -nan     0.1000    0.0055
    ##     40        0.3172            -nan     0.1000    0.0007
    ##     60        0.2863            -nan     0.1000    0.0007
    ##     80        0.2689            -nan     0.1000    0.0004
    ##    100        0.2528            -nan     0.1000    0.0003
    ##    120        0.2411            -nan     0.1000    0.0001
    ##    140        0.2348            -nan     0.1000    0.0000
    ##    150        0.2301            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2481            -nan     0.1000    0.0695
    ##      2        1.1332            -nan     0.1000    0.0573
    ##      3        1.0391            -nan     0.1000    0.0474
    ##      4        0.9593            -nan     0.1000    0.0399
    ##      5        0.8918            -nan     0.1000    0.0339
    ##      6        0.8342            -nan     0.1000    0.0287
    ##      7        0.7848            -nan     0.1000    0.0248
    ##      8        0.7421            -nan     0.1000    0.0211
    ##      9        0.7044            -nan     0.1000    0.0189
    ##     10        0.6726            -nan     0.1000    0.0161
    ##     20        0.4865            -nan     0.1000    0.0056
    ##     40        0.3801            -nan     0.1000    0.0011
    ##     60        0.3477            -nan     0.1000    0.0003
    ##     80        0.3389            -nan     0.1000    0.0003
    ##    100        0.3316            -nan     0.1000    0.0000
    ##    120        0.3263            -nan     0.1000    0.0000
    ##    140        0.3208            -nan     0.1000    0.0000
    ##    150        0.3188            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2427            -nan     0.1000    0.0721
    ##      2        1.1254            -nan     0.1000    0.0591
    ##      3        1.0272            -nan     0.1000    0.0491
    ##      4        0.9442            -nan     0.1000    0.0412
    ##      5        0.8736            -nan     0.1000    0.0350
    ##      6        0.8135            -nan     0.1000    0.0299
    ##      7        0.7621            -nan     0.1000    0.0258
    ##      8        0.7165            -nan     0.1000    0.0227
    ##      9        0.6776            -nan     0.1000    0.0195
    ##     10        0.6426            -nan     0.1000    0.0175
    ##     20        0.4383            -nan     0.1000    0.0064
    ##     40        0.3404            -nan     0.1000    0.0005
    ##     60        0.3135            -nan     0.1000    0.0002
    ##     80        0.3002            -nan     0.1000    0.0006
    ##    100        0.2859            -nan     0.1000    0.0001
    ##    120        0.2757            -nan     0.1000    0.0002
    ##    140        0.2667            -nan     0.1000    0.0001
    ##    150        0.2629            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2374            -nan     0.1000    0.0737
    ##      2        1.1159            -nan     0.1000    0.0603
    ##      3        1.0135            -nan     0.1000    0.0512
    ##      4        0.9286            -nan     0.1000    0.0425
    ##      5        0.8555            -nan     0.1000    0.0365
    ##      6        0.7923            -nan     0.1000    0.0313
    ##      7        0.7385            -nan     0.1000    0.0271
    ##      8        0.6923            -nan     0.1000    0.0231
    ##      9        0.6513            -nan     0.1000    0.0203
    ##     10        0.6138            -nan     0.1000    0.0187
    ##     20        0.4221            -nan     0.1000    0.0058
    ##     40        0.3259            -nan     0.1000    0.0013
    ##     60        0.2975            -nan     0.1000    0.0004
    ##     80        0.2776            -nan     0.1000    0.0007
    ##    100        0.2629            -nan     0.1000    0.0001
    ##    120        0.2509            -nan     0.1000    0.0003
    ##    140        0.2406            -nan     0.1000    0.0001
    ##    150        0.2366            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2479            -nan     0.1000    0.0689
    ##      2        1.1349            -nan     0.1000    0.0563
    ##      3        1.0406            -nan     0.1000    0.0467
    ##      4        0.9614            -nan     0.1000    0.0393
    ##      5        0.8948            -nan     0.1000    0.0332
    ##      6        0.8379            -nan     0.1000    0.0285
    ##      7        0.7886            -nan     0.1000    0.0243
    ##      8        0.7471            -nan     0.1000    0.0209
    ##      9        0.7104            -nan     0.1000    0.0182
    ##     10        0.6791            -nan     0.1000    0.0156
    ##     20        0.4956            -nan     0.1000    0.0061
    ##     40        0.3866            -nan     0.1000    0.0017
    ##     60        0.3562            -nan     0.1000    0.0003
    ##     80        0.3459            -nan     0.1000    0.0001
    ##    100        0.3388            -nan     0.1000    0.0000
    ##    120        0.3325            -nan     0.1000    0.0001
    ##    140        0.3283            -nan     0.1000    0.0001
    ##    150        0.3261            -nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2436            -nan     0.1000    0.0718
    ##      2        1.1245            -nan     0.1000    0.0593
    ##      3        1.0256            -nan     0.1000    0.0495
    ##      4        0.9419            -nan     0.1000    0.0416
    ##      5        0.8719            -nan     0.1000    0.0351
    ##      6        0.8102            -nan     0.1000    0.0305
    ##      7        0.7582            -nan     0.1000    0.0259
    ##      8        0.7116            -nan     0.1000    0.0231
    ##      9        0.6726            -nan     0.1000    0.0194
    ##     10        0.6371            -nan     0.1000    0.0178
    ##     20        0.4306            -nan     0.1000    0.0060
    ##     40        0.3317            -nan     0.1000    0.0012
    ##     60        0.3048            -nan     0.1000    0.0002
    ##     80        0.2892            -nan     0.1000    0.0001
    ##    100        0.2774            -nan     0.1000   -0.0000
    ##    120        0.2687            -nan     0.1000    0.0001
    ##    140        0.2608            -nan     0.1000    0.0002
    ##    150        0.2576            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2361            -nan     0.1000    0.0752
    ##      2        1.1137            -nan     0.1000    0.0608
    ##      3        1.0130            -nan     0.1000    0.0501
    ##      4        0.9267            -nan     0.1000    0.0428
    ##      5        0.8535            -nan     0.1000    0.0368
    ##      6        0.7906            -nan     0.1000    0.0310
    ##      7        0.7346            -nan     0.1000    0.0283
    ##      8        0.6854            -nan     0.1000    0.0244
    ##      9        0.6452            -nan     0.1000    0.0202
    ##     10        0.6084            -nan     0.1000    0.0182
    ##     20        0.4121            -nan     0.1000    0.0046
    ##     40        0.3190            -nan     0.1000    0.0013
    ##     60        0.2886            -nan     0.1000    0.0003
    ##     80        0.2708            -nan     0.1000    0.0005
    ##    100        0.2502            -nan     0.1000    0.0001
    ##    120        0.2399            -nan     0.1000    0.0001
    ##    140        0.2298            -nan     0.1000    0.0001
    ##    150        0.2258            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2477            -nan     0.1000    0.0697
    ##      2        1.1334            -nan     0.1000    0.0569
    ##      3        1.0384            -nan     0.1000    0.0474
    ##      4        0.9586            -nan     0.1000    0.0396
    ##      5        0.8919            -nan     0.1000    0.0335
    ##      6        0.8345            -nan     0.1000    0.0287
    ##      7        0.7856            -nan     0.1000    0.0245
    ##      8        0.7436            -nan     0.1000    0.0209
    ##      9        0.7057            -nan     0.1000    0.0190
    ##     10        0.6734            -nan     0.1000    0.0162
    ##     20        0.4921            -nan     0.1000    0.0049
    ##     40        0.3833            -nan     0.1000    0.0012
    ##     60        0.3541            -nan     0.1000    0.0005
    ##     80        0.3448            -nan     0.1000    0.0002
    ##    100        0.3385            -nan     0.1000    0.0001
    ##    120        0.3340            -nan     0.1000    0.0000
    ##    140        0.3287            -nan     0.1000    0.0001
    ##    150        0.3264            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2427            -nan     0.1000    0.0720
    ##      2        1.1248            -nan     0.1000    0.0589
    ##      3        1.0262            -nan     0.1000    0.0492
    ##      4        0.9432            -nan     0.1000    0.0413
    ##      5        0.8738            -nan     0.1000    0.0346
    ##      6        0.8136            -nan     0.1000    0.0299
    ##      7        0.7621            -nan     0.1000    0.0255
    ##      8        0.7166            -nan     0.1000    0.0227
    ##      9        0.6765            -nan     0.1000    0.0199
    ##     10        0.6422            -nan     0.1000    0.0169
    ##     20        0.4369            -nan     0.1000    0.0059
    ##     40        0.3379            -nan     0.1000    0.0006
    ##     60        0.3127            -nan     0.1000    0.0003
    ##     80        0.2969            -nan     0.1000    0.0006
    ##    100        0.2862            -nan     0.1000    0.0003
    ##    120        0.2775            -nan     0.1000    0.0001
    ##    140        0.2697            -nan     0.1000    0.0001
    ##    150        0.2670            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2382            -nan     0.1000    0.0739
    ##      2        1.1177            -nan     0.1000    0.0599
    ##      3        1.0150            -nan     0.1000    0.0512
    ##      4        0.9286            -nan     0.1000    0.0431
    ##      5        0.8537            -nan     0.1000    0.0373
    ##      6        0.7911            -nan     0.1000    0.0312
    ##      7        0.7362            -nan     0.1000    0.0276
    ##      8        0.6905            -nan     0.1000    0.0227
    ##      9        0.6490            -nan     0.1000    0.0207
    ##     10        0.6132            -nan     0.1000    0.0175
    ##     20        0.4179            -nan     0.1000    0.0051
    ##     40        0.3251            -nan     0.1000    0.0010
    ##     60        0.2953            -nan     0.1000    0.0004
    ##     80        0.2790            -nan     0.1000    0.0002
    ##    100        0.2627            -nan     0.1000    0.0001
    ##    120        0.2489            -nan     0.1000    0.0001
    ##    140        0.2406            -nan     0.1000    0.0002
    ##    150        0.2356            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2459            -nan     0.1000    0.0704
    ##      2        1.1298            -nan     0.1000    0.0577
    ##      3        1.0342            -nan     0.1000    0.0479
    ##      4        0.9542            -nan     0.1000    0.0403
    ##      5        0.8861            -nan     0.1000    0.0342
    ##      6        0.8279            -nan     0.1000    0.0291
    ##      7        0.7781            -nan     0.1000    0.0249
    ##      8        0.7354            -nan     0.1000    0.0214
    ##      9        0.6983            -nan     0.1000    0.0182
    ##     10        0.6647            -nan     0.1000    0.0167
    ##     20        0.4803            -nan     0.1000    0.0061
    ##     40        0.3696            -nan     0.1000    0.0017
    ##     60        0.3379            -nan     0.1000    0.0006
    ##     80        0.3269            -nan     0.1000    0.0002
    ##    100        0.3195            -nan     0.1000    0.0001
    ##    120        0.3151            -nan     0.1000    0.0001
    ##    140        0.3112            -nan     0.1000    0.0000
    ##    150        0.3093            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2397            -nan     0.1000    0.0735
    ##      2        1.1195            -nan     0.1000    0.0598
    ##      3        1.0207            -nan     0.1000    0.0494
    ##      4        0.9360            -nan     0.1000    0.0422
    ##      5        0.8647            -nan     0.1000    0.0355
    ##      6        0.8040            -nan     0.1000    0.0303
    ##      7        0.7512            -nan     0.1000    0.0265
    ##      8        0.7040            -nan     0.1000    0.0234
    ##      9        0.6635            -nan     0.1000    0.0201
    ##     10        0.6262            -nan     0.1000    0.0184
    ##     20        0.4211            -nan     0.1000    0.0062
    ##     40        0.3215            -nan     0.1000    0.0011
    ##     60        0.2975            -nan     0.1000    0.0003
    ##     80        0.2827            -nan     0.1000    0.0003
    ##    100        0.2720            -nan     0.1000    0.0000
    ##    120        0.2622            -nan     0.1000    0.0003
    ##    140        0.2546            -nan     0.1000    0.0001
    ##    150        0.2512            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2329            -nan     0.1000    0.0762
    ##      2        1.1080            -nan     0.1000    0.0626
    ##      3        1.0066            -nan     0.1000    0.0511
    ##      4        0.9195            -nan     0.1000    0.0437
    ##      5        0.8435            -nan     0.1000    0.0377
    ##      6        0.7784            -nan     0.1000    0.0324
    ##      7        0.7233            -nan     0.1000    0.0274
    ##      8        0.6764            -nan     0.1000    0.0235
    ##      9        0.6334            -nan     0.1000    0.0212
    ##     10        0.5974            -nan     0.1000    0.0179
    ##     20        0.3954            -nan     0.1000    0.0055
    ##     40        0.3023            -nan     0.1000    0.0006
    ##     60        0.2736            -nan     0.1000    0.0004
    ##     80        0.2590            -nan     0.1000    0.0004
    ##    100        0.2457            -nan     0.1000    0.0001
    ##    120        0.2325            -nan     0.1000    0.0004
    ##    140        0.2228            -nan     0.1000    0.0001
    ##    150        0.2192            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2452            -nan     0.1000    0.0702
    ##      2        1.1301            -nan     0.1000    0.0577
    ##      3        1.0342            -nan     0.1000    0.0480
    ##      4        0.9541            -nan     0.1000    0.0403
    ##      5        0.8863            -nan     0.1000    0.0339
    ##      6        0.8286            -nan     0.1000    0.0289
    ##      7        0.7788            -nan     0.1000    0.0252
    ##      8        0.7360            -nan     0.1000    0.0213
    ##      9        0.6982            -nan     0.1000    0.0190
    ##     10        0.6649            -nan     0.1000    0.0166
    ##     20        0.4785            -nan     0.1000    0.0056
    ##     40        0.3691            -nan     0.1000    0.0014
    ##     60        0.3404            -nan     0.1000    0.0002
    ##     80        0.3306            -nan     0.1000    0.0000
    ##    100        0.3241            -nan     0.1000    0.0000
    ##    120        0.3185            -nan     0.1000    0.0001
    ##    140        0.3142            -nan     0.1000    0.0000
    ##    150        0.3123            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2392            -nan     0.1000    0.0733
    ##      2        1.1193            -nan     0.1000    0.0599
    ##      3        1.0195            -nan     0.1000    0.0496
    ##      4        0.9341            -nan     0.1000    0.0426
    ##      5        0.8616            -nan     0.1000    0.0360
    ##      6        0.7998            -nan     0.1000    0.0308
    ##      7        0.7464            -nan     0.1000    0.0262
    ##      8        0.7001            -nan     0.1000    0.0231
    ##      9        0.6600            -nan     0.1000    0.0198
    ##     10        0.6244            -nan     0.1000    0.0176
    ##     20        0.4209            -nan     0.1000    0.0059
    ##     40        0.3214            -nan     0.1000    0.0012
    ##     60        0.2977            -nan     0.1000    0.0002
    ##     80        0.2856            -nan     0.1000    0.0003
    ##    100        0.2735            -nan     0.1000    0.0001
    ##    120        0.2651            -nan     0.1000    0.0001
    ##    140        0.2554            -nan     0.1000    0.0013
    ##    150        0.2512            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2352            -nan     0.1000    0.0756
    ##      2        1.1124            -nan     0.1000    0.0611
    ##      3        1.0106            -nan     0.1000    0.0506
    ##      4        0.9219            -nan     0.1000    0.0443
    ##      5        0.8474            -nan     0.1000    0.0370
    ##      6        0.7832            -nan     0.1000    0.0321
    ##      7        0.7273            -nan     0.1000    0.0278
    ##      8        0.6784            -nan     0.1000    0.0243
    ##      9        0.6351            -nan     0.1000    0.0214
    ##     10        0.5979            -nan     0.1000    0.0185
    ##     20        0.4009            -nan     0.1000    0.0050
    ##     40        0.3107            -nan     0.1000    0.0011
    ##     60        0.2767            -nan     0.1000    0.0003
    ##     80        0.2569            -nan     0.1000    0.0002
    ##    100        0.2436            -nan     0.1000    0.0003
    ##    120        0.2308            -nan     0.1000    0.0003
    ##    140        0.2230            -nan     0.1000    0.0002
    ##    150        0.2176            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2456            -nan     0.1000    0.0702
    ##      2        1.1298            -nan     0.1000    0.0577
    ##      3        1.0341            -nan     0.1000    0.0481
    ##      4        0.9537            -nan     0.1000    0.0401
    ##      5        0.8855            -nan     0.1000    0.0340
    ##      6        0.8277            -nan     0.1000    0.0291
    ##      7        0.7779            -nan     0.1000    0.0246
    ##      8        0.7341            -nan     0.1000    0.0217
    ##      9        0.6958            -nan     0.1000    0.0192
    ##     10        0.6630            -nan     0.1000    0.0162
    ##     20        0.4757            -nan     0.1000    0.0057
    ##     40        0.3607            -nan     0.1000    0.0011
    ##     60        0.3300            -nan     0.1000    0.0000
    ##     80        0.3177            -nan     0.1000    0.0002
    ##    100        0.3115            -nan     0.1000    0.0000
    ##    120        0.3060            -nan     0.1000    0.0000
    ##    140        0.3024            -nan     0.1000    0.0000
    ##    150        0.3005            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2415            -nan     0.1000    0.0725
    ##      2        1.1222            -nan     0.1000    0.0594
    ##      3        1.0233            -nan     0.1000    0.0493
    ##      4        0.9395            -nan     0.1000    0.0413
    ##      5        0.8676            -nan     0.1000    0.0355
    ##      6        0.8069            -nan     0.1000    0.0300
    ##      7        0.7542            -nan     0.1000    0.0261
    ##      8        0.7074            -nan     0.1000    0.0232
    ##      9        0.6675            -nan     0.1000    0.0198
    ##     10        0.6332            -nan     0.1000    0.0169
    ##     20        0.4271            -nan     0.1000    0.0053
    ##     40        0.3277            -nan     0.1000    0.0005
    ##     60        0.3016            -nan     0.1000    0.0004
    ##     80        0.2848            -nan     0.1000    0.0001
    ##    100        0.2761            -nan     0.1000    0.0002
    ##    120        0.2683            -nan     0.1000    0.0002
    ##    140        0.2577            -nan     0.1000    0.0001
    ##    150        0.2537            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2380            -nan     0.1000    0.0738
    ##      2        1.1141            -nan     0.1000    0.0619
    ##      3        1.0116            -nan     0.1000    0.0514
    ##      4        0.9248            -nan     0.1000    0.0432
    ##      5        0.8512            -nan     0.1000    0.0368
    ##      6        0.7892            -nan     0.1000    0.0305
    ##      7        0.7355            -nan     0.1000    0.0267
    ##      8        0.6891            -nan     0.1000    0.0233
    ##      9        0.6482            -nan     0.1000    0.0204
    ##     10        0.6105            -nan     0.1000    0.0186
    ##     20        0.4103            -nan     0.1000    0.0053
    ##     40        0.3158            -nan     0.1000    0.0011
    ##     60        0.2846            -nan     0.1000    0.0004
    ##     80        0.2677            -nan     0.1000    0.0003
    ##    100        0.2527            -nan     0.1000    0.0000
    ##    120        0.2414            -nan     0.1000    0.0003
    ##    140        0.2305            -nan     0.1000    0.0000
    ##    150        0.2263            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2483            -nan     0.1000    0.0693
    ##      2        1.1351            -nan     0.1000    0.0569
    ##      3        1.0405            -nan     0.1000    0.0473
    ##      4        0.9606            -nan     0.1000    0.0397
    ##      5        0.8932            -nan     0.1000    0.0336
    ##      6        0.8367            -nan     0.1000    0.0286
    ##      7        0.7885            -nan     0.1000    0.0241
    ##      8        0.7453            -nan     0.1000    0.0214
    ##      9        0.7085            -nan     0.1000    0.0183
    ##     10        0.6760            -nan     0.1000    0.0164
    ##     20        0.4948            -nan     0.1000    0.0054
    ##     40        0.3909            -nan     0.1000    0.0012
    ##     60        0.3607            -nan     0.1000    0.0003
    ##     80        0.3498            -nan     0.1000    0.0004
    ##    100        0.3433            -nan     0.1000    0.0001
    ##    120        0.3374            -nan     0.1000    0.0001
    ##    140        0.3335            -nan     0.1000    0.0000
    ##    150        0.3308            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2424            -nan     0.1000    0.0722
    ##      2        1.1244            -nan     0.1000    0.0595
    ##      3        1.0268            -nan     0.1000    0.0489
    ##      4        0.9438            -nan     0.1000    0.0415
    ##      5        0.8733            -nan     0.1000    0.0351
    ##      6        0.8130            -nan     0.1000    0.0297
    ##      7        0.7601            -nan     0.1000    0.0262
    ##      8        0.7141            -nan     0.1000    0.0229
    ##      9        0.6751            -nan     0.1000    0.0195
    ##     10        0.6400            -nan     0.1000    0.0174
    ##     20        0.4355            -nan     0.1000    0.0051
    ##     40        0.3332            -nan     0.1000    0.0009
    ##     60        0.3071            -nan     0.1000    0.0004
    ##     80        0.2945            -nan     0.1000    0.0002
    ##    100        0.2860            -nan     0.1000    0.0001
    ##    120        0.2761            -nan     0.1000    0.0003
    ##    140        0.2689            -nan     0.1000    0.0001
    ##    150        0.2660            -nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2368            -nan     0.1000    0.0742
    ##      2        1.1159            -nan     0.1000    0.0603
    ##      3        1.0126            -nan     0.1000    0.0516
    ##      4        0.9264            -nan     0.1000    0.0430
    ##      5        0.8535            -nan     0.1000    0.0361
    ##      6        0.7892            -nan     0.1000    0.0321
    ##      7        0.7361            -nan     0.1000    0.0265
    ##      8        0.6902            -nan     0.1000    0.0228
    ##      9        0.6502            -nan     0.1000    0.0199
    ##     10        0.6149            -nan     0.1000    0.0172
    ##     20        0.4152            -nan     0.1000    0.0049
    ##     40        0.3222            -nan     0.1000    0.0007
    ##     60        0.2942            -nan     0.1000    0.0003
    ##     80        0.2764            -nan     0.1000    0.0004
    ##    100        0.2620            -nan     0.1000    0.0001
    ##    120        0.2491            -nan     0.1000    0.0001
    ##    140        0.2398            -nan     0.1000    0.0001
    ##    150        0.2352            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2452            -nan     0.1000    0.0707
    ##      2        1.1294            -nan     0.1000    0.0579
    ##      3        1.0328            -nan     0.1000    0.0479
    ##      4        0.9526            -nan     0.1000    0.0402
    ##      5        0.8844            -nan     0.1000    0.0340
    ##      6        0.8264            -nan     0.1000    0.0289
    ##      7        0.7765            -nan     0.1000    0.0249
    ##      8        0.7335            -nan     0.1000    0.0215
    ##      9        0.6966            -nan     0.1000    0.0184
    ##     10        0.6645            -nan     0.1000    0.0160
    ##     20        0.4816            -nan     0.1000    0.0054
    ##     40        0.3732            -nan     0.1000    0.0012
    ##     60        0.3429            -nan     0.1000    0.0006
    ##     80        0.3330            -nan     0.1000    0.0001
    ##    100        0.3278            -nan     0.1000    0.0000
    ##    120        0.3222            -nan     0.1000   -0.0000
    ##    140        0.3185            -nan     0.1000    0.0000
    ##    150        0.3167            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2408            -nan     0.1000    0.0729
    ##      2        1.1228            -nan     0.1000    0.0592
    ##      3        1.0249            -nan     0.1000    0.0493
    ##      4        0.9412            -nan     0.1000    0.0417
    ##      5        0.8708            -nan     0.1000    0.0352
    ##      6        0.8097            -nan     0.1000    0.0303
    ##      7        0.7582            -nan     0.1000    0.0259
    ##      8        0.7115            -nan     0.1000    0.0232
    ##      9        0.6721            -nan     0.1000    0.0195
    ##     10        0.6365            -nan     0.1000    0.0176
    ##     20        0.4374            -nan     0.1000    0.0071
    ##     40        0.3373            -nan     0.1000    0.0006
    ##     60        0.3133            -nan     0.1000    0.0002
    ##     80        0.2983            -nan     0.1000    0.0001
    ##    100        0.2882            -nan     0.1000    0.0003
    ##    120        0.2760            -nan     0.1000    0.0001
    ##    140        0.2699            -nan     0.1000    0.0001
    ##    150        0.2667            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2331            -nan     0.1000    0.0762
    ##      2        1.1092            -nan     0.1000    0.0622
    ##      3        1.0048            -nan     0.1000    0.0521
    ##      4        0.9188            -nan     0.1000    0.0430
    ##      5        0.8431            -nan     0.1000    0.0378
    ##      6        0.7788            -nan     0.1000    0.0318
    ##      7        0.7235            -nan     0.1000    0.0274
    ##      8        0.6765            -nan     0.1000    0.0234
    ##      9        0.6338            -nan     0.1000    0.0215
    ##     10        0.5956            -nan     0.1000    0.0188
    ##     20        0.3986            -nan     0.1000    0.0049
    ##     40        0.3082            -nan     0.1000    0.0010
    ##     60        0.2816            -nan     0.1000    0.0001
    ##     80        0.2642            -nan     0.1000    0.0000
    ##    100        0.2534            -nan     0.1000    0.0001
    ##    120        0.2416            -nan     0.1000    0.0005
    ##    140        0.2297            -nan     0.1000    0.0001
    ##    150        0.2261            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2471            -nan     0.1000    0.0699
    ##      2        1.1326            -nan     0.1000    0.0570
    ##      3        1.0374            -nan     0.1000    0.0476
    ##      4        0.9587            -nan     0.1000    0.0397
    ##      5        0.8906            -nan     0.1000    0.0334
    ##      6        0.8329            -nan     0.1000    0.0288
    ##      7        0.7841            -nan     0.1000    0.0246
    ##      8        0.7410            -nan     0.1000    0.0212
    ##      9        0.7048            -nan     0.1000    0.0182
    ##     10        0.6732            -nan     0.1000    0.0157
    ##     20        0.4953            -nan     0.1000    0.0055
    ##     40        0.3894            -nan     0.1000    0.0016
    ##     60        0.3581            -nan     0.1000    0.0003
    ##     80        0.3467            -nan     0.1000    0.0001
    ##    100        0.3402            -nan     0.1000    0.0001
    ##    120        0.3347            -nan     0.1000    0.0000
    ##    140        0.3304            -nan     0.1000    0.0000
    ##    150        0.3280            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2439            -nan     0.1000    0.0715
    ##      2        1.1255            -nan     0.1000    0.0595
    ##      3        1.0269            -nan     0.1000    0.0495
    ##      4        0.9431            -nan     0.1000    0.0414
    ##      5        0.8730            -nan     0.1000    0.0353
    ##      6        0.8125            -nan     0.1000    0.0301
    ##      7        0.7604            -nan     0.1000    0.0260
    ##      8        0.7151            -nan     0.1000    0.0223
    ##      9        0.6768            -nan     0.1000    0.0193
    ##     10        0.6419            -nan     0.1000    0.0175
    ##     20        0.4412            -nan     0.1000    0.0051
    ##     40        0.3406            -nan     0.1000    0.0011
    ##     60        0.3159            -nan     0.1000    0.0003
    ##     80        0.3016            -nan     0.1000    0.0001
    ##    100        0.2905            -nan     0.1000    0.0002
    ##    120        0.2807            -nan     0.1000    0.0001
    ##    140        0.2710            -nan     0.1000    0.0001
    ##    150        0.2674            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2369            -nan     0.1000    0.0746
    ##      2        1.1172            -nan     0.1000    0.0597
    ##      3        1.0153            -nan     0.1000    0.0511
    ##      4        0.9296            -nan     0.1000    0.0429
    ##      5        0.8568            -nan     0.1000    0.0364
    ##      6        0.7938            -nan     0.1000    0.0312
    ##      7        0.7410            -nan     0.1000    0.0264
    ##      8        0.6934            -nan     0.1000    0.0236
    ##      9        0.6511            -nan     0.1000    0.0211
    ##     10        0.6164            -nan     0.1000    0.0172
    ##     20        0.4240            -nan     0.1000    0.0055
    ##     40        0.3291            -nan     0.1000    0.0008
    ##     60        0.3011            -nan     0.1000    0.0002
    ##     80        0.2844            -nan     0.1000    0.0001
    ##    100        0.2713            -nan     0.1000    0.0002
    ##    120        0.2600            -nan     0.1000    0.0002
    ##    140        0.2495            -nan     0.1000    0.0003
    ##    150        0.2443            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2457            -nan     0.1000    0.0701
    ##      2        1.1315            -nan     0.1000    0.0570
    ##      3        1.0367            -nan     0.1000    0.0473
    ##      4        0.9569            -nan     0.1000    0.0397
    ##      5        0.8899            -nan     0.1000    0.0338
    ##      6        0.8324            -nan     0.1000    0.0287
    ##      7        0.7828            -nan     0.1000    0.0247
    ##      8        0.7408            -nan     0.1000    0.0210
    ##      9        0.7044            -nan     0.1000    0.0181
    ##     10        0.6719            -nan     0.1000    0.0163
    ##     20        0.4910            -nan     0.1000    0.0061
    ##     40        0.3843            -nan     0.1000    0.0012
    ##     60        0.3549            -nan     0.1000    0.0007
    ##     80        0.3442            -nan     0.1000    0.0002
    ##    100        0.3377            -nan     0.1000    0.0001
    ##    120        0.3323            -nan     0.1000   -0.0000
    ##    140        0.3283            -nan     0.1000    0.0000
    ##    150        0.3262            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2426            -nan     0.1000    0.0718
    ##      2        1.1228            -nan     0.1000    0.0596
    ##      3        1.0235            -nan     0.1000    0.0494
    ##      4        0.9397            -nan     0.1000    0.0417
    ##      5        0.8696            -nan     0.1000    0.0351
    ##      6        0.8094            -nan     0.1000    0.0300
    ##      7        0.7579            -nan     0.1000    0.0257
    ##      8        0.7127            -nan     0.1000    0.0226
    ##      9        0.6734            -nan     0.1000    0.0194
    ##     10        0.6394            -nan     0.1000    0.0169
    ##     20        0.4356            -nan     0.1000    0.0067
    ##     40        0.3343            -nan     0.1000    0.0010
    ##     60        0.3074            -nan     0.1000    0.0004
    ##     80        0.2911            -nan     0.1000    0.0004
    ##    100        0.2807            -nan     0.1000    0.0001
    ##    120        0.2697            -nan     0.1000    0.0001
    ##    140        0.2612            -nan     0.1000    0.0001
    ##    150        0.2581            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2375            -nan     0.1000    0.0743
    ##      2        1.1159            -nan     0.1000    0.0606
    ##      3        1.0136            -nan     0.1000    0.0509
    ##      4        0.9278            -nan     0.1000    0.0427
    ##      5        0.8558            -nan     0.1000    0.0360
    ##      6        0.7916            -nan     0.1000    0.0320
    ##      7        0.7373            -nan     0.1000    0.0269
    ##      8        0.6892            -nan     0.1000    0.0240
    ##      9        0.6489            -nan     0.1000    0.0200
    ##     10        0.6128            -nan     0.1000    0.0180
    ##     20        0.4184            -nan     0.1000    0.0048
    ##     40        0.3271            -nan     0.1000    0.0011
    ##     60        0.2939            -nan     0.1000    0.0005
    ##     80        0.2787            -nan     0.1000    0.0003
    ##    100        0.2656            -nan     0.1000    0.0002
    ##    120        0.2532            -nan     0.1000    0.0001
    ##    140        0.2426            -nan     0.1000    0.0001
    ##    150        0.2370            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2488            -nan     0.1000    0.0688
    ##      2        1.1355            -nan     0.1000    0.0566
    ##      3        1.0423            -nan     0.1000    0.0469
    ##      4        0.9631            -nan     0.1000    0.0393
    ##      5        0.8966            -nan     0.1000    0.0334
    ##      6        0.8402            -nan     0.1000    0.0283
    ##      7        0.7913            -nan     0.1000    0.0243
    ##      8        0.7495            -nan     0.1000    0.0207
    ##      9        0.7130            -nan     0.1000    0.0184
    ##     10        0.6815            -nan     0.1000    0.0158
    ##     20        0.4966            -nan     0.1000    0.0057
    ##     40        0.3872            -nan     0.1000    0.0017
    ##     60        0.3560            -nan     0.1000    0.0005
    ##     80        0.3448            -nan     0.1000    0.0000
    ##    100        0.3367            -nan     0.1000    0.0000
    ##    120        0.3316            -nan     0.1000    0.0001
    ##    140        0.3266            -nan     0.1000    0.0000
    ##    150        0.3252            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2428            -nan     0.1000    0.0716
    ##      2        1.1231            -nan     0.1000    0.0598
    ##      3        1.0234            -nan     0.1000    0.0495
    ##      4        0.9411            -nan     0.1000    0.0415
    ##      5        0.8695            -nan     0.1000    0.0357
    ##      6        0.8086            -nan     0.1000    0.0303
    ##      7        0.7569            -nan     0.1000    0.0262
    ##      8        0.7106            -nan     0.1000    0.0231
    ##      9        0.6713            -nan     0.1000    0.0193
    ##     10        0.6362            -nan     0.1000    0.0174
    ##     20        0.4369            -nan     0.1000    0.0068
    ##     40        0.3367            -nan     0.1000    0.0009
    ##     60        0.3063            -nan     0.1000    0.0007
    ##     80        0.2909            -nan     0.1000    0.0002
    ##    100        0.2799            -nan     0.1000    0.0001
    ##    120        0.2690            -nan     0.1000    0.0001
    ##    140        0.2603            -nan     0.1000    0.0002
    ##    150        0.2561            -nan     0.1000    0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2355            -nan     0.1000    0.0755
    ##      2        1.1130            -nan     0.1000    0.0611
    ##      3        1.0118            -nan     0.1000    0.0502
    ##      4        0.9268            -nan     0.1000    0.0425
    ##      5        0.8527            -nan     0.1000    0.0372
    ##      6        0.7883            -nan     0.1000    0.0321
    ##      7        0.7343            -nan     0.1000    0.0267
    ##      8        0.6883            -nan     0.1000    0.0229
    ##      9        0.6469            -nan     0.1000    0.0203
    ##     10        0.6092            -nan     0.1000    0.0188
    ##     20        0.4084            -nan     0.1000    0.0057
    ##     40        0.3147            -nan     0.1000    0.0013
    ##     60        0.2847            -nan     0.1000    0.0004
    ##     80        0.2670            -nan     0.1000    0.0002
    ##    100        0.2512            -nan     0.1000    0.0003
    ##    120        0.2413            -nan     0.1000    0.0003
    ##    140        0.2317            -nan     0.1000    0.0001
    ##    150        0.2272            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2488            -nan     0.1000    0.0685
    ##      2        1.1358            -nan     0.1000    0.0565
    ##      3        1.0421            -nan     0.1000    0.0466
    ##      4        0.9633            -nan     0.1000    0.0393
    ##      5        0.8963            -nan     0.1000    0.0329
    ##      6        0.8394            -nan     0.1000    0.0282
    ##      7        0.7906            -nan     0.1000    0.0240
    ##      8        0.7488            -nan     0.1000    0.0207
    ##      9        0.7115            -nan     0.1000    0.0187
    ##     10        0.6791            -nan     0.1000    0.0161
    ##     20        0.4947            -nan     0.1000    0.0053
    ##     40        0.3848            -nan     0.1000    0.0010
    ##     60        0.3539            -nan     0.1000    0.0006
    ##     80        0.3434            -nan     0.1000    0.0003
    ##    100        0.3372            -nan     0.1000    0.0003
    ##    120        0.3334            -nan     0.1000    0.0001
    ##    140        0.3288            -nan     0.1000    0.0001
    ##    150        0.3266            -nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2424            -nan     0.1000    0.0718
    ##      2        1.1259            -nan     0.1000    0.0579
    ##      3        1.0288            -nan     0.1000    0.0481
    ##      4        0.9469            -nan     0.1000    0.0409
    ##      5        0.8772            -nan     0.1000    0.0348
    ##      6        0.8167            -nan     0.1000    0.0303
    ##      7        0.7650            -nan     0.1000    0.0259
    ##      8        0.7195            -nan     0.1000    0.0226
    ##      9        0.6800            -nan     0.1000    0.0197
    ##     10        0.6457            -nan     0.1000    0.0170
    ##     20        0.4451            -nan     0.1000    0.0066
    ##     40        0.3468            -nan     0.1000    0.0007
    ##     60        0.3174            -nan     0.1000    0.0005
    ##     80        0.3030            -nan     0.1000    0.0001
    ##    100        0.2898            -nan     0.1000    0.0002
    ##    120        0.2795            -nan     0.1000    0.0000
    ##    140        0.2708            -nan     0.1000    0.0000
    ##    150        0.2665            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2345            -nan     0.1000    0.0761
    ##      2        1.1118            -nan     0.1000    0.0615
    ##      3        1.0089            -nan     0.1000    0.0518
    ##      4        0.9211            -nan     0.1000    0.0439
    ##      5        0.8489            -nan     0.1000    0.0363
    ##      6        0.7849            -nan     0.1000    0.0318
    ##      7        0.7313            -nan     0.1000    0.0267
    ##      8        0.6825            -nan     0.1000    0.0244
    ##      9        0.6421            -nan     0.1000    0.0201
    ##     10        0.6041            -nan     0.1000    0.0186
    ##     20        0.4066            -nan     0.1000    0.0049
    ##     40        0.3118            -nan     0.1000    0.0011
    ##     60        0.2834            -nan     0.1000    0.0003
    ##     80        0.2668            -nan     0.1000    0.0001
    ##    100        0.2535            -nan     0.1000   -0.0000
    ##    120        0.2402            -nan     0.1000    0.0000
    ##    140        0.2303            -nan     0.1000    0.0003
    ##    150        0.2260            -nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2422            -nan     0.1000    0.0720
    ##      2        1.1236            -nan     0.1000    0.0595
    ##      3        1.0245            -nan     0.1000    0.0493
    ##      4        0.9411            -nan     0.1000    0.0418
    ##      5        0.8707            -nan     0.1000    0.0350
    ##      6        0.8090            -nan     0.1000    0.0306
    ##      7        0.7571            -nan     0.1000    0.0261
    ##      8        0.7127            -nan     0.1000    0.0218
    ##      9        0.6725            -nan     0.1000    0.0200
    ##     10        0.6373            -nan     0.1000    0.0174
    ##     20        0.4363            -nan     0.1000    0.0053
    ##     40        0.3380            -nan     0.1000    0.0011
    ##     60        0.3128            -nan     0.1000    0.0006
    ##     80        0.3012            -nan     0.1000    0.0002
    ##    100        0.2907            -nan     0.1000    0.0001
    ##    120        0.2793            -nan     0.1000   -0.0000
    ##    140        0.2710            -nan     0.1000    0.0001
    ##    150        0.2674            -nan     0.1000    0.0001

``` r
# GBM
pulsar_sgb_up_predictions <- predict(fit.pulsar.sgb_up,test_features)
confusionMatrix(pulsar_sgb_up_predictions, as.factor(test_target), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3163   27
    ##          1   95  294
    ##                                           
    ##                Accuracy : 0.9659          
    ##                  95% CI : (0.9594, 0.9716)
    ##     No Information Rate : 0.9103          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.8094          
    ##                                           
    ##  Mcnemar's Test P-Value : 1.312e-09       
    ##                                           
    ##             Sensitivity : 0.91589         
    ##             Specificity : 0.97084         
    ##          Pos Pred Value : 0.75578         
    ##          Neg Pred Value : 0.99154         
    ##              Prevalence : 0.08969         
    ##          Detection Rate : 0.08215         
    ##    Detection Prevalence : 0.10869         
    ##       Balanced Accuracy : 0.94336         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
# GBM predicted 294 / 321 (91.59%) class 1's accurately
```

Looks like, both KNN & GBM algorithm’s are the best classifiers with
highest accuracy among all.

GBM is the best, with an accuracy(correctly predicting 1) of 91.59% and
an AUC of 94.34

``` r
results <- resamples(list(lda=fit.pulsar.lda_up, logistic=fit.glm_up,
                          svm=fit.pulsar.svmradial_up, knn=fit.pulsar.knn_up, nb=fit.pulsar.nb_up, cart=fit.pulsar.cart_up, bagging=fit.pulsar.treebag_up, rf=fit.pulsar.rf_up, gbm=fit.pulsar.sgb_up))
# Table comparison
print(summary(results))
```

    ## 
    ## Call:
    ## summary.resamples(object = results)
    ## 
    ## Models: lda, logistic, svm, knn, nb, cart, bagging, rf, gbm 
    ## Number of resamples: 30 
    ## 
    ## Accuracy 
    ##               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## lda      0.9664804 0.9736338 0.9769633 0.9762785 0.9783633 0.9881285    0
    ## logistic 0.9615653 0.9657821 0.9678883 0.9686660 0.9704958 0.9832402    0
    ## svm      0.9608939 0.9712047 0.9748516 0.9745790 0.9783482 0.9846369    0
    ## knn      0.9224860 0.9350218 0.9385475 0.9398699 0.9434358 0.9546089    0
    ## nb       0.9330077 0.9427374 0.9462478 0.9458998 0.9495197 0.9608939    0
    ## cart     0.9490223 0.9643668 0.9671788 0.9675253 0.9713787 0.9818436    0
    ## bagging  0.9643855 0.9741440 0.9762570 0.9760690 0.9776536 0.9853352    0
    ## rf       0.9713687 0.9764400 0.9801046 0.9797705 0.9818436 0.9874302    0
    ## gbm      0.9574022 0.9631764 0.9668298 0.9670135 0.9699668 0.9769553    0
    ## 
    ## Kappa 
    ##               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## lda      0.8063120 0.8438639 0.8618359 0.8584399 0.8715227 0.9293082    0
    ## logistic 0.7802365 0.8103896 0.8219893 0.8244516 0.8348988 0.9037893    0
    ## svm      0.7866689 0.8386247 0.8526127 0.8538138 0.8721677 0.9112263    0
    ## knn      0.6445236 0.6835552 0.6996961 0.7029216 0.7233493 0.7688992    0
    ## nb       0.6741628 0.7047081 0.7236511 0.7213426 0.7336644 0.7905733    0
    ## cart     0.7261620 0.7955477 0.8120303 0.8143586 0.8339580 0.8922486    0
    ## bagging  0.7935269 0.8418377 0.8541697 0.8552984 0.8678208 0.9126748    0
    ## rf       0.8259663 0.8541672 0.8766767 0.8753104 0.8907717 0.9233295    0
    ## gbm      0.7725682 0.7996968 0.8148709 0.8175932 0.8336137 0.8714920    0

``` r
# boxplot comparison
print(bwplot(results))
```

![](pulsar_stars_CLASSIFICATION_notebook_files/figure-markdown_github/unnamed-chunk-10-1.png)

``` r
# Dot-plot comparison
print(dotplot(results))
```

![](pulsar_stars_CLASSIFICATION_notebook_files/figure-markdown_github/unnamed-chunk-10-2.png)

``` r
# Plotting Variable importance using Random Forest Model

plot(varImp(fit.pulsar.rf_up))
```

![](pulsar_stars_CLASSIFICATION_notebook_files/figure-markdown_github/unnamed-chunk-11-1.png)
