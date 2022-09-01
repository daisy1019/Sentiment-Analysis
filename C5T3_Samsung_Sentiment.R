## Required doParallel Packaga
library(doParallel)
library(caret)
library(plotly)
library(corrplot)
library(dplyr)
library(tibble)
library(e1071)
library(C50)
library(randomForest)
library(kknn)


# Find how many cores are on your machine
detectCores()  # typically 4 to 6
# [1] 8

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes.
cl <- makeCluster(6)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers()
# [1] 6

# Stop Cluster. After performing your tasks, stop your cluster. 
 stopCluster(cl)

## import data
samsungDF <- read.csv("~/Desktop/smallmatrix_labeled_8d/galaxy_smallmatrix_labeled_9d.csv")

## structure of the data
str(samsungDF)
summary(samsungDF)

## plot the y variable
plot_ly(iphoneDF, x= ~iphoneDF$iphonesentiment, type='histogram')
plot_ly(samsungDF, x= ~samsungDF$galaxysentiment, type='histogram')


# find sentiment counts for iphone & samsung datasets
summary(samsungDF$galaxysentiment)
# 1    2    3    4    5    6 
# 1696  382  450 1175 1417 7791 



### Pre-processing for 2 data sets

## check for missing values
is.na(samsungDF) # no missing values

## change  galaxysentiment to Factor
samsungDF$galaxysentiment <- as.factor (samsungDF$galaxysentiment)
str(samsungDF)



##### Correlation Feature Selection 
# set max print to 1000000
# options(max.print=1000000)

#iphoneDF$iphonesentiment <- as.integer (iphoneDF$iphonesentiment)
# str(iphoneDF)
# samsungDF$galaxysentiment <- as.integer (samsungDF$galaxysentiment)
#str(samsungDF)


Corrmatrix = cor(iphoneDF)
findCorrelation(Corrmatrix, cutoff = .8, verbose = TRUE, names = TRUE)

Corrmatrix2 = cor(samsungDF)
findCorrelation(Corrmatrix2, cutoff = .8, verbose = TRUE, names = TRUE)

iphoneCOR <- select(iphoneDF, c(samsungdisneg, samsungperneg, samsungdispos, htcdisneg, googleperneg, 
                                googleperpos, samsungdisunc, samsungcamunc, htcperpos, nokiacamunc, 
                                nokiadisneg, nokiadispos, nokiaperunc, nokiacampos, nokiadisunc, nokiaperneg,
                                nokiacamneg, iphonedisneg, iphonedispos, sonydispos, iosperunc, iosperneg, ios, 
                                htcphone, iphonesentiment))

samsungCOR <- select(samsungDF, c(samsungdisneg, samsungperneg, samsungdispos, htcdisneg, googleperneg, 
                                  googleperpos, samsungdisunc, samsungcamunc, htcperpos, nokiacamunc, nokiadisneg,
                                  nokiadispos, nokiaperunc, nokiacampos, nokiadisunc, nokiaperneg, nokiacamneg,
                                  iphonedisneg, iphonedispos,  sonyperpos, iosperunc, iosperneg, sonydisneg, ios, 
                                  htcphone, galaxysentiment))



# display correlation matrix
corrplot(cor(iphoneDF), method="shade",shade.col=NA, tl.col="black", tl.srt=45)
corrplot(cor(iphoneCOR), method="shade",shade.col=NA, tl.col="black", tl.srt=45)
corrplot(cor(samsungDF), method="shade",shade.col=NA, tl.col="black", tl.srt=45)
corrplot(cor(samsungCOR), method="shade",shade.col=NA, tl.col="black", tl.srt=45)


## iphoneCOR$featureToRemove <- NULL   # no feature to be removed



#####  nearZeroVar()
### nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: frequency ratio, percentage unique, zero variance and near zero variance 
## get iphoneNZV data set
nzvMetrics <- nearZeroVar(iphoneDF, saveMetrics = TRUE)
nzvMetrics
# freqRatio percentUnique zeroVar   nzv
# iphone             5.041322    0.20812457   FALSE FALSE
# samsunggalaxy     14.127336    0.05395822   FALSE FALSE
# sonyxperia        44.170732    0.03854159   FALSE  TRUE
# nokialumina      497.884615    0.02312495   FALSE  TRUE
# htcphone          11.439614    0.06937486   FALSE FALSE
# ios               27.735294    0.04624990   FALSE  TRUE
# googleandroid     61.247573    0.04624990   FALSE  TRUE
# iphonecampos      10.524697    0.23124952   FALSE FALSE
# samsungcampos     93.625000    0.08479149   FALSE  TRUE
# sonycampos       348.729730    0.05395822   FALSE  TRUE
# nokiacampos     1850.142857    0.08479149   FALSE  TRUE
# htccampos         79.272152    0.16958298   FALSE  TRUE
# iphonecamneg      19.517529    0.13104139   FALSE  TRUE
# samsungcamneg    100.132812    0.06937486   FALSE  TRUE
# sonycamneg      1851.285714    0.04624990   FALSE  TRUE
# nokiacamneg     2158.833333    0.06166654   FALSE  TRUE
# htccamneg         93.444444    0.11562476   FALSE  TRUE
# iphonecamunc      16.764205    0.16187466   FALSE FALSE
# samsungcamunc     74.308140    0.06937486   FALSE  TRUE
# sonycamunc       588.318182    0.03854159   FALSE  TRUE
# nokiacamunc     2591.200000    0.05395822   FALSE  TRUE
# htccamunc         50.548000    0.12333308   FALSE  TRUE
# iphonedispos       6.792440    0.24666615   FALSE FALSE
# samsungdispos     97.061069    0.13104139   FALSE  TRUE
# sonydispos       331.076923    0.06937486   FALSE  TRUE
# nokiadispos     1438.777778    0.09249981   FALSE  TRUE
# htcdispos         64.694301    0.20041625   FALSE  TRUE
# iphonedisneg      10.084428    0.18499961   FALSE FALSE
# samsungdisneg     99.155039    0.10791644   FALSE  TRUE
# sonydisneg      2159.333333    0.06937486   FALSE  TRUE
# nokiadisneg     1850.142857    0.08479149   FALSE  TRUE
# htcdisneg         88.492958    0.14645803   FALSE  TRUE
# iphonedisunc      11.471875    0.20812457   FALSE FALSE
# samsungdisunc     74.255814    0.09249981   FALSE  TRUE
# sonydisunc       719.222222    0.05395822   FALSE  TRUE
# nokiadisunc     1619.375000    0.04624990   FALSE  TRUE
# htcdisunc         50.590361    0.13874971   FALSE  TRUE
# iphoneperpos       9.297834    0.19270793   FALSE FALSE
# samsungperpos     94.200000    0.10791644   FALSE  TRUE
# sonyperpos       416.870968    0.06166654   FALSE  TRUE
# nokiaperpos     2158.000000    0.08479149   FALSE  TRUE
# htcperpos         74.279762    0.19270793   FALSE  TRUE
# iphoneperneg      11.054137    0.16958298   FALSE FALSE
# samsungperneg    101.650794    0.10020812   FALSE  TRUE
# sonyperneg      2159.666667    0.07708317   FALSE  TRUE
# nokiaperneg     3237.250000    0.09249981   FALSE  TRUE
# htcperneg         94.428571    0.15416635   FALSE  TRUE
# iphoneperunc      13.018349    0.12333308   FALSE FALSE
# samsungperunc     86.500000    0.09249981   FALSE  TRUE
# sonyperunc      3240.250000    0.04624990   FALSE  TRUE
# nokiaperunc     1850.428571    0.06937486   FALSE  TRUE
# htcperunc         50.055556    0.15416635   FALSE  TRUE
# iosperpos        153.373494    0.09249981   FALSE  TRUE
# googleperpos      98.592308    0.06937486   FALSE  TRUE
# iosperneg        141.744444    0.09249981   FALSE  TRUE
# googleperneg      99.403101    0.08479149   FALSE  TRUE
# iosperunc        135.893617    0.07708317   FALSE  TRUE
# googleperunc      96.443609    0.07708317   FALSE  TRUE
# iphonesentiment    3.843017    0.04624990   FALSE FALSE


# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(iphoneDF, saveMetrics = FALSE)
nzv

# create a new data set and remove near zero variance features
iphoneNZV <- iphoneDF[,-nzv]
str(iphoneNZV)
# 'data.frame':	12973 obs. of  12 variables:
#   $ iphone         : int  1 1 1 1 1 41 1 1 1 1 ...
# $ samsunggalaxy  : int  0 0 0 0 0 0 0 0 0 0 ...
# $ htcphone       : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonecampos   : int  0 0 0 0 0 1 1 0 0 0 ...
# $ iphonecamunc   : int  0 0 0 0 0 7 1 0 0 0 ...
# $ iphonedispos   : int  0 0 0 0 0 1 13 0 0 0 ...
# $ iphonedisneg   : int  0 0 0 0 0 3 10 0 0 0 ...
# $ iphonedisunc   : int  0 0 0 0 0 4 9 0 0 0 ...
# $ iphoneperpos   : int  0 1 0 1 1 0 5 3 0 0 ...
# $ iphoneperneg   : int  0 0 0 0 0 0 4 1 0 0 ...
# $ iphoneperunc   : int  0 0 0 1 0 0 5 0 0 0 ...
# $ iphonesentiment: int  0 0 0 0 0 4 4 0 0 0 ...


## get samsungNZV data set
nzvMetrics2 <- nearZeroVar(samsungDF, saveMetrics = TRUE)
nzvMetrics2
# freqRatio percentUnique zeroVar   nzv
# iphone             5.039313    0.20912400   FALSE FALSE
# samsunggalaxy     14.090164    0.05421733   FALSE FALSE
# sonyxperia        44.111888    0.03872667   FALSE  TRUE
# nokialumina      495.500000    0.02323600   FALSE  TRUE
# htcphone          11.427740    0.06970800   FALSE FALSE
# ios               27.662132    0.04647200   FALSE  TRUE
# googleandroid     61.248780    0.04647200   FALSE  TRUE
# iphonecampos      10.526217    0.23236000   FALSE FALSE
# samsungcampos     93.176471    0.08519867   FALSE  TRUE
# sonycampos       347.081081    0.05421733   FALSE  TRUE
# nokiacampos     1841.285714    0.08519867   FALSE  TRUE
# htccampos         79.401274    0.17039734   FALSE  TRUE
# iphonecamneg      19.660473    0.13167067   FALSE  TRUE
# samsungcamneg     99.648438    0.06970800   FALSE  TRUE
# sonycamneg      1842.428571    0.04647200   FALSE  TRUE
# nokiacamneg     2148.500000    0.06196267   FALSE  TRUE
# htccamneg         92.992593    0.11618000   FALSE  TRUE
# iphonecamunc      16.805436    0.16265200   FALSE FALSE
# samsungcamunc     73.953488    0.06970800   FALSE  TRUE
# sonycamunc       585.545455    0.03872667   FALSE  TRUE
# nokiacamunc     2578.800000    0.05421733   FALSE  TRUE
# htccamunc         50.510040    0.12392533   FALSE  TRUE
# iphonedispos       6.797333    0.24785067   FALSE FALSE
# samsungdispos     96.595420    0.13167067   FALSE  TRUE
# sonydispos       329.512821    0.06196267   FALSE  TRUE
# nokiadispos     1431.888889    0.09294400   FALSE  TRUE
# htcdispos         64.383420    0.20137867   FALSE  TRUE
# iphonedisneg      10.104816    0.18588800   FALSE FALSE
# samsungdisneg     98.674419    0.10843467   FALSE  TRUE
# sonydisneg      2149.000000    0.06970800   FALSE  TRUE
# nokiadisneg     1841.285714    0.08519867   FALSE  TRUE
# htcdisneg         88.063380    0.14716134   FALSE  TRUE
# iphonedisunc      11.527865    0.20912400   FALSE FALSE
# samsungdisunc     74.333333    0.09294400   FALSE  TRUE
# sonydisunc       757.941176    0.05421733   FALSE  TRUE
# nokiadisunc     1611.625000    0.04647200   FALSE  TRUE
# htcdisunc         50.757085    0.13941600   FALSE  TRUE
# iphoneperpos       9.299184    0.18588800   FALSE FALSE
# samsungperpos     93.748148    0.10843467   FALSE  TRUE
# sonyperpos       414.903226    0.06196267   FALSE  TRUE
# nokiaperpos     2147.666667    0.08519867   FALSE  TRUE
# htcperpos         74.371257    0.19363334   FALSE  TRUE
# iphoneperneg      11.037910    0.17039734   FALSE FALSE
# samsungperneg    101.158730    0.10068933   FALSE  TRUE
# sonyperneg      2149.333333    0.07745333   FALSE  TRUE
# nokiaperneg     3221.750000    0.09294400   FALSE  TRUE
# htcperneg         93.969925    0.15490667   FALSE  TRUE
# iphoneperunc      13.034602    0.12392533   FALSE FALSE
# samsungperunc     86.087838    0.09294400   FALSE  TRUE
# sonyperunc      3225.000000    0.04647200   FALSE  TRUE
# nokiaperunc     1841.571429    0.06970800   FALSE  TRUE
# htcperunc         50.015936    0.15490667   FALSE  TRUE
# iosperpos        152.626506    0.09294400   FALSE  TRUE
# googleperpos      98.115385    0.06970800   FALSE  TRUE
# iosperneg        141.055556    0.09294400   FALSE  TRUE
# googleperneg      98.922481    0.08519867   FALSE  TRUE
# iosperunc        135.234043    0.07745333   FALSE  TRUE
# googleperunc      95.977444    0.07745333   FALSE  TRUE
# galaxysentiment    4.593750    0.04647200   FALSE FALSE

nzv2 <- nearZeroVar(samsungDF, saveMetrics = FALSE) 
nzv2
# [1]  3  4  6  7  9 10 11 12 13 14 15 16 17 19 20 21 22 24 25 26 27 29 30 31 32 34 35 36 37 39 40 41 42 44 45 46 47 49 50 51 52 53 54
# [44] 55 56 57 58

samsungNZV <- samsungDF[,-nzv2]
str(samsungNZV)





##### Recursive Feature Elimination 
# Let's sample the data before using RFE

## change iphonesentiment / galaxysentiment to Factor
# iphoneDF$iphonesentiment <- as.factor (iphoneDF$iphonesentiment)
# str(iphoneDF)
# 
# samsungDF$galaxysentiment <- as.factor (samsungDF$galaxysentiment)
# str(samsungDF)

## get samsungRFE dataset
set.seed(123)
samsungSample <- samsungDF[sample(1:nrow(samsungDF), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 1,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults2 <- rfe(samsungSample[,1:58], 
                   samsungSample$galaxysentiment, 
                   sizes=(1:58), 
                   rfeControl=ctrl)

# Get results
rfeResults2
# Recursive feature selection
# 
# Outer resampling method: Cross-Validated (10 fold, repeated 1 times) 
# 
# Resampling performance over subset size:
#   
#   Variables Accuracy  Kappa AccuracySD KappaSD Selected
# 1   0.7081 0.3107    0.01941 0.05720         
# 2   0.7131 0.3282    0.01712 0.06179         
# 3   0.7202 0.3583    0.02528 0.08079         
# 4   0.7342 0.3941    0.02736 0.08684         
# 5   0.7372 0.4021    0.03080 0.09288         
# 6   0.7372 0.4030    0.03080 0.09443         
# 7   0.7372 0.4024    0.03080 0.09375         
# 8   0.7382 0.4058    0.03398 0.10046         
# 9   0.7443 0.4393    0.03523 0.09294         
# 10   0.7493 0.4508    0.03404 0.09161         
# 11   0.7493 0.4491    0.03481 0.09103         
# 12   0.7432 0.4370    0.03534 0.09299         
# 13   0.7483 0.4485    0.03519 0.09238         
# 14   0.7473 0.4462    0.03574 0.09375         
# 15   0.7503 0.4525    0.03503 0.09341         
# 16   0.7503 0.4595    0.03258 0.08483         
# 17   0.7532 0.4656    0.02950 0.07612        *
#   18   0.7512 0.4614    0.02950 0.07456         
# 19   0.7523 0.4597    0.03275 0.08697         
# 20   0.7513 0.4568    0.03172 0.08614         
# 21   0.7503 0.4539    0.03543 0.09235         
# 22   0.7473 0.4441    0.03759 0.10089         
# 23   0.7452 0.4394    0.03540 0.09843         
# 24   0.7472 0.4429    0.03380 0.09292         
# 25   0.7513 0.4571    0.03271 0.08999         
# 26   0.7492 0.4512    0.03282 0.09114         
# 27   0.7492 0.4499    0.03288 0.09048         
# 28   0.7492 0.4501    0.03160 0.08681         
# 29   0.7462 0.4404    0.03236 0.08835         
# 30   0.7452 0.4374    0.03465 0.09304         
# 31   0.7443 0.4318    0.03706 0.09852         
# 32   0.7433 0.4302    0.03617 0.09689         
# 33   0.7433 0.4289    0.03732 0.09885         
# 34   0.7423 0.4273    0.03641 0.09718         
# 35   0.7452 0.4334    0.03480 0.09369         
# 36   0.7432 0.4313    0.03591 0.09816         
# 37   0.7472 0.4405    0.03658 0.09613         
# 38   0.7452 0.4362    0.03329 0.08970         
# 39   0.7443 0.4319    0.03625 0.09554         
# 40   0.7453 0.4348    0.03595 0.09510         
# 41   0.7443 0.4319    0.03625 0.09554         
# 42   0.7443 0.4319    0.03625 0.09554         
# 43   0.7432 0.4275    0.03503 0.09454         
# 44   0.7452 0.4332    0.03480 0.09388         
# 45   0.7452 0.4321    0.03313 0.08923         
# 46   0.7443 0.4319    0.03625 0.09554         
# 47   0.7452 0.4320    0.03267 0.08942         
# 48   0.7433 0.4281    0.03524 0.09444         
# 49   0.7442 0.4317    0.03389 0.09220         
# 50   0.7423 0.4266    0.03852 0.10178         
# 51   0.7433 0.4296    0.03751 0.09865         
# 52   0.7433 0.4296    0.03751 0.09865         
# 53   0.7443 0.4311    0.03686 0.09699         
# 54   0.7443 0.4309    0.03615 0.09708         
# 55   0.7423 0.4269    0.03661 0.09592         
# 56   0.7443 0.4309    0.03615 0.09708         
# 57   0.7433 0.4282    0.03524 0.09434         
# 58   0.7443 0.4297    0.03454 0.09264         
# 
# The top 5 variables (out of 17):
#   iphone, samsunggalaxy, googleandroid, htcphone, iphonedisunc

# Plot results
plot(rfeResults2, type=c("g", "o"))


samsungRFE <- samsungDF[,predictors(rfeResults2)]

# add the dependent variable to iphoneRFE
samsungRFE$galaxysentiment <- samsungDF$galaxysentiment

# review outcome
str(samsungRFE)  # $ galaxysentiment: Factor w/ 6 levels "1","2","3","4",..: 6 4 4 1 2 1 4 6 6 6 ...

predictors(rfeResults2)
# [1] "iphone"        "samsunggalaxy" "googleandroid" "iphonedisneg"  "iphonedisunc"  "iphonedispos"  "htcphone"     
# [8] "iphoneperpos"  "iphonecamneg"  "ios"           "sonyxperia"    "iphoneperunc"  "iphonecamunc"  "iphoneperneg" 
# [15] "htcperpos"     "iphonecampos"  "htccampos"     "htcperneg"     "htcdisunc"     "htccamneg"     "htcperunc"    
# [22] "htccamunc"     "iosperneg"     "samsungdispos" "samsungcampos"



## convert sentiment attribute to Factor before modeling
str(samsungCOR)
str(samsungNZV)
str(samsungRFE)

iphoneCOR$iphonesentiment <- as.factor (iphoneCOR$iphonesentiment) # $ iphonesentiment: Factor w/ 6 levels "1","2","3","4",..: 1 1 1 1 1 5 5 1 1 1 ...
samsungCOR$galaxysentiment <- as.factor (samsungCOR$galaxysentiment) # $ galaxysentiment: Factor w/ 6 levels "1","2","3","4",..: 6 4 4 1 2 1 4 6 6 6 ...
samsungNZV$galaxysentiment <- as.factor (samsungNZV$galaxysentiment) # $ galaxysentiment: Factor w/ 6 levels "1","2","3","4",..: 6 4 4 1 2 1 4 6 6 6 ...



##### Build Training / Predictive Models on Original Samsung Dataset #####

### KKNN Model
set.seed(123)

#create a 20% sample of the data (not creating sample data)
# iphoneDF <- iphoneDF[sample(1:nrow(iphoneDF), 3000, replace=FALSE),]
# str(iphoneDFsample)


# define an 70%/30% train/test split of the dataset
inTraining2 <- createDataPartition(samsungDF$galaxysentiment, p = .70, list = FALSE)
training2 <- samsungDF[inTraining,]
testing2 <- samsungDF[-inTraining,]

#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

# use this to omit missling value in predictor
data <- na.omit(training2)

# train KNN model 
system.time(kknnFit_samsungDF <- train(galaxysentiment~., data = data, method = "kknn", 
                                      trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 3.174   0.079  13.390 

kknnFit_samsungDF
# k-Nearest Neighbors 
# 
# 9044 samples
# 58 predictor
# 6 classes: '0', '1', '2', '3', '4', '5' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8141, 8139, 8139, 8139, 8139, 8141, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.6645338  0.4133324
# 
# Tuning parameter 'kmax' was held constant at a value of 5
# Tuning parameter 'distance' was held constant at a value of
# 2
# Tuning parameter 'kernel' was held constant at a value of optimal



## Random Forest
system.time(rfFit_samsungDF <- train(galaxysentiment~., data = data, method = "rf", 
                                    trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 11.258   0.288  37.545 

rfFit_samsungDF
# Random Forest 
# 
# 9044 samples
# 58 predictor
# 6 classes: '0', '1', '2', '3', '4', '5' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8140, 8138, 8139, 8141, 8140, 8141, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7583009  0.5014555
# 
# Tuning parameter 'mtry' was held constant at a value of 7





## C5.0
system.time(c50Fit_samsungDF <- train(galaxysentiment~., data = data, method = "C5.0", 
                                     trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 0.763   0.036   5.030 


c50Fit_samsungDF
# C5.0 
# 
# 9044 samples
# 58 predictor
# 6 classes: '0', '1', '2', '3', '4', '5' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8140, 8140, 8140, 8140, 8141, 8138, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7679087  0.5308357
# rules   TRUE   0.7674677  0.5303594
# tree   FALSE   0.7672456  0.5305268
# tree    TRUE   0.7666937  0.5297474
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.
 

## SVM(e1071)
system.time(svmFit_samsungDF <- train(galaxysentiment~., data = data, method = "svmLinear", 
                                     trControl=fitControl, tuneLength = 1))  
# user  system elapsed 
# 5.154   0.108  18.880 

svmFit_samsungDF 
# Support Vector Machines with Linear Kernel 
# 
# 9044 samples
# 58 predictor
# 6 classes: '0', '1', '2', '3', '4', '5' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8139, 8141, 8140, 8142, 8139, 8139, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7188179  0.4056713
# 
# Tuning parameter 'C' was held constant at a value of 1










##### prediction base on the model #####
kknnFit_samsungDFpreds <- predict(kknnFit_samsungDF, samsungDF)
confusionMatrix(kknnFit_samsungDFpreds, samsungDF$galaxysentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0 1236    6    7   12   23  125
# 1   15   55   12   12   31  211
# 2    9   10  107   10   20  120
# 3   18    6    9  762   26  199
# 4   78   53   43   64  642  890
# 5  340  252  272  315  675 6246
# 
# Overall Statistics
# 
# Accuracy : 0.7008          
# 95% CI : (0.6928, 0.7087)
# No Information Rate : 0.6034          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4871          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.72877  0.14398 0.237778  0.64851  0.45307   0.8017
# Specificity           0.98457  0.97757 0.986438  0.97802  0.90186   0.6379
# Pos Pred Value        0.87722  0.16369 0.387681  0.74706  0.36271   0.7711
# Neg Pred Value        0.96001  0.97400 0.972853  0.96527  0.93044   0.6789
# Prevalence            0.13136  0.02959 0.034854  0.09101  0.10975   0.6034
# Detection Rate        0.09573  0.00426 0.008288  0.05902  0.04973   0.4838
# Detection Prevalence  0.10913  0.02602 0.021377  0.07900  0.13709   0.6274
# Balanced Accuracy     0.85667  0.56078 0.612108  0.81326  0.67747   0.7198





rfFit_samsungDFpreds <- predict(rfFit_samsungDF, samsungDF)
confusionMatrix(rfFit_samsungDFpreds, samsungDF$galaxysentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0 1191    3    6    8   20   84
# 1    0    3    0    0    0    0
# 2    2    0   57    1    2    4
# 3    5    2    1  486    8   34
# 4    8    1    1    2  434   19
# 5  490  373  385  678  953 7650
# 
# Overall Statistics
# 
# Accuracy : 0.7607         
# 95% CI : (0.7532, 0.768)
# No Information Rate : 0.6034         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.5082         
# 
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
#                      Class: 0  Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.70224 0.0078534 0.126667  0.41362  0.30628   0.9819
# Specificity           0.98921 1.0000000 0.999278  0.99574  0.99730   0.4377
# Pos Pred Value        0.90777 1.0000000 0.863636  0.90672  0.93333   0.7266
# Neg Pred Value        0.95646 0.9706384 0.969404  0.94432  0.92102   0.9408
# Prevalence            0.13136 0.0295872 0.034854  0.09101  0.10975   0.6034
# Detection Rate        0.09225 0.0002324 0.004415  0.03764  0.03361   0.5925
# Detection Prevalence  0.10162 0.0002324 0.005112  0.04151  0.03602   0.8155
# Balanced Accuracy     0.84573 0.5039267 0.562972  0.70468  0.65179   0.7098




c50Fit_samsungDFpreds <- predict(c50Fit_samsungDF, samsungDF)
confusionMatrix(c50Fit_samsungDFpreds, samsungDF$galaxysentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0 1166    3    6   11   24   89
# 1    0    0    0    0    0    0
# 2    2    0   53    1    2    5
# 3    7    7    3  698   19   75
# 4   12    1    2    3  425   42
# 5  509  371  386  462  947 7580
# 
# Overall Statistics
# 
# Accuracy : 0.7685          
# 95% CI : (0.7611, 0.7757)
# No Information Rate : 0.6034          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5346          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.68750  0.00000 0.117778  0.59404  0.29993   0.9729
# Specificity           0.98814  1.00000 0.999197  0.99054  0.99478   0.4775
# Pos Pred Value        0.89761      NaN 0.841270  0.86279  0.87629   0.7392
# Neg Pred Value        0.95436  0.97041 0.969100  0.96059  0.92017   0.9206
# Prevalence            0.13136  0.02959 0.034854  0.09101  0.10975   0.6034
# Detection Rate        0.09031  0.00000 0.004105  0.05406  0.03292   0.5871
# Detection Prevalence  0.10061  0.00000 0.004880  0.06266  0.03756   0.7943
# Balanced Accuracy     0.83782  0.50000 0.558488  0.79229  0.64735   0.7252


svmFit_samsungDFpreds <- predict(svmFit_samsungDF, samsungDF)
confusionMatrix(svmFit_samsungDFpreds, samsungDF$galaxysentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0 1144    9    9   34   45  195
# 1    0    1    1    0    1    0
# 2    1    1   10    0    0    2
# 3    8    3   46  331    8   29
# 4    8    1    3    2  247   11
# 5  535  367  381  808 1116 7554
# 
# Overall Statistics
# 
# Accuracy : 0.7193         
# 95% CI : (0.7115, 0.727)
# No Information Rate : 0.6034         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.4117         
# 
# Mcnemar's Test P-Value : < 2.2e-16      
# 
# Statistics by Class:
# 
#                      Class: 0  Class: 1  Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.67453 2.618e-03 0.0222222  0.28170  0.17431   0.9696
# Specificity           0.97396 9.998e-01 0.9996790  0.99199  0.99782   0.3736
# Pos Pred Value        0.79666 3.333e-01 0.7142857  0.77882  0.90809   0.7020
# Neg Pred Value        0.95190 9.705e-01 0.9658835  0.93240  0.90743   0.8898
# Prevalence            0.13136 2.959e-02 0.0348540  0.09101  0.10975   0.6034
# Detection Rate        0.08861 7.745e-05 0.0007745  0.02564  0.01913   0.5851
# Detection Prevalence  0.11122 2.324e-04 0.0010843  0.03292  0.02107   0.8335
# Balanced Accuracy     0.82425 5.012e-01 0.5109506  0.63685  0.58607   0.6716



###  postResample Values
postResample(kknnFit_samsungDFpreds, obs = samsungDF$galaxysentiment)
# Accuracy     Kappa 
# 0.7007978 0.4870813 

postResample(rfFit_samsungDFpreds, obs = samsungDF$galaxysentiment)
# Accuracy     Kappa 
# 0.7606692 0.5081844 

postResample(c50Fit_samsungDFpreds, obs = samsungDF$galaxysentiment)
# Accuracy     Kappa 
# 0.7684920 0.5346444

postResample(svmFit_samsungDFpreds, obs = samsungDF$galaxysentiment)
# Accuracy     Kappa 
# 0.7193091 0.4116551


## feature importance using RF on iphoneRFE
varImp(c50Fit_samsungRFE, scale = FALSE)
plot(varImp(c50Fit_samsungRFE))
# C5.0 variable importance
# 
# only 20 most important variables shown (out of 25)
# 
# Overall
# samsungcampos   99.72
# iphone          16.18
# iphoneperpos    11.94
# iphonedispos    11.71
# googleandroid   10.15
# iphonedisneg    10.00
# sonyxperia       7.53
# iphonedisunc     7.07
# iphoneperunc     4.63
# iphonecampos     4.56
# htccampos        4.48
# ios              3.68
# htcphone         3.06
# iphoneperneg     2.06
# htcperpos        1.86
# iphonecamneg     1.64
# samsunggalaxy    1.59
# iphonecamunc     1.51
# samsungdispos    0.00
# htccamneg        0.00








++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
##### Modeling with samsungCOR / samsungNZV / samsungRFE
  
## #### train with dataset samsungCOR
set.seed(123)

#create a 20% sample of the data
# iphoneCOR <- iphoneCOR[sample(1:nrow(iphoneCOR), 150, replace=FALSE),] # no need to sample according to POA

# use this to omit missling value in predictor
# data <- na.omit(training2)

# define an 70%/30% train/test split of the dataset
inTrainingCORsamsung <- createDataPartition(samsungCOR$galaxysentiment, p = .70, list = FALSE)
trainingCORsamsung <- samsungCOR[inTraining,]
testingCORsamsung <- samsungCOR[-inTraining,]

#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats = 5)

##  Model Random Forest
system.time(c50Fit_samsungCOR <- train(galaxysentiment~., data = na.omit(trainingCORsamsung), method = "C5.0", 
                                     trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 0.640   0.047   8.682  

c50Fit_samsungCOR
# C5.0 
# 
# 9044 samples
# 25 predictor
# 6 classes: '1', '2', '3', '4', '5', '6' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 5 times) 
# Summary of sample sizes: 8139, 8140, 8138, 8139, 8139, 8142, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.6847871  0.3020627
# rules   TRUE   0.6844993  0.3011212
# tree   FALSE   0.6840357  0.3052844
# tree    TRUE   0.6837042  0.3040543
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.



## iphone NZV
inTrainingNZVsamsung <- createDataPartition(samsungNZV$galaxysentiment, p = .70, list = FALSE)
trainingNZVsamsung <- samsungNZV[inTraining,]
testingNZVsamsung <- samsungNZV[-inTraining,]

#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats = 5)

##  Model Random Forest
system.time(c50Fit_samsungNZV <- train(galaxysentiment~., data = na.omit(trainingNZVsamsung), method = "C5.0", 
                                       trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 0.580   0.038   6.038

c50Fit_samsungNZV
# C5.0 
# 
# 9044 samples
# 11 predictor
# 6 classes: '1', '2', '3', '4', '5', '6' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 5 times) 
# Summary of sample sizes: 8141, 8138, 8139, 8139, 8141, 8138, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7530753  0.4949592
# rules   TRUE   0.7525890  0.4937774
# tree   FALSE   0.7528324  0.4956471
# tree    TRUE   0.7525894  0.4952636
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.



#### iphoneRFE
inTrainingRFEsamsung <- createDataPartition(samsungRFE$galaxysentiment, p = .70, list = FALSE)
trainingRFEsamsung <- samsungRFE[inTraining,]
testingRFEsamsung <- samsungRFE[-inTraining,]

#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats = 5)

##  Model Random Forest
system.time(c50Fit_samsungRFE <- train(galaxysentiment~., data = na.omit(trainingRFEsamsung), method = "C5.0", 
                                       trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 0.685   0.050  10.782 

c50Fit_samsungRFE
# C5.0 
# 
# 9044 samples
# 25 predictor
# 6 classes: '0', '1', '2', '3', '4', '5' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 5 times) 
# Summary of sample sizes: 8140, 8140, 8139, 8138, 8141, 8140, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7673369  0.5297335
# rules   TRUE   0.7667178  0.5285403
# tree   FALSE   0.7661863  0.5287806
# tree    TRUE   0.7660542  0.5284865
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.



#### postResample & ConfusionMatrix for Model RF with iphoneCOR/NZV/RFE
## samsungCOR
c50Fit_samsungCORpreds <- predict(c50Fit_samsungCOR, samsungCOR)
postResample(c50Fit_samsungCORpreds, obs = samsungCOR$galaxysentiment)
# Accuracy     Kappa 
# 0.6859267 0.3109697

confusionMatrix(c50Fit_samsungCORpreds, samsungCOR$galaxysentiment)
#  Accuracy : 0.6859 
#  Kappa : 0.311  


## samsungNZV
c50Fit_samsungNZVpreds <- predict(c50Fit_samsungNZV, samsungNZV)
postResample(c50Fit_samsungNZVpreds, obs = samsungNZV$galaxysentiment)
# Accuracy     Kappa 
# 0.7574162 0.5090933 

confusionMatrix(c50Fit_samsungNZVpreds, samsungNZV$galaxysentiment)
# Accuracy : 0.7574 
# Kappa : 0.5091


## samsungRFE
c50Fit_samsungRFEpreds <- predict(c50Fit_samsungRFE, samsungRFE)
postResample(c50Fit_samsungRFEpreds, obs = samsungRFE$galaxysentiment)
# Accuracy     Kappa 
# 0.7707381 0.5408266 

confusionMatrix(c50Fit_samsungRFEpreds, samsungRFE$galaxysentiment)
# Accuracy : 0.7707
# Kappa : 0.5408 




# ******************************************************************************

# prediction with LargeMatrix dataset
# import data
library(caret)
samsungLargeDF <- read.csv("~/Desktop/samsungLargeMatrix.csv")

# data preprocessing
str(samsungLargeDF)
summary(samsungLargeDF)



### change empty column iphonesentiment to Factor
# Code to build iphoneLargeDF_vector
samsungLargeDF_vector <- c("0", "1", "2", "3", "4", "5")
factor_samsungLargeDF_vector <- factor(samsungLargeDF_vector)

# Specify the levels of factor_survey_vector
levels(samsungLargeDF_vector) <- 6

factor_samsungLargeDF_vector

samsungLargeDF$galaxysentiment <- factor_samsungLargeDF_vector
str(samsungLargeDF$galaxysentiment)

# remove extra columns
# iphoneLargeDF <- subset (iphoneLargeDF, select = -iphonesentiment5)
# iphoneLargeDF <- subset (iphoneLargeDF, select = -iphonesentiment_vector)
# str(iphoneLargeDF)

is.na(samsungLargeDF)

str(samsungLargeDF)

# create new data set with rfe recommended features
samsungLargeDFRFE <- samsungLargeDF[,predictors(rfeResults2)]
str(samsungLargeDFRFE)
# add the dependent variable to iphoneRFE
samsungLargeDFRFE$galaxysentiment <- samsungLargeDF$galaxysentiment

samsungLargepreds <- predict(c50Fit_samsungRFE, samsungLargeDFRFE)
summary(samsungLargepreds)
# 0     1     2     3     4     5 
# 13643     0  1280  1984   209  7664 


stopCluster(cl)

