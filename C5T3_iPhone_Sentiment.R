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
# stopCluster(cl)

## import data
iphoneDF <- read.csv("~/Desktop/smallmatrix_labeled_8d/iphone_smallmatrix_labeled_8d.csv")
samsungDF <- read.csv("~/Desktop/smallmatrix_labeled_8d/galaxy_smallmatrix_labeled_9d.csv")

## structure of the data
str(iphoneDF) # all integer values
summary(iphoneDF)
str(samsungDF)
summary(samsungDF)

## plot the y variable
plot_ly(iphoneDF, x= ~iphoneDF$iphonesentiment, type='histogram')
plot_ly(samsungDF, x= ~samsungDF$galaxysentiment, type='histogram')

# find sentiment counts for iphone & samsung datasets
summary(iphoneDF$iphonesentiment)
# 0    1    2    3    4    5 
# 1962  390  454 1188 1439 7540 
summary(samsungDF$galaxysentiment)
# 1    2    3    4    5    6 
# 1696  382  450 1175 1417 7791 



### Pre-processing for 2 data sets

## check for missing values
is.na(iphoneDF) # no missing values
is.na(samsungDF) # no missing values

## change iphonesentiment / galaxysentiment to Factor
iphoneDF$iphonesentiment <- as.factor (iphoneDF$iphonesentiment)
str(iphoneDF)

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

## get iphoneRFE dataset
set.seed(123)
iphoneSample <- iphoneDF[sample(1:nrow(iphoneDF), 1000, replace = FALSE), ]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs,
                   method =  "repeatedcv",
                   repeats = 1,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[, 1:58], 
                  iphoneSample$iphonesentiment,
                  sizes = (1:58),
                  rfeControl = ctrl)

# Get results
rfeResults
# Recursive feature selection
# 
# Outer resampling method: Cross-Validated (10 fold, repeated 1 times) 
# 
# Resampling performance over subset size:
#   
#   Variables Accuracy  Kappa AccuracySD KappaSD Selected
# 1   0.6671 0.2612    0.02421 0.06526         
# 2   0.6741 0.2959    0.02940 0.08485         
# 3   0.6822 0.3214    0.03141 0.08772         
# 4   0.7161 0.4179    0.03433 0.08579         
# 5   0.7171 0.4225    0.03070 0.08287         
# 6   0.7342 0.4585    0.02798 0.06717         
# 7   0.7381 0.4738    0.02729 0.06269         
# 8   0.7380 0.4699    0.02787 0.07012         
# 9   0.7540 0.5121    0.02245 0.05360         
# 10   0.7580 0.5196    0.02681 0.06425         
# 11   0.7621 0.5247    0.02360 0.05769         
# 12   0.7632 0.5299    0.03520 0.08023         
# 13   0.7631 0.5284    0.02697 0.06325         
# 14   0.7682 0.5384    0.03661 0.08206        *
#   15   0.7641 0.5299    0.02553 0.05753         
# 16   0.7662 0.5383    0.03219 0.07396         
# 17   0.7672 0.5395    0.03613 0.07904         
# 18   0.7662 0.5355    0.03493 0.07783         
# 19   0.7682 0.5390    0.03422 0.07689         
# 20   0.7631 0.5269    0.02784 0.06235         
# 21   0.7621 0.5236    0.02452 0.05532         
# 22   0.7601 0.5180    0.02496 0.05741         
# 23   0.7551 0.5053    0.02832 0.06669         
# 24   0.7561 0.5086    0.02945 0.06922         
# 25   0.7621 0.5256    0.02693 0.06199         
# 26   0.7621 0.5243    0.02738 0.06184         
# 27   0.7621 0.5228    0.02584 0.06078         
# 28   0.7611 0.5204    0.02526 0.05941         
# 29   0.7561 0.5078    0.02583 0.06110         
# 30   0.7571 0.5108    0.02586 0.06063         
# 31   0.7511 0.4963    0.02998 0.06882         
# 32   0.7531 0.5008    0.02865 0.06654         
# 33   0.7511 0.4962    0.02910 0.06781         
# 34   0.7511 0.4957    0.02802 0.06550         
# 35   0.7501 0.4928    0.02863 0.06623         
# 36   0.7541 0.5033    0.02870 0.06669         
# 37   0.7521 0.4978    0.02846 0.06665         
# 38   0.7521 0.4978    0.02846 0.06665         
# 39   0.7511 0.4957    0.02802 0.06550         
# 40   0.7491 0.4913    0.02957 0.06707         
# 41   0.7511 0.4956    0.02802 0.06614         
# 42   0.7501 0.4936    0.02863 0.06664         
# 43   0.7501 0.4936    0.02863 0.06664         
# 44   0.7501 0.4933    0.02863 0.06602         
# 45   0.7501 0.4933    0.02863 0.06602         
# 46   0.7471 0.4860    0.03063 0.07003         
# 47   0.7451 0.4803    0.03196 0.07413         
# 48   0.7461 0.4815    0.03171 0.07665         
# 49   0.7491 0.4918    0.02957 0.06675         
# 50   0.7491 0.4918    0.02957 0.06675         
# 51   0.7501 0.4928    0.02863 0.06623         
# 52   0.7491 0.4905    0.03032 0.07110         
# 53   0.7491 0.4897    0.02957 0.06838         
# 54   0.7430 0.4769    0.02833 0.06700         
# 55   0.7461 0.4813    0.03087 0.07335         
# 56   0.7451 0.4791    0.03242 0.07686         
# 57   0.7381 0.4628    0.02799 0.06878         
# 58   0.7401 0.4677    0.02926 0.06964         
# 
# The top 5 variables (out of 20):
#   iphone, samsunggalaxy, googleandroid, iphonedispos, iphonedisunc


# Plot results
plot(rfeResults, type=c("g", "o"))

# create new data set with rfe recommended features
iphoneRFE <- iphoneDF[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphoneDF$iphonesentiment

iphoneRFE

predictors(rfeResults)

# review outcome
str(iphoneRFE) # $ iphonesentiment: Factor w/ 6 levels "0","1","2","3",..: 1 1 1 1 1 5 5 1 1 1 ...



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

# predictors
predictors(rfeResults2)
# [1] "iphone"        "samsunggalaxy" "googleandroid" "iphonedisneg"  "iphonedisunc"  "iphonedispos"  "htcphone"     
# [8] "iphoneperpos"  "iphonecamneg"  "ios"           "sonyxperia"    "iphoneperunc"  "iphonecamunc"  "iphoneperneg" 
# [15] "htcperpos"     "iphonecampos"  "htccampos"     "htcperneg"     "htcdisunc"     "htccamneg"     "htcperunc"    
# [22] "htccamunc"     "iosperneg"     "samsungdispos" "samsungcampos"


## convert sentiment attribute to Factor before modeling
str(iphoneCOR)
str(iphoneNZV)
str(iphoneRFE)
str(samsungCOR)
str(samsungNZV)
str(samsungRFE)

iphoneCOR$iphonesentiment <- as.factor (iphoneCOR$iphonesentiment) # $ iphonesentiment: Factor w/ 6 levels "1","2","3","4",..: 1 1 1 1 1 5 5 1 1 1 ...
samsungCOR$galaxysentiment <- as.factor (samsungCOR$galaxysentiment) # $ galaxysentiment: Factor w/ 6 levels "1","2","3","4",..: 6 4 4 1 2 1 4 6 6 6 ...
samsungNZV$galaxysentiment <- as.factor (samsungNZV$galaxysentiment) # $ galaxysentiment: Factor w/ 6 levels "1","2","3","4",..: 6 4 4 1 2 1 4 6 6 6 ...




##### Build Training / Predictive Models #####

### KKNN Model
set.seed(123)

#create a 20% sample of the data (not creating sample data)
# iphoneDF <- iphoneDF[sample(1:nrow(iphoneDF), 3000, replace=FALSE),]
# str(iphoneDFsample)


# define an 70%/30% train/test split of the dataset
inTraining <- createDataPartition(iphoneDF$iphonesentiment, p = .70, list = FALSE)
training <- iphoneDF[inTraining,]
testing <- iphoneDF[-inTraining,]

#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)


# train KNN model 
system.time(kknnFit_iphoneDF <- train(iphonesentiment~., data = training, method = "kknn", 
                                      trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 3.494   0.066  12.376 

kknnFit_iphoneDF
# k-Nearest Neighbors 
# 
# 9083 samples
# 58 predictor
# 6 classes: '0', '1', '2', '3', '4', '5' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8176, 8174, 8175, 8175, 8177, 8174, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.3087063  0.1521264
# 
# Tuning parameter 'kmax' was held constant at a value of 5
# Tuning parameter 'distance' was held constant at a value of
# 2
# Tuning parameter 'kernel' was held constant at a value of optimal



## Random Forest
system.time(rfFit_iphoneDF <- train(iphonesentiment~., data = training, method = "rf", 
                                      trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 10.853   0.286  36.669 

rfFit_iphoneDF
# Random Forest 
# 
# 9083 samples
# 58 predictor
# 6 classes: '0', '1', '2', '3', '4', '5' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8174, 8175, 8175, 8175, 8174, 8176, ... 
# Resampling results:
#   
#   Accuracy   Kappa   
# 0.7695714  0.548053
# 
# Tuning parameter 'mtry' was held constant at a value of 7


## C5.0
system.time(c50Fit_iphoneDF <- train(iphonesentiment~., data = training, method = "C5.0", 
                                    trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 0.381   0.007   0.742 

c50Fit_iphoneDF
# C5.0 
# 
# 703 samples
# 58 predictor
# 6 classes: '1', '2', '3', '4', '5', '6' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 633, 632, 634, 633, 633, 631, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7482029  0.5096496
# rules   TRUE   0.7396096  0.4930975
# tree   FALSE   0.7510002  0.5167924
# tree    TRUE   0.7396096  0.4930975
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = tree and winnow = FALSE.


## SVM(e1071)
system.time(svmFit_iphoneDF <- train(iphonesentiment~., data = training, method = "svmLinear", 
                                     trControl=fitControl, tuneLength = 1)) 
# user  system elapsed 
# 5.552   0.121  18.229 

svmFit_iphoneDF 
# Support Vector Machines with Linear Kernel 
# 
# 9083 samples
# 58 predictor
# 6 classes: '0', '1', '2', '3', '4', '5' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8174, 8173, 8177, 8176, 8173, 8175, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7078091  0.4100513
# 
# Tuning parameter 'C' was held constant at a value of 1



#### Variable Importance
varImp(kknnFit_iphoneDF, scale = FALSE)
plot(varImp(kknnFit_iphoneDF))
# ROC curve variable importance
# 
# variables are sorted by maximum importance across the classes
# only 20 most important variables shown (out of 58)
# 
# X0     X1     X2     X3     X4     X5
# iphone        0.6637 0.6640 0.7186 0.6637 0.6637 0.6640
# htcphone      0.7076 0.7076 0.7076 0.7076 0.7076 0.7071
# iphonedisunc  0.5099 0.5908 0.6723 0.5071 0.5071 0.5908
# iphonedisneg  0.5116 0.5137 0.6658 0.5052 0.5129 0.5137
# iphonedispos  0.5122 0.5926 0.6513 0.5024 0.5154 0.5926
# samsunggalaxy 0.6497 0.6492 0.6492 0.6492 0.6492 0.6497
# iphonecamunc  0.5297 0.5297 0.6387 0.5345 0.5297 0.5281
# iphonecamneg  0.5341 0.5315 0.6293 0.5253 0.5244 0.5341
# iphonecampos  0.5314 0.5392 0.6122 0.5336 0.5314 0.5392
# htcdispos     0.5945 0.5945 0.5945 0.5951 0.5945 0.5929
# htccampos     0.5900 0.5896 0.5896 0.5908 0.5896 0.5900
# htcperpos     0.5907 0.5907 0.5907 0.5907 0.5907 0.5894
# iphoneperpos  0.5258 0.5274 0.5790 0.5302 0.5224 0.5274
# htcdisneg     0.5790 0.5790 0.5790 0.5790 0.5790 0.5776
# htcperneg     0.5779 0.5779 0.5779 0.5779 0.5779 0.5765
# htcdisunc     0.5717 0.5717 0.5717 0.5717 0.5717 0.5700
# htccamneg     0.5713 0.5713 0.5713 0.5713 0.5713 0.5713
# iphoneperneg  0.5195 0.5247 0.5708 0.5169 0.5179 0.5247
# googleandroid 0.5383 0.5285 0.5285 0.5298 0.5705 0.5383
# iphoneperunc  0.5117 0.5233 0.5695 0.5079 0.5141 0.5233

varImp(rfFit_iphoneDF, scale = FALSE)
# rf variable importance
# 
# only 20 most important variables shown (out of 58)
# 
# Overall
# iphone         455.07
# samsunggalaxy  204.52
# htcphone       172.91
# iphonedisunc   140.12
# googleandroid  130.73
# iphonedisneg   110.41
# iphonedispos    85.15
# iphonecamunc    72.08
# iphoneperpos    62.34
# sonyxperia      57.59
# ios             54.94
# iphonecamneg    53.45
# iphonecampos    45.75
# htccampos       44.13
# htcdispos       42.19
# iphoneperneg    39.74
# iphoneperunc    37.55
# htcperpos       29.71
# htcperneg       20.62
# htcdisneg       20.43

varImp(c50Fit_iphoneDF, scale = FALSE)
# C5.0 variable importance
# 
# only 20 most important variables shown (out of 58)
# 
# Overall
# iphone         100.00
# ios             92.46
# samsunggalaxy   90.47
# htcphone        88.62
# iphonedispos    85.63
# iphonedisunc    81.79
# iphonecampos    73.97
# googleandroid   72.97
# iphonedisneg    10.67
# iphoneperpos     5.41
# sonydisunc       0.00
# iosperpos        0.00
# nokiadispos      0.00
# nokiaperneg      0.00
# iphoneperneg     0.00
# iosperunc        0.00
# sonyperunc       0.00
# htccampos        0.00
# htccamunc        0.00
# samsungperpos    0.00

varImp(svmFit_iphoneDF, scale = FALSE)
# ROC curve variable importance
# 
# variables are sorted by maximum importance across the classes
# only 20 most important variables shown (out of 58)
# 
# X1     X2     X3     X4     X5     X6
# htcphone      0.7379 0.7379 0.7379 0.7379 0.7379 0.7379
# iphone        0.6648 0.6992 0.7321 0.6682 0.6648 0.6992
# iphonedisunc  0.5340 0.5883 0.6857 0.5340 0.5340 0.5883
# iphonedisneg  0.5284 0.5111 0.6706 0.5262 0.5420 0.5284
# iphonedispos  0.5134 0.6198 0.6648 0.5120 0.5112 0.6198
# iphonecamunc  0.5388 0.5388 0.6512 0.5388 0.5388 0.5241
# samsunggalaxy 0.6456 0.6456 0.6456 0.6456 0.6456 0.6456
# iphonecamneg  0.5437 0.5437 0.6285 0.5437 0.5437 0.5366
# htccampos     0.6165 0.6165 0.6165 0.6165 0.6165 0.6165
# iphonecampos  0.5457 0.5457 0.6088 0.5457 0.5457 0.5307
# iphoneperpos  0.5936 0.5123 0.5960 0.5621 0.6078 0.5936
# htcdispos     0.6068 0.6068 0.6068 0.6068 0.6068 0.6068
# htcperpos     0.6019 0.6019 0.6019 0.6019 0.6019 0.6019
# iphoneperunc  0.5340 0.5340 0.5846 0.5340 0.5340 0.5104
# iphoneperneg  0.5458 0.5161 0.5781 0.5424 0.5602 0.5458
# sonyxperia    0.5777 0.5777 0.5777 0.5777 0.5777 0.5777
# htcperneg     0.5777 0.5777 0.5777 0.5777 0.5777 0.5777
# htccamneg     0.5728 0.5728 0.5728 0.5728 0.5728 0.5728
# htcdisneg     0.5728 0.5728 0.5728 0.5728 0.5728 0.5728
# googleandroid 0.5492 0.5292 0.5194 0.5194 0.5714 0.5492



##### prediction base on the model #####
kknnFit_iphoneDFpreds <- predict(kknnFit_iphoneDF, iphoneDF)
confusionMatrix(kknnFit_iphoneDFpreds, iphoneDF$iphonesentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0 1877  304  280  306  701 5732
# 1   20   57   15   16   33  257
# 2    8    4  128    6   11   95
# 3    4    2    3  818   10   38
# 4   18   11    6    6  622  130
# 5   35   12   22   36   62 1288
# 
# Overall Statistics
# 
# Accuracy : 0.3692          
# 95% CI : (0.3609, 0.3776)
# No Information Rate : 0.5812          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2241          
# 
# Mcnemar's Test P-Value : <2e-16          
# 
# Statistics by Class:
# 
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity            0.9567 0.146154 0.281938  0.68855  0.43224  0.17082
# Specificity            0.3349 0.972900 0.990095  0.99516  0.98517  0.96926
# Pos Pred Value         0.2040 0.143216 0.507937  0.93486  0.78436  0.88522
# Neg Pred Value         0.9775 0.973519 0.974373  0.96942  0.93292  0.45720
# Prevalence             0.1512 0.030062 0.034996  0.09157  0.11092  0.58121
# Detection Rate         0.1447 0.004394 0.009867  0.06305  0.04795  0.09928
# Detection Prevalence   0.7092 0.030679 0.019425  0.06745  0.06113  0.11216
# Balanced Accuracy      0.6458 0.559527 0.636017  0.84186  0.70871  0.57004


rfFit_iphoneDFpreds <- predict(rfFit_iphoneDF, iphoneDF)
confusionMatrix(rfFit_iphoneDFpreds, iphoneDF$iphonesentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0 1289    0    2    1    9   14
# 1    0    1    0    0    0    0
# 2    0    1   66    0    0    0
# 3    6    1    1  773    6    7
# 4    5    0    1    1  477    2
# 5  662  387  384  413  947 7517
# 
# Overall Statistics
# 
# Accuracy : 0.7803          
# 95% CI : (0.7731, 0.7874)
# No Information Rate : 0.5812          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5716          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 0  Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.65698 2.564e-03 0.145374  0.65067  0.33148   0.9969
# Specificity           0.99764 1.000e+00 0.999920  0.99822  0.99922   0.4859
# Pos Pred Value        0.98023 1.000e+00 0.985075  0.97355  0.98148   0.7291
# Neg Pred Value        0.94227 9.700e-01 0.969936  0.96592  0.92296   0.9914
# Prevalence            0.15124 3.006e-02 0.034996  0.09157  0.11092   0.5812
# Detection Rate        0.09936 7.708e-05 0.005087  0.05959  0.03677   0.5794
# Detection Prevalence  0.10136 7.708e-05 0.005165  0.06120  0.03746   0.7947
# Balanced Accuracy     0.82731 5.013e-01 0.572647  0.82445  0.66535   0.7414

c50Fit_iphoneDFpreds <- predict(c50Fit_iphoneDF, iphoneDF)
confusionMatrix(c50Fit_iphoneDFpreds, iphoneDF$iphonesentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0 1284    0    1    3   14   17
# 1    0    0    0    0    0    0
# 2    0    0   62    0    0    0
# 3    8    1    5  775    5   12
# 4    5    0    1    5  484   13
# 5  665  389  385  405  936 7498
# 
# Overall Statistics
# 
# Accuracy : 0.7788          
# 95% CI : (0.7715, 0.7859)
# No Information Rate : 0.5812          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5696          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.65443  0.00000 0.136564  0.65236  0.33634   0.9944
# Specificity           0.99682  1.00000 1.000000  0.99737  0.99792   0.4883
# Pos Pred Value        0.97346      NaN 1.000000  0.96154  0.95276   0.7295
# Neg Pred Value        0.94182  0.96994 0.969638  0.96606  0.92339   0.9844
# Prevalence            0.15124  0.03006 0.034996  0.09157  0.11092   0.5812
# Detection Rate        0.09897  0.00000 0.004779  0.05974  0.03731   0.5780
# Detection Prevalence  0.10167  0.00000 0.004779  0.06213  0.03916   0.7923
# Balanced Accuracy     0.82563  0.50000 0.568282  0.82486  0.66713   0.7414

svmFit_iphoneDFpreds <- predict(svmFit_iphoneDF, iphoneDF)
confusionMatrix(svmFit_iphoneDFpreds, iphoneDF$iphonesentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0 1278    6    7   57   39  123
# 1    2    1    0    0    1    0
# 2    2    0   11    0    3    1
# 3   10    2   55  334   10   16
# 4    1    1    2    1  292   19
# 5  669  380  379  796 1094 7381
# 
# Overall Statistics
# 
# Accuracy : 0.7166          
# 95% CI : (0.7088, 0.7244)
# No Information Rate : 0.5812          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4302          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 0  Class: 1  Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.65138 2.564e-03 0.0242291  0.28114  0.20292   0.9789
# Specificity           0.97893 9.998e-01 0.9995207  0.99211  0.99792   0.3893
# Pos Pred Value        0.84636 2.500e-01 0.6470588  0.78220  0.92405   0.6899
# Neg Pred Value        0.94033 9.700e-01 0.9658073  0.93193  0.90938   0.9301
# Prevalence            0.15124 3.006e-02 0.0349958  0.09157  0.11092   0.5812
# Detection Rate        0.09851 7.708e-05 0.0008479  0.02575  0.02251   0.5690
# Detection Prevalence  0.11640 3.083e-04 0.0013104  0.03291  0.02436   0.8247
# Balanced Accuracy     0.81515 5.012e-01 0.5118749  0.63663  0.60042   0.6841


summary(kknnFit_iphoneDFpreds)
# 0    1    2    3    4    5 
# 9200  398  252  875  793 1455 

summary(rfFit_iphoneDFpreds)
# 0     1     2     3     4     5 
# 1315     1    67   794   486 10310 

summary(c50Fit_iphoneDFpreds)
# 0     1     2     3     4     5 
# 1319     0    62   806   508 10278 

summary(svmFit_iphoneDFpreds)
# 0     1     2     3     4     5 
# 1510     4    17   427   316 10699



###  postResample Values
postResample(kknnFit_iphoneDFpreds, obs = iphoneDF$iphonesentiment)
# Accuracy     Kappa 
# 0.3692284 0.2241460 

postResample(rfFit_iphoneDFpreds, obs = iphoneDF$iphonesentiment)
# Accuracy     Kappa 
# 0.7803130 0.5716144 

postResample(c50Fit_iphoneDFpreds, obs = iphoneDF$iphonesentiment)
# Accuracy     Kappa 
# 0.7787713 0.5695566 

postResample(svmFit_iphoneDFpreds, obs = iphoneDF$iphonesentiment)
# Accuracy     Kappa 
# 0.7166423 0.4302036 


## feature importance using RF on iphoneRFE
varImp(rfFit_iphoneRFE, scale = FALSE)
plot(varImp(rfFit_iphoneRFE))
# rf variable importance
# 
# Overall
# iphone         563.07
# samsunggalaxy  234.54
# htcphone       203.64
# iphonedisunc   188.88
# iphonedisneg   146.05
# googleandroid  145.76
# iphonedispos   121.06
# iphoneperpos   101.89
# iphonecamunc    91.91
# iphonecampos    67.53
# htccampos       65.22
# iphonecamneg    64.40
# iphoneperneg    61.42
# iphoneperunc    57.83
# sonyxperia      57.60
# ios             56.20
# htcperpos       42.31
# htccamneg       18.06
# htcdisunc       17.92
# htcperneg       14.95






++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


##### Modeling with iphoneCOR / iphoneNZV / iphoneRFE

## #### train with dataset iphoneCOR
set.seed(123)


#create a 20% sample of the data
# iphoneCOR <- iphoneCOR[sample(1:nrow(iphoneCOR), 150, replace=FALSE),] # no need to sample according to POA

# define an 70%/30% train/test split of the dataset
inTrainingCOR <- createDataPartition(iphoneCOR$iphonesentiment, p = .70, list = FALSE)
trainingCOR <- iphoneCOR[inTraining,]
testingCOR <- iphoneCOR[-inTraining,]

#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats = 5)

##  Model Random Forest
system.time(rfFit_iphoneCOR <- train(iphonesentiment~., data = trainingCOR, method = "rf", 
                                      trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 5.018   0.347  55.857   

rfFit_iphoneCOR
# Random Forest 
# 
# 9083 samples
# 24 predictor
# 6 classes: '1', '2', '3', '4', '5', '6' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 5 times) 
# Summary of sample sizes: 8174, 8175, 8175, 8175, 8174, 8176, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.6759888  0.3247135
# 
# Tuning parameter 'mtry' was held constant at a value of 4


## iphone NZV
inTrainingNZV <- createDataPartition(iphoneNZV$iphonesentiment, p = .70, list = FALSE)
trainingNZV <- iphoneNZV[inTraining,]
testingNZV <- iphoneNZV[-inTraining,]
system.time(rfFit_iphoneNZV <- train(iphonesentiment~., data = trainingNZV, method = "rf", 
                                     trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 3.269   0.147  35.946   

rfFit_iphoneNZV
# Accuracy   Kappa    
# 0.7599461  0.5292822



#### iphoneRFE
inTrainingRFE <- createDataPartition(iphoneRFE$iphonesentiment, p = .70, list = FALSE)
trainingRFE <- iphoneRFE[inTraining,]
testingRFE <- iphoneRFE[-inTraining,]
system.time(rfFit_iphoneRFE <- train(iphonesentiment~., data = trainingRFE, method = "rf", 
                                      trControl=fitControl, tuneLength = 1))
# user  system elapsed 
# 5.169   0.244  58.255

rfFit_iphoneRFE
# Accuracy   Kappa    
# 0.7764855  0.5660296
# 
# Tuning parameter 'mtry' was held constant at a value of 4



#### postResample & ConfusionMatrix for Model RF with iphoneCOR/NZV/RFE
rfFit_iphoneCORpreds <- predict(rfFit_iphoneCOR, iphoneCOR)
postResample(rfFit_iphoneCORpreds, obs = iphoneCOR$iphonesentiment)
# Accuracy     Kappa 
# 0.6824173 0.3364190

confusionMatrix(rfFit_iphoneCORpreds, iphoneCOR$iphonesentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    1    2    3    4    5    6
# 1  981    7   12   21   36  130
# 2    0    0    0    0    0    0
# 3    1    0    8    0    0    1
# 4    3    0    0   29    6    2
# 5    7    0    4    2  463   35
# 6  970  383  430 1136  934 7372
# 
# Overall Statistics
# 
# Accuracy : 0.6824          
# 95% CI : (0.6743, 0.6904)
# No Information Rate : 0.5812          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.3364          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 1 Class: 2  Class: 3 Class: 4 Class: 5 Class: 6
# Sensitivity           0.50000  0.00000 0.0176211 0.024411  0.32175   0.9777
# Specificity           0.98129  1.00000 0.9998402 0.999067  0.99584   0.2908
# Pos Pred Value        0.82645      NaN 0.8000000 0.725000  0.90607   0.6567
# Neg Pred Value        0.91677  0.96994 0.9655944 0.910384  0.92168   0.9039
# Prevalence            0.15124  0.03006 0.0349958 0.091575  0.11092   0.5812
# Detection Rate        0.07562  0.00000 0.0006167 0.002235  0.03569   0.5683
# Detection Prevalence  0.09150  0.00000 0.0007708 0.003083  0.03939   0.8653
# Balanced Accuracy     0.74065  0.50000 0.5087307 0.511739  0.65879   0.6343

## ---- postResample & confusionMatrix for iphoneNZV + RFE not working---- ##


rfFit_iphoneNZVpreds <- predict(rfFit_iphoneNZV, iphoneNZV)
postResample(rfFit_iphoneNZVpreds, obs = iphoneNZV$iphonesentiment)
# Accuracy     Kappa 
# 0.7761505 0.5637566 

confusionMatrix(rfFit_iphoneNZVpreds, iphoneCOR$iphonesentiment)
# Accuracy : 0.7762


rfFit_iphoneNZVpreds <- predict(rfFit_iphoneNZV, iphoneNZV)
postResample(rfFit_iphoneNZVpreds, obs = iphoneNZV$iphonesentiment)
# Accuracy     Kappa 
# 0.6824173 0.3364190

confusionMatrix(rfFit_iphoneCORpreds, iphoneCOR$iphonesentiment)
# Accuracy   Kappa  
#  0.7762    0.5638


rfFit_iphoneRFEpreds <- predict(rfFit_iphoneRFE, iphoneRFE)
postResample(rfFit_iphoneRFEpreds, obs = iphoneRFE$iphonesentiment)
# Accuracy     Kappa 
# 0.7877900 0.5891535 

confusionMatrix(rfFit_iphoneRFEpreds, iphoneRFE$iphonesentiment)
# Accuracy     Kappa 
# 0.7878       0.5892 




++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# prediction with LargeMatrix dataset
# import data
library(caret)
iphoneLargeDF <- read.csv("~/Desktop/iphoneLargeMatrix.csv")

# data preprocessing
str(iphoneLargeDF)
summary(iphoneLargeDF)

iphoneLargeDF$iphonesentiment <- as.factor(iphoneLargeDF$iphonesentiment)
str(iphoneLargeDF$iphonesentiment)

is.na(iphoneLargeDF)


### change empty column iphonesentiment to Factor
# Code to build iphoneLargeDF_vector
iphoneLargeDF_vector <- c("0", "1", "2", "3", "4", "5")
factor_iphoneLargeDF_vector <- factor(iphoneLargeDF_vector)

# Specify the levels of factor_survey_vector
levels(iphoneLargeDF_vector) <- 6

factor_iphoneLargeDF_vector

iphoneLargeDF$iphonesentiment <- factor_iphoneLargeDF_vector
str(iphoneLargeDF$iphonesentiment)

# remove extra columns
# iphoneLargeDF <- subset (iphoneLargeDF, select = -iphonesentiment5)
# iphoneLargeDF <- subset (iphoneLargeDF, select = -iphonesentiment_vector)
# str(iphoneLargeDF)


# create new data set with rfe recommended features
iphoneLargeDFRFE <- iphoneLargeDF[,predictors(rfeResults)]
str(iphoneLargeDFRFE)
# add the dependent variable to iphoneRFE
iphoneLargeDFRFE$iphonesentiment <- iphoneLargeDF$iphonesentiment

iphoneLargepreds <- predict(rfFit_iphoneRFE, iphoneLargeDFRFE)
summary(iphoneLargepreds)
# 1     2     3     4     5     6 
# 14296     0   586  1932   219  7747 



stopCluster(cl)

