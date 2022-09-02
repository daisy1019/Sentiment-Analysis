**Introduction**

Hello is an app developer which is working with a government health agency to create a suit of smart phone medical apps for use by aid workers in developing countries. These apps will enable the aid workers to manage local health conditions by facilitating communication with medical professionals located elsewhere. Hello is in the process of evaluating potential handset models to determine which one to bundle their software with. 




**Findings** 

Prior to analyzing the big dataset, Alert! Analytics has an analyst who’s manually labeled the sentiment scale for 2 small datasets iPhone and Samsung. These 2 datasets have over 12,000 observations and 59 variables, these variables represent devices features such as camera, display, performance etc., then the analyst interpreted the sentiment and categorized them into the following rankings:

0: sentiment unclear

1: very negative

2: somewhat negative

3: neutral

4: somewhat positive

5: very positive


Below table and graph show the sentiment value based on the two labeled small datasets, iPhone and Samsung. The sentiment value is very close for the two devices, they both have high scale on the ‘very positive’ ranking, followed by ‘sentiment unclear’
, ‘neutral’ and ‘somewhat positive’ have close rankings, users for both devices have very low rankings on ‘very negative’ and ‘somewhat negative’. Based on the data collected, it’s almost impossible to tell which device has better sentiment value.

![image](https://user-images.githubusercontent.com/80385435/188042013-eae53d88-e6e3-4f74-9add-5125154593e0.png)

![image](https://user-images.githubusercontent.com/80385435/188042167-e768f210-7731-48f4-8a4f-57b5a2b5b639.png)



**Methodology**

Since our training dataset is comparably large (over 12,000), additional three datasets were created using their unique features, Correlation (COR), Near Zero Variance (NZV) and Recursive Feature Elimination (RFE) datasets were generated to have the training set for predicting the final sentiment value. Before we created the featured
datasets, we obtained the accuracy and kappa score from the original data so we can compare the accuracy scores with the 3 featured datasets.

Four classification models were utilized to generate the best accuracy and kappa score. They are: Weighted K-Nearest Neighbors (KKNN), RandomForest (RF), C5.0 and Support Vector Machine (SVM). 
This was performed separately for iPhone and Samsung.

 __iPhone__
![image](https://user-images.githubusercontent.com/80385435/188044166-f1179e2b-a958-4eeb-abac-8b224fc0b12a.png)

__Samsung__

![image](https://user-images.githubusercontent.com/80385435/188067899-62031424-cde0-4755-8b51-39db1f5abaec.png)




**Prediction**

The following histogram and pie charts illustrate the sentiment rankings on large unlabeled dataset. Both devices showing higher counts of ‘sentiment unclear’ while iPhone has slightly higher than Samsung; they both have relatively good ‘very positive’ rankings; some users have ‘neutral’ ranking for both devices; Samsung has more ‘somewhat negative’ ranking than iPhone.

 iPhone                                            Samsung

![image](https://user-images.githubusercontent.com/80385435/188069055-af188d49-6824-492a-a868-5a86a93d7af5.png)

![image](https://user-images.githubusercontent.com/80385435/188069267-7b426a73-abd1-4543-bd82-ff215299d297.png)
![image](https://user-images.githubusercontent.com/80385435/188069278-52c88a0c-b927-4f48-96d4-664537114f19.png)




**Confidence and Implication**

The sentiment counts for both devices are quite similar for the small training and large predicting datasets. The major difference is that the ‘sentiment unclear’ takes more counts than the ‘very positive’ category, this is reversed from the small training dataset. 

The following graphs show the feature importance of the 2 datasets. Since different iPhone and Samsun emphasize on different features, this may imply the higher popularity of the 2 devices and the reason why the final device is chosen. 

**Feature Importance**

Samsung                                               
![image](https://user-images.githubusercontent.com/80385435/188069854-0a59d68b-62cc-4da0-a0ee-f3e70a238898.png)

iPhone

![image](https://user-images.githubusercontent.com/80385435/188069893-fe8e1fdd-8430-4a5c-a28d-a345118f523f.png)



**Summary**

Alert! Analytics has performed a detailed analysis on the datasets and is confident to recommend iPhone to Hello as their smart phone device to be used with health apps development to assist aid workers in developing countries.

The reasons for selecting iPhone are below:
iPhone shows predominant preference from Samsung feature importance graph
iPhone has slightly higher ranking on ‘very positive’ sentiment count
iPhone has less count on ‘somewhat negative, low than 50% comparing to Samsung




