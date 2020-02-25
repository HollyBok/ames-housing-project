# Project 2 Code: Modeling Property Sale Prices in Ames, Iowa
#### Author: Holly Bok



## Problem Statement

This project concerns sale prices for properties in Ames, Iowa. **The goal of this project is to create a supervised machine learning model that can accurately predict the Sale Price of a property given information about its features.** This model will be created using housing data from Ames, Iowa that includes 79 independent features as well as the sale price of the property. A selection of independent features will be used to predict values for the target variable, sale price. 70% of the housing dataset has been put aside as "train" data. This data will be used twofold, 1) to identify strong correlations between independent features and sale price, and 2) to fit machine learning models that closely predict sale price. The remaining 30% of the data has been put aside in a different dataset called "test." This dataset does not include the sale prices of the properties and will be used to test the strength of the model. 

**Predictions will be made using 3 supervised, machine learning models: Linear Regression, LASSO, and RidgeCV** Linear Regression models are supervised, machine learning models that create best-fit lines based on existing information (independent variables) with the purpose of making predictions about a target variable. In this project, the target variable is 'SalePrice', the sale price of the properties. Linear Regression requires manual selection of features through data exploration. The two other models shown here, LASSO and RidgeCV models, input all numeric features and regularize them to create appropriate coefficients for each feature. Features that are good predictors of price are given higher coefficients and features that are not good predictors of price are given coefficients near, or at, 0. LASSO and other regularized models are different from Linear Regression because they include penalties for features that are not as strong predictors of the target variable.

Success for these models will be evaluated using R2 scores*. Higher R2 scores (with a max of 1) indicate more successful models. The R2 score is the performance of the model in comparison to the mean of the target variable. Put another way, the R2 scores are interpreted as the variability in the target feature that is explained by our model in comparison to the average value of that feature. R2 scores closer to 1 indicate higher explanation of variabilty by the model; An R2 score of 1 would mean that 100% of the variability in the response feature is explined by the model.

* *Also known as the coefficient of determination.* 
*Equal to the sum of squared errors divided by the sum of total squares*

In this project I will identify features of the Ames housing dataset that have linear relationships with sale price and fit a Linear Model using these features. I will also fit two regularized models, a LASSO model and a RidgeCV model, using polynomial features to include interaction terms. The models will be assessed for their success and the best model will be chosen and discussed. Furthermore, I will identify the specific features that best predit sale price. Lastly, I will make recommendations for property listing companies and other interested parties as well as discuss issues in the data and make suggestions for further research. 


## Data Loading and Cleaning

Housing data has been obtained from https://www.kaggle.com/c/dsi-us-10-project-2-regression-challenge/data. The data dictionary is available for this housing data set therefore it is not outlined in detail here. Exploration and discussion of specific categories and/or aspects of the data will be outlined here for relevant variables. 

Train and test data has been read in from the datasets folder of this repository. The original train data has been read from filepath 'datasets/train.csv' and the test data has been read from filepath 'datasets/test.csv' Both dataframes have been read as pandas dataframes. 

The training data has 2,050 rows and 81 columns including the dependent variable 'SalePrice'. The test data has 878 rows and 80 columns. Each row in the data corresponds to a property that has been sold. Of the 81 columns in the training data set, 78 are possible dependent variables (the remaining 3 are SalePrice, ID, and Property ID). All properties were sold in Ames, Iowa between 2006 and 2010. Data has been obtained from the Ames Assessors Office. 

Several variables in the raw data set are ranked on non-numerical, but hierarchical, string grading scales. The most common of these grading scales is a quality ranking of Po, Fa, TA, Gd, and Ex for Poor, Fair, Typical / Average, Good, and Excellent. This scale is used for 8 variables (Exterior Quality, Exterior Condition, Heating Quality, Kitchen Quality, Garage Quality, Garage Condition, Pool Quality, and Basement Condition). Each of these variables have been replaced with a 1-5 scale (where 1 is Poor and 5 is Excellent) and converted from objects to integers for use in the modeling process. The assumption made henceforth is that null values represent a lack of the variable they describe (for example, a score of 0 for Garage Quality will represent the lack of a garage). Null values for all of these categories have been replaced with a score of 0. A few other ranking scales exist for non-numerical, heirarchical categories in this data including Paved Drive, Functional, and Garage Finish. The specific conversions for these variables are available as notation in the '01 Data Clean' file of this repository. One variable, PID, was read as numeric data and converted to the more appropriate object datatype. 9,822 total null values are present in the raw dataset. All null values are assumed as 0 for all variables.

The cleaned dataset has been written to .csv as clean_train. This dataset is used for the modeling process for both the multi linear regression model and the LASSO model. Predictions for the Sale Price of properties in the test dataset have been saved in two separate datsets, one for the linear regression and one for the LASSO model. These files can be found in the datasets folder of this repository as 'clean_train.csv', 'Holly_Preds_LR', and 'Holly_Preds_LASSO'



## Modeling


#### Linear Regression Model:

The majority of EDA is done during the selection process for the variables in the multi linear regression (MLR) model. The tools used for initial feature selection and feature analysis for this model include a seaborn heatmap, histograms of dependent variables, and a seaborn pairplot.

A seaborn heatmap is a visual created to present the linear correlations between all possible independent variables and the dependent variable 'SalePrice.' The strongest correlated variables are identified by their correlation scores; The features selected have scores above .40 or below -.40. The threshold of .40/-.40 is chosen because this provides a number of features that will not cause the model to be overfit. Of these variables 10 are selected as the independent features for the model. The process of selecting or eliminating these variables is in the '02 Linear Regression' file of this repository, under the "EDA and Feature Selection" subheader.

**The features selected for the model include:** Overall Quality, Exterior Quality, Above ground living area, Kitchen Qual, Garage Area, Total basement SF, Year Built, Year Remodeled, Full Bath, and Masonry Veneer Area. A new dataframe, called 'train_1' is created to include only these features and Sale Price. All MLR modeling is done using this 'train_1' dataframe rather than the original, raw 'train' dataframe.

A logarithmic transformation is done to the two independent variables with the highest correlation scores. These variables are Overall Quality and Exterior Quality. A logarithmic transformation is applied to the target variable as well. The purpose of the logarithmic transformation is to create a more homoskedastic model. In addition, outliers are identified that could affect the normality and homoskedasticity of the data. These outliers are identified and removed for the Linear Regression model as well as each model hereafter.

A train_test_split function has been applied to our train_1 dataframe to create a training and test subset of train_1. A Linear Regression model (called 'my_model') is instantiated and fit to the training subset using our X and y variables (where X is all the independent variables included in train_1 and y is the dependent variable). Using this model a set of predictions is made for the train_test_split test subset. A cross_val_score (with 5 folds), training score, and test score are all produced from my_model. A detailed summary of the performance and coefficients for each independent variable can be found in the '02 Linear Regression' file of this repository under the "Model Scoring" subheader. 

**The Linear Regression model, my_model, preformed as follows*:**
cross_val_train().mean() : 0.8644905636927491
train subset: 0.8705915566195076
test subset: 0.8319067601632991

* All scores presented are R2 scores


The original test data is transformed to match the training data for the MLR in order to make appropriate predictions for the test data using my_model. Kitchen Quality and Exterior Quality are transformed from non-numerical, heirarchical objects to ranked integers in the same way the train data was. Overall Quality and Exterior Quality are transformed through log functions, just as the train data. Predictions are made for the Sale Price of the original test data using my_model.predict(). These predictions are 
converted back from log functions to return to the original units. A dataframe consisting of property IDs and predicted sale prices is written as a .csv file called 'Holly_Preds_LR'


#### LASSO Model:

A second, regularized machine learning model is created from the clean_train dataset. This model is created and tested in competition with the MLR model to produce a better fit model with higher cross_val_train, train subset, and test subset R2 scores. Features are not selected using EDA as they are in the MLR model. Instead, all numerical independent variables are input as features into the LASSO regularization model and their coefficients are scaled to fit their correlation to sale price. Additionally, this model is fit with polynomial interaction features. Polynomial interaction features add features for the interaction terms between all independent variables. A new dataframe called 'numeric_train' is created that includes all numeric variables from clean_train and all new polynomial features. This results in 1,274 independent model variables. A train_test_split is run on the numeric_train dataset, creating a training and testing subset. 

A LASSO model is instantiated (with max iterations set at 1500) and fit. A requirement of the LASSO model is that all variables have been scaled to ensure that all variables are on the same numeric scale. A standard scaler is used to accomplish this on the training and testing subsets. The model is then fit and scored. The coefficients for each variable are shown in a dataframe called 'lasso_coef' is shown in the '03 LASSO Model' file of this repository under the subheader 'Fitting and Scoring the LASSO Model." More details on the specifics of the LASSO model, including a list of features that were regularized to a coefficient larger or smaller than 0, can be found there. 

**The LASSO model, lcv, preformed as follows*:**
cross_val_train().mean() : 0.916942076487099
train subset: 0.9400702246522403
test subset: 0.8930934913207212

* All scores presented are R2 scores

As with the Linear Regression model, the original test dataset is manipulated to match the dataset numeric_train that the LASSO model is fit to. As this process is outlined above under the Linear Regression Model subheader, it is not detailed here. Specifics can be found in the '03 LASSO Model' file of this repository under the 'Generating Predictions for Test Dataset' subheader. The property IDs and sale price predictions are saved to a dataframe and written as a .csv file named 'Holly_Preds_LASSO.csv' 


#### RidgeCV Model

A RidgeCV model is created and tested in competition with the LASSO model. This model is created in a very similar fashion to the LASSO model, as the RidgeCV and LASSO models are both regularized models. RidgeCV and LASSO models differ in the measures they use to assign penalties to coefficients that are not good predictors of the target variable. The ridgeCV model will not assign a coefficient of 0 to any feature but will instead spread the coefficients out smoothly towards 0. 

The Ridge CV Model did not score as high as the LASSO model or the Linear Regression model.

**The RIDGECV model, ridge_cv, preformed as follows*:**
cross_val_train().mean() : 0.6570085100344006
train subset: 0.885667792472507
test subset: 0.880501632158954

* All scores presented are R2 scores

Predictions from the RidgeCV model are saved under 'Holly_Preds_Ridge' 



## Conclusions


#### Overall

The goal of this project was to create the best possible supervised, machine learning model to predict the sale price of properties sold in Ames, Iowa. These models were created by manipulating a series of informational variables representing the presence and/or quality of housing features. Model success was evaluated using R2 scores, a measure of how strong a model is at making predictions. Each model was tested and given an R2 score based on its ability to make accurate predictions.

**The R2 scores for these models suggest that the LASSO model is the best at predicting housing prices**, followed by the Linear Regression model and the RidgeCV model. The R2 scores for the LASSO model were highest in all three scoring categories (cross_val_score, test subset, and train subset). This means that the LASSO model is the most successful of our models. The Linear Regression model preformed second best and the RidgeCV model preformed the worst. The highest scores for all 3 models were seen in the training subsets. The highest R2 score received on the test subset is 0.89 using the LASSO model. 

The LASSO model also balanced bias and variance better than the other two models. This suggests that the best models for predicting the sale price of houses need to be careful to not overfit or underfit. The LASSO model choice is much like the choice of porridge that Goldilocks must make. The LASSO model is the 'just right' model, while the Linear Regression model has not been engineered enough to more closely examine prices (the 'too cold' model), and the RidgeCV model has been overengineered to the point that the testing data is unlikely to follow the same pattern (the 'too hot' model). In selecting the best model (and in turn, the best independent features) we will be able to more accurately predict price.

Polynomial features preformed higher than the original features in the LASSO model, indicating polynomial features are better predictors of Sale Price. The strongest indicators of Sale Price among all models included Overall Quality, Exterior Quality, Above Ground SF, Total Basement SF, Kitchen Quality, and Masonry Veneer Area. These features are seen independently and in many of the strong interaction features that were created during polynomial transformation. 


#### LASSO Model

The LASSO model preformed the strongest of all 3 models. The model preformed best both overall and in each 3 seperate scoring categories (cross_val_score, train subset, and test subset). The strongest R2 score for the LASSO model was the training subset, with a score of 0.94. The cross_val_score was slightly lower, at 0.92, and the test subset score was the lowest at 0.89. The test subset R2 score is lower than the training and cross_val scores. This suggests that slight overfitting to the training data has resulted in higher variance. However, as all 3 scores are relatively consistent (and all 3 scores are high) we can conclude that the LASSO model is the best at predicting Sale Price as compared to the other models. 

The LASSO model uses all numerical features as input and assigns coefficients for each features based on how that variable affects the dependent variable. Indepdent features that are determined to have low effect on the target variable are assigned coefficients of 0, meaning they do not have an effect on Sale Price. Of 1,274 features (including polynomial features) that are input into the LASSO model, only 123 features are assigned coefficients. The distribution of these coefficients can be seen in the '02 Linear Regression' file of this repository under the "Fitting and Scoring to Lasso Model" subheader. The majority of coefficients are small and cluster towards 0 with very few coefficients above the 8,000 mark or below the 2,000 mark. This suggests that the feature with high coefficients are doing the majority of the work in predicting Sale Prices. 

Almost all strong coefficients in the LASSO model are polynomial features. All of the top 50 positively and bottom 50 negatively most correlated coefficients are polynomial features. The strongest 5 coefficients are as follows:

Masonry Veneer Area x Pool Quality:          - 20990   
Overall Quality x Above Ground Living Area:    13669
Overall Quality x Total Basement SF:           11474
Masonry Veneer Area x Miscelaneous Value:     -10918 
Above Ground Living Area x Kitchen Quality:   - 8592

Although the large coefficients are all polynomial features, the polynomial features are interaction terms between many of the feature variables identified in the Linear Regression model. This suggests that the feature selection in the Linear Regression model resulted in features that affect Sale Price in all models. For example, Overall Quality, Above Ground Living Area, Total Basement SF, Kitchen Quality, and Masonry Veneer Area are all features that were selected for the Linear Regression models. These features are seen frequently in interaction terms in the LASSO model. 
    

#### Linear Regression Model

The Linear Regression does not preform as well as the LASSO model but is still a high preforming model. The strongest R2 score for the LASSO model was the training subset, with a score of 0.871. The cross_val_score was slightly lower, at 0.865, and the test subset score was the lowest at 0.83. Much like the LASSO model, this suggests slight variance and overfitting, but the model is preforming well regardless. 

Show some of the coefficients and how some are really not that strong. Of the 10 features selected for the Linear Regression model, only 3 features have coefficienst above .01. In the case of Linear Regression, we can interpret the value of these coefficients as "a one unit increase of this feature will lead to a .46 unit increase in the target variable." For example, in this Linear Regression, a 1 unit increase Overall Quality will result in a .46 unit increase in predicted Sale Price.

The strongest coefficients of this model were:
Overall Quality :  .46
Exterior Quality*: .06
Kitchen Quality:   .05

*The relationship between Exterior Quality and Sale Price has a p value above .05

Among the variables with small coefficients we see a higher instance of large scale variables. The variables with large coefficients (Overall Quality, Exterior Quality, and Kitchen Quality) tend to be variables with rank scales (1-10) as opposed to variables with square footage or other, large scales. It is difficult to interpret the importance of the variables on coefficients alone for this reason. This problem is avoided in the LASSO and RidgeCV models as those models are regularized and keep all of the feature variables on the same scale.

There are several possible reasons that the Linear Regression is not as strong as the LASSO model. Assumptions of Linear Regression are violated in the dataset. Many of the selected features are not completely linearly related to the target variable, Sale Price. Although we do see somewhat linear coorelations, they are not completely linear and thus violate one of the assumptions of the regression model. Linear relationships can be observed in the '02 Linear Regression' file of this repository under the "EDA and Feature Selection" subheader. Additionally, our original, raw data is not distributed normally. This results in a non-normal distribution of errors and an increase in heteroskedasticity. While steps are taken to reduce the effects of these problems, we still see our assumptions violated with this data. Lastly, we are violating the independence of variables assumption of LR. Some redundancies are eliminated by dropping features that are redundant, but it is near impossible to avoid non-independence of independent variables in the case of housing data. These assumption violations likely cause the Linear Regression model to underpreform as compared to the LASSO model. 
    

#### RidgeCV Model

The third model, RidgeCV, preformed the worst of the three models. The highest R2 score for the RidgeCV model was 0.89 for the training subset, followed by 0.88 for the testing subset, and 0.66 for cross_val_score.

While initially this cross_val_score seems incredibly low, this is likely due to the presence of an outlier in the testing data. The cross_val_score is generated by average the R2 scores of a 3-fold train/test split. The average R2 score is presenting as very low compared to the other scores (0.66), but this is because the cross_val_score is an *average* of a few R2 scores created in different train/test folds. The average cross_val_score for the RidgeCV model is thrown off because of the individual R2 scores: 0.85403624, 0.85500432, and 0.26198497. The average of is brought down by the R2 score for the 3rd fold, 0.27. 

The preformance of the RidgeCV model is closer to the preformance of the Linear Model while the LASSO model outpreformed both. The RidgeCV model and LASSO model do not preform the same way even though they are both regularized models. The goal of both models is to raise the coefficients of important features and lower the coefficients of less important features. However, while the LASSO model will create coefficients of *exactly* 0, the RidgeCV model will only bring them *very close* to 0. Because of this, the RidgeCV model is including many, many more independent features (all 1274) because it is not fully rejecting any one feature. This is likely why we saw better preformance with the LASSO model than the RidgeCV model; the RidgeCV model has so many inputs that it becomes complicated and it is less likely to make accurate predictions. 

In general, the coefficients in the RidgeCV model showed consistency of predictor features with the other two models. Overall quality and exterior quality made up the majority of the polynomial features that were assigned the largest coefficients. We see negative correlations here with masonry veneer area and pool quality. 



## Recommendations 


#### Listing Corporations:

**In order to best predict property sale prices I recommend the use if the LASSO model. This model is the best choice for a home listing corporation, such as a real estate firm, to accurately predict the sale price of their properties.** This is in the interest of such corporations because it balances the expectations of both the home buyers and home owners and will provide the best results for ensuring clients on both sides are satisfied with their listing or buying experience. This will also increase turnover and result in more homes being sold and listed through the company.  

More generally, when designing supervised machine learning models to predict housing prices it is best to use models that regularize as opposed to models that require manual feature selection. This is the best way to optimize the preformance of the model and minimizes the chance of error due to hidden features of the data. However, when utilizing regularized models care should be taken in which data is regularized. It is important to be discriminatory when selecting features that go into the model or to use a model that will set coefficients of 0 for less important features.

When getting information about properties that will be sold it is important to ask questions first and foremost about the overall quality of the home. The interior and exterior quality / condition of the property overall as well as the quality of specific features are good indicators of sale price. Variables that have information about the size of the property are also good indicators of sale price. Many of the variables that are indicators of price concern quality, size, or interaction terms of one or both. It is important to note that some features, such as masonry veneer area and number of baths, are indirect indicators of size. For example, a property with a large value for masonry veneer square footage is likely also going to be an otherwise very large property. Variables of size and condition are successful in both the Linear Regression and the regularized models.  

While it is good to ask about more specific features such as driveway quality or alley access, these do little to predict sale price. These features may attract specific buyers to spend more money, but home buyers are more likey to weigh quality and size when doing the cost / benefit analysis of purchasing a new home.

**In order to make the best possible predictions for the prices of homes that will be listed, property listing corporations and interests should focus on the overall quality of the property as well as the size.** 


#### Housing Buyers:

In order to purchase a home of high quality for the best value it is good practice to look for homes that are within a reasonable size. It can be tempting to want to purchase as large of a house as possible, but these homes are often priced high because of their size. Instead of purchasing the largest home you can find under budget, purchase a slightly smaller home with excellent internal features. Similarly, the home that will be the most valuable for the sale price will need repairs on "appearance quality" features. A paint job will likely be cheaper than replacing a bad HVAC system, but a home with fresh paint is going to be valued higher than a home with great HVAC.


#### For Sale by Owner:

In order to sell a home at a high price while investing as little as possible into renovations, focus on the overall appearance and quality of the home. It is best to prepare the property for sale starting with the largest and most damaged feature, as repairing these will likely result in a good return in sale price increase. In addition, increasing the square footage of the home, even in non-expensive ways such as setting up a carport or cutting back brush / wilderness, can increase the sale price. 



#### Further exploration

I recommend that this dataset be explored further in several ways. The data dictionary provided has very little information on the meaning of null and 0 values. Several homes have square footage listed as 0. It is impossible to know looking at this data whether this represents a lack of information or if it supposed to represent an empty lot or a home that is still in the process of construction. Additionally, many homes list bedrooms and/or baths above 0 while still listing square footage as 0. This is impossible and points bad data collection. This data needs to be corrected or a new, similar dataset needs to be created through new research. It is also not known for any feature, such as 'Paved Driveway,' if the presence of a null value means that that feature does not exist at this property or if the reporter does not know if the property has this information or it is not disclosed for another reason. 

I recommend that this general model be broken down into several smaller models. The predicts for sale price will likey be more accurate if there is a separate model for extremely large homes or homes with many great features. Other models could be made to predict the sale price of foreclosures, vacant lots, or homes to and from family members. For further research I would like to construct a similar LASSO model that only includes homes that have no null data in the raw dataset. 

In addition, the models used could be improved. The linear regression model could be improved by including more interaction or polynomial terms, as well as eliminating more outliers and improving the homoskedasticity and normality of the data. I would like to build another RidgeCV model with a smaller selection of polynomial features. I would also like to identify the data in the RidgeCV dataset that is causing the cross_val_score preform so low in one fold. Identifying and removing this variable could improve the overall preformance of the model. Lastly, I would like to create a model that does not include masonry veneer area. This feature seemed to be somewhat inconsistent between models and often had a negative cooeficient. This is likely because homes with large masonry veneer areas will be very large, but not all very large homes will have masonry veneer area. This heavily violates the independence of variables term and I would like to explore the effects of this further. 








*Note: small parts of this code may need to be adapted to fit the requirements of packages or programs in future use. These code warnings appear in red when code is run and do not affect the current preformance of the code.*




