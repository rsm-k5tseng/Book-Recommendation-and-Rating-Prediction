# Book-Recommendation-and-Rating-Prediction

[Predicting Ratings]
1. Splitting Data:
The dataset allRatings is divided into training (ratingsTrain) and validation (ratingsValid) sets. 
95% of the data is used for training, while 5% is reserved for validation. 
Additionally, the ratings are organized by users (ratingsPerUser) and items (ratingsPerItem) using defaultdicts.

2. Preparing Data for Surprise:
The allRatings data is converted into a pandas DataFrame for compatibility with the Surprise library. 
A Reader object is created to define the rating scale, and the dataset is loaded using Dataset.load_from_df.

3. Grid Search for Hyperparameter Tuning:
A grid search is conducted using GridSearchCV with the SVD algorithm. 
The parameter grid includes factors such as the number of latent factors (n_factors), learning rate (lr_all), regularization (reg_all), and epochs (n_epochs). 
The best parameters are identified based on RMSE scores.

4. Cross-Validation:
K-Fold Cross-Validation is used with 5 folds to evaluate the SVD model’s performance. 
The results are displayed to ensure the model is performing well.

5. Model Training:
The best estimator from the grid search is trained on the full dataset. 
This ensures that the final model leverages all available training data.

6. Prediction Logic:
Fallback logic is implemented to handle cases where either a user or an item is not in the training set:
	•	If both user and item are in the training set, predictions are made using the trained model.
	•	If only the user is in the training set, their average rating is used.
	•	If only the item is in the training set, its average rating is used.
	•	If neither is available, the global average rating (mu) is used as a fallback.

7. Generating Predictions:
Predictions for the test dataset (pairs_Rating.csv) are generated and saved to a CSV file (predictions_Rating.csv) in the required format.

[Predicting Read/ No Read]
1. Data Splitting and Initialization:
	•	The data is split into training (ratingsTrain) and validation (ratingsValid) sets, with 95% of the data used for training.
	•	ratingsPerUser and ratingsPerItem store mappings of users to their rated books and books to their users, respectively.

2. Negative Sampling:
	•	Positive samples from the validation set are used to create negative samples. For each user, a random book not already rated by them is selected as a negative sample.
	•	The validation set is then extended to include both positive and negative samples for evaluation.

3. Data Structure Optimization:
	•	A custom dictionary class, my_large_dict, is implemented to transform input data into a more efficient structure for user-book relationships.

4. Improved Prediction Logic with Thresholds:
	•	An ImprovedPredictor class is introduced, utilizing:
	•	Jaccard Similarity: Measures overlap between users of two books.
	•	Book Popularity: Uses the number of users who have rated a book.
	•	The prediction function combines Jaccard similarity and popularity thresholds to decide whether a user will interact with a book.

5. Threshold Tuning:
	•	Jaccard and popularity thresholds are tuned by testing combinations over a range of values.
	•	Accuracy is calculated for each combination, and the best thresholds are identified based on maximum accuracy.

6. K-Fold Cross-Validation:
	•	5-fold cross-validation is performed to evaluate the predictor’s performance. Each fold updates usersPerBook and bookPopularity based on the training split.
	•	Accuracy for each fold is calculated, and the average accuracy is reported.

7. Test Predictions:
	•	The final model with optimal thresholds is used to predict interactions for the test dataset (read_test).
	•	Predictions are saved to a CSV file (predictions_Read.csv).

8. Output:
	•	Final predictions are stored with the columns: userID, bookID, and prediction. The predictions indicate whether a user is expected to interact with a book (1 for interaction, 0 otherwise).