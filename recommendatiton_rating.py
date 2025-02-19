# %%
import gzip
from collections import defaultdict
import random
import pandas as pd
import csv

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        parts = l.strip().split(',')
        if len(parts) != 3:
            print(f"Skipping malformed line: {l.strip()}")
            continue
        u, b, r = parts
        r = int(r)  # Convert rank to integer
        yield u, b, r

# %%
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

# %%
len(allRatings)

# %% [markdown]
# **predict rating**

# %%
ratingsTrain = allRatings[:int(len(allRatings)*0.95)]
ratingsValid = allRatings[int(len(allRatings)*0.95):]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

# %%
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV, cross_validate, KFold

allRatings_df = pd.DataFrame(allRatings, columns=["userID", "bookID", "rating"])

reader = Reader(rating_scale=(allRatings_df['rating'].min(), allRatings_df['rating'].max()))
data = Dataset.load_from_df(allRatings_df[['userID', 'bookID', 'rating']], reader)

param_grid = {
    'n_factors': [10, 20, 30],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.05, 0.1, 0.2],
    'n_epochs': [50, 100]
}

kf = KFold(n_splits=5, random_state=42, shuffle=True)

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=kf, n_jobs=-1)
gs.fit(data)

best_params = gs.best_params['rmse']
print(f"Best Parameters: {best_params}")

algo = gs.best_estimator['rmse']

cv_results = cross_validate(algo, data, measures=['rmse'], cv=kf, verbose=True)

trainset = data.build_full_trainset()
algo.fit(trainset)

mu = allRatings_df['rating'].mean()
u_avg_rating = allRatings_df.groupby('userID')['rating'].mean().to_dict()
b_avg_rating = allRatings_df.groupby('bookID')['rating'].mean().to_dict()

# Define prediction function
def predict_rating(user, book):
    if user in u_avg_rating and book in b_avg_rating:
        return algo.predict(user, book).est
    elif user in u_avg_rating:
        return u_avg_rating[user]
    elif book in b_avg_rating:
        return b_avg_rating[book]
    else:
        return mu  # Fallback to global average rating

# Load test data
test_data = pd.read_csv('pairs_Rating.csv')

# Save predictions to CSV
predictions_path = "predictions_Rating.csv"
with open(predictions_path, 'w') as predictions:
    predictions.write("userID,bookID,prediction\n")
    for _, row in test_data.iterrows():
        u, b = row['userID'], row['bookID']
        pred = predict_rating(u, b)
        predictions.write(f"{u},{b},{pred:.4f}\n")

print("Predictions saved to predictions_Rating.csv")

# %% [markdown]
# **predict reading**

# %%
ratingsTrain = allRatings[:int(len(allRatings)*0.95)]
ratingsValid = allRatings[int(len(allRatings)*0.95):]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

# %%
import random

booksPerUser = defaultdict(set)
usersPerbook = defaultdict(set)
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

for user, book, _ in ratingsTrain:
    booksPerUser[user].add(book)
    usersPerbook[book].add(user)

positive_samples = [(user, book) for user, book, _ in ratingsValid]
negative_samples = []

for user, book in positive_samples:
    while True:
        negative_book = random.choice(list(bookCount.keys()))
        if negative_book not in booksPerUser[user]:  # Ensure this book wasn't read by the user
            negative_samples.append((user, negative_book))
            break

validation_set = positive_samples + negative_samples

# %%
from collections import UserDict, defaultdict
class my_large_dict(UserDict):
    def __init__(self, input_data=None):
        super().__init__()
        if input_data:
            self.data = self.transform_data(input_data)
        else:
            self.data = {}

    def transform_data(self, input_data):
        book_record = defaultdict(list)
        for user_id, book_id, _ in input_data:
            book_record[book_id].append(user_id)
        
        return book_record

    def add_item(self, user_id, book_id, _):
        if book_id in self.data:
            self.data[book_id].append(user_id)
        else:
            self.data[book_id] = [user_id]
        return self.data

# %%
allBooks = set(book for _, book, _ in ratingsTrain)
negativeSamples = []
for user, book, _ in ratingsValid:
    while True:
        negativeBook = random.choice(list(allBooks))
        if negativeBook not in [b for b, _ in ratingsPerUser[user]]:
            negativeSamples.append((user, negativeBook, -1))
            break

validationSet = ratingsValid + negativeSamples
random.shuffle(validationSet)
labels = [1] * len(positive_samples) + [0] * len(negative_samples)  

# %%
read_test = pd.read_csv('pairs_Read.csv')
read_test = list(zip(read_test['userID'], read_test['bookID']))
read_test[:2]

# %%
usersPerBook = defaultdict(set)
bookPopularity = defaultdict(int)
for user, book, _ in ratingsTrain:
    usersPerBook[book].add(user)
    bookPopularity[book] += 1

class ImprovedPredictor:
    def __init__(self, jaccard_threshold, popularity_threshold):
        self.jaccard_threshold = jaccard_threshold
        self.popularity_threshold = popularity_threshold

    def jaccard_similarity(self, book1, book2):
        users1 = usersPerBook[book1]
        users2 = usersPerBook[book2]
        intersection = len(users1.intersection(users2))
        union = len(users1.union(users2))
        return intersection / union if union != 0 else 0

    def predict(self, user, book):
        books_read_by_user = [b for b, _ in ratingsPerUser[user]]
        max_similarity = 0
        for b_prime in books_read_by_user:
            similarity = self.jaccard_similarity(book, b_prime)
            max_similarity = max(max_similarity, similarity)
        
        book_popularity = bookPopularity[book]
        
        return (max_similarity >= self.jaccard_threshold) or (book_popularity >= self.popularity_threshold)

best_accuracy = 0
best_jaccard_threshold = 0
best_popularity_threshold = 0

jaccard_thresholds = [i / 100 for i in range(1, 101)] 
popularity_thresholds = [i / 10 for i in range(100, 1001, 10)]

for jt in jaccard_thresholds:
    for pt in popularity_thresholds:
        predictor = ImprovedPredictor(jaccard_threshold=jt, popularity_threshold=pt)
        
        predictions = [predictor.predict(user, book) for user, book in validation_set]
        
        correct = sum([pred == label for pred, label in zip(predictions, labels)])
        accuracy = correct / len(validation_set)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_jaccard_threshold = jt
            best_popularity_threshold = pt

print("Best Jaccard threshold:", best_jaccard_threshold)
print("Best Popularity threshold:", best_popularity_threshold)
print("Best accuracy:", best_accuracy)

# %%
usersPerBook = defaultdict(set)
bookPopularity = defaultdict(int)
for user, book, _ in ratingsTrain:
    usersPerBook[book].add(user)
    bookPopularity[book] += 1

class ImprovedPredictor:
    def __init__(self, jaccard_threshold, popularity_threshold):
        self.jaccard_threshold = jaccard_threshold
        self.popularity_threshold = popularity_threshold

    def jaccard_similarity(self, book1, book2):
        users1 = usersPerBook[book1]
        users2 = usersPerBook[book2]
        intersection = len(users1.intersection(users2))
        union = len(users1.union(users2))
        return intersection / union if union != 0 else 0

    def predict(self, user, book):
        books_read_by_user = [b for b, _ in ratingsPerUser[user]]
        max_similarity = 0
        for b_prime in books_read_by_user:
            similarity = self.jaccard_similarity(book, b_prime)
            max_similarity = max(max_similarity, similarity)
        
        book_popularity = bookPopularity[book]
        
        return (max_similarity >= self.jaccard_threshold) or (book_popularity >= self.popularity_threshold)

predictor = ImprovedPredictor(jaccard_threshold=0.05, popularity_threshold=30)
        
predictions = [(user, book, predictor.predict(user, book)) for user, book in read_test]
predictions

# %%
from sklearn.model_selection import KFold
from collections import defaultdict

usersPerBook = defaultdict(set)
bookPopularity = defaultdict(int)

for user, book, _ in ratingsTrain:
    usersPerBook[book].add(user)
    bookPopularity[book] += 1

class ImprovedPredictor:
    def __init__(self, jaccard_threshold, popularity_threshold):
        self.jaccard_threshold = jaccard_threshold
        self.popularity_threshold = popularity_threshold

    def jaccard_similarity(self, book1, book2):
        users1 = usersPerBook[book1]
        users2 = usersPerBook[book2]
        intersection = len(users1.intersection(users2))
        union = len(users1.union(users2))
        return intersection / union if union != 0 else 0

    def predict(self, user, book):
        books_read_by_user = [b for b, _ in ratingsPerUser[user]]
        max_similarity = 0
        for b_prime in books_read_by_user:
            similarity = self.jaccard_similarity(book, b_prime)
            max_similarity = max(max_similarity, similarity)

        book_popularity = bookPopularity[book]
        
        return (max_similarity >= self.jaccard_threshold) or (book_popularity >= self.popularity_threshold)

kf = KFold(n_splits=5, random_state=42, shuffle=True)
jaccard_threshold = 0.05
popularity_threshold = 30

fold_results = []
for fold, (train_index, valid_index) in enumerate(kf.split(ratingsTrain)):
    train_data = [ratingsTrain[i] for i in train_index]
    valid_data = [ratingsTrain[i] for i in valid_index]

    usersPerBook_fold = defaultdict(set)
    bookPopularity_fold = defaultdict(int)
    ratingsPerUser_fold = defaultdict(list)
    
    for user, book, rating in train_data:
        usersPerBook_fold[book].add(user)
        bookPopularity_fold[book] += 1
        ratingsPerUser_fold[user].append((book, rating))

    predictor = ImprovedPredictor(jaccard_threshold, popularity_threshold)
    
    predictions = [(user, book, predictor.predict(user, book)) for user, book, _ in valid_data]
    
    accuracy = sum((pred == (rating > 0.5)) for (_, _, rating), (_, _, pred) in zip(valid_data, predictions)) / len(valid_data)
    fold_results.append(accuracy)
    print(f"Fold {fold + 1}: Accuracy = {accuracy:.4f}")

average_accuracy = sum(fold_results) / len(fold_results)
print(f"Average Accuracy across all folds: {average_accuracy:.4f}")

predictor = ImprovedPredictor(jaccard_threshold, popularity_threshold)
predictions = [(user, book, predictor.predict(user, book)) for user, book in read_test]

print("Predictions completed.")

# %%

with open("predictions_Read.csv", "w", newline="") as predictions_file:
    writer = csv.writer(predictions_file)
    
    writer.writerow(["userID", "bookID", "prediction"])
    
    for user, book, prediction in predictions:
        writer.writerow([user, book, int(prediction)])

print("Predictions have been written to 'predictions_Read.csv'.")


