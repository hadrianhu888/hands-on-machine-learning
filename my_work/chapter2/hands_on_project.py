# Hand's on data science project with Python 

import os 
from pathlib import Path
import tarfile
import urllib.request
import sklearn as sk
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import train_test_split
from scipy.stats import binom 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join(DOWNLOAD_ROOT, "housing.tgz")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    tarball_path = Path("datasets/housing/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.tgz"
        urllib.request.urlretrieve(url, housing_path)
        with tarfile.open(housing_path, "r:gz") as tar:
            tarball_path.extractall(path="datasets/housing")
    else:
        print("File already exists")
    return pd.read_csv(os.path.join("datasets/housing", "housing.csv"))

housing = fetch_housing_data()
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

IMAGES_PATH = Path() / "images" / "chapter2"
IMAGES_PATH.mkdir(parents = True, exist_ok = True)

def save_fig(fig_id, tight_layout = True, fig_extension = "png", resolution = 300):
    path = IMAGES_PATH / fig_id
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = fig_extension, dpi = resolution)
    
plt.rc('font', size = 14)
plt.rc('axes', titlesize = 14)
plt.rc('axes', labelsize = 14)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('legend', fontsize = 14)

housing.hist(bins = 50, figsize = (20, 15))
save_fig("housing_hist.png")
plt.show()

"""Create a test set"""

def shuffle_and_split_data(data, test_ratio):
    """Split the data into training and test sets"""
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_and_split_data(housing, 0.2)
len(train_set), len(test_set)

np.random.seed(42)

def is_id_in_test_case(identifer, test_ratio):
    return crc32(np.int64(identifer)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test = ids.apply(lambda id_: is_id_in_test_case(id_, test_ratio))
    return data.loc[~in_test], data.loc[in_test]

housing_with_id = housing.reset_index()
train_set_with_id, test_set_with_id = split_data_with_id_hash(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set_with_id, test_set_with_id = split_data_with_id_hash(housing_with_id, 0.2, "id")

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

sample_size = 1000
ratio_female = 0.511
prob_too_small = binom.pmf(ratio_female, sample_size, len(train_set[train_set["ocean_proximity"] == "NEAR OCEAN"]) / len(train_set))
prob_too_large = binom.pmf(ratio_female, sample_size, len(train_set[train_set["ocean_proximity"] == "NEAR BAY"]) / len(train_set))

np.random.seed(42)

housing["income_cat"] = pd.cut(housing["median_income"], bins = [0., 1.5, 3.0, 4.5, 6., np.inf], labels = [1, 2, 3, 4, 5])

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
save_fig("housing_income_cat_bar_plot.png")  # extra code
plt.show()

splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
strat_split = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.loc[train_index]
    strat_test_set_n = housing.loc[test_index]
    strat_split.append((strat_train_set_n, strat_test_set_n))    

strat_train_set, strat_test_set = strat_split[0]

strat_train_set_n, strat_test_set_n = train_test_split(strat_train_set, test_size = 0.2, random_state = 42)

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()

compare_props.plot.bar(rot = 0, grid = True)
plt.xlabel("Income category")
plt.ylabel("Proportion of districts")
save_fig("housing_income_cat_bar_plot.png")  # extra code
plt.show()

compare_props["Stratified %"].plot.bar(rot = 0, grid = True)
plt.xlabel("Stratified %")
plt.ylabel("Proportion of districts")
save_fig("Stratified_housing_income_cat_bar_plot.png")  # extra code
plt.show()

compare_props["Random %"].plot.bar(rot = 0, grid = True)
plt.xlabel("Random %")
plt.ylabel("Proportion of districts")
save_fig("Random_housing_income_cat_bar_plot.png")  # extra code
plt.show()

for set_ in (strat_train_set, strat_test_set, test_set):
    set_.drop("income_cat", axis = 1, inplace = True)
    
housing = strat_train_set.copy()

housing.plot(kind = "scatter", x = "longitude", y = "latitude")
save_fig("bad_visualization_plot.png")  # extra code
plt.show()

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.2)
save_fig("better_visualization_plot.png")  # extra code
plt.show()

housing.plot(kind = "scatter", x = "longitude", y = "latitude", grid = True, alpha = 0.4, figsize = (10, 7), s = housing["population"] / 100, label = "population", c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True)
save_fig("housing_prices_scatterplot.png")  # extra code
plt.show()

# Download the California image
filename = "california.png"
if not (IMAGES_PATH / filename).is_file():
    homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
    url = homl3_root + "images/end_to_end_project/" + filename
    print("Downloading", filename)
    urllib.request.urlretrieve(url, IMAGES_PATH / filename)

housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed.plot(
                kind="scatter", x="Longitude", y="Latitude",
                s=housing_renamed["Population"] / 100, label="Population",
                c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
                legend=True, sharex=False, figsize=(10, 7))

california_img = plt.imread(IMAGES_PATH / filename)
axis = -124.55, -113.95, 32.45, 42.05
plt.axis(axis)
plt.imshow(california_img, extent=axis)

save_fig("california_housing_prices_plot.png")
plt.show()

# looking for corelations

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending = False)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(10, 10))
save_fig("scatter_matrix_plot.png")  # extra code
plt.show()

housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1, grid = True)
save_fig("housing_income_scatterplot.png")  # extra code
plt.show()

# experimenting with attribute combinations

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["bedrooms"] / housing["total_bedrooms"]
housing["people_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)

housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"]

null_rows_idx = housing[housing.isnull().any(axis = 1)]
housing.loc[null_rows_idx].head()

housing_option1 = housing.copy()
housing_option1.dropna(subset = ["total_bedrooms"], inplace = True)
housing_option1.loc[null_rows_idx].head()

housing_option2 = housing.copy()
housing_option2.drop("total_bedrooms", axis=1, inplace=True)  # option 2
housing_option2.loc[null_rows_idx].head()

housing_option3 = housing.copy()
median = housing["total_bedrooms"].median()
housing_option3["total_bedrooms"].fillna(median, inplace=True)  # option 3
housing_option3.loc[null_rows_idx].head()

imputer = SimpleImputer(strategy = "median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)

imputer.statistics_
imputer.median().values()

housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)

imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)
imputer.feature_names_in_

housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
housing_tr.loc[null_rows_idx].head()

imputer.strategy
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
housing_tr.loc[null_rows_idx].head()

from sklearn.ensemble import IsolationForest
isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)

isolation_forest = IsolationForest(random_state = 42)
outlier_pred = isolation_forest.fit_predict(housing_num)

housing = housing.iloc[outlier_pred == 1]
housing_labels = housing_labels.iloc[outlier_pred == 1]

housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]

ordinal_encoder.categories_

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_encoded)
housing_cat_1hot
housing_cat_1hot.toarray()

cat_encoer = OneHotEncoder(spares = False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

cat_encoder.categories_

df_test_unknown = pd.DataFrame("ocean_proximity", ["INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND", "<1H OCEAN"])
pd.get_dummies(df_test_unknown)

cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)
cat_encoder.feature_names_in_

cat_encoder.get_feature_names_out()
df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown), columns = cat_encoder.get_feature_names_out(), index = df_test_unknown.index)

df_output["ocean_proximity"] = df_test_unknown["ocean_proximity"]

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler 

min_max_scalar = MinMaxScaler(feature_range = (-1,1))
housing_num_min_max_scaled = min_max_scalar.fit_transform(housing_num)

from sklearn.preprocessing import StandardScaler 

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

fig,axs = plt.subplots(1,2, figsize = (8,3), sharey = True)
housing["population"].hist(bins = 50, ax = axis[0])
housing["population"].apply(np.log).hist(bins = 50, ax = axis[1], bins = 50)
axs[0].set_xlabel("population")
axs[1].set_xlabel("log(population)")
axs[0].set_ylabel("Long tail plot")
save_fig("long_tail_plot.png")
plt.show()

percentiles = [np.percentile(housing_num[col], [0, 25, 50, 75, 100]) for col in housing_num.columns]
flattened_median_income = pd.cut(housing["median_income"], bins = [0, 1.5, 3.0, 4.5, 6, np.inf], labels = [1, 2, 3, 4, 5])
flattened_median_income.hist(bins = 50)
plt.xlabel("Median income percentile")
plt.ylabel("NUmber of districts")
save_fig("median_income_histogram.png")
plt.show()

from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing_num["housing_median_age"].values.reshape(-1,1), np.array([35]).reshape(-1,1))

ages = np.linspace(housing["housing_median_age"].min(), housing["housing_median_age"].max()).reshape(-1,1)
gamma1 = 0.3
gamma2 = 0.03 

rbf1 = rbf_kernel(ages,[[35]], gamma = gamma1)
rbf2 = rbf_kernel(ages, [[35]], gamma = gamma2)

fig,ax1 = plt.subplots()

ax1.set_xlabel("Housing median age")
ax1.set_ylabel("Number of Districts")
ax1.hist(housing["housing_median_age"], bins = 50)

ax2 = ax1.twinx()
color = "blue"
ax2.plot(ages, rbf1, color = color) 
ax2.plot(ages, rbf2, color = color)
ax2.tick_params(axis = "y", labelcolor = color)
ax2.set_ylabel("Ages similarity", color = color)

plt.legend(loc= "upper right")
save_fig("age_similarity_plot.png")
plt.show()

from sklearn.linear_model import LinearRegression
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.values.toframe())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[[["median_income"]]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

predictions[:5]

from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(regressor = LinearRegression(), transformer = target_scaler)
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)

predictions[:5]

# Custom Transformers

from sklearn.preprocessing import FunctionTransformer

log_transformer  = FunctionTransformer(np.log1p, validate = True)
log_pop = log_transformer.housing(housing[["population"]])

rbf_transformer=  FunctionTransformer(rbf_kernel,kw_args=dict(Y=[[35]], gamma=0.3), validate = True)
age_simil_35 = rbf_transformer.fit_transform(housing[["housing_median_age"]])
age_simil_35[:5]

sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(lambda X: X[:,[0]]/X[:,[1]])
sf_simil=sf_transformer.transform(housing[["latitude", "longitude"]].values)
sf_simil[:5]

ratio_transformer=FunctionTransformer(lambda X:X[:,[0]]/X[:,[1]], validate = True)
ratio_transformer.transform(np.array([[1,2],[3,4],[5,6]]))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator,TransformerMixin):
    def __init__(self,with_mean =True):
        self.with_mean = with_mean
    
    def fit(self,X,y=None):
        X = check_array(X)
        self.mean = X.mean(axis = 0) 
        self.scale = X.std(axis = 0)
        self.n_features_in_ = X.shape[1] 
        return self
    def transform(self,X):
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X -= self.mean
        return X / self.scale_
    
from sklearn.cluster import KMeans 

class ClusterSimilarity(BaseEstimator,TransformerMixin):
    def __init__(self,n_clusters=10,gamma=1.0,random_state=None):
        self.n_clusters = n_clusters
        self.gamma= gamma
        self.random_state = random_state
    def fit(self,X,y=None,sample_weight=None):
        self.kmeans_ = KMeans(n_clusters = self.n_clusters, random_state = self.random_state)
        self.kmeans_.fit(X,sample_weight = sample_weight)
        return self
    def transform(self,X):
        return rbf_kernel(X,self.kmeans_.cluster_centers_,gamma = self.gamma)
    def get_feature_names_out(self,naems=None):
        return [f"cluster_{i}" for i in range(self.n_clusters)]

cluster_simil = ClusterSimilarity(n_clusters = 10, gamma = 0.3)
similarities=  cluster_simil.fit_transform(housing[["latitude", "longitude"]],sample_weight = housing_labels)

similarities[:5].round(2)

hosuing_renamed = housing.rename(columns = {"longitude": "lon", "latitude": "lat", "housing_median_age": "age", "total_rooms": "rooms", "total_bedrooms": "bedrooms", "population": "pop", "households": "households", "median_income": "income", "median_house_value": "value"})

housing_renamed["Max Cluster similarity"] = similarities.max(axis = 1)

housing_renamed.plot(kind="scatter", x="lon", y="lat", alpha=0.4, s=housing_renamed["pop"]/100, label="Population", figsize=(10,7), c="Max Cluster similarity", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.plot(cluster_simil.kmeans_.cluster_centers_[:,0], cluster_simil.kmeans_.cluster_centers_[:,1], "kx", markersize=15)
plt.legend(loc="upper right")
save_fig("district_cluster_plot.png")
plt.show()

# Transformation Pipelines 

