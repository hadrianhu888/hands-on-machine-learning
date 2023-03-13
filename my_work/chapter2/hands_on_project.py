# Hand's on data science project with Python 

import os 
from pathlib import Path
import tarfile
import urllib.request
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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
HOUSING_URL = os.path.join(DOWNLOAD_ROOT,"datasets/housing/housing.tgz")

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
housing["population"].apply(np.log).hist(bins = 50, ax = axis[1])
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
from sklearn.pipeline import Pipeline
from sklearn import set_config
set_config(display='diagram')
# set up the pipeline
num_pipeline = set_config()

housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:5].round(2)

def monkey_patch_get_signature_names_out():
    """Monkey patch some classes which did not handle get_feature_names_out()
       correctly in Scikit-Learn 1.0.*."""
    from inspect import Signature, signature, Parameter
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    default_get_feature_names_out = StandardScaler.get_feature_names_out

    if not hasattr(SimpleImputer, "get_feature_names_out"):
      print("Monkey-patching SimpleImputer.get_feature_names_out()")
      SimpleImputer.get_feature_names_out = default_get_feature_names_out

    if not hasattr(FunctionTransformer, "get_feature_names_out"):
        print("Monkey-patching FunctionTransformer.get_feature_names_out()")
        orig_init = FunctionTransformer.__init__
        orig_sig = signature(orig_init)

        def __init__(*args, feature_names_out=None, **kwargs):
            orig_sig.bind(*args, **kwargs)
            orig_init(*args, **kwargs)
            args[0].feature_names_out = feature_names_out

        __init__.__signature__ = Signature(
            list(signature(orig_init).parameters.values()) + [
                Parameter("feature_names_out", Parameter.KEYWORD_ONLY)])

        def get_feature_names_out(self, names=None):
            if callable(self.feature_names_out):
                return self.feature_names_out(self, names)
            assert self.feature_names_out == "one-to-one"
            return default_get_feature_names_out(self, names)

        FunctionTransformer.__init__ = __init__
        FunctionTransformer.get_feature_names_out = get_feature_names_out
        
monkey_patch_get_signature_names_out()

df_housing_num_prepared = pd.DataFrame(housing_num_prepared, columns = num_pipeline.get_feature_names_out(),index=housing_num.index)

df_housing_num_prepared.head(2)

num_pipeline.steps

num_pipeline[1]
num_pipeline[:-1]

make_pipeline = Pipeline([("simpleimputer",SimpleImputer(strategy="mean")),("standard_scaler",StandardScaler())])

num_pipeline.named_steps["simpleimputer"]

num_pipeline.set_params(simpleimputer__strategy="median")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
num_attributes=["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"]
cat_attribues=["ocean_proximity"]
cat_pipeline=make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknnown="ignore"))
preprocessing=ColumnTransformer([("num",num_pipeline,num_attributes),("cat",cat_pipeline,cat_attribues)])
housing_prepared=preprocessing.fit_transform(housing)
housing_prepared_fr =pd.DataFrame(housing_prepared,columns=num_attributes+list(preprocessing.named_transformers_["cat"].named_steps["onehotencoder"].get_feature_names_out(cat_attribues)),index=housing.index)
housing_prepared_fr.head(2)

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())
    
make_column_selector = lambda pattern: lambda df: [col for col in df.columns if re.search(pattern, col)]

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                        StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

housing_prepared=preprocessing.fit_transform(housing)
housing_prepared.shape

preprocessing.get_feature_names_out()

# Select and Train a Model 

from sklearn.linear_model import LinearRegression

lin_reg = Pipeline(['preprocessing', preprocessing, 'linear_regression', LinearRegression()])
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2)  # -2 = rounded to the nearest hundred

housing_labels.iloc[:5].values

# extra code – computes the error ratios discussed in the book
error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))

from sklearn.metrics import mean_squared_error

lin_rmse = mean_squared_error(housing_labels, housing_predictions,squared=False)
lin_rmse

from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions,squared=False)
tree_rmse

#Fine tune your model

from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([("preprocessing", preprocessing), ("linear_regression", LinearRegression())])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
        'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
        'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,scoring = "neg_mean_squared_error",return_train_score=True)
grid_search.fit(housing, housing_labels)

print(str(full_pipeline.get_params().keys())[1:-1] + ", scoring")

grid_search.best_params_

grid_search.best_estimator_

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

cv_res = cv_res[["mean_test_score", "std_test_score", "params"]]
score_cols = np.sqrt(-cv_res["mean_test_score"])
cv_res.columns = ["mean_test_score", "std_test_score", "params"]
cv_res[score_cols] = -cv_res[score_cols].round(2).astype(np.int64)

# Randomized Search

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
# import plt 
import re 
import pandas as pd

param_distribs = {'preprocessing__geo__n_clusters': randint(low=5, high=15),'random_forest__max_features': randint(low=4, high=10)}
rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,scoring = "neg_mean_squared_error",return_train_score=True)
rnd_search.fit(housing, housing_labels)

cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_preprocessing__geo__n_clusters", "mean_test_score", "std_test_score"]]
cv_res.columns = ["n_clusters", "mean_test_score", "std_test_score"]
cv_res[score_cols] = -cv_res[score_cols].round(2).astype(np.int64)
cv_res.head()

from scipy.stats import randint, uniform, geom, expon

xs1 = np.arange(0, 7 + 1)
randint_distrib = randint(0, 7 + 1).pmf(xs1)

xs2 = np.linspace(0, 7, 500)
uniform_distrib = uniform(0, 7).pdf(xs2)

xs3 = np.arange(0, 7 + 1)
geom_distrib = geom(0.5).pmf(xs3)

xs4 = np.linspace(0, 7, 500)
expon_distrib = expon(scale=1).pdf(xs4)

plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.bar(xs1, randint_distrib, label="scipy.randint(0, 7 + 1)")
plt.ylabel("Probability")
plt.legend()
plt.axis([-1, 8, 0, 0.2])

plt.subplot(2, 2, 2)
plt.fill_between(xs2, uniform_distrib, label="scipy.uniform(0, 7)")
plt.ylabel("PDF")
plt.legend()
plt.axis([-1, 8, 0, 0.2])

plt.subplot(2, 2, 3)
plt.bar(xs3, geom_distrib, label="scipy.geom(0.5)")
plt.xlabel("Hyperparameter value")
plt.ylabel("Probability")
plt.legend()
plt.axis([0, 7, 0, 1])

plt.subplot(2, 2, 4)
plt.fill_between(xs4, expon_distrib, label="scipy.expon(scale=1)")
plt.xlabel("Hyperparameter value")
plt.ylabel("PDF")
plt.legend()
plt.axis([0, 7, 0, 1])

plt.show()

from scipy.stats import reciprocal

xs1 = np.linspace(0, 7, 500)
expon_distrib = expon(scale=1).pdf(xs1)

log_xs2 = np.linspace(-5, 3, 500)
log_expon_distrib = np.exp(log_xs2 - np.exp(log_xs2))

xs3 = np.linspace(0.001, 1000, 500)
reciprocal_distrib = reciprocal(0.001, 1000).pdf(xs3)

log_xs4 = np.linspace(np.log(0.001), np.log(1000), 500)
log_reciprocal_distrib = uniform(np.log(0.001), np.log(1000)).pdf(log_xs4)

plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.fill_between(xs1, expon_distrib,
                 label="scipy.expon(scale=1)")
plt.ylabel("PDF")
plt.legend()
plt.axis([0, 7, 0, 1])

plt.subplot(2, 2, 2)
plt.fill_between(log_xs2, log_expon_distrib,
                 label="log(X) with X ~ expon")
plt.legend()
plt.axis([-5, 3, 0, 1])

plt.subplot(2, 2, 3)
plt.fill_between(xs3, reciprocal_distrib,
                 label="scipy.reciprocal(0.001, 1000)")
plt.xlabel("Hyperparameter value")
plt.ylabel("PDF")
plt.legend()
plt.axis([0.001, 1000, 0, 0.005])

plt.subplot(2, 2, 4)
plt.fill_between(log_xs4, log_reciprocal_distrib,
                 label="log(X) with X ~ reciprocal")
plt.xlabel("Log of hyperparameter value")
plt.legend()
plt.axis([-8, 1, 0, 0.2])

plt.show()

# Analyze the Best Models and Their Errors

final_model = rnd_search.best_estimator_
feature_importances = final_model.named_steps["random_forest"].feature_importances_
feature_importances.round(2)

sorted(zip(feature_importances, final_model.named_steps["preprocessing"].transformers_[0][1].get_feature_names()), reverse=True)

# Evaluate Your System on the Test Set

X_test = strat_test_set.drop("median_house_value", axis=1)
Y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_remse = mean_squared_error(Y_test, final_predictions, squared=False)
final_remse
print(final_remse)

from scipy import stats 

confidence = 0.95
squared_errors = (final_predictions - Y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,loc=squared_errors.mean(), scale=stats.sem(squared_errors)))

# Extra code - shows how to compute the confidence inverval for the RMSE 

m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)

import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import haversine_distances

def column_ratio(X):
    return X[:, 0] / X[:, 1]

final_model_reloaded = joblib.load("my_california_housing_model.pkl")

new_data = housing.iloc[:5]
predictions = final_model_reloaded.predict(new_data)

predictions

# try with a vector machine regressor

from sklearn import svm

housing_data = housing.drop("median_house_value", axis=1)
housing_data.head()

housing_labels = housing["median_house_value"].copy()
# use SVM regressor
housing_SVM_regressor = svm.SVR()
# plot the regressor 
housing_SVM_regressor.fit(housing_data, housing_labels)
plot = housing_SVM_regressor.predict(housing_data)
plt.plot(plot)
plt.show()

# use RandomizedSearchCV to find the best hyperparameters for the SVM regressor

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': reciprocal(20, 200000), 'gamma': expon(scale=1.0), 'epsilon': expon(scale=1.0)}
housing_randomized_search_cv = RandomizedSearchCV(housing_SVM_regressor, param_distributions=param_distributions, n_iter=10, cv=5, verbose=2, n_jobs=-1)
housing_randomized_search_cv.fit(housing_data, housing_labels)
housing_randomized_search_cv.best_estimator_
print(housing_randomized_search_cv.best_estimator_)

# Create a transformer that prepares a pipeline that finds the most important features

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

class FeatureSelectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_features):
        self.num_features = num_features

    def fit(self, X, y=None):
        selector = SelectKBest(f_classif, k=self.num_features)
        selector.fit(X, y)
        self.selected_features_ = selector.get_support()
        return self

    def transform(self, X, y=None):
        return X[:,self.selected_features_]

pipeline = Pipeline([
    ('feature_selector', FeatureSelectorTransformer(num_features=10)),
    ('classifier', RandomForestClassifier())
])

# implement the transformation pipeline above to find the most important features in the California housing dataset

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

RandomForestRegressor = RandomForestClassifier()

# Load the California housing dataset
dataset = fetch_california_housing(as_frame=True)
X = dataset.data
y = dataset.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('feature_selector', FeatureSelectorTransformer(num_features=5)),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)

# Complete the pipeline with a preprocessing step that scales the data

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the California housing dataset
dataset = fetch_california_housing(as_frame=True)
X = dataset.data
y = dataset.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
numeric_transformer = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_regression, k=3))
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']),
    ('cat', categorical_transformer, ['OceanProximity'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)

# Plot the feature importances
feature_importances = pipeline.named_steps['regressor'].feature_importances_
features = preprocessor.transformers_[0][2] + list(preprocessor.transformers_[1][1].get_feature_names(['OceanProximity']))
sorted_indices = feature_importances.argsort()
plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices])
plt.yticks(range(len(sorted_indices)), [features[i] for i in sorted_indices])
plt.xlabel('Feature importance')
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR 

param_grid = [{'svr__kernel': ['linear'], 'svr__C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.]}, {'svr__kernel': ['rbf'], 'svr__C': [1.0, 3.0, 10., 30., 100., 300., 1000.], 'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}, {'svr__kernel': ['poly'], 'svr__C': [1.0, 3.0, 10., 30., 100., 300., 1000.], 'svr__degree': [2, 3], 'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}]

svr_pipeline = Pipeline(["preprocessing", preprocessing], ["svr", SVR()])
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(housing.iloc[:, :-1], housing["median_house_value"])

svr_grid_search_rmse = -grid_search.best_score_
svr_grid_search_rmse

grid_search.best_params_

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': reciprocal(20, 200_000),
        'svr__gamma': expon(scale=1.0),
    }

rnd_search = RandomizedSearchCV(svr_pipeline,
                                param_distributions=param_distribs,
                                n_iter=50, cv=3,
                                scoring='neg_root_mean_squared_error',
                                random_state=42)
rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

svr_rnd_search_rmse = -rnd_search.best_score_
svr_rnd_search_rmse

rnd_search.best_params_

np.random.seed(42)

s = expon(scale=1).rvs(100_000)  # get 100,000 samples
((s > 0.105) & (s < 2.29)).sum() / 100_000


selector_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('selector', SelectFromModel(RandomForestRegressor(random_state=42),
                                 threshold=0.005)),  # min feature importance
    ('svr', SVR(C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"])),
])

selector_rmses = -cross_val_score(selector_pipeline,
                                  housing.iloc[:5000],
                                  housing_labels.iloc[:5000],
                                  scoring="neg_root_mean_squared_error",
                                  cv=3)
pd.Series(selector_rmses).describe()

from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import MetaEstimatorMixin, clone

class FeatureFromRegressor(MetaEstimatorMixin, BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        estimator_ = clone(self.estimator)
        estimator_.fit(X, y)
        self.estimator_ = estimator_
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator, "feature_names_in_"):
            self.feature_names_in_ = self.estimator.feature_names_in_
        return self  # always return self!
    
    def transform(self, X):
        check_is_fitted(self)
        predictions = self.estimator_.predict(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def get_feature_names_out(self, names=None):
        check_is_fitted(self)
        n_outputs = getattr(self.estimator_, "n_outputs_", 1)
        estimator_class_name = self.estimator_.__class__.__name__
        estimator_short_name = estimator_class_name.lower().replace("_", "")
        return [f"{estimator_short_name}_prediction_{i}"
                for i in range(n_outputs)]
        
from sklearn.utils.estimator_checks import check_estimator

check_estimator(FeatureFromRegressor(KNeighborsRegressor()))

knn_reg = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn_transformer = FeatureFromRegressor(knn_reg)
geo_features = housing[["latitude", "longitude"]]
knn_transformer.fit_transform(geo_features, housing_labels)

knn_transformer.get_feature_names_out()

from sklearn.base import clone

transformers = [(name, clone(transformer), columns)
                for name, transformer, columns in preprocessing.transformers]
geo_index = [name for name, _, _ in transformers].index("geo")
transformers[geo_index] = ("geo", knn_transformer, ["latitude", "longitude"])

new_geo_preprocessing = ColumnTransformer(transformers)

new_geo_pipeline = Pipeline([
    ('preprocessing', new_geo_preprocessing),
    ('svr', SVR(C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"])),
])

new_pipe_rmses = -cross_val_score(new_geo_pipeline,
                                  housing.iloc[:5000],
                                  housing_labels.iloc[:5000],
                                  scoring="neg_root_mean_squared_error",
                                  cv=3)
pd.Series(new_pipe_rmses).describe()

param_distribs = {
    "preprocessing__geo__estimator__n_neighbors": range(1, 30),
    "preprocessing__geo__estimator__weights": ["distance", "uniform"],
    "svr__C": reciprocal(20, 200_000),
    "svr__gamma": expon(scale=1.0),
}

new_geo_rnd_search = RandomizedSearchCV(new_geo_pipeline,
                                        param_distributions=param_distribs,
                                        n_iter=50,
                                        cv=3,
                                        scoring='neg_root_mean_squared_error',
                                        random_state=42)

new_geo_rnd_search_rmse = -new_geo_rnd_search.best_score_
new_geo_rnd_search_rmse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X_orig = X
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        if hasattr(X_orig, "columns"):
            self.feature_names_in_ = np.array(X_orig.columns, dtype=object)
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError("Unexpected number of features")
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
    
    def inverse_transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError("Unexpected number of features")
        X = X * self.scale_
        return X + self.mean_ if self.with_mean else X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return getattr(self, "feature_names_in_",
                           [f"x{i}" for i in range(self.n_features_in_)])
        else:
            if len(input_features) != self.n_features_in_:
                raise ValueError("Invalid number of features")
            if hasattr(self, "feature_names_in_") and not np.all(
                self.feature_names_in_ == input_features
            ):
                raise ValueError("input_features ≠ feature_names_in_")
            return input_features
        
from sklearn.utils.estimator_checks import check_estimator

check_estimator(StandardScalerClone())

np.random.seed(42)
X = np.random.rand(1000, 3)

scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(X)

assert np.allclose(X_scaled, (X - X.mean(axis=0)) / X.std(axis=0))

scaler = StandardScalerClone(with_mean=False)
X_scaled_uncentered = scaler.fit_transform(X)

assert np.allclose(X_scaled_uncentered, X / X.std(axis=0))

scaler = StandardScalerClone()
X_back = scaler.inverse_transform(scaler.fit_transform(X))

assert np.allclose(X, X_back)

assert np.all(scaler.get_feature_names_out() == ["x0", "x1", "x2"])
assert np.all(scaler.get_feature_names_out(["a", "b", "c"]) == ["a", "b", "c"])

df = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(df)

assert np.all(scaler.feature_names_in_ == ["a", "b"])
assert np.all(scaler.get_feature_names_out() == ["a", "b"])