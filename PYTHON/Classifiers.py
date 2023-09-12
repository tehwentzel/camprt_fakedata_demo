from numpy.random import seed
seed(1)

from PatientSet import PatientSet
from Constants import Constants
import Metrics
from analysis import *
from Models import *
from copy import copy
import numpy as np
import pandas as pd

from collections import namedtuple, OrderedDict

from dependencies.NCA import NeighborhoodComponentsAnalysis
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import softmax
from sklearn.metrics import silhouette_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile


from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB



class MetricLearningClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components = 'auto',
                 random_state = 1,
                 resampler = None,
                 resample_after_nca = True,
                 use_softmax = True):
        self.n_components = n_components
        if n_components is not 'auto':
            self.transformer = NeighborhoodComponentsAnalysis(n_components = n_components)
        self.group_parameters = namedtuple('group_parameters', ['means', 'inv_covariance', 'max_dist'])
        self.resampler = resampler
        self.use_softmax = use_softmax
        self.resample_after_nca = resample_after_nca

    def get_optimal_components(self, x, y):
        n_components = x.shape[1]
        def get_score():
            nca = NeighborhoodComponentsAnalysis(n_components = n_components)
            nca.fit(x,y)
            return silhouette_score(nca.transform(x), y), nca
        score, nca = get_score()
        while True:
            if n_components <= 2:
                return nca
            n_components -= 1
            new_score, new_nca = get_score()
            if new_score > 1.1*score:
                score = new_score
                nca = new_nca
            else:
                return nca

    def resample(self, x, y):
        if self.resampler is not None:
            xtemp, ytemp = self.resampler.fit_resample(x,y)
            if len(np.unique(ytemp)) == len(np.unique(y)):
                x = xtemp
                y = ytemp
        return x, y

    def fit(self, x, y):
        if not self.resample_after_nca:
            x, y = self.resample(x,y)
        if self.n_components == 'auto':
            self.transformer = self.get_optimal_components(x, y)
        self.transformer.fit(x, y)
        self.groups = OrderedDict()
        if self.resample_after_nca:
            x, y = self.resample(x,y)
        for group in np.unique(y):
            self.groups[group] = self.group_params(x, y, group)

    def group_params(self, x, y, group):
        targets = np.argwhere(y == group).ravel()
        x_target = self.transformer.transform(x[targets])
        fmeans = x_target.mean(axis = 0)
        inv_cov = np.linalg.pinv(np.cov(x_target.T))
        train_dists = self.mahalanobis_distances(x, self.group_parameters(fmeans, inv_cov, 0))
        parameters = self.group_parameters(fmeans, inv_cov, train_dists.max())
        return parameters

    def mahalanobis_distances(self, x, group):
        x_offset = self.transformer.transform(x) - group.means
        left_term = np.dot(x_offset, group.inv_covariance)
        mahalanobis = np.dot(left_term, x_offset.T).diagonal()
        return mahalanobis

    def predict_proba(self, x):
        all_distances = []
        for group_id, group_params in self.groups.items():
            distances = self.mahalanobis_distances(x, group_params)
            proximity = np.clip(1 - (distances/group_params.max_dist), 0.00001, 1)
            all_distances.append(proximity)
        output = np.hstack(all_distances).reshape(-1, len(self.groups.keys()))
        if self.use_softmax:
            output = softmax(output)
        else:
            output = output/output.sum(axis = 1).reshape(-1,1)
        return output

    def predict(self, x):
        labels = list(self.groups.keys())
        probs = self.predict_proba(self, x)
        max_probs =  np.argmax(probs, axis = 1).ravel()
        ypred = np.zeros(max_probs.shape).astype(np.dtype(labels[0]))
        for i in range(max_probs.shape[0]):
            ypred[i] = labels[max_probs[i]]
        return ypred[i]

    def fit_predict(self, x, y):
        self.fit(x,y)
        return self.predict(x)

class BayesWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, bayes = BernoulliNB(alpha = 0), n_categories = None):
        if n_categories is None:
            self.encoder = OneHotEncoder(categories = 'auto',
                                         sparse = False,
                                         handle_unknown = 'ignore')
        else:
            self.encoder = KBinsDiscretizer(n_bins = n_categories, encode = 'ordinal')
        self.bayes = bayes

    def fit(self, x, y):
        x = self.encoder.fit_transform(x)
        self.bayes.fit(x,y)
        return self

    def predict(self, x):
        xpred = self.encoder.transform(x)
        return self.bayes.predict(xpred)

    def predict_proba(self, x):
        xpred = self.encoder.transform(x)
        return self.bayes.predict_proba(xpred)

    def fit_predict(self, x, y, probability = False):
        self.fit(x,y)
        if probability:
            return self.predict_proba(x)
        return self.predict(x)

class RecallBasedModel:

    def __init__(self, model, recall_threshold = None, feature_selection = False):
        self.model = copy(model)
        self.recall_threshold = recall_threshold
        self.probability_threshold = None
        self.feature_selection = feature_selection

    def fit(self, x, y):
        self.get_feature_args(x,y)
        xfit = x[:, self.features_to_use]
        self.model.fit(xfit, y.ravel())
        if self.recall_threshold is not None:
            self.tune_threshold(xfit, y.ravel())

    def predict(self, x):
        xsubset = x[:, self.features_to_use]
        if self.probability_threshold is not None:
            probs = self.model.predict_proba(xsubset)[:,1].ravel()
            prediction = probs >= self.probability_threshold
        else:
            prediction = self.model.predict(xsubset)
        return prediction.astype('bool')

    def fit_predict(self, x, y):
        self.fit(x,y)
        return self.predict(x,y);

    def tune_threshold(self, x, y):
        ypred = self.model.predict_proba(x)
        sorted_scores = sorted(ypred[:,1], key = lambda x: -x)
        threshold_i = 0
        ythresh = ypred[:,1] >= sorted_scores[threshold_i]
        while recall_score(y, ythresh) < self.recall_threshold and threshold_i < len(sorted_scores) - 1:
            threshold_i += 1
            ythresh = ypred[:,1] >= sorted_scores[threshold_i]
        self.probability_threshold = sorted_scores[threshold_i]
        self.all_thresholds = sorted_scores

    def increment_threshold(self, increment = 1):
        current_index = self.all_thresholds.index(self.probability_threshold)
        new_index = np.clip(0, current_index + increment, len(self.all_thresholds) -1)
        self.probability_threshold = self.all_thresholds[new_index]

    def get_feature_args(self, x, y, percentile = 80, k = 40):
        if self.feature_selection == 'info':
            info_score = mutual_info_classif(x, y)
            self.features_to_use = np.argwhere(info_score > 0).ravel()
            if len(self.features_to_use) <= 1:
                self.features_to_use = np.argwhere(x.std(axis = 0) > 0).ravel()
        elif self.feature_selection == 'percentile':
            selector = SelectPercentile(percentile = percentile)
            selector.fit(x,y)
            self.features_to_use = np.argwhere(selector.get_support()).ravel()
        elif self.feature_selection == 'kbest':
            k = np.min([int(np.ceil(percentile*x.shape[1]/100)), k])
            selector = SelectKBest(k = k).fit(x,y)
            self.features_to_use = np.argwhere(selector.get_support()).ravel()
        else:
            self.features_to_use = np.argwhere(x.std(axis = 0) > 0).ravel()

    def __str__(self):
        string = str(self.model)
        string += '\n num features ' + str(len(self.features_to_use))
        string += '\n threshold ' + str(self.probability_threshold)
        string += '\n num_in_threshold: ' + str(self.all_thresholds.index(self.probability_threshold))
        return string + '\n'

class StackedClassifier:

    def __init__(self, default_models,
                 min_misses = 0,
                 recall_threshold = None,
                 feature_selection = False,
                 num_feature_splits = 2):
        self.gen_models = lambda: [RecallBasedModel(copy(m), recall_threshold, feature_selection) for m in default_models]
        self.min_misses = min_misses
        self.num_models = len(default_models)
        self.num_feature_splits = num_feature_splits

    def fit(self, x, y):
        models = []
        x_groups = self.split_features(x, y)
        for x_set in x_groups:
            model_set = self.gen_models()
            model_group = []
            for model in model_set:
                model.fit(x_set, y)
                model_group.append(model)
            models.append(model_group)
        self.models = models

    def predict(self, x, min_votes = None):
        min_votes = self.num_models if min_votes is None else min_votes
        x_groups= self.split_features(x)
        assert(len(x_groups) == len(self.models))
        predictions = []
        for group in range(len(x_groups)):
            model_set = self.models[group]
            x_set = x_groups[group]
            for model in model_set:
                ypred = model.predict(x_set)
                predictions.append(ypred.reshape(-1,1))
        predictions = np.hstack(predictions)
        ypred = predictions.sum(axis = 1) >= min_votes
        return ypred

    def fit_predict(self, x,y, min_votes = None):
        self.fit(x,y)
        return self.predict(x, min_votes)

    def split_features(self, x,y = None):
        #feature clusteirng or whatever here
        #just split if train is false, find grups if true
        if self.num_feature_splits <= 1:
            return [x]
        if y is not None:
            info_score = np.nan_to_num(mutual_info_classif(x, y))
            scores = np.argsort(-info_score)
            self.feature_group_args = [[] for g in range(self.num_feature_splits)]
            current_feature = 0
            while current_feature < x.shape[1]:
                for group in range(self.num_feature_splits):
                    self.feature_group_args[group].append(scores[current_feature])
                    current_feature += 1
                    if current_feature >= x.shape[1]:
                        break
            #this code was for clustering the features first
#            self.feature_group_args = []
#            clusterer = AgglomerativeClustering(n_clusters = self.num_feature_splits)
#            features = x.transpose() #should I regularize here?
#            groups = clusterer.fit_predict(features)
#            for g in np.unique(groups):
#                args = np.argwhere(groups == g).ravel()
#                self.feature_group_args.append(args)
        x_groups = []
        for group_args in self.feature_group_args:
            x_groups.append(x[:, group_args])
        return x_groups

    def __repr__(self):
        string = ""
        for modelset in self.models:
            for model in modelset:
                string += str(model)
        return string

class IterativeClassifier:

    def __init__(self, default_models,
                 min_misses = 0,
                 recall_threshold = None,
                 feature_selection = False,
                 num_feature_splits = 1):
        self.gen_ensemble = lambda: StackedClassifier(default_models,
                                                      min_misses,recall_threshold,
                                                      feature_selection,
                                                      num_feature_splits)
        self.min_misses = min_misses
        self.num_models = len(default_models)
        self.num_feature_splits = num_feature_splits

    def fit(self, x, y):
        current_model = self.gen_ensemble()
        models = [current_model]
        y_pred = current_model.fit_predict(x,y)
        while not self.is_done(y, y_pred) and len(models) < 7:
            args = np.argwhere(y_pred == 0).ravel()
#            print(len(args))
            if len(args) < 5:
                break
            xsubset = x[args,:]
            ysubset = y[args]
            new_model = self.gen_ensemble()
            new_predict = new_model.fit_predict(xsubset, ysubset)
            new_args = np.argwhere(new_predict > 0)
            if len(new_args) <= 0:
                break
            y_pred[args[new_args]] = True
            models.append(new_model)
        self.models = models
#        print()

    def predict(self, x):
        y_pred = np.zeros((x.shape[0],))
        for model in self.models:
            valid = np.argwhere(model.predict(x) > 0).ravel()
            y_pred[valid] = 1
        return y_pred

    def predict_proba(self, x):
        y_pred = np.zeros((x.shape[0],))
        model_pos = 1
        for model in self.models:
            valid = np.argwhere(model.predict(x) > 0).ravel()
            y_pred[valid] = 1/model_pos
            model_pos += 1
        return y_pred

    def fit_predict(self, x, y):
        self.fit(x,y)
        return self.predict(x)

    def is_done(self, y, y_pred):
        false_predictions = np.argwhere(y_pred == 0).ravel()
        false_negatives = y[false_predictions].sum()
        return (false_negatives <= self.min_misses)

    def test(self, x, y):
        ypred = self.fit_predict(x, y)
        yproba = self.predict_proba(x)
        print('AUC score', roc_auc_score(y, yproba))
        print('recall ', recall_score(y, ypred))
        print('f1 ', f1_score(y, ypred))
        return yproba