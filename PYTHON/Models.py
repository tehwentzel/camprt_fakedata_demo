# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:15:21 2019

@author: Andrew
"""
from numpy.random import seed
seed(1)
import numpy as np
from Constants import Constants
from ErrorChecker import ErrorChecker
import Metrics
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, AgglomerativeClustering
from copy import copy
from sklearn.preprocessing import quantile_transform
from sklearn.tree import DecisionTreeClassifier
from multiprocessing import Pool


class Estimator(ABC):

    def __init__(self, match_threshold = .94, match_type = 'threshold', min_matches = 8):
        #match type is how the number of matches are selected
        #give clusters to make it based on the class.
        #threshold uses similarity score, uses max(min_matches, patients with score > match threshold)
        self.min_matches = min_matches
        #similarity needed to be a match
        self.match_threshold = match_threshold
        self.match_type = match_type
        return

    @abstractmethod
    def predict_doses(self, similarity, data):
        pass

    @abstractmethod
    def get_matches(self, similarity_matrix, data):
        pass

    def get_error(self, predicted_doses, dose_matrix):
        differences = np.abs(predicted_doses - dose_matrix)
        percent_error = np.sum(differences, axis = 1)/np.sum(dose_matrix, axis = 1)
        return percent_error

    def evaluate(self, similarity_matrix, data):
        predicted_doses = self.predict_doses(similarity_matrix, data)
        percent_error = self.get_error(predicted_doses, data.doses)
        return(percent_error)

class SupervisedModel(ABC):

    @abstractmethod
    def get_true_matches(self, data):
        pass

    def get_match_error(self, data):
        #get error in between patient dose distrubtions
        #data can be the dose matrix or a patientset
        if isinstance(data, np.ndarray):
            doses = data
        else:
            doses = data.doses
        n_patients = doses.shape[0]
        error_matrix = np.zeros((n_patients, n_patients))
        for p1 in range(n_patients):
            for p2 in range(p1 + 1, n_patients):
                dose_difference = np.abs(doses[p1,:] - doses[p2, :])
                error_matrix[p1, p2] = np.sum(dose_difference)/np.sum(doses[p1, :])
        error_matrix += error_matrix.transpose()
        return error_matrix

class SymmetryAugmentedModel(Estimator):

    def __init__(self, match_threshold = .94, match_type = 'threshold', min_matches = 8):
        #match type is how the number of matches are selected
        #give clusters to make it based on the class.
        #threshold uses similarity score, uses max(min_matches, patients with score > match threshold)
        super().__init__(match_threshold, match_type, min_matches)


    def tsim(self, d1, d2, adjacency):
        scores = []
        if d2.sum() == np.inf:
            return 0
        for organ_set in adjacency:
            scores.append(Metrics.jaccard_distance(d1[organ_set], d2[organ_set]))
        return np.mean(scores)

    def predict_doses(self, similarity, data):
        flip_args = Metrics.get_flip_args()
        adjacency = TJaccardModel().get_adjacency_lists(data.organ_distances,
                                 np.arange(Constants.num_organs))
        normal_distances = data.tumor_distances
        flipped_distances = data.tumor_distances[:, flip_args]
        flipped_doses = data.doses[:, flip_args]
        dose_predictions = np.zeros((data.get_num_patients(),
                                     Constants.num_organs))
        for p1 in range(data.get_num_patients()):
            matches = []
            for p2 in range(0, data.get_num_patients()):
                if p1 == p2:
                    continue
                match = self.get_patient_similarity(p1, p2,
                                                 normal_distances,
                                                 flipped_distances,
                                                 data.doses,
                                                 flipped_doses,
                                                 adjacency)
                matches.append(match)
            matches = sorted(matches, key = lambda x: -x[0])
            n_matches = self.get_num_matches(p1, matches, data.classes)
            prediction = np.array([x[1] for x in matches[0:n_matches]])
            weights = np.array([x[0] for x in matches[0:n_matches]]).reshape(-1,1)
            if weights.mean() <= 0:
                print(weights, p1, [x[0] for x in matches])
            dose_predictions[p1,:] = np.mean(prediction*weights, axis = 0)/np.mean(weights)
        return(dose_predictions)

    def get_num_matches(self, p, matches, clusters):
        #for later better use probs
        if self.match_type == 'threshold':
            good_matches= np.sum([1 for m in matches if m[0] > self.match_threshold])
            matches = max([self.min_matches, good_matches])
        elif self.match_type == 'clusters':
            num_cluster_values = len(np.where(clusters == clusters[p])[0])
            matches = max([int(np.sqrt(num_cluster_values) + 1), self.min_matches])
        else:
            matches = self.min_matches
        return matches

    def get_matches(self, similarity_matrix, data):
        flip_args = Metrics.get_flip_args()
        adjacency = TJaccardModel().get_adjacency_lists(data.organ_distances,
                                 np.arange(Constants.num_organs))
        normal_distances = data.tumor_distances
        flipped_distances = data.tumor_distances[:, flip_args]
        flipped_doses = data.doses[:, flip_args]
        all_matches = []
        for p1 in range(data.get_num_patients()):
            matches = []
            for p2 in range(0, data.get_num_patients()):
                match = self.get_patient_similarity(p1, p2,
                                                 normal_distances,
                                                 flipped_distances,
                                                 data.doses,
                                                 flipped_doses,
                                                 adjacency)
                matches.append(match)
            n_matches = self.get_num_matches(p1, matches, data.classes)
            similarities = np.array([-m[0] for m in matches])#this is inverse because argsort is ascending
            match_args = (np.argsort(similarities))[:n_matches]
            all_matches.append(match_args)
        return all_matches

    def get_patient_similarity(self, p1, p2,
                            normal_distances, flipped_distances,
                            doses, flipped_doses, adjacency):
        if p1 == p2:
            return (-np.inf, doses[p2])
        base_similarity = self.tsim(normal_distances[p1], normal_distances[p2], adjacency)
        flipped_similarity = self.tsim(normal_distances[p1], flipped_distances[p2], adjacency)
        if base_similarity > flipped_similarity:
            match = (base_similarity, doses[p2])
        else:
            match = (flipped_similarity, flipped_doses[p2])
        return match


class KnnEstimator(Estimator):
    #class that works as a modified knn predictor for doses
    #mainn function is evaluate, which gives you the precent prediction error given a patientset and a similarity matrix
    def __init__(self, match_threshold = .94, match_type = 'threshold', min_matches = 8):
        #match type is how the number of matches are selected
        #give clusters to make it based on the class.
        #threshold uses similarity score, uses max(min_matches, patients with score > match threshold)
        super().__init__(match_threshold, match_type, min_matches)

    def predict_doses(self, similarity_matrix, data):
        n_patients = data.get_num_patients()
        dose_matrix = data.doses
        outliers = ErrorChecker().get_data_outliers(data.doses)
        is_augmented = similarity_matrix.shape[0] > n_patients
        if is_augmented: #if matrix is agumented
            dose_matrix = Metrics.augment_mirrored(dose_matrix)
            outliers = outliers | set([o + n_patients for o in outliers])
        similarity = np.copy(similarity_matrix[:n_patients, :])
        clusters = data.classes
        predicted_doses = np.zeros((n_patients, dose_matrix.shape[1]))
        for p in range( n_patients ):
            scores = similarity[p, :]
            if p not in outliers:
                scores[list(outliers)] = 0
            num_matches = self.get_num_matches(p, scores, clusters)
            args = np.argsort(-scores)
            args = args[0 : num_matches]

            predicted_doses[p, :] = self.get_prediction(dose_matrix, scores, args, p)
        return(predicted_doses)

    def get_prediction(self, dose_matrix, scores, args, patient):
        #default, l is bascially a flag to do the version that attempts optimization
        matched_scores = scores[args].reshape(len(args), 1)
        matched_doses = dose_matrix[args, :]
        if matched_scores.mean() > 0:
            predicted_doses = np.mean(matched_scores*matched_doses, axis = 0)/matched_scores.mean()
        else:
#            print(patient, 'doesnt have any matches')
            predicted_doses = dose_matrix.mean(axis=0) #return average if bad
        return predicted_doses

    def get_prediction_with_optimization(self,data,scores,args,patient):
        #version of matching that attempts to change scoring so the mean matches distance matrix lines up better
        #shouldn't be used, but I took like 5 hours to get this to work so I don't want to delete it
        dose_matrix = data.doses
        matched_scores = scores[args].reshape(len(args), 1)
        matched_doses = dose_matrix[args, :]
        matched_distances = data.tumor_distances[args, :]
        current_distances = data.tumor_distances[patient,:]
        def loss_function(score_vector):
            weighted_distances = np.mean(matched_distances*score_vector.reshape(-1,1), axis = 0)/score_vector.mean()
            distance_loss = np.linalg.norm(weighted_distances - current_distances)/np.linalg.norm(current_distances)
            score_loss = np.sum((score_vector - matched_scores)**2)
            return self.l*distance_loss**2 + score_loss
        bounds = [(0,1) for dummy in matched_scores]
        x0 = np.copy(matched_scores)
        optimized = minimize(loss_function, x0, bounds = bounds)
        if optimized.success:
            weights = optimized.x.reshape(-1,1)
        else:
            print('failure')
            weights = matched_scores
        return np.mean(weights*matched_doses, axis = 0)/weights.mean()


    def get_matches(self, similarity_matrix, data):
        dose_matrix = data.doses
        clusters = data.classes
        #should return a list of matched patients
        matches = []
        outliers = ErrorChecker().get_data_outliers(data.doses)
        similarity = np.copy(similarity_matrix)
        for p in range( dose_matrix.shape[0] ):
            patient_matches = self.get_patient_matches(p, similarity[p], data, outliers, clusters)
            matches.append(patient_matches)
        return(matches)

    def get_patient_matches(self, p, scores, data, outliers, clusters):
        num_matches = self.get_num_matches(p, scores, clusters)
        if p not in outliers:
            scores[list(outliers)] = 0
        args = np.argsort(-scores)
        args = args[0 : num_matches] + 1
        return args

    def get_num_matches(self, p, score_vector, clusters):
        #for later better use probs
        if self.match_type == 'threshold':
            good_matches= len(np.where(score_vector > self.match_threshold)[0])
            matches = max([self.min_matches, good_matches])
        elif self.match_type == 'clusters':
            num_cluster_values = len(np.where(clusters == clusters[p])[0])
            matches = max([int(np.sqrt(num_cluster_values) + 1), self.min_matches])
        else:
            matches = self.min_matches
        return matches

class PSUpsampler():

    def __init__(self, clusterer = None):
        if clusterer is None:
            self.clusterer = AgglomerativeClustering(n_clusters = 4)
        else:
            self.clusterer = clusterer

    def get_cluster_ratios(self, clusters):
        max_cluster_count = np.max([len(np.argwhere(clusters == c)) for c in np.unique(clusters)])
        class_offset ={c: max_cluster_count - len(np.argwhere(clusters == c)) for c in sorted(np.unique(clusters))}
        return class_offset, max_cluster_count

    def unsupervised_upsample(self, train_x, target_x):
        clusters = self.clusterer.fit_predict(np.vstack(train_x, target_x))
        target_class = clusters[-1]
        train_y = clusters[: train_x.shape[0] ]
        return self.upsample(train_x, train_y, target_class = target_class)

    def upsample(self, x, y, target_class = None):
        offsets, max_cluster_count = self.get_cluster_ratios(y)
        if target_class is None:
            target_class = np.unique(y)
        upsampled_x = []
        upsampled_y = []
        for c, additional_samples in offsets.items():
            if c not in target_class:
                continue
            class_args = np.argwhere(y == c).ravel()
            for sample in range(additional_samples + 1):
                sample_arg = np.random.choice(class_args)
                upsampled_x.append(x[sample_arg])
                upsampled_y.append(y[sample_arg])
        upsampled_x = np.vstack(upsampled_x)
        upsampled_y = np.vstack(upsampled_y)
        return np.vstack([x,upsampled_x]), np.vstack([y.reshape(-1,1), upsampled_y]).ravel()


class TreeKnnEstimator(KnnEstimator, SupervisedModel):

    def __init__(self, match_threshold = .95,
                 match_type = 'threshold',
                 min_matches = 5, min_true_matches = 2,
                 match_model = None):
        #match type is how the number of matches are selected
        #give clusters to make it based on the class.
        #threshold uses similarity score, uses max(min_matches, patients with score > match threshold)
        super().__init__(match_threshold, match_type, min_matches)
        self.min_true_matches = min_true_matches
        if match_model is not None:
            self.match_model = match_model
        else:
            self.match_model = DecisionTreeClassifier(max_depth = 3,
                                                      class_weight = 'balanced',
                                                      random_state=1)

    def predict_doses(self, similarity_list, data, weight_matrix_loc = None):
        qunatile = lambda x: quantile_transform(x, axis = 1, copy = True, n_quantiles = 20)
        similarity_list = [qunatile(s) for s in similarity_list]
        n_patients = data.get_num_patients()
        dose_matrix = data.doses
        outliers = ErrorChecker().get_data_outliers(data.doses)
        is_augmented = similarity_list[0].shape[0] > n_patients
        if is_augmented: #if matrix is agumented
            dose_matrix = Metrics.augment_mirrored(dose_matrix)
            outliers = outliers | set([o + n_patients for o in outliers])
        predicted_doses = np.zeros((n_patients, dose_matrix.shape[1]))
        clusters = self.get_feature_clusters(data)
        features, labels, positions = self.extract_features(similarity_list, dose_matrix,
                                         clusters, is_augmented)
        for p in range(n_patients):

            patient_positions = np.argwhere( np.any([p == positions, p + n_patients == positions], axis = 0) )[:,0]
            train_x = np.delete(features, patient_positions, axis = 0)
            train_labels = np.delete(labels, patient_positions, axis = 0)
            upsampled_x, upsampled_y = self.upsample_clusters(train_x, train_labels, clusters[p])
            self.match_model.fit(upsampled_x, upsampled_y)
            patient_args = np.argwhere( p == positions[:,0] ).ravel()

            match_probs = self.match_model.predict_proba(features[patient_args])[:,1]
            match_probs = np.insert(match_probs, p, 0)
            if is_augmented:
                match_probs = np.insert(match_probs, p + n_patients, 0)
            match_args = self.get_match_args(match_probs)
            if weight_matrix_loc is None:
                match_weights = match_probs
            else:
                match_weights = similarity_list[weight_matrix_loc][p, :]
            dose_prediction = self.get_individual_dose_prediction(dose_matrix, match_weights, match_args)
            predicted_doses[p,:] = dose_prediction
        return predicted_doses

    def upsample_clusters(self, x, y, cluster, ratio = 2):
        #specifically upsample the features in the same spaial group from the training class
        canidates = np.argwhere(x[:, 0] == cluster).ravel()
        count = ratio*(x.shape[0] - len(canidates) + 1)
        upsampled_x = np.zeros((count, x.shape[1]))
        upsampled_y = np.zeros((count,))
        for s in range(count):
            sample_arg = np.random.choice(canidates)
            upsampled_x[s] = x[sample_arg]
            upsampled_y[s] = y[sample_arg]
        return np.vstack([x, upsampled_x]), np.hstack([y, upsampled_y])

    def get_individual_dose_prediction(self, doses, weights, args):
        match_doses = doses[args]
        match_weights = weights[args].reshape(-1,1)
        if np.sum(match_weights) > 0:
            weighted_doses = np.sum(match_doses*match_weights, axis = 0)/np.sum(match_weights)
        else:
            weighted_doses = np.mean(doses, axis = 0)
        return weighted_doses

    def get_match_args(self, scores):
        threshold = copy(self.match_threshold)
        num_matches = 0
        while num_matches < self.min_matches:
            num_matches = len(np.argwhere(scores > threshold))
            threshold = threshold - .01
        match_args = np.argsort(-scores)[:num_matches]
        return match_args

    def get_feature_clusters(self, data):
        dose_pca = Metrics.pca(data.doses, n_components = 3)
        clusters = KMeans(n_clusters = 5, random_state = 1).fit_predict(data.tumor_distances)
        return clusters.ravel()

    def get_true_matches(self, doses, negative_class = 0, error_threshold = .1):
        dose_error = self.get_match_error(doses)
        match_matrix = np.zeros(dose_error.shape).astype('int32') + negative_class
        n_patients = doses.shape[0]
        dose_error[np.arange(n_patients), np.arange(n_patients)] = 1
        def get_error(p, m_args):
            pred = np.mean(doses[m_args], axis = 0)
            return np.sum(np.abs(doses[p] - pred))/np.sum(doses[p])
        for p in range(n_patients):
            ranked_args = np.argsort(dose_error[p])
            num_matches = 1
            match_error = np.inf
            while True:
                old_error = match_error + 0
                match_args = ranked_args[:num_matches]
                match_error = get_error(p, match_args)
                if num_matches > self.min_true_matches and match_error > old_error:
                    num_matches -= 1
                    break
                num_matches += 1
            match_matrix[p,ranked_args[:num_matches]] = 1
        return match_matrix


    def extract_features(self, similarities, doses, clusters, is_augmented):
        true_similarity = self.get_true_matches(doses)
        num_patients = int(doses.shape[0]/(is_augmented + 1))
        x = []
        y = []
        positions = []
        for p1 in range(num_patients):
            for p2 in range(num_patients*(1 + is_augmented)):
                if p1 == p2%num_patients:
                    continue
                x_row = [clusters[p2%num_patients]]
                for similarity in similarities:
                    x_row.append(similarity[p1, p2])
                x.append(x_row)
                y.append(true_similarity[p1, p2])
                positions.append([p1,p2])
        x = np.array(x)
        y = np.array(y)
        positions = np.array(positions)
        return [x, y, positions]
    
def tsim_score(model, patient1, patient2, distances, volumes, clusters, adjacency):
    model_sim = model.pairwise_similarity(patient1, patient2, distances, volumes, clusters, adjacency)
    return patient1, patient2, model_sim

class TsimModel():
    #orginal-ish similarity model that gives a similarity matrix from get_simirity based on a spatial ssim
    def __init__(self, max_distance = 50, patients = None, organs = None,
                 similarity_function = None, use_classes = False, n_jobs = None):
        self.max_distance = max_distance
        #for subsetting the data later
        self.patients = patients
        self.organs = organs
        self.use_classes = use_classes
        if similarity_function is None:
            self.similarity_function = Metrics.local_ssim
        else:
            self.similarity_function = similarity_function
        self.n_jobs = n_jobs

    def get_adjacency_lists(self, organ_distance_matrix, organs = None):
        if organs is None:
            organs = self.organs
            if organs is None:
                organs = range(Constants.num_organs)
        #this code is much simpler than expected
        organ_distances = organ_distance_matrix[organs][:, organs]
        adjacency_lists = []
        for row in organ_distances:
            adjacent_args = np.argwhere(row < self.max_distance)
            adjacency_lists.append(adjacent_args.ravel())
        return adjacency_lists

    def get_similarity(self, data, augment = False):
        #data is assumed to be a patientset object for now
        if self.patients is None:
            patients = range(len(data.ids))
        else:
            patients = self.patients
        if self.organs is None:
            organs = range(Constants.num_organs)
        else:
            organs = self.organs
        index = np.ix_(patients, organs)
        adjacency = self.get_adjacency_lists(data.organ_distances, organs)
        distances = data.tumor_distances[index]
        volumes = data.volumes[index]
        clusters = data.classes[patients]
        if augment:
            organ_list = list(np.array(Constants.organ_list)[organs])
            distances = Metrics.augment_mirrored(distances, organ_list = organ_list)
            volumes = Metrics.augment_mirrored(volumes, organ_list = organ_list)
        scores = self.similarity(adjacency, distances, volumes, clusters)
        return scores
    
    def similarity(self, adjacency, distances, volumes, clusters, similarity_function = None):
        num_patients, num_organs = distances.shape
        num_original_patients = len(clusters)
        score_matrix = np.zeros((num_patients, num_patients))
        with Pool(self.n_jobs) as pool:
            score_results = []
            for patient1 in range(0, num_patients - 1):
                for patient2 in range(patient1 + 1, num_patients):
                    if (patient1 % num_original_patients) == (patient2 % num_original_patients):
                        continue
                    score_result = pool.apply_async(tsim_score, 
                                                   args = (self, patient1, 
                                                           patient2, distances, 
                                                           volumes,clusters, 
                                                           adjacency)
                                                  )
                    score_results.append(score_result)
            for res in score_results:
                (p1, p2, score) = res.get(10000)
                score_matrix[p1, p2] = score
        score_matrix += np.transpose(score_matrix)
#        scale to between 0 and .99
        score_matrix = (score_matrix - score_matrix.min())
        score_matrix = .99*score_matrix/score_matrix.max()
        for p in range(num_original_patients):
            score_matrix[p,p] = 0
            if num_original_patients < num_patients:
                mirror_pos = p+num_original_patients
                score_matrix[p, mirror_pos] = 0
                score_matrix[mirror_pos, p] = 0
                score_matrix[mirror_pos, mirror_pos] = 0
        return score_matrix

    def pairwise_similarity(self, patient1, patient2, distances, volumes, clusters, adjacency = None):
        if self.use_classes:
            p1 = patient1 % len(clusters)
            p2 = patient2 % len(clusters)
            if clusters[p1] != clusters[p2]:
                return 0
        scores = []
        for organ in range(distances.shape[1]):
            adjacent_args = adjacency[organ]
            if len(adjacent_args) < 1:
                continue
            d1 = distances[patient1, adjacent_args]
            d2 = distances[patient2, adjacent_args]
            v1 = volumes[patient1, adjacent_args]
            v2 = volumes[patient2, adjacent_args]
            scores.append( self.similarity_function(d1,d2,v1,v2) )
        return np.mean(scores)

class TJaccardModel(TsimModel):
    #smalle variant of the tsim model that uses jaccard by default.  saves like two lines of work later
    def __init__(self, max_distance = 50, patients = None, organs = None,
                 similarity_function = None, use_classes = False):
        super(TJaccardModel, self).__init__(max_distance, patients, organs, similarity_function, use_classes)
        self.similarity_function = Metrics.jaccard_distance
        if organs is None:
            self.organs = [Constants.organ_list.index(o) for o in Constants.optimal_jaccard_organs]

class OsimModel(TsimModel):
    #variant that calculates the tsim similarity using organ-organ distances, rather than tumor-organ distances
    def get_similarity(self, data):
        #data is assumed to be a patientset object for now
        if self.patients is None:
            patients = range(len(data.ids))
        else:
            patients = self.patients
        if self.organs is None:
            organs = range(Constants.num_organs)
        else:
            organs = self.organs
        adjacency = self.get_adjacency_lists(data.organ_distances, organs)
        distances = data.all_organ_distances
        volumes = data.volumes
        clusters = data.classes
        scores = self.similarity(adjacency, distances, volumes, clusters)
        return scores

    def similarity(self, adjacency, distances, volumes, clusters, similarity_function = None):
        if similarity_function is None:
            similarity_function = self.local_ssim
        num_patients, num_organs = (distances.shape[2], distances.shape[0])
        score_matrix = np.zeros((num_patients, num_patients))
        for patient1 in range(0, num_patients - 1):
            for patient2 in range(patient1 + 1, num_patients):
#                if clusters[patient1] != clusters[patient2]:
#                    continue
                scores = []
                for organ in range(num_organs):
                    adjacent_args = adjacency[organ]
                    if len(adjacent_args) < 1:
                        continue
                    d1 = distances[adjacent_args, :, patient1][:, adjacent_args]
                    d2 = distances[adjacent_args, :, patient2][:, adjacent_args]
                    similarity_score = similarity_function(d1,d2)
                    scores.append( similarity_score )
                score_matrix[patient1, patient2] = np.mean(scores)
        score_matrix += np.transpose(score_matrix)
        #scale to between 0 and .99
        score_matrix = .99*(score_matrix - score_matrix.min())/(score_matrix.max() - score_matrix.min())
        return score_matrix

class SimilarityFuser(SupervisedModel):
    #class that uses (logistic regression) to map a list of similarity scores to a single score
    #attempts to classify each vector as a neighbor (dose error < some number)
    #and the match probability is used as the similarity
    def __init__(self, model = None, min_matches = 4, max_error = .1):
        self.min_matches = min_matches
        self.max_error = max_error
        if model is None:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(class_weight = 'balanced',
                                       solver = 'lbfgs',
                                       max_iter=500,
                                       random_state = 0)
        self.model = model

    def get_similarity(self, db, similarity_matrices, classes = None):
        [x,y, positions] = self.extract_features(similarity_matrices, db.get_num_patients(), db, classes)
        final_similarity = np.zeros(similarity_matrices[0].shape)
        x = (x-x.min(axis=0))/(x.max(axis=0) - x.min(axis=0))
        for p in range(db.get_num_patients() - 1):
            test_pairs = np.where(positions == p)[0]
            x_train = np.delete(x, test_pairs, axis = 0)
            y_train = np.delete(y, test_pairs, axis = 0)
            predict_pairs  = np.where(positions[:,0] == p)[0]
            x_test = x[predict_pairs,:]
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict_proba(x_test)[:, 1]
            final_similarity[p, :] = np.insert(y_pred, p, 0)
#        final_similarity += final_similarity.transpose()
        return(final_similarity)


    def extract_features(self, similarities, num_patients, data, classes):
        true_similarity = self.get_true_matches(data)
        if classes is not None:
            classes = (classes - np.min(classes))/(np.max(classes) - np.min(classes))
        x = []
        y = []
        positions = []
        for p1 in range(num_patients):
            for p2 in range(num_patients):
                if p1 == p2:
                    continue
                x_row = []
                for similarity in similarities:
                    x_row.append(similarity[p1, p2])
                if classes is not None:
                    target_class = classes[p2]
                    x_row.append(target_class)
                x.append(x_row)
                y.append(true_similarity[p1, p2])
                positions.append([p1,p2])
        x = np.array(x)
        y = np.array(y)
        positions = np.array(positions)
        return [x, y, positions]

    def get_true_matches(self, data, negative_class = 0):
        min_matches = self.min_matches
        dose_error = self.get_match_error(data)
        match_matrix = np.zeros(dose_error.shape) + negative_class
        n_patients = data.get_num_patients()
        for p in range(n_patients):
            max_error = self.max_error
            errors = dose_error[p, :]
            matches = []
            max_error = max_error
            while len(matches) < min_matches:
                matches = np.where(errors < max_error)[0]
                max_error = max_error + .01
            match_matrix[p, matches] = 1
        return match_matrix.astype('int32')

class SimilarityBooster(SimilarityFuser):

    def __init__(self, model = None):
        self.model = model
        if model is None:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(n_estimators = 20)
#            from sklearn.ensemble import AdaBoostRegressor
#            self.model = AdaBoostRegressor(learning_rate=.1, loss = 'square')

    def get_similarity(self, db, similarity_matrices):
        [x,y, positions] = self.extract_features(similarity_matrices, db.get_num_patients(), db)
        final_similarity = np.zeros(similarity_matrices[0].shape)
        x = (x-x.min(axis=0))/(x.max(axis=0) - x.min(axis=0))
        for p in range(db.get_num_patients()):
            test_pairs = np.where(positions == p)[0]
            x_train = np.delete(x, test_pairs, axis = 0)
            y_train = np.delete(y, test_pairs, axis = 0)
            predict_pairs  = np.where(positions[:,0] == p)[0]
            x_test = x[predict_pairs,:]
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            print(y_pred.shape)
            final_similarity[p, :] = np.insert(y_pred, p, 0)
#        final_similarity += final_similarity.transpose()
        return Metrics.dist_to_sim(final_similarity)


    def extract_features(self, similarities, num_patients, data):
        true_similarity = (self.get_true_matches(data))
        x = []
        y = []
        positions = []
        for p1 in range(num_patients):
            for p2 in range(num_patients):
                if p1 == p2:
                    continue
                x_row = []
                for similarity in similarities:
                    x_row.append(similarity[p1, p2])
                x.append(x_row)
                y.append(true_similarity[p1, p2])
                positions.append([p1,p2])
        x = np.array(x)
        y = np.array(y)
        positions = np.array(positions)
        return [x, y, positions]

    def get_true_matches(self, data):
        error = self.get_match_error(data)
        sim = Metrics.dist_to_sim(error)
        return error
