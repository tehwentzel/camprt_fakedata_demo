# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:11:27 2019

@author: Andrew
"""
import numpy as np
from Constants import Constants
from Models import *
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import ConvexHull, procrustes
import copy
import pandas as pd
from re import match, sub, search
from dependencies.NCA import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

#misc functions?
def pca(points, n_components = 2):
    points = points - np.mean(points, axis = 0)
    cov = np.cov(points, rowvar = False)
    ev, eig = np.linalg.eigh(cov)
    args = np.argsort(ev)[::-1]
    ev = ev[args[0:n_components]]
    eig = eig[:, args[0:n_components]]
    principle_components = np.dot(points, eig)
    return(principle_components)

def discretize(x, n_bins = 10, encode = 'ordinal', strategy = 'kmeans'):
    discretizer = KBinsDiscretizer(n_bins = n_bins,
                                   encode = encode,
                                   strategy = strategy)
    return discretizer.fit_transform(x)

def dist_to_sim(distance):
    #converts a distance matrix to a similarity matrix with 0 in the diagonal
    distance = np.copy(distance)
    distance -= distance.min()
    distance /= distance.max()
    diagonals = ( np.arange(distance.shape[0]), np.arange(distance.shape[1]) )
    sim = 1-distance
    sim[diagonals] = 0
    return sim

def local_ssim(x,y,v = None, w = None):
    c1 = .000001
    c2  = .000001
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    covariance = np.cov(x,y)
    numerator = (2*mean_x*mean_y + c1) * (covariance[0,1] + covariance[1,0] + c2)
    denominator = (mean_x**2 + mean_y**2 + c1)*(np.var(x) + np.var(y) + c2)
    if v is not None and w is not None:
        mean_v = np.mean(v)
        mean_w = np.mean(w)
        numerator *= (2*mean_v*mean_w + c1)
        denominator *= (mean_v**2 + mean_w**2 + c1)
    if denominator > 0:
        return numerator/denominator
    else:
        print('error, zero denomiator in ssim function')
        return 0

def root_kernel(matrix):
    matrix = matrix/(matrix.sum(axis = 1) + .000001).reshape(-1,1)
    return np.sqrt(matrix)

def minmax_scale(matrix):
    m = matrix - matrix.min()
    return m/m.max()

def get_flip_args(organ_list = None):
    #get arguments from an organ list that will swap left and right oriented organs
    #assumes naming conventions Rt_ and Lt_ for right and left
    if organ_list is None:
        organ_list = Constants.organ_list
    flip_args = np.arange(len(organ_list))
    for organ in organ_list:
        for pattern in [('Rt_', 'Lt_'), ('Lt_', 'Rt_')]:
            if match(pattern[0], organ) is not None:
                other_organ = sub(pattern[0], pattern[1], organ)
                if other_organ in organ_list:
                    idx1 = organ_list.index(organ)
                    idx2 = organ_list.index(other_organ)
                    flip_args[idx1] = idx2
    return flip_args

def lcr_args(organ_list = None, exclude = None):
    #gets indexes of organs for different regions of the head
    #also assumes Rt_, Lt_ naming conventions for lateral organs
    if organ_list is None:
        organ_list = Constants.organ_list
    left = []
    right = []
    center = []
    left_pattern = 'Lt_'
    right_pattern = 'Rt_'
    for organ in organ_list:
        #exclude should be a list of patterns to skip over
        if exclude is not None:
            for pattern in exclude:
                if search(pattern, organ) is not None:
                    continue
        position = organ_list.index(organ)
        if match(left_pattern, organ) is not None:
            left.append(position)
            continue
        elif match(right_pattern, organ) is not None:
            right.append(position)
        else:
            center.append(position)
    return (left, center, right)

def augment_mirrored(matrix, organ_list = None):
    matrix = np.array(matrix)
    if organ_list is None:
        organ_list = Constants.organ_list
    if len(matrix.shape) > 1 and matrix.shape[1] == len(organ_list):
        flip_args = get_flip_args(organ_list)
        return np.vstack([matrix, matrix[:, flip_args]])
    elif matrix.ndim > 1:
        return np.vstack([matrix, matrix])
    else:
        return np.hstack([matrix,matrix])

#full similarity measures

def get_sim(db, similarity_function):
    #takes a function and the database and returns a similarity or distance matrix
    #assumes it's symmetric, so it only compares once
    num_patients = db.get_num_patients()
    similarity_matrix = np.zeros((num_patients, num_patients))
    for p1 in range(num_patients):
        for p2 in range(p1 + 1, num_patients):
            similarity_matrix[p1,p2] = similarity_function(db, p1, p2)
    similarity_matrix += similarity_matrix.transpose()
    return similarity_matrix

def augmented_sim(feature, similarity_function, organ_list = None):
    #like get sim but needs to explicity give the data matrix (e.g. tumor distances)
    #so it can augment the data with mirrored things
    augmented_features = augment_mirrored(feature, organ_list)
    n_patients = feature.shape[0]
    n_features = augmented_features.shape[0]
    similarity_matrix = np.zeros((n_features, n_features))
    for p1 in range(similarity_matrix.shape[0]):
        data1 = augmented_features[p1]
        for p2 in range(similarity_matrix.shape[1]):
            if (p1%n_patients) == (p2%n_patients):
                continue
            data2 = augmented_features[p2]
            similarity_matrix[p1,p2] = similarity_function(data1, data2)
    similarity_matrix = minmax_scale(similarity_matrix)
    for p in range(n_patients):
        similarity_matrix[p,p] = 0
    return similarity_matrix

def reduced_augmented_sim(features, similarity_function, organ_list = None, distance = False):
    big_sim = augmented_sim(features, similarity_function, organ_list)
    n_patients = features.shape[0]
    merge_func = np.minimum if distance else np.maximum
    reduced = merge_func(big_sim[0:n_patients, 0:n_patients], big_sim[0:n_patients, n_patients:])
    axis = np.arange(n_patients)
#    if distance is False:
#        reduced[axis, axis] = 0
    return reduced

def lymph_similarity(db, file = 'data/spatial_lymph_scores.csv'):
    #loads in lymph similarity from the lymnph node project data into a matrix
    #uses 0? is patients are missing
    lymph_df = pd.read_csv(file, index_col = 0)
    all_patients = set(lymph_df.index)
    similarity = np.zeros((db.get_num_patients(), db.get_num_patients()))
    for p1 in range( db.get_num_patients() ):
        name1 = 'Patient ' + str(db.ids[p1])
        if name1 not in all_patients:
            print(name1, 'Not in Lymph Data', db.n_categories[p1], db.therapy_type[p1], db.gtvs[p1][1].volume)
            continue
        for p2 in range(p1 + 1, db.get_num_patients()):
            name2 = 'Patient ' + str(db.ids[p2])
            if name2 not in all_patients:
                continue
            similarity[p1, p2] = lymph_df.loc[name1, name2]
    return similarity + similarity.transpose()

#patient-wise distance comparisons.  versions used in the default similarty method (tsim) should take
#4 arguments.  2 vectors of distances, and two (optinally) of volumes
#methods passed to get_sim need to accept two patients and a database. use lambda functions for more args
def mse(x,y,w=None,v=None):
    return np.mean((x.ravel() - y.ravel())**2)

def jaccard_distance(x, y, w = None, v = None):
    numerator = x.dot(y)
    denominator = x.dot(x) + y.dot(y) - x.dot(y)
    if numerator == 0 or denominator == 0:
        return 0
    return numerator/denominator

def local_ssim(x,y,v = None, w = None):
    c1 = .000001
    c2  = .000001
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    covariance = np.cov(x,y)
    numerator = (2*mean_x*mean_y + c1) * (covariance[0,1] + covariance[1,0] + c2)
    denominator = (mean_x**2 + mean_y**2 + c1)*(np.var(x) + np.var(y) + c2)
    if v is not None and w is not None:
        mean_v = np.mean(v)
        mean_w = np.mean(w)
        numerator *= (2*mean_v*mean_w + c1)
        denominator *= (mean_v**2 + mean_w**2 + c1)
    if denominator > 0:
        return numerator/denominator
    else:
        print('error, zero denomiator in ssim function')
        return 0

def harmonic_sum(values):
    return 1/np.sum([1/v for v in values])

def undirected_hausdorff_distance(db,p1,p2,centroids):
    h1 = directed_hausdorff(centroids[p1], centroids[p2], 1)
    h2 = directed_hausdorff(centroids[p2], centroids[p1], 1)
    return max([h1[0], h2[0]])

def procrustes_distance(db,p1,p2,centroids, max_points = 1000):
    #max points is in case I want to use a different # later
    #centroids should be a list of n x 3 np arrays
    c1 = centroids[p1]
    c2 = centroids[p2]
    n_points = max([len(c1), len(c2), max_points])
    def scale_centroid(c):
        if len(c) < n_points:
            n_dummy_points = n_points - len(c)
            dummy_values = np.zeros((n_dummy_points, c.shape[1]))
            c = np.vstack([c, dummy_values])
        elif len(c) > n_points:
            c = c[:n_points, :]
        return c
    c1 = scale_centroid(c1)
    c2 = scale_centroid(c2)
    return procrustes(c1, c2)[2]

def n_category_dist(db, p1, p2):
    #distance score for n category between patients
    n_categories = db.n_categories.astype('int32')
    n1 = n_categories[p1]
    n2 = n_categories[p2]
    normalized_difference = abs(n1 - n2)/n_categories.max()
    return normalized_difference

def t_category_dist(db,p1,p2):
    #gives a value giving the distance between t-categories fore each patient
    t_category_map = {'Tis': 0, 'Tx': 1, 'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}
    t1 = db.t_categories[p1]
    t2 = db.t_categories[p2]
    normalized_difference = (abs(t_category_map.get(t1, 0) - t_category_map.get(t2, 0))/5)
    return normalized_difference


def gtv_volume_dist(db,p1,p2):
    #similarity giving the percent different
    gtvns1 = db.gtvs[p1]
    gtvns2 = db.gtvs[p2]
    vol1 = sorted([gtv.volume for gtv in gtvns1], key = lambda x: -x)
    vol2 = sorted([gtv.volume for gtv in gtvns2], key = lambda x: -x)
    if max([vol1, vol2]) == 0:
        return 1
    return np.abs(np.sum(vol1) - np.sum(vol2))

def gtv_organ_sim(db,p1,p2):
    #makes a binary vector denoting the organs that most overlap with each tumor
    #computes a jaccard/tanimoto similarity based on that vector
    def vectorify(p):
        v = np.zeros((Constants.num_organs,))
        for gtv in db.gtvs[p]:
            if gtv.organ in Constants.organ_list:
                pos = Constants.organ_list.index(gtv.organ)
                v[pos] = 1
        return v
    v1 = vectorify(p1)
    v2 = vectorify(p2)
    return jaccard_distance(v1,v2) #1 if np.linalg.norm(v1 - v2) == 0 else 0

def gtv_count_sim(db,p1,p2):
    #similarity based one the difference between number of organs
    gtvs1 = db.gtvs[p1]
    gtvs2 = db.gtvs[p2]
    count1 = 0
    count2 = 0
    for gtv in gtvs1:
        if gtv.volume > 0:
            count1 += 1
    for gtv in gtvs2:
        if gtv.volume > 0:
            count2 += 1
    return min([count1, count2])/max([count1, count2])

def gtv_volume_jaccard_sim(db,p1,p2):
    #tanimoto/jaccard similarity between the vector of tumor volumes beteen patients
    vols1 = [gtv.volume for gtv in db.gtvs[p1]]
    vols2 = [gtv.volume for gtv in db.gtvs[p2]]
    vector_len = np.max([len(vols2), len(vols1)])
    volume_array1 = np.zeros((vector_len,))
    volume_array2 = np.zeros((vector_len,))
    volume_array1[0:len(vols1)] = vols1
    volume_array2[0:len(vols2)] = vols2
    return jaccard_distance(volume_array1, volume_array2)

def vector_sim(db, p1, p2):
    #cosine similarity between the vector between the main and secondary tumors
    vectors = get_gtv_vectors(db)
    return np.dot(vectors[p1, 3:], vectors[p2, 3:])

#misc functions/features
def single_convex_hull_projection(point_cloud, centroids, cuttoff = 15):
    #computes the projection of a given tumor onto the conve
    hull = ConvexHull(point_cloud)
    def project(point, plane):
        distance = np.dot(plane, np.append(point, 1))
        displacement = distance*plane[0:3]/np.linalg.norm(plane[0:3])
        return (point + displacement, distance)
    projections = np.zeros(centroids.shape)
    for idx in range(centroids.shape[0]):
        point = centroids[idx]
        min_dist = np.inf
        for hull_idx in range(hull.equations.shape[0]):
            plane = hull.equations[hull_idx]
            projection, distance = project(point,plane)
            if np.abs(distance) < min_dist and projection[1] < cuttoff: #check if point is roughly in anerior of the head
                min_dist = distance
                projections[idx,:] = projection
    return projections

def convex_hull_projection(point_cloud, centroids):
    #looks at every plane in the convex hull and finds the projection
    #of the closest tumor?
    hull = ConvexHull(point_cloud)
    def project(point, plane):
        distance = np.dot(plane, np.append(point, 1))
        displacement = distance*plane[0:3]/np.linalg.norm(plane[0:3])
        return (point + displacement, distance)
    projections = []
    for hull_idx in range(hull.equations.shape[0]):
        plane = hull.equations[hull_idx]
        min_dist = np.inf
        curr_projection = np.zeros((3,))
        for idx in range(centroids.shape[0]):
            point = centroids[idx]
            projection, distance = project(point,plane)
            if distance < min_dist:
                curr_projection = projection
                min_dist = distance
        projections.append(curr_projection)
    return np.vstack(projections)

def get_lr_tumors(db):
    tumor_sets = np.zeros((db.get_num_patients(), Constants.num_organs, 2))
    for p in range(db.get_num_patients()):
        gtvs = db.gtvs[p]
        left = np.inf*np.ones((Constants.num_organs,))
        right = copy.copy(left)
        #position[0] > 0 is left side
        for gtv in gtvs:
            if gtv.position[0] > 0:
                left = np.minimum(left, gtv.dists)
            else:
                right = np.minimum(right, gtv.dists)
        tumor_sets[p, :, 0] = left
        tumor_sets[p, :, 1] = right
    return tumor_sets

def get_gtv_vectors(db):
    #pretty sure this gets a matrix of tumor vectors of the centroid of the main tumor and slope? between the main
    #tumor and a weighted value of the secondary tumors
    vectors = np.zeros((db.get_num_patients(), 6))
    for p in range(db.get_num_patients()):
        gtvs = db.gtvs[p]
        center = np.mean([x.position*x.volume for x in gtvs], axis = 0)/np.sum([x.volume for x in gtvs], axis = 0)
        secondary_points = np.zeros((3,))
        secondary_tumor_volume = np.sum([tumor.volume for tumor in gtvs])
        if secondary_tumor_volume > 0:
            for t in range(len(gtvs)):
                weight = gtvs[t].volume/secondary_tumor_volume
                secondary_points = secondary_points + weight*(gtvs[t].position - center)
            slope = secondary_points/np.linalg.norm(secondary_points)
        else:
            slope = np.zeros((3,))
        vectors[p] = np.hstack([center, slope])
    return vectors

def get_max_tumor_ssim(patient1, patient2):
    #todo: remember what his does exactly

    options = set()
    scores = -np.ones((len(patient1),))
    similarities = []
    for p1 in range(len(patient1)):
        t1 = patient1[p1]
        for p2 in range(len(patient2)):
            t2 = patient2[p2]
            options.add(p2)
            similarity = local_ssim(t1.dists, t2.dists, t1.volume, t2.volume)
            similarities.append((similarity, p1, p2))
    similarities = sorted(similarities, key = lambda x: -x[0])
    for (s, p1, p2) in similarities:
        if scores[p1] == -1 and p2 in options:
            scores[p1] = s
            options.remove(p2)
    t_volumes = np.array([t.volume for t in patient1])
    max_similarity = np.mean((scores*t_volumes)/t_volumes.sum())
    return max_similarity

def dose_similarity(dose_predictions, distance_metric = None, similarity = True):
    if distance_metric is None:
        distance_metric = mse
    n_patients = dose_predictions.shape[0]
    dists = np.zeros((n_patients, n_patients))
    for p1 in range(n_patients):
        d1 = dose_predictions[p1]
        for p2 in range(p1+1, n_patients):
            d2 = dose_predictions[p2]
            dists[p1,p2] = distance_metric(d1, d2)
    dists += dists.transpose()
    if similarity:
        similarity = dist_to_sim(dists)
        return similarity
    else:
        return dists

def gtv_overlap_vectors(db, use_nonempty = False):
    gtvs = db.gtvs
    vects = np.zeros((db.get_num_patients(), Constants.num_organs))
    for row in range(vects.shape[0]):
        gtvset = gtvs[row]
        for gtv in gtvset:
            if gtv.volume <= 0:
                continue
            valid_overlap = (gtv.dists <= .0001) & (db.volumes[row] > 0)
            overlap_args = np.argwhere(valid_overlap).ravel()
            vects[row, overlap_args] = 1
    if use_nonempty:
        nonempty = np.argwhere(vects.sum(axis = 0) > 0).ravel()
        vects = vects[:, nonempty]
    return vects

def tumor_cosine_similarity(p1, p2, t_o_vectors, adjacency):
    vects1 = t_o_vectors[p1]
    vects2 = t_o_vectors[p2]
    dist = []
    for organ in range(t_o_vectors.shape[1]):
        overlap = np.linalg.norm(np.dot(vects1[organ], vects2[organ]))
        dist.append(overlap)
    return np.mean(dist)


def nca_cv_dist(x, y, n_components = 15):
    nca = NeighborhoodComponentsAnalysis(n_components = n_components, max_iter= 300)
    loo = LeaveOneOut()
    loo.get_n_splits(x)
    nca_dist = np.zeros((len(y), len(y)))
    for train_index, test_index in loo.split(x):
        nca.fit(x[train_index], y[train_index])
        xfit = nca.transform(x)
        for p in range(len(y)):
            nca_dist[test_index, p] = np.linalg.norm(xfit[test_index] - xfit[p])
    return nca_dist

def nca_cv_sim(x, y, n_components = 15, quantile = False):
    nca_dist = nca_cv_dist(x, y, n_components)
    nca_sim = dist_to_sim(nca_dist)
    if quantile:
        nca_sim = quantile_transform(nca_sim, axis = 1)
    return nca_sim

def downsample(x,y,target, ratio):
    to_downsample = np.argwhere(y == target).ravel()
    n_keep = int(ratio*len(to_downsample))
    choices = np.random.choice(to_downsample, (n_keep,), replace = False)
    return x[choices], y[choices]

def get_model_auc(x, y, model):
    ypred = cross_val_predict(model, x, y, cv = LeaveOneOut(), method = 'predict_proba')
    ypred = ypred[:,1]
    roc_score = roc_auc_score(y, ypred)
    fpr, tpr, thresholds = roc_curve(y, ypred)
    return fpr, tpr, thresholds, roc_score

def rescale(x1, x2 = None):
    scale = lambda x: (x - x1.min(axis = 0))/(x1.max(axis = 0) - x1.min(axis = 0))
    if x2 is not None:
        return scale(x1), scale(x2)
    return scale(x1)

def normalize(x1, x2 = None):
    normalize = lambda x: (x - x1.mean(axis = 0))/x1.std(axis = 0)
    if x2 is not None:
        return normalize(x1), normalize(x2)
    return normalize(x1)

def normalize_and_drop(x1, x2 = None):
    args = np.argwhere(x1.std(axis = 0) > 0).ravel()
    x1 = x1[:,args]
    if x2 is not None:
        x2 = x2[:,args]
    return rescale(x1, x2)