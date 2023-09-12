# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:22:31 2019
@author: Andrew
"""
from numpy.random import seed
seed(1)
from preprocessing import *
from glob import glob
from re import findall, match, sub
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from Constants import Constants
from Patient import Patient
from ErrorChecker import ErrorChecker
from Metrics import lcr_args, get_flip_args
import copy
import pickle

class PatientSet():

    def __init__(self, outliers = [], root = 'data/patients_v*/',
                 use_distances = False, use_clean_subset = True, denoise = True, additional_features = False):
        self.classes = None
        self.num_classes = 0
        self.left, center, self.right = lcr_args()
        self.read_patient_data(root, outliers, use_distances)
        if use_clean_subset:
            self.clean_values()
        if denoise:
            self.denoise_tumor_distances()
        print('\npatient data loaded...\n')
        self.additional_features = additional_features
        if additional_features:
            self.add_features()

    def read_patient_data(self, root, outliers, use_distances, save_distances = False):

        #sorts by size of largest integer string, which is the id for our files
        file_sort = lambda x: sorted(x, key =
                                     lambda file:
                                         max([int(x) for x in findall("[0-9]+", file)])
                                )
        distance_files = file_sort(glob(root + '**/*distances.csv'))
        dose_files = file_sort(glob(root + '**/*centroid*.csv'))
        #maps ids to position so I can do that too?
        ids = self.delete_outliers(outliers, distance_files, dose_files)
        metadata_file = Constants.patient_info_file
        assert(len(distance_files) == len(dose_files))
        #maps a position 0-len(files) to the dummy id for a patient
        num_patients = len(ids)
        metadata = pd.read_csv(metadata_file,
                               index_col = 0, #index is the "Dummy ID"
                               usecols = [0,1,2,3,4,5,6,7,8,9,10,11,12,
                                          13,14,15,16, 17,18,31,32,35,36,37,38]
                               ).loc[ids]
#        print(metadata.columns)

        #super inefficient way of reading in the data
        patients = OrderedDict()
        dose_matrix = np.zeros((num_patients, Constants.num_organs))
        max_dose_matrix = np.zeros((num_patients, Constants.num_organs))
        min_dose_matrix = np.zeros((num_patients, Constants.num_organs))
        centroid_matrix = np.zeros((num_patients, Constants.num_organs, 3))
        total_dose_vector = np.zeros((num_patients,))
        prescribed_dose_vector = np.zeros((num_patients,))
        tumor_distance_matrix = np.zeros((num_patients, Constants.num_organs))
        organ_distance_matrix = np.zeros((Constants.num_organs, Constants.num_organs, num_patients))
        volume_matrix = np.zeros((num_patients,Constants.num_organs))
        node_matrix = np.zeros((num_patients, Constants.num_node_types))

        pathological_grade_vector = np.zeros((num_patients,)).astype(str)
        therapy_type_vector = np.zeros((num_patients,)).astype(str)
        n_category_vector = np.zeros((num_patients,)).astype(str)
        t_category_vector = np.zeros((num_patients)).astype(str)
        gender_vector = np.zeros((num_patients,)).astype(str)
        age_vector = np.zeros((num_patients,)).astype('int32')
        self.ajcc8 = np.zeros((num_patients,))
        self.hpv = np.zeros((num_patients,))
        self.dose_fractions = np.zeros((num_patients,))
        self.has_gtvp = np.zeros((num_patients,))
    
        self.aspiration = np.zeros((num_patients,))
        self.aspiration_change = np.zeros((num_patients,))
        self.smoking = np.zeros((num_patients,))
        self.packs_per_year = np.zeros((num_patients,))

        laterality_list = []
        subsite_list = []
        gtv_list = []
        classes = np.zeros((num_patients,))
        self.feeding_tubes = np.zeros((num_patients,)).astype('bool')
        
        self.race = np.zeros((num_patients,)).astype('str')

        self.mean_tumor_distances = np.copy(tumor_distance_matrix)
        self.max_tumor_distances = np.copy(tumor_distance_matrix)
        #putting all the data into a patient object for further objectification
      
        if not use_distances:
            o_dists = self.load_saved_distances()
            if not isinstance(o_dists, bool):
                self.organ_distances = o_dists
                self.all_organ_distances = None
            else:
                use_distances = True
                save_distances = True

        for patient_index in range(0, num_patients):
            dataset_version = int(findall('patients_v([0-9])', distance_files[patient_index])[0])
            assert(dataset_version in [2,3])
            #these are indexed by name of organ
            #we only use 3 rows but half of them have a comma missing in the header between the last two rows
            distances = pd.read_csv(distance_files[patient_index],
                                    usecols = [0,1,2]).dropna()
            #renames anything that is equivalent to GTVp/GTVn to the correct format
            distances = self.fix_tumor_names(distances)
            doses = pd.read_csv(dose_files[patient_index],
                                usecols = [0,1,2,3,4,5,6,7]).dropna()
            #pateints_v3 dataset has a different way of ording the columns (and different spelling)
            if dataset_version == 2:
                doses.columns = Constants.centroid_file_names_v2
            elif dataset_version == 3:
                doses.columns = Constants.centroid_file_names_v3
            doses = self.fix_tumor_names(doses)
            #misc patient info - laterality, subsite, total dose, etc
            info = metadata.loc[ids[patient_index]]
            group = self.get_patient_class(ids[patient_index], doses.set_index('ROI').mean_dose)
            #uses a new patient class to acually parse/process data
            new_patient = Patient(distances, doses,
                                  ids[patient_index], group,
                                  info, use_distances = use_distances)
            patients[patient_index] = new_patient
            classes[patient_index] = group
            laterality_list.append(new_patient.laterality)
            subsite_list.append(new_patient.tumor_subsite)
            gtv_list.append(new_patient.gtvs)

            dose_matrix[patient_index, :] = new_patient.doses
            max_dose_matrix[patient_index, :] = new_patient.max_doses
            min_dose_matrix[patient_index, :] = new_patient.min_doses
            tumor_distance_matrix[patient_index, :] = new_patient.tumor_distances
            total_dose_vector[patient_index] = new_patient.total_dose
            prescribed_dose_vector[patient_index] = new_patient.prescribed_dose
            volume_matrix[patient_index, :] = new_patient.volumes
            centroid_matrix[patient_index, :, :] = new_patient.centroids
            node_matrix[patient_index, :] = new_patient.node_vector

            pathological_grade_vector[patient_index] = new_patient.pathological_grade
            therapy_type_vector[patient_index] = new_patient.therapy_type
            n_category_vector[patient_index] = new_patient.n_stage
            t_category_vector[patient_index] = new_patient.t_category
            gender_vector[patient_index] = new_patient.gender
            age_vector[patient_index] = new_patient.age
            self.ajcc8[patient_index] = new_patient.ajcc8
            self.hpv[patient_index] = new_patient.hpv
            self.feeding_tubes[patient_index] = (new_patient.feeding_tube.lower() == 'y')
            self.dose_fractions[patient_index] = new_patient.dose_fractions
            self.smoking[patient_index] = new_patient.smoking
            self.packs_per_year[patient_index] = new_patient.packs_per_year

            self.aspiration[patient_index] = new_patient.aspiration
            self.aspiration_change[patient_index] = new_patient.aspiration_change

            self.max_tumor_distances[patient_index] = new_patient.max_tumor_distances
            self.mean_tumor_distances[patient_index] = new_patient.mean_tumor_distances

            self.has_gtvp[patient_index] = new_patient.gtvs[0].volume > 0
            self.race[patient_index] = new_patient.race
            if use_distances:
                organ_distance_matrix[:, :, patient_index] = np.nan_to_num(new_patient.distances)
        self.doses = np.nan_to_num(dose_matrix)
        self.max_doses = np.nan_to_num(max_dose_matrix)
        self.min_doses = np.nan_to_num(min_dose_matrix)
        self.tumor_distances = np.nan_to_num(tumor_distance_matrix)
        self.volumes = np.nan_to_num(volume_matrix)
        self.lymph_nodes = np.nan_to_num(node_matrix)
        self.classes = np.nan_to_num(classes)

        self.pathological_grades = pathological_grade_vector
        self.therapy_type = therapy_type_vector
        self.n_categories = n_category_vector
        self.t_categories = t_category_vector
        self.genders = gender_vector
        self.ages = np.nan_to_num(age_vector)
        self.dose_fractions = np.nan_to_num(self.dose_fractions)

        if use_distances:
            self.all_organ_distances = np.nan_to_num(organ_distance_matrix)
            self.organ_distances = self.all_organ_distances.mean(axis = 2)
            if save_distances:
                try:
                    self.save_organ_distances()
                except:
                    print('error saving organ distances?')
        
        self.prescribed_doses = np.nan_to_num(prescribed_dose_vector)
        self.centroids = np.nan_to_num(centroid_matrix)
        self.lateralities = np.array(laterality_list)
        self.subsites = np.array(subsite_list)
        self.ids = np.array(ids)
        self.gtvs = gtv_list

    def clean_values(self):
        #subsets to the values approved by the error checker object
        error_checker = ErrorChecker()
        p = error_checker.get_clean_subset(self)
        p = sorted(p)
        self.subset(p)

    def subset(self, p):
        #take of list of indices and just subsets all the data to match
        #used when getting the error checker output for cleaning
        #if new values are added, make sure to add them in here
        self.doses = self.doses[p]
        self.max_doses = self.max_doses[p]
        self.min_doses = self.min_doses[p]
        self.tumor_distances = self.tumor_distances[p]
        self.volumes = self.volumes[p]
        self.classes = self.classes[p]
        self.prescribed_doses = self.prescribed_doses[p]
        self.centroids = self.centroids[p]
        self.lateralities = self.lateralities[p]
        self.subsites = self.subsites[p]
        self.lymph_nodes = self.lymph_nodes[p]
        self.ids = self.ids[p]
        new_gtvs = []
        for patient in p:
            new_gtvs.append(self.gtvs[patient])
        self.gtvs = new_gtvs

        self.ages = self.ages[p]
        self.genders = self.genders[p]
        self.t_categories = self.t_categories[p]
        self.n_categories = self.n_categories[p]
        self.therapy_type = self.therapy_type[p]
        self.ajcc8 = self.ajcc8[p]
        self.pathological_grades = self.pathological_grades[p]
        self.hpv = self.hpv[p]
        self.feeding_tubes = self.feeding_tubes[p]
        self.dose_fractions = self.dose_fractions[p]
        self.aspiration = self.aspiration[p]
        self.aspiration_change = self.aspiration_change[p]

        self.mean_tumor_distances = self.mean_tumor_distances[p]
        self.max_tumor_distances = self.max_tumor_distances[p]
        self.packs_per_year = self.packs_per_year[p]
        self.smoking = self.smoking[p]
        self.race = self.race[p]

        self.has_gtvp = self.has_gtvp[p]
        if self.all_organ_distances is not None:
            self.all_organ_distances = self.all_organ_distances[:,:,p]
            
    def add_features(self):
        from analysis import tsim_prediction
        #add more fields for use in the clustering scripts for feature testing
        self.t_volumes = np.array([np.sum([g.volume for g in gtvs]) for gtvs in self.gtvs]).reshape(-1,1)
        self.bilateral = self.lateralities == 'B'
        self.total_volumes = self.volumes.sum(axis = 1)
        self.toxicity = self.feeding_tubes + self.aspiration > 0
        self.tsimdoses = tsim_prediction(self)
        self.neck_width = np.linalg.norm(self.centroids[:,Constants.organ_list.index('Lt_Sternocleidomastoid_M'),:] - self.centroids[:,Constants.organ_list.index('Rt_Sternocleidomastoid_M'), :], axis = 1)
        self.additional_features = True

    def get_num_patients(self):
        return( self.doses.shape[0] )

    def load_saved_distances(self, file = None):
        if file is None:
            file = Constants.mean_organ_distance_file
        try:
            distances = pd.read_csv(file, index_col = 0)
            distances = distances.values
        except:
            print('error, no mean-organ distance file found')
            return False
        return distances

    def get_patient_class(self, patient_id, doses):
        #if a vector of classes is used
        group = self.get_default_class(patient_id, doses)
        if self.classes is not None:
            try:
                subclass = self.classes[patient_id]
                group = (group-1)*self.num_classes + subclass
            except:
                pass
#                print('patient ', patient_id, 'not in class list, defaulting to 0')
        return int(group)

    def get_default_class(self, patient_id, dose_vector):
        full_dose, left_biased = self.check_if_full_dose(dose_vector)
        if not full_dose:
            group = (3 if left_biased == True else 4)
        elif patient_id in Constants.v2_high_throat_dose:
            group = 2
        else:
            group = 1
        return group

    def get_xerostima(self):
        return self.feeding_tubes + self.aspiration > 1

    def check_if_full_dose(self, dose_vector):
        #checks difference in sternoceldomastoids to seperate out unilaterally dosed patients?
        #may be used for getting classes eventually?
        def confirm():
            try:
                if isinstance(dose_vector, pd.core.series.Series):
                    ls = dose_vector.loc['Lt_Sternocleidomastoid_M']
                    rs = dose_vector.loc['Rt_Sternocleidomastoid_M']
                else:
                    ls_pos = Constants.organ_list.index('Lt_Sternocleidomastoid_M')
                    rs_pos = Constants.organ_list.index('Rt_Sternocleidomastoid_M')
                    ls = dose_vector[ls_pos]
                    rs = dose_vector[rs_pos]
            except:
#                print('error in getting dose?')
                ls = 1
                rs = 1
            if np.abs(ls - rs)/max([ls, rs]) < .6:
                full_dose = True
            else:
                full_dose = False
            return(full_dose, (ls > rs))
        try:
            if isinstance(dose_vector, pd.core.series.Series):
                left, center, right = lcr_args(list(dose_vector.index), exclude = ['[e,E]ye'])
                if len(left) != len(right):
                    return confirm()
                ls = dose_vector[left].values
                rs = dose_vector[right].values
            else:
                ls = dose_vector[self.left]
                rs = dose_vector[self.right]
        except:
#            print('error in getting dose?')
            return(True, False)
        full_dose = bool(np.abs(ls - rs).mean() < 22)
        left_dominant = bool(ls.mean() > rs.mean())
        return(full_dose, left_dominant)

    def delete_outliers(self, outliers, distance_files, dose_files):
        id_map = {max([int(x) for x in findall('[0-9]+', file)]): distance_files.index(file)  for file in distance_files}
        ids = sorted(list(id_map.keys()))
        #delete patient files with an id in the outliers
        for outlier_id in sorted(outliers, reverse = True):
            if outlier_id in ids:
                pos = id_map[outlier_id]
                del distance_files[pos]
                del dose_files[pos]
                del ids[pos]
        return(ids)

    def fix_tumor_names(self, dataframe):
        #this should probably not need to return anything, but does.
        #replace the aliases for GTVp or GTVn(1) with a consistent name
        dataframe.replace(Constants.tumor_aliases, inplace = True)
        dataframe.replace(to_replace = r'GTV.*N', value = 'GTVn', regex = True, inplace = True)
        return dataframe

    def change_classes(self, class_name = None, class_file = 'data/clusters2.csv'):
        if class_name is not None:
            classes = pd.read_csv(class_file,
                                       index_col = 1)
            classes = classes.drop(labels = ['Unnamed: 0'], axis = 1)
            classes = classes.sort_index()
            classes.columns = classes.columns.str.strip()
            self.classes = classes[class_name].values.astype('int32')
            self.num_classes = len(self.classes)
        else:
            for p in range(self.get_num_patients()):
                self.classes[p] = self.get_default_class(self.ids[p], self.doses[p,:])

    def save_organ_distances(self, file = None):
        if file is None:
            file = Constants.mean_organ_distance_file
        if self.organ_distances.ndim > 2:
            mean_dists = self.organ_distances.mean(axis = 2)
        else:
            mean_dists = self.organ_distances
        if np.sum(mean_dists) == 0:
            print('error, trying to save emtpy organ list')
            return
        else:
            organ_dist_df = pd.DataFrame(mean_dists, index = Constants.organ_list, columns = Constants.organ_list)
        organ_dist_df.to_csv(file)

    def get_all_tumor_distances(self):
        distances = []
        for gtvset in self.gtvs:
            for gtv in gtvset:
                distances.append(gtv.dists)
        distances = np.array(distances)
        return distances

    def denoise_tumor_distances(self):
        #passes tumors through a densoiing autoencoder.
        #will change self.tumor_distance but not self.gtvs
        try:
            distances = self.get_all_tumor_distances()
            distances = Denoiser(normalize = False, noise = .5).fit_transform(distances, lr = .0001)
            i = 0
            #p = 0
            new_tumor_distances = np.zeros(self.tumor_distances.shape)
            all_tumor_distances = []
            for p in range(self.get_num_patients()):
                p_dists = []
                count = len(self.gtvs[p])
                new_dists = np.inf*np.ones((self.tumor_distances.shape[1]))
                for c in range(count):
                    p_dists.append(distances[i])
                    new_dists = np.minimum(new_dists, distances[i])
                    i += 1
                new_tumor_distances[p,:] = new_dists
                all_tumor_distances.append( np.vstack(p_dists))
            self.tumor_distances = new_tumor_distances
            self.stack_tumor_distances  = all_tumor_distances
        except Exception as e:
            print("Error denoising values:")
            print(e)
            
            
    def tumorcount_patients(self, min_tumors = 3):
        #gets all patients with more than a given number of tumors
        mtumors = []
        for p in range(self.get_num_patients()):
            n_tumors = 0
            gtvset = self.gtvs[p]
            for gtv in gtvset:
                if gtv.volume > 0.001:
                    n_tumors += 1
            if n_tumors >= min_tumors:
                mtumors.append(p)
        return mtumors

    def to_dataframe(self, attributes, to_merge = None, organ_list = None, merge_mirrored_organs = False):
        #tries to convert internal variables to a dataframe with patient ids as the index
        #attributes should be a list of strings of the member names
        #assumes internal vectors are 1d arrays or 2d arrays of type (n_patientsxn_organs)
        #so passing centroids shuuldn't work
        #to merge is a dataframe also indexed on patient id
        organ_list = Constants.organ_list if organ_list is None else organ_list
        data = {}
        for attr in attributes:
            values = getattr(self, attr)
            if values.ndim == 1:
                try:
                    data[attr] = pd.to_numeric(values)
                except:
                    for encoding in np.unique(values):
                        name = attr + '_' + encoding
                        data[name] = (values == encoding).astype('int32')
            elif values.ndim == 2 and values.shape[1] == Constants.num_organs:
                for organ in organ_list:
                    if organ in Constants.organ_list:
                        pos = Constants.organ_list.index(organ)
                        data[organ+'_'+attr] = values[:, pos]
            else:
                try:
                    values = values.reshape(values.shape[0], -1)
                    for i in range(values.shape[1]):
                        data[attr + '_' + str(i)] = values[:,i]
                except:
                    print('error getting values for ' + attr)
        df = pd.DataFrame(index = self.ids, data = data)
        if merge_mirrored_organs:
            organs_to_drop = []
            for col in df.columns:
                if match('Rt_*', col):
                    base_organ = sub('Rt_','',col)
                    for col2 in df.columns:
                        if match('Lt_*', col2):
                            if base_organ == sub('Lt_', '', col2):

                                df[base_organ + '_combined'] = np.sqrt(df[col]**2 + df[col2]**2)
                                organs_to_drop.extend([col,col2])
                                break
            df = df.drop(organs_to_drop, axis = 1)
        if to_merge is not None:
            df = df.join(to_merge, how = 'inner')
        return df
    
def save_patientset(db = None, add_features = True):
    if db is None:
        db = PatientSet()
    try:
        if not db.additional_features and add_features:
            db.add_features()
        pickle.dump(db, open(Constants.patient_set_pickle, 'wb'))
        print("Patient Set Save successfully")
        return db
    except:
        print("Error saving PatientSet")
        return False
        
def load_patientset(add_features = True):
    try:
        new_db = pickle.load(open(Constants.patient_set_pickle, 'rb'))
        if not new_db.additional_features and add_features:
            new_db.add_features()
        return new_db
    except:
        print("Error loading patientset, trying to resave it...")
        try:
            new_db = save_patientset()
            return new_db
        except:
            print("Could not resave patientset.  you have bugs in yo code")
            return False
        
