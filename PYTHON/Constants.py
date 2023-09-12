# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:47:15 2019

@author: Andrew
"""

class Constants():
    #all constants I use in the code, so I can make sure they are universal. Organ list is used a lot

    #all non-tumor organs being included (some datasets have more)
    #organ list set via affinity propogation clustering
    organ_list = ['Esophagus',
         'Spinal_Cord',
         'Lt_Brachial_Plexus',
         'Rt_Brachial_Plexus',
         'Cricopharyngeal_Muscle',
         'Lt_thyroid_lobe',
         'Rt_thyroid_lobe',
         'Cricoid_cartilage',
         'IPC',
         'MPC',
         'Brainstem',
         'Larynx',
         'Thyroid_cartilage',
         'Rt_Sternocleidomastoid_M',
         'Rt_Mastoid',
         'Rt_Parotid_Gland',
         'Rt_Medial_Pterygoid_M',
         'Rt_Lateral_Pterygoid_M',
         'Rt_Masseter_M',
         'Lt_Sternocleidomastoid_M',
         'Lt_Mastoid',
         'Lt_Parotid_Gland',
         'Lt_Submandibular_Gland',
         'Lt_Medial_Pterygoid_M',
         'Lt_Lateral_Pterygoid_M',
         'Lt_Masseter_M',
         'Supraglottic_Larynx',
         'SPC',
         'Rt_Submandibular_Gland',
         'Hyoid_bone',
         'Soft_Palate',
         'Genioglossus_M',
         'Tongue',
         'Rt_Ant_Digastric_M',
         'Lt_Ant_Digastric_M',
         'Mylogeniohyoid_M',
         'Extended_Oral_Cavity',
         'Mandible',
         'Hard_Palate',
         'Lt_Posterior_Seg_Eyeball',
         'Rt_Posterior_Seg_Eyeball',
         'Lt_Anterior_Seg_Eyeball',
         'Rt_Anterior_Seg_Eyeball',
         'Lower_Lip',
         'Upper_Lip']

    num_organs = len(organ_list)
    v2_high_throat_dose = [184, 276, 5084, 246, 289, 283, 5068, 197, 5007, 100, 10054, 10, 145, 10199]
    #GTVn analgoues: 'GTV node', 'GTV-N', 'GTV_n', 'GTVn2', 'GTVn1'
    #GTVp analogues: 'GTV primary', 'GTV-P', 'GTV_p'
    tumor_aliases = {'GTV node': 'GTVn',
                     'GTV-N': 'GTVn',
                     'GTV_n': 'GTVn',
                     'GTVn1': 'GTVn',
                     'GTV primary': 'GTVp',
                     'GTV-P': 'GTVp',
                     'GTV_p': 'GTVp',
                     'GTV_P': 'GTVp',
                     'GTV P': 'GTVp',
                     'GTV nodes': 'GTVn',
                     'GTV-N1': 'GTVn',
                     'GTV_N1': 'GTVn',
                     'GTV N': 'GTVn',
                     'GTV-NR': 'GTVn2', #I am only aware of this for 10144 and 10022, may need more robust solution later
                     'GTV-NL': 'GTVn3'
                     }
    num_node_types = 39 #number of unique tumor nodes? L/R 2 isn't actually 2A/B in data, put legend says that's what it i
    #names to use for the dataframe when I read a centroid file.  it's cleaner
    centroid_file_names_v2 = ['ROI', 'x', 'y', 'z', 'mean_dose', 'volume', 'min_dose', 'max_dose']
    centroid_file_names_v3 = ['ROI', 'x', 'y', 'z', 'volume', 'min_dose', 'mean_dose', 'max_dose']
    centroid_file_names_v4 = ['ROI', 'x', 'y', 'z']
    #patients without organs, using their sorted order (12th patient in the sorted list ect)
    missing_organs = {}
    no_tumor = []

    subsite_map = {'BOT': 0, 'GPS': 1, 'NOS': 2, 'Soft palate': 3, 'Tonsil': 4}
    laterality_map = {'B': 0, 'L': 1, 'R': 2}
    toxicity_log_file_root = 'data/toxicity_logs/tox_log'
    patient_set_pickle = 'data/patientSetAugmented.p'
    baseline_feature_file = 'data/baselineClustering.csv'
    patient_info_file = 'data/patient_info.csv'
    mean_organ_distance_file = 'data/mean_organ_distances.csv'