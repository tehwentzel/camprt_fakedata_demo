# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:18:10 2019

@author: Andrew
"""
import numpy as np
from Constants import Constants
from collections import OrderedDict, namedtuple

GTV = namedtuple('GTV', ['name', 'volume', 'position', 'doses', 'dists', 'organ'])

class Patient():
    ##class holds information for each patient.
    ##is pased a series of dataframes (distacnes, doses, info) and extracts info
    ##p_id is a dummy id, position is the position of the patient in the whole dataset
    node_binarizer = {'R RPLN': 0,
        'L RPLN': 1,
        'R1A': 2,
        'R1B': 3,
        'R2': 4,
        'R3': 5,
        'R4': 6,
        'R5A': 7,
        'R5B': 8,
        'L1A': 9,
        'L1B': 10,
        'L2': 11,
        'L3': 12,
        'L4': 13,
        'L5A': 14,
        'L5B': 15,
        'L2/3 R1B': 3}
#        'L2/3': [11,12],
#        'L2/3 R1B': [11,12,3],
#        'R2-R4': [4,5,6],
#        'R2/3': [4,5],
#        'R3/4': [5,6],
#        'R3/R4': [5,6],
#        'R2/3/4': [4,5,6]}

    node_adjacency = {'R1A': ['R1B', 'L1A'],
      'R1B': [ 'R2', 'R3'],
      'R2': ['R3', 'R5A'],
      'R3': ['R4', 'R5A', 'R5B'],
      'R4': ['R4', 'R5B'],
      'R5A': ['R5B'],
      'L1A': ['L1B'],
      'L1B': [ 'L2', 'L3'],
      'L2': ['L3', 'L5A'],
      'L3': ['L4', 'L5A', 'L5B'],
      'L4': ['L4', 'L5B'],
      'L5A': ['L5B']
      }

    ajcc8_map = {'I': 1,
                 'II': 2,
                 'III': 3,
                 'IV': 4,
                 'V': 5}

    hpv_map = {'Positive': 1,
               'Negative': -1,
               'Unknown': 0}

    smoking_map = {'Former': 0,
                 'Never': -1,
                 'Current': 1}

    def __init__(self, distances, doses, p_id, group, info, use_distances = False):
        #patient ID number
        self.id = p_id
        self.group = group
        self.neck_boost = (info['Neck boost (Y/N)'] == 'Y')
        ##self.high_throat_dose = 1 if p_id in Constants.v2_high_throat_dose else 0
        #basically ordinality of the id, so where it will be in an index
        self.prescribed_dose = info['Total dose']
        try:
            n_stage = info['N-category']
            self.n_stage = int(n_stage[1])
        except:
            print('error reading nstage for ', p_id)
            self.n_stage = -1
        self.tumor_subsite = info['Tumor subsite (BOT/Tonsil/Soft Palate/Pharyngeal wall/GPS/NOS)']
        self.age = info['Age at Diagnosis (Calculated)']
        self.pathological_grade = info['Pathological Grade']
        self.gender = info['Gender']
        self.race = info['Race']
        self.dose_fractions = info['Total fractions']
        self.ajcc8 = Patient.ajcc8_map.get(info['AJCC 8th edition'], 0)
        self.therapy_type = info['Therapeutic combination']
        self.feeding_tube = info['Feeding tube 6m']
        self.t_category = info['T-category']
        self.aspiration = info['Aspiration rate(Y/N)'] == 'Y'
        self.aspiration_change = info['Aspiration rate Pre-therapy'] == info['Aspiration rate Post-therapy']
        self.hpv = Patient.hpv_map.get(info['HPV/P16 status'], 0)
        try:
            self.packs_per_year = int(info['Smoking status (Packs/Year)'])
        except:
            self.packs_per_year = 0
        self.smoking = Patient.smoking_map.get(info['Smoking status at Diagnosis (Never/Former/Current)'],0)
        centroid_data = self.get_doses_file_info(doses, distances)
        #I make a new matrix, so the order of centroid data isn't the same as the orginal csv
        self.doses = centroid_data[:, 4]
        self.min_doses = centroid_data[:, 5]
        self.max_doses = centroid_data[:, 6]
        #this is the actual total dose for all organs, 'total dose' from the data csv (66 or 70) is prescribed dose
        self.total_dose = np.sum(self.doses)
        self.volumes = centroid_data[:, 3]
        self.centroids = centroid_data[:, 0:3]
        self.get_lymph_node_data(info)
        #distances is a symetric matrix sorted by the Constants.organ_list
        if use_distances:
            self.distances = self.gen_distance_matrix(distances)
        #store the entries without gtvp for future study
        (self.tumor_volume, self.tumor_distances, self.mean_tumor_distances, self.max_tumor_distances, self.tumor_position) = self.get_main_tumor()
        self.laterality = self.get_laterality(self.gtvs)
        self.check_missing_organs(doses)
        ##self.check_if_full_dose()
        #report if there is no primary tumor
        if self.tumor_volume == 0 or np.sum(self.tumor_distances) == 0:
            Constants.no_tumor.append(self.id)

    def get_laterality(self, gtvs):
        sides = set([])
        for gtv in gtvs:
            if gtv.position[0] > 0:
                sides.add('L')
            elif gtv.position[0] < 0:
                sides.add('R')
        if 'L' in sides:
            if 'R' in sides:
                return 'B'
            else:
                return 'L'
        elif 'R' in sides:
            return 'R'
        return 'NA'

    def get_lymph_node_data(self, info):

        lymph_nodes = info['Affected Lymph node cleaned']
        node_vector = np.zeros((Constants.num_node_types))
        if isinstance(lymph_nodes, str):
            nodes = lymph_nodes.split(',')
            for node in nodes:
                node = node.strip()
                if node in Patient.node_binarizer:
                    node_vector[Patient.node_binarizer[node]] = 1
                else:
                    pass
#                     print('notation not accounted for in lymph nodes:', node)
        k = 0
        for val in Patient.node_binarizer.values():
            if isinstance(val, int):
                k = max([k,val])
        for node, adjacent_nodes in Patient.node_adjacency.items():
            x = Patient.node_binarizer[node]
            for adjacent_node in adjacent_nodes:
                k = k+1
                y = Patient.node_binarizer[adjacent_node]
                bigram_val = node_vector[x]*node_vector[y]
                node_vector[k] = bigram_val
        self.node_vector = node_vector


    def check_missing_organs(self, doses):
        #check if any organs are missing using the dose file, and store them
        organs = set(Constants.organ_list[:])
        dose_organs = set(doses['ROI'].unique())
        diff = organs - dose_organs
        if len(diff) > 0:
            Constants.missing_organs[self.id] = {'organs': diff}
        return

    def get_doses_file_info(self, doses, dists):
        #rename the columns so they're consistent, now done in dataset so I can read which set it's from
#        doses.columns = Constants.centroid_file_names
        #move centroids so the center of the cloud is at zero?
        centroids = self.center_centroids(doses)
        centroids = centroids.set_index('ROI')
        #extract a secondary tumor (only gets the first one?)
        #several patients have no gtvp but a gtvn

        def getGTV(name):
            try:
                gtv = centroids.loc[name]
                if not isinstance(gtv.volume, float):
                    Constants.multiple_gtv.append(self.id)
                    gtv = gtv.iloc[0]
                volume = gtv.volume
                position = gtv[['x','y','z']].values
                doses = gtv[['min_dose','mean_dose','max_dose']]
                distances = self.get_tumor_distances(name, dists)
                min_dist = np.argmin(distances)
                organ = Constants.organ_list[min_dist]
            except:
                volume = float(0)
                position = np.array([0,0,0])
                doses = np.array([0,0,0])
                distances = np.zeros((Constants.num_organs,))
                organ = 'None'
            return( GTV(name, volume, position, doses, distances, organ) )
        self.gtvs = [getGTV('GTVp'), getGTV('GTVn')]
        gtvn_count = 2
        while(True):
            gtv_name = 'GTVn' + str(gtvn_count)
            newGTV = getGTV(gtv_name)
            if newGTV.volume > 0:
                self.gtvs.append(newGTV)
                gtvn_count += 1
            else:
                break
        #merge overlapping secondary tumors
        self.merge_gtvns(self.gtvs, dists)
        #get the info the centers, volumes, nad doses for all the things
        centroid_matrix = np.zeros((Constants.num_organs,7)) #row = x,y,z,volume,dose
        for idx in range(0, Constants.num_organs):
            organ = Constants.organ_list[idx]
            try:
                organ_entry = centroids.loc[organ]
                centroid_matrix[idx, 0:3] = organ_entry[['x','y','z']].values
                centroid_matrix[idx, 3] = organ_entry.volume
                centroid_matrix[idx, 4] = organ_entry.mean_dose
                centroid_matrix[idx, 5] = organ_entry.min_dose
                centroid_matrix[idx, 6] = organ_entry.max_dose
            except:
                pass
                #print('patient ', self.id, ' is missing organ ', organ, ' centroid data')
        return(centroid_matrix)

    def merge_gtvns(self, gtvs, dists):
        #merges gtvs where there is overlap between them into a single gtv
        #calculates which gtvs overlap, and uses combine gtvs to get a single gtv
        if len(gtvs) <= 2:
            return gtvs
        dists = dists.set_index(['Reference ROI', 'Target ROI']).sort_index()
        new_gtvs = []
        for i in range(0,len(gtvs)):
            gtv1 = gtvs[i]
            new_gtvs.append(set([i,i]))
            for ii in range(i+1, len(gtvs)): # I think I need to include itself?
                gtv2 = gtvs[ii]
                if self.gtv_overlap(gtv1.name, gtv2.name, dists):
                    new_gtvs.append(set([i,ii]))
        if len(new_gtvs) > 1:
            for idx in np.arange(len(new_gtvs) - 1, 0, -1):
                if not new_gtvs[idx - 1].isdisjoint(new_gtvs[idx]):
                    new_gtvs[idx - 1] = new_gtvs[idx - 1].union(new_gtvs[idx])
                    del new_gtvs[idx]
        temp_gtvs = []
        tumor_number = 1
        for gtvset in new_gtvs:
            if 0 in gtvset:
                name = 'GTVp'
            else:
                name = 'GTVn' if tumor_number == 1 else 'GTVn' + str(tumor_number)
                tumor_number += 1
            new_tumor = self.combine_gtvs(gtvs, gtvset, name = name)
            temp_gtvs.append(new_tumor)
        if len(gtvs) > len(temp_gtvs):
#             print(self.id, new_gtvs)
            self.gtvs = temp_gtvs

    def combine_gtvs(self, gtvs, gtvset, name = None):
        #takes the set of gtvs from the data and indexes, merges the gtvs in gtvset
        gtvset = sorted(gtvset, key = lambda x: gtvs[x].name)
        if len(gtvset) < 2:
            return gtvs[gtvset[0]]
        volumes = np.array([gtvs[i].volume for i in gtvset])
        total_volume = volumes.sum()
        weights = volumes/total_volume #so now these are weightedd
        if name is None:
            if 0 in gtvset:
                name = gtvs[0].name
            else:
                name = gtvs[gtvset[0]].name

        doses = weights[0]*gtvs[gtvset[0]].doses
        distances = weights[0]*gtvs[gtvset[0]].dists
        position = weights[0]*gtvs[gtvset[0]].position
        for k in range(1, len(gtvset)):
            this_gtv = gtvs[gtvset[k]]
            w = weights[k]
            doses = doses + w*this_gtv.doses
            distances = np.minimum(this_gtv.dists, distances)
            position = position + w*this_gtv.position
        organ = Constants.organ_list[np.argmin(distances)]
        combined_gtv = GTV(name, total_volume, position, doses, distances, organ)
        return combined_gtv

    def gtv_overlap(self, name1, name2, dists):
        try:
            distance = (dists.loc[name1, name2])['Eucledian Distance (mm)']
        except:
            try:
                distance = (dists.loc[name2, name1])['Eucledian Distance (mm)']
            except:
                return False
        return (distance <= 0)

    def center_centroids(self, centroids):
        #subtract off the mean so the pointcloud is centered at 0
        #should I just use a reference organ instead?  or rotate?
        centroids.x -= centroids.x.mean()
        centroids.y -= centroids.y.mean()
        centroids.z -= centroids.z.mean()
        return(centroids)

    def gen_distance_matrix(self, dists):
        #generates a symetric 45x45 matrix of organ-organ distances
        dist_matrix = np.zeros(( Constants.num_organs, Constants.num_organs))
        dists = dists.set_index(['Reference ROI', 'Target ROI']).sort_index()
        alphabetical_organ_list = sorted(Constants.organ_list)
        for row in range(0, Constants.num_organs):
            for col in range(row + 1, Constants.num_organs):
                organ1 = alphabetical_organ_list[row]
                organ2 = alphabetical_organ_list[col]
                try:
                    dist_matrix[row, col] = (dists.loc[organ1, organ2])['Eucledian Distance (mm)']
                except:
                    try:
                        dist_matrix[row, col] = (dists.loc[organ2, organ1])['Eucledian Distance (mm)']
                    except:
                        dist_matrix[row, col] = 0
                        print(self.id, ' ', organ1, ' ', organ2, ' missing')
        dist_matrix += np.transpose(dist_matrix)
        return(dist_matrix)

    def get_tumor_distances(self, name, dists):
        #gets the tumor-organ distances
        gtv_dists = np.zeros((Constants.num_organs,))
        dists = dists.set_index(['Reference ROI', 'Target ROI']).sort_index()
        for idx in range(0, Constants.num_organs):
            organ = Constants.organ_list[idx]
            try:
                tumor_row = dists.loc[name, organ]
                gtv_dists[idx] = tumor_row['Eucledian Distance (mm)']
            except:
                try:
                    tumor_row = dists.loc[organ, name]
                    gtv_dists[idx] = tumor_row['Eucledian Distance (mm)']
                except:
                    gtv_dists[idx] = float(0)
        return(gtv_dists)

    def get_main_tumor(self):
        #basically gives a proxy so we use only the most important tumor
        tumor_volume = 0.0
        mean_tumor_distances = np.zeros((Constants.num_organs,))
        min_tumor_distances = self.gtvs[0].dists[:]
        max_tumor_distances = self.gtvs[0].dists[:]
        tumor_position = np.zeros((3,))
        try:
            for gtv in self.gtvs:
                tumor_volume += gtv.volume
                mean_tumor_distances += gtv.volume*gtv.dists
                min_tumor_distances = np.minimum(gtv.dists, min_tumor_distances)
                max_tumor_distances = np.maximum(gtv.dists, max_tumor_distances)
                tumor_position += gtv.volume*gtv.position
            mean_tumor_distances /= tumor_volume
            tumor_position /= tumor_volume
        except:
            print('error reading tumor volume for ', self.id)
            tumor_volume = 0.0
            min_tumor_distances = np.zeros((Constants.num_organs,))
            mean_tumor_distances = np.zeros((Constants.num_organs,))
            max_tumor_distances = np.zeros((Constants.num_organs,))
            tumor_position = np.zeros((3,))
        return(tumor_volume, min_tumor_distances, mean_tumor_distances,
               max_tumor_distances, tumor_position)