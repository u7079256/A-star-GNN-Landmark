import math
import numpy as np


def scoring_with_avg(feature_dict):
    """
    Calculating the avgerage distance between specific landmark to all other nodes, then passing through sigmoid
    function to resize scale. The lower the avergae distance, the higher the output value
    :param feature_dict: This dictionary is of the following format: {(source_node,target_node):
    {landmark1:estimated_distance, landmark2:estimated_distance2....}}
    :return: A dict, {landmark1:score1.....}
    """
    landmark_score = {}
    for key, value in feature_dict.items():
        for landmark, distance in value.items():
            if landmark not in landmark_score:
                landmark_score[landmark] = distance
            else:
                landmark_score[landmark] += distance
    for key, value in landmark_score.items():
        landmark_score[key] = - math.tanh(value) + 1
    return landmark_score


def scoring_with_ordering(feature_dict):
    """
    For every node pairs, we could get the order of the landmark estimated distance, every landmark will get a value
    indicates their order in specific pair, we then add them up and reorder the landmark. From nearest
    (the lowest order) to furthest (the highest order), the final value will start from 1 to n
    :param feature_dict: This dictionary is of the following format: {(source_node,target_node):
    {landmark1:estimated_distance, landmark2:estimated_distance2....}}
    :param feature_dict:
    :return: A dict, {landmark1:score1.....} n >= score_n >= 1
    """
    landmark_score = {}
    #print("=============================")
    #print(feature_dict)
    #print(":+++++++++++++++++++++++++++++++")
    for key, value_dict in feature_dict.items():
        #print(value_dict)
        distance_list = sorted(value_dict.items(), key=lambda x: x[1], reverse=True)
        score_default = 0
        for pairs in distance_list:
            if pairs[0] not in landmark_score:
                landmark_score[pairs[0]] = score_default
                score_default += 1
            else:
                landmark_score[pairs[0]] += score_default
                score_default += 1
    for key, value in landmark_score.items():
        landmark_score[key] = - math.tanh(value) + 1
    return landmark_score
