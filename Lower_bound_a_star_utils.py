def scoring_with_avg(feature_dict):
    """
    Calculating the avgerage distance between specific landmark to all other nodes, then passing through sigmoid
    function to resize scale. The lower the avergae distance, the higher the output value
    :param feature_dict: This dictionary is of the following format: {(source_node,target_node):
    {landmark1:estimated_distance, landmark2:estimated_distance2....}}
    :return: A dict, {landmark1:score1.....}
    """
    # TODO
    return


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
    # TODO
    return
