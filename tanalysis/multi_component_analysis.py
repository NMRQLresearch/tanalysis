import numpy as np
import tncontract as tn
from .reconstructions import *


def extract_core_tensor_via_common_features(data_tensor,left_features,right_features):
    """
    Performs dimensionality reduction of the data (pre tensorized) by extracting a core-tensor via pre-exisiting left and right features
    These pre-existing features are usually obtained by a mixed canonical decomposition of training data

    :param data_tensor: The tensorized data that we want to be reduced via MCA. The partition must match the MCS.
    :param left_features: The list of left tensors from the previously obtained mixed canonical state
    :param right_features: The list of right tensors from the previously obtained mixed canonical state
    :return data-reduced: The extracted reduced data set
    """
    
    left_consolidated = reconstruct_to_tensor(left_features)
    left_consolidated.replace_label("b", "l")
    left_consolidated.move_index("l", 0)

    right_consolidated = reconstruct_to_tensor(right_features)
    right_consolidated.replace_label("a", "r")
    
    batch_size_position = np.size(left_features)
    partition_size = np.size(data_tensor.labels)

    label_list = ["batchsize"]
    label_list.extend([str(j + 1) for j in range(partition_size - 1)])


    right_labeling = label_list[batch_size_position + 1:partition_size]
    left_labeling = label_list[1:batch_size_position + 1]
    right_consolidated.replace_label([right_consolidated.labels[j] for j in range(1, np.size(right_consolidated.labels))], right_labeling)
    left_consolidated.replace_label([left_consolidated.labels[j] for j in range(1, np.size(left_consolidated.labels))], left_labeling)

    extracted_core_int = tn.contract(left_consolidated, data_tensor, left_labeling, left_labeling)
    extracted_core = tn.contract(right_consolidated, extracted_core_int, right_labeling, right_labeling)
    extracted_core.move_index("l", 0)
    extracted_core.replace_label("batchsize", "c")

    data_reduced = tn.tensor_to_matrix(extracted_core, extracted_core.labels[2])

    return data_reduced


def extract_core_tensor_via_common_features_from_matrix(data_matrix, left, right):
    """
    Performs dimensionality reduction of the data by extracting a core-tensor via pre-existing left and right features
    These pre-exsiting features are usually obtained by a mixed canonical decomposition of training data
    This version ensures that the partitioning of the test data matches up accordingly
    ---> it is preferable to use this over "extract_core_tensor_via_common_features"

    :param data_matrix: The data matrix whose features we want extracted. Partition is extracted automatically.
    :param left: The list of left tensors from the previously obtained mixed canonical state
    :param right: The list of left tensors from the previously obtained mixed canonical state
    :return data-reduced: The extracted reduced data set
    """

    batch_size_position = np.size(left)
    partition = [np.shape(data_matrix)[0]]

    left_partition = [np.shape(left[i].data)[2] for i in range(np.size(left))]
    right_partition = [np.shape(right[i].data)[2] for i in range(np.size(right))]

    partition.extend(left_partition)
    partition.extend(right_partition)

    tensor_labels = ["batchsize"]
    tensor_labels.extend([str(j + 1) for j in range(np.size(partition) - 1)])

    data_tensor = tn.matrix_to_tensor(data_matrix, partition, labels=tensor_labels)
    data_tensor.move_index("batchsize", batch_size_position)

    data_reduced = extract_core_tensor_via_common_features(data_tensor, left, right)

    return data_reduced
