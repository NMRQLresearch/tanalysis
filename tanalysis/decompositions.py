import numpy as np
import tncontract as tn
from .partitioning import *

def mixed_canonical_full_withdiagnostics(data_tensor,max_bond_dimension,batch_size_position):
    """
    Performs a full mixed canonical MPS decomposition of a pre-partitioned data tensor.
    Also provides some diagnostics concerning the truncation.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param max_bond_dimension: The maximum bond dimension of the output MPS.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return left: A list of left canonical 3-tensors from the left hand edge up to the core tensor.
    :return right: A list of right canonical 3-tensors from the right hand edge up to the core tensor.
    :return core: The core tensor of the mixed canonical MPS.
    :return svd_thresholds: A list of the ratios of retained singular values to all singular values per bond.
    :return original_bonds: a list of original bond dimensions before truncation
    :return new_bond_percentage: a list of the ratios of truncated bond dimensions over original bond dimensions.
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    left = [None] * (batch_size_position)
    right_count = num_cores - batch_size_position - 1
    right = [None] * (right_count)
    svd_thresholds = [None] * (num_cores - 1)
    original_bonds = [None] * (num_cores - 1)
    new_bond_percentage = [None] * (num_cores - 1)

    total_count_1 = 0
    for j in range(batch_size_position):

        if j == 0:

            left[j], working_tensor, svd_thresholds[total_count_1], original_bonds[total_count_1], new_bond_percentage[
                total_count_1] = tn.truncated_svd(working_tensor, [working_tensor.labels[0]], chi=max_bond_dimension)
            
            left[j].add_dummy_index("a", position=0)
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")
            total_count_1 += 1

        else:

            left[j], working_tensor, svd_thresholds[total_count_1], original_bonds[total_count_1], new_bond_percentage[
                total_count_1] = tn.truncated_svd(working_tensor, [working_tensor.labels[0], working_tensor.labels[1]],
                                              chi=max_bond_dimension)

            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")
            total_count_1 += 1

    total_count_2 = 0
    for j in range(right_count):
        ind = right_count - j - 1

        if j == 0:

            working_tensor, right[ind], svd_thresholds[num_cores - 2 - total_count_2], original_bonds[num_cores - 2 - total_count_2], \
            new_bond_percentage[num_cores - 2 - total_count_2] = tn.truncated_svd(working_tensor, [working_tensor.labels[k] for k in
                                                                                        range(np.size(
                                                                                            working_tensor.labels) - 1)],
                                                                        chi=max_bond_dimension, absorb_singular_values='left')
            right[ind].add_dummy_index("b", position=1)
            right[ind].replace_label(["svd_out", right[ind].labels[2]], ["a", "c"])

            working_tensor.replace_label("svd_in", "b")
            total_count_2 += 1

        else:

            working_tensor, right[ind], svd_thresholds[num_cores - 2 - total_count_2], original_bonds[num_cores - 2 - total_count_2], \
            new_bond_percentage[num_cores - 2 - total_count_2] = tn.truncated_svd(working_tensor, [working_tensor.labels[k] for k in
                                                                                        range(np.size(
                                                                                            working_tensor.labels) - 2)],
                                                                        chi=max_bond_dimension, absorb_singular_values='left')
            right[ind].replace_label(["svd_out", right[ind].labels[1]], ["a", "c"])
            right[ind].move_index("c", 2)

            working_tensor.replace_label("svd_in", "b")
            total_count_2 += 1

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return left, right, core, svd_thresholds, original_bonds, new_bond_percentage


def mixed_canonical_core_only_withdiagnostics(data_tensor, max_bond_dimension, batch_size_position):
    """
    Performs a mixed canonical MPS decomposition of a pre-partitioned data tensor, and only stores the core tensor.
    Also provides some diagnostics concerning the truncation.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param max_bond_dimension: The maximum bond dimension of the output MPS.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return core: The core tensor of the mixed canonical MPS.
    :return svd_thresholds: A list of the ratios of retained singular values to all singular values per bond.
    :return original_bonds: a list of original bond dimensions before truncation
    :return new_bond_percentage: a list of the ratios of truncated bond dimensions over original bond dimensions.
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    right_count = num_cores - batch_size_position - 1
    svd_thresholds = [None] * (num_cores - 1)
    original_bonds = [None] * (num_cores - 1)
    new_bond_percentage = [None] * (num_cores - 1)

    total_count_1 = 0
    for j in range(batch_size_position):

        if j == 0:

            left, working_tensor, svd_thresholds[total_count_1], original_bonds[total_count_1], new_bond_percentage[
                total_count_1] = tn.truncated_svd(working_tensor, [working_tensor.labels[0]], chi=max_bond_dimension)


            working_tensor.replace_label("svd_out", "a")
            total_count_1 += 1

        else:

            left, working_tensor, svd_thresholds[total_count_1], original_bonds[total_count_1], new_bond_percentage[
                total_count_1] = tn.truncated_svd(working_tensor, [working_tensor.labels[0], working_tensor.labels[1]],
                                                  chi=max_bond_dimension)

            working_tensor.replace_label("svd_out", "a")
            total_count_1 += 1


    for j in range(right_count):

        if j == 0:

            working_tensor, right, svd_thresholds[total_count_1], original_bonds[total_count_1], \
            new_bond_percentage[total_count_1] = tn.truncated_svd(working_tensor,
                                                                                  [working_tensor.labels[k] for k in
                                                                                   range(np.size(
                                                                                       working_tensor.labels) - 1)],
                                                                                  chi=max_bond_dimension,
                                                                                  absorb_singular_values='left')


            working_tensor.replace_label("svd_in", "b")
            total_count_1 += 1

        else:

            working_tensor, right, svd_thresholds[total_count_1], original_bonds[total_count_1], \
            new_bond_percentage[total_count_1] = tn.truncated_svd(working_tensor,
                                                                                  [working_tensor.labels[k] for k in
                                                                                   range(np.size(
                                                                                       working_tensor.labels) - 2)],
                                                                                  chi=max_bond_dimension,
                                                                                  absorb_singular_values='left')


            working_tensor.replace_label("svd_in", "b")
            total_count_1 += 1

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return core, svd_thresholds, original_bonds, new_bond_percentage

def mixed_canonical_core_only_nodiagnostics(data_tensor, max_bond_dimension, batch_size_position):
    """
    Performs a mixed canonical MPS decomposition of a pre-partitioned data tensor, and only stores the core tensor.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param max_bond_dimension: The maximum bond dimension of the output MPS.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return core: The core tensor of the mixed canonical MPS.
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    right_count = num_cores - batch_size_position - 1

    total_count_1 = 0
    for j in range(batch_size_position):

        if j == 0:

            left, working_tensor= tn.truncated_svd_eff(working_tensor, [working_tensor.labels[0]], chi=max_bond_dimension)

            working_tensor.replace_label("svd_out", "a")
            total_count_1 += 1

        else:

            left, working_tensor = tn.truncated_svd_eff(working_tensor, [working_tensor.labels[0], working_tensor.labels[1]],
                                                  chi=max_bond_dimension)

            working_tensor.replace_label("svd_out", "a")
            total_count_1 += 1


    for j in range(right_count):

        if j == 0:

            working_tensor, right = tn.truncated_svd_eff(working_tensor,[working_tensor.labels[k] for k in
                                                                                   range(np.size(
                                                                                       working_tensor.labels) - 1)],
                                                                                  chi=max_bond_dimension,
                                                                                  absorb_singular_values='left')


            working_tensor.replace_label("svd_in", "b")
            total_count_1 += 1

        else:

            working_tensor, right = tn.truncated_svd_eff(working_tensor,[working_tensor.labels[k] for k in
                                                                                   range(np.size(
                                                                                       working_tensor.labels) - 2)],
                                                                                  chi=max_bond_dimension,
                                                                                  absorb_singular_values='left')


            working_tensor.replace_label("svd_in", "b")
            total_count_1 += 1

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return core

def core_compression(data_matrix, maximum_bond_dimension):
    """
    Performs dimensionality reduction by extracting the core of a mixed canonical representation of the data tensor.
    In this implementation the data tensor is partitioned via the longest possible prime partition.
    The new representation has maximum_bond_dimension^2 number of features.

    :param data_matrix: A data array with rows as instances and columns as features
    :param max_bond_dimension: The maximum bond dimension of the output MPS.
    :return data_compressed: A compressed representation of the initial data array.
    """

    batch_size = np.shape(data_matrix)[0]  # This is not actually the batch size (TrainingSize)

    pre_partition = symmetrize(raw_partition(np.shape(data_matrix)[1]))
    partition = [batch_size]
    partition.extend(pre_partition)

    tensor_labels = ["batchsize"]
    tensor_labels.extend([str(j + 1) for j in range(np.size(pre_partition))])

    num_cores = np.size(partition)
    batch_size_position = int(round((num_cores - 1) / 2))

    data_tensor = tn.matrix_to_tensor(data_matrix, partition, labels=tensor_labels)
    data_tensor.move_index("batchsize", batch_size_position)

    core_tensor = mixed_canonical_core_only_nodiagnostics(data_tensor, maximum_bond_dimension, batch_size_position)
    data_compressed = tn.tensor_to_matrix(core_tensor, "c")

    return data_compressed



