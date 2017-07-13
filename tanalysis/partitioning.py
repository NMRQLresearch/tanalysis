import numpy as np
from sympy import factorint
import tncontract as tn

def raw_partition(N):
    """
    Generates the longest possible product-partition of N.
    i.e. the longest list of prime numbers partition such that prod(partition) = N

    :param N: A postive integer.
    :return partition: The longest prime product-partition of N (A list of prime numbers).
    """
    factor_dict = factorint(N)
    partition = []

    for key in factor_dict:
        for j in range(factor_dict[key]):
            partition.append(key)
    return partition


def symmetrize(partition):
    """
    Symmetrizes a given partition.
    i.e. permute the partition such that if plotted the resulting shape would be pyramidal.

    :param partition: A list of integers.
    :return sym_partition: The symmetrized initial list.
    """

    partition_length = np.size(partition)
    sorted_partition = np.sort(partition)
    sym_partition = [None] * partition_length

    count_left = 0
    count_right = 0
    for j in range(partition_length):
        if np.mod(j, 2) == 0:
            sym_partition[count_left] = int(sorted_partition[j])
            count_left += 1
        else:
            sym_partition[partition_length - 1 - count_right] = int(sorted_partition[j])
            count_right += 1

    return sym_partition

def create_random_permutations(partition,N):
    """
    Creates a list of N random permutations of the partition

    :param partition: A list of integers.
    :return partition_list: a list of N random permutations of partition (i.e. a list of lists)
    """

    partition_list = [np.ndarray.tolist(np.random.permutation(partition)) for j in range(N)]
    return partition_list


def compress_partition(partition, steps, step_size, sym=True):
    """
    Compresses a given partition.
    Compresses from the outside in, by using a given number of steps, of a certain step size.
    Each step multiplies together "step_size" entries of the partition to create a new entry.
    If the partition cannot be compressed as requested, then the original partition is returned and a flag is raised.

    :param partition: A list of integers.
    :param steps: The number of steps (of size "step_size") to take in the compression process
    :param step_size: The size of each compression step
    :param sym: If "True" then the resulting partition is always symmetric
    :return new_partition: The compressed partition (a list of integers)
    :return compression_success: If "True" then the compression was possible and was executed.
    """

    partition_length = np.size(partition)
    compression_success = True

    if np.mod(partition_length, 2) == 0:
        threshhold = partition_length / 2
    else:
        threshhold = (partition_length - 1) / 2

    if step_size * steps > threshhold:
        return partition, False
    else:

        new_length = partition_length - (step_size - 1) * 2 * steps
        new_partition = [None] * (new_length)

        for j in range(steps):
            new_partition[j] = np.prod(partition[j * step_size:(j + 1) * step_size])
            new_partition[new_length - 1 - j] = np.prod(partition[partition_length - (j + 1) * step_size:partition_length - j * step_size])

        new_partition[steps:new_length - steps] = partition[steps * step_size:(partition_length - steps * step_size)]

        if steps == 1 and sym:
            new_partition = symmetrize(new_partition)

        return new_partition, compression_success


def full_partition(data):
    """
    Performs full virtual tensorization of a data matrix via the symmetrized raw partition.
    This function also centers the batch_size_postition by default.

    :param data: A matrix of data with rows as instances and features as columns.
    :return data_tensor: A tensorized form of the data, tensorized via the symmetric raw partition.
    :return batch_size_position: The position of the batch size index in the partition.
    """

    training_set_size = np.shape(data)[0]
    pre_partition = symmetrize(raw_partition(np.shape(data)[1]))
    partition = [training_set_size]
    partition.extend(pre_partition)

    tensor_labels = ["batchsize"]
    tensor_labels.extend([str(j + 1) for j in range(np.size(pre_partition))])

    num_cores = np.size(partition)
    batch_size_position = int(round((num_cores - 1) / 2))

    data_tensor = tn.matrix_to_tensor(data, partition, labels=tensor_labels)
    data_tensor.move_index("batchsize", batch_size_position)

    return data_tensor, batch_size_position
