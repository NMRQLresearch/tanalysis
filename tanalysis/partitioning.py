import numpy as np
from sympy import factorint

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