import numpy as np
import tncontract as tn


def mixed_consolidate(left, right, core):
    """
    Consolidates a mixed canonical tensor stored as a seperate left, right and core into a single MPS (list of tensors).

    :param left: A list of tensors left of the core tensor.
    :param core: The core tensor.
    :param right: A list of tensors right of the core tensor.
    :return batch_size_position: The required index of the core tensor
    :return mc_mps: A list of 3 tensors encoding the MPS from left to right.
    """

    batch_size_position = np.size(left)
    num_cores = np.size(left) + np.size(right) + 1
    mc_mps = [None] * num_cores

    count = 0
    for j in range(num_cores):

        if j < batch_size_position:
            mc_mps[j] = left[j].copy()
        elif j == batch_size_position:
            mc_mps[j] = core.copy()
        else:
            mc_mps[j] = right[count].copy()
            count += 1

    return mc_mps


def reconstruct_to_tensor(mps):

    """
    Reconstructs a multi-dimensional tensor from a MPS.

    :param mps: A matrix product state.
    :return tensor_form: the high dimensional tensor corresponding to the MPS.
    """

    num_cores = np.size(mps)

    tensor_form = mps[0].copy()
    tensor_form.replace_label(["c"], ["0"])

    for j in range(1, num_cores):
        tensor_form = tn.contract(tensor_form, mps[j], "b", "a")
        tensor_form.replace_label(["c"], [str(j)])

    tensor_form.remove_all_dummy_indices(labels=None)

    return tensor_form


def reconstruct_to_matrix(mps,batch_size_position):
    """
    Reconstructs a data matrix from a MPS.

    :param mps: A matrix product state.
    :param batch_size_position: The index number of the tensor index corresponding to batch size
    :return matrix_form: A matrix with instances as rows and features as columns
    """

    tensor_form = reconstruct_to_tensor(mps)
    matrix_form = tn.tensor_to_matrix(tensor_form,str(batch_size_position))

    return matrix_form


def reconstruct_to_matrix_from_mc(left, right, core):
    """
    Reconstructs a data matrix from a mixed canonical MPS stored as three seperate lists (left, right, core)

    :param left: A list of tensors left of the core tensor.
    :param core: The core tensor.
    :param right: A list of tensors right of the core tensor.
    :return matrix_form: A matrix with instances as rows and features as columns
    """

    batch_size_position = np.size(left)
    mps = mixed_consolidate(left, right, core)
    matrix_form = reconstruct_to_matrix(mps,batch_size_position)

    return matrix_form


