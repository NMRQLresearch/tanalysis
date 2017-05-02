import numpy as np
import tncontract as tn
from .partitioning import *
from .reconstructions import *


# --------------------------------------------------------------------------------------
# The following two functions provide slightly modified versions of tncontract functions

def truncated_svd_ret_sv(tensor, row_labels, chi=0, threshold=1e-15,
                         absorb_singular_values="right", absolute=True):
    """
    This function overwrites tn.truncated_svd. It performs an svd of the given tensor, as per tn.tensor_svd,
    but then truncates according to both the specified maximum number of singular values and a relative
    or absolute singular value threshold. This version returns both the original and retained singular values.

    :param tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param row_labels: The labels of the tensor which will be reshaped and joined into the matrix row index.
    :param chi: The maximum number of singular values -> chi=0 implies np truncation.
    :param threshold: The absolute or relative threshold for singular value truncation.
    :param absorb_singular_values: Determines whether S is incorporated into U or V
    :param absolute: Determines absolute or relative thresholding.
    :return U_new: The resulting truncated U matrix
    :return V_new: The resulting truncated V matrix.
    :return singular_values: A list of all the original singular values.
    :return singular_values_to_keep: a list of the retained singular values
    """

    U, S, V = tn.tensor_svd(tensor, row_labels)

    singular_values = np.diag(S.data)

    # Truncate to maximum number of singular values

    if chi:
        singular_values_to_keep = singular_values[:chi]
    else:
        singular_values_to_keep = singular_values

    # Thresholding

    if absolute:
        singular_values_to_keep = singular_values_to_keep[singular_values_to_keep > threshold]
    else:
        rel_thresh = singular_values[0] * threshold
        singular_values_to_keep = singular_values_to_keep[singular_values_to_keep > rel_thresh]

    S.data = np.diag(singular_values_to_keep)

    U.move_index("svd_in", 0)
    U.data = U.data[0:len(singular_values_to_keep)]  # U.data=U.data[:,:,0:len(singular_values_to_keep)]
    U.move_index("svd_in", (np.size(U.labels) - 1))
    V.data = V.data[0:len(singular_values_to_keep)]

    # Absorb singular values S into either V or U
    # or take the square root of S and absorb into both (default)

    if absorb_singular_values == "left":
        U_new = tn.contract(U, S, ["svd_in"], ["svd_out"])
        V_new = V
    elif absorb_singular_values == "right":
        V_new = tn.contract(S, V, ["svd_in"], ["svd_out"])
        U_new = U
    else:
        sqrtS = S.copy()
        sqrtS.data = np.sqrt(sqrtS.data)
        U_new = tn.contract(U, sqrtS, ["svd_in"], ["svd_out"])
        V_new = tn.contract(sqrtS, V, ["svd_in"], ["svd_out"])

    return U_new, V_new, singular_values, singular_values_to_keep


def truncated_svd_eff(tensor, row_labels, chi=0, threshold=1e-15,
                      absorb_singular_values="right", absolute=True):
    """
    An efficient version of truncated_svd which does not return singular values.

    :param tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param row_labels: The labels of the tensor which will be reshaped and joined into the matrix row index.
    :param chi: The maximum number of singular values -> chi=0 implies np truncation.
    :param threshold: The absolute or relative threshold for singular value truncation.
    :param absorb_singular_values: Determines whether S is incorporated into U or V
    :param absolute: Determines absolute or relative thresholding.
    :return U_new: The resulting truncated U matrix
    :return V_new: The resulting truncated V matrix.
    :return svd_thresholds: A list of the ratios of retained singular values to all singular values per bond.
    :return original_bonds: a list of original bond dimensions before truncation
    :return new_bond_percentage: a list of the ratios of truncated bond dimensions over original bond dimensions.
    """

    U, S, V = tn.tensor_svd(tensor, row_labels)

    singular_values = np.diag(S.data)

    # Truncate to maximum number of singular values

    if chi:
        singular_values_to_keep = singular_values[:chi]
    else:
        singular_values_to_keep = singular_values

    # Thresholding

    if absolute:
        singular_values_to_keep = singular_values_to_keep[singular_values_to_keep > threshold]
    else:
        rel_thresh = singular_values[0] * threshold
        singular_values_to_keep = singular_values_to_keep[singular_values_to_keep > rel_thresh]

    S.data = np.diag(singular_values_to_keep)

    U.move_index("svd_in", 0)
    U.data = U.data[0:len(singular_values_to_keep)]  # U.data=U.data[:,:,0:len(singular_values_to_keep)]
    U.move_index("svd_in", (np.size(U.labels) - 1))
    V.data = V.data[0:len(singular_values_to_keep)]

    # Absorb singular values S into either V or U

    if absorb_singular_values == "left":
        U_new = tn.contract(U, S, ["svd_in"], ["svd_out"])
        V_new = V
    elif absorb_singular_values == "right":
        V_new = tn.contract(S, V, ["svd_in"], ["svd_out"])
        U_new = U
    else:
        sqrtS = S.copy()
        sqrtS.data = np.sqrt(sqrtS.data)
        U_new = tn.contract(U, sqrtS, ["svd_in"], ["svd_out"])
        V_new = tn.contract(sqrtS, V, ["svd_in"], ["svd_out"])

    return U_new, V_new


# ----------------------------------------------------------------------------------------------------

def mixed_canonical_full_no_truncation_ret_sv(data_tensor, batch_size_position):
    """
    Performs a full mixed canonical MPS decomposition of a pre-partitioned data tensor.
    No truncation of any of the bonds is performed.
    Also provides the singular values (both retained and full) from the decomposition procedure.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param core_bond_dimension: The bond dimension of the output core tensor.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return left: A list of left canonical 3-tensors from the left hand edge up to the core tensor.
    :return right: A list of right canonical 3-tensors from the right hand edge up to the core tensor.
    :return core: The core tensor of the mixed canonical MPS.
    :return full_singular_values: A list of lists of singular values per bond, before truncation.
    :return retained_singular_values: A list of lists of singular values per bond, after truncation
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    left = [None] * (batch_size_position)
    right_count = num_cores - batch_size_position - 1
    right = [None] * (right_count)

    full_singular_values = [[] for k in range(num_cores - 1)]
    retained_singular_values = [[] for k in range(num_cores - 1)]

    for j in range(batch_size_position):

        if j == 0:

            left[j], S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0]])
            left[j].add_dummy_index("a", position=0)
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            full_singular_values[j] = np.diag(S.data)
            retained_singular_values[j] = np.diag(S.data)

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

        else:

            left[j], S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0], working_tensor.labels[1]])
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            full_singular_values[j] = np.diag(S.data)
            retained_singular_values[j] = np.diag(S.data)

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

    for j in range(right_count):
        ind = right_count - j - 1
        ind_2 = num_cores - j - 2

        if j == 0:

            U, S, right[ind] = tn.tensor_svd(working_tensor, [working_tensor.labels[k] for k in
                                                              range(np.size(working_tensor.labels) - 1)])
            right[ind].add_dummy_index("b", position=1)
            right[ind].replace_label(["svd_out", right[ind].labels[2]], ["a", "c"])

            full_singular_values[ind_2] = np.diag(S.data)
            retained_singular_values[ind_2] = np.diag(S.data)

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

        else:

            U, S, right[ind] = tn.tensor_svd(working_tensor, [working_tensor.labels[k] for k in
                                                              range(np.size(working_tensor.labels) - 2)])
            right[ind].replace_label(["svd_out", right[ind].labels[1]], ["a", "c"])
            right[ind].move_index("c", 2)

            full_singular_values[ind_2] = np.diag(S.data)
            retained_singular_values[ind_2] = np.diag(S.data)

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return left, right, core, full_singular_values, retained_singular_values


def mixed_canonical_full(data_tensor, max_bond_dimension, batch_size_position):
    """
    Performs a full mixed canonical MPS decomposition of a pre-partitioned data tensor.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param max_bond_dimension: The maximum bond dimension of the output MPS.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return left: A list of left canonical 3-tensors from the left hand edge up to the core tensor.
    :return right: A list of right canonical 3-tensors from the right hand edge up to the core tensor.
    :return core: The core tensor of the mixed canonical MPS.
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    left = [None] * (batch_size_position)
    right_count = num_cores - batch_size_position - 1
    right = [None] * (right_count)

    for j in range(batch_size_position):

        if j == 0:

            left[j], working_tensor = truncated_svd_eff(
                working_tensor, [working_tensor.labels[0]], chi=max_bond_dimension)

            left[j].add_dummy_index("a", position=0)
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")

        else:

            left[j], working_tensor = truncated_svd_eff(
                working_tensor, [working_tensor.labels[0], working_tensor.labels[1]], chi=max_bond_dimension)

            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")

    for j in range(right_count):
        ind = right_count - j - 1

        if j == 0:

            working_tensor, right[ind] = truncated_svd_eff(working_tensor,
                [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 1)],
                chi=max_bond_dimension,
                absorb_singular_values='left')

            right[ind].add_dummy_index("b", position=1)
            right[ind].replace_label(["svd_out", right[ind].labels[2]], ["a", "c"])

            working_tensor.replace_label("svd_in", "b")

        else:

            working_tensor, right[ind] = truncated_svd_eff(working_tensor,
                                              [working_tensor.labels[k] for k in
                                               range(np.size(
                                                   working_tensor.labels) - 2)],
                                              chi=max_bond_dimension,
                                              absorb_singular_values='left')

            right[ind].replace_label(["svd_out", right[ind].labels[1]], ["a", "c"])
            right[ind].move_index("c", 2)

            working_tensor.replace_label("svd_in", "b")

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return left, right, core


def mixed_canonical_full_ret_sv(data_tensor, max_bond_dimension, batch_size_position):
    """
    Performs a full mixed canonical MPS decomposition of a pre-partitioned data tensor.
    Also returns the singular values from the decomposition procedure.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param max_bond_dimension: The maximum bond dimension of the output MPS.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return left: A list of left canonical 3-tensors from the left hand edge up to the core tensor.
    :return right: A list of right canonical 3-tensors from the right hand edge up to the core tensor.
    :return core: The core tensor of the mixed canonical MPS.
    :return full_singular_values: A list of lists of singular values per bond, before truncation.
    :return retained_singular_values: A list of lists of singular values per bond, after truncation
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    left = [None] * (batch_size_position)
    right_count = num_cores - batch_size_position - 1
    right = [None] * (right_count)

    full_singular_values = [[] for k in range(num_cores - 1)]
    retained_singular_values = [[] for k in range(num_cores - 1)]

    for j in range(batch_size_position):

        if j == 0:

            left[j], working_tensor, full_singular_values[j], retained_singular_values[j] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[0]], chi=max_bond_dimension)

            left[j].add_dummy_index("a", position=0)
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")

        else:

            left[j], working_tensor, full_singular_values[j], retained_singular_values[j] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[0], working_tensor.labels[1]], chi=max_bond_dimension)

            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")

    for j in range(right_count):
        ind = right_count - j - 1
        ind_2 = num_cores - j - 2

        if j == 0:

            working_tensor, right[ind], full_singular_values[ind_2], retained_singular_values[
                ind_2] = truncated_svd_ret_sv(
                working_tensor,
                [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 1)],
                chi=max_bond_dimension,
                absorb_singular_values='left')

            right[ind].add_dummy_index("b", position=1)
            right[ind].replace_label(["svd_out", right[ind].labels[2]], ["a", "c"])

            working_tensor.replace_label("svd_in", "b")

        else:

            working_tensor, right[ind], full_singular_values[ind_2], retained_singular_values[
                ind_2] = truncated_svd_ret_sv(working_tensor,
                                              [working_tensor.labels[k] for k in
                                               range(np.size(
                                                   working_tensor.labels) - 2)],
                                              chi=max_bond_dimension,
                                              absorb_singular_values='left')

            right[ind].replace_label(["svd_out", right[ind].labels[1]], ["a", "c"])
            right[ind].move_index("c", 2)

            working_tensor.replace_label("svd_in", "b")

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return left, right, core, full_singular_values, retained_singular_values


def mixed_canonical_core_only_ret_sv(data_tensor, max_bond_dimension, batch_size_position):
    """
    Performs a mixed canonical MPS decomposition of a pre-partitioned data tensor, and only stores the core tensor.
    Also provides some diagnostics concerning the truncation.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param max_bond_dimension: The maximum bond dimension of the output MPS.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return core: The core tensor of the mixed canonical MPS.
    :return full_singular_values: A list of lists of singular values per bond, before truncation.
    :return retained_singular_values: A list of lists of singular values per bond, after truncation
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    right_count = num_cores - batch_size_position - 1

    full_singular_values = [[] for k in range(num_cores - 1)]
    retained_singular_values = [[] for k in range(num_cores - 1)]

    for j in range(batch_size_position):

        if j == 0:

            left, working_tensor, full_singular_values[j], retained_singular_values[j] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[0]], chi=max_bond_dimension)

            working_tensor.replace_label("svd_out", "a")

        else:

            left, working_tensor, full_singular_values[j], retained_singular_values[j] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[0], working_tensor.labels[1]], chi=max_bond_dimension)

            working_tensor.replace_label("svd_out", "a")

    for j in range(right_count):
        ind_2 = num_cores - j - 2

        if j == 0:

            working_tensor, right, full_singular_values[ind_2], retained_singular_values[
                ind_2] = truncated_svd_ret_sv(working_tensor,
                        [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 1)],
                         chi=max_bond_dimension,
                         absorb_singular_values='left')

            working_tensor.replace_label("svd_in", "b")

        else:

            working_tensor, right, full_singular_values[ind_2], retained_singular_values[
                ind_2] = truncated_svd_ret_sv(working_tensor,
                        [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 2)],
                         chi=max_bond_dimension,
                         absorb_singular_values='left')

            working_tensor.replace_label("svd_in", "b")


    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return core, full_singular_values, retained_singular_values


def mixed_canonical_core_only(data_tensor, max_bond_dimension, batch_size_position):
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

            left, working_tensor = truncated_svd_eff(working_tensor, [working_tensor.labels[0]],
                                                     chi=max_bond_dimension)

            working_tensor.replace_label("svd_out", "a")
            total_count_1 += 1

        else:

            left, working_tensor = truncated_svd_eff(working_tensor,
                                                     [working_tensor.labels[0], working_tensor.labels[1]],
                                                     chi=max_bond_dimension)

            working_tensor.replace_label("svd_out", "a")
            total_count_1 += 1

    for j in range(right_count):

        if j == 0:

            working_tensor, right = truncated_svd_eff(working_tensor, [working_tensor.labels[k] for k in
                                                                       range(np.size(
                                                                           working_tensor.labels) - 1)],
                                                      chi=max_bond_dimension,
                                                      absorb_singular_values='left')

            working_tensor.replace_label("svd_in", "b")
            total_count_1 += 1

        else:

            working_tensor, right = truncated_svd_eff(working_tensor, [working_tensor.labels[k] for k in
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


def mixed_canonical_full_core_truncation_only(data_tensor, core_bond_dimension, batch_size_position):
    """
    Performs a full mixed canonical MPS decomposition of a pre-partitioned data tensor.
    Only truncation of the core tensor bonds is performed

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param core_bond_dimension: The bond dimension of the output core tensor.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return left: A list of left canonical 3-tensors from the left hand edge up to the core tensor.
    :return right: A list of right canonical 3-tensors from the right hand edge up to the core tensor.
    :return core: The core tensor of the mixed canonical MPS.
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    left = [None] * (batch_size_position)
    right_count = num_cores - batch_size_position - 1
    right = [None] * (right_count)

    for j in range(batch_size_position):

        if j == 0 and j != (batch_size_position - 1):

            left[j], S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0]])
            left[j].add_dummy_index("a", position=0)
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

        elif j == 0 and j == (batch_size_position - 1):

            left[j], working_tensor = truncated_svd_eff(
                working_tensor, [working_tensor.labels[0]], chi=core_bond_dimension)

            left[j].add_dummy_index("a", position=0)
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")

        elif j > 0 and j == (batch_size_position - 1):

            left[j], working_tensor = truncated_svd_eff(
                working_tensor, [working_tensor.labels[0], working_tensor.labels[1]],
                chi=core_bond_dimension)

            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")

        else:

            left[j], S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0], working_tensor.labels[1]])
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

    for j in range(right_count):
        ind = right_count - j - 1

        if j == 0 and j != (right_count - 1):

            U, S, right[ind] = tn.tensor_svd(working_tensor, [working_tensor.labels[k] for k in
                                                              range(np.size(working_tensor.labels) - 1)])
            right[ind].add_dummy_index("b", position=1)
            right[ind].replace_label(["svd_out", right[ind].labels[2]], ["a", "c"])

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

        elif j == 0 and j == (right_count - 1):

            working_tensor, right[ind] = truncated_svd_eff(working_tensor,
                                                           [working_tensor.labels[k] for k in
                                                            range(np.size(working_tensor.labels) - 1)],
                                                           chi=core_bond_dimension, absorb_singular_values='left')

            right[ind].add_dummy_index("b", position=1)
            right[ind].replace_label(["svd_out", right[ind].labels[2]], ["a", "c"])

            working_tensor.replace_label("svd_in", "b")

        elif j > 0 and j == (right_count - 1):

            working_tensor, right[ind] = truncated_svd_eff(working_tensor,
                                                           [working_tensor.labels[k] for k in
                                                            range(np.size(working_tensor.labels) - 2)],
                                                           chi=core_bond_dimension, absorb_singular_values='left')

            right[ind].replace_label(["svd_out", right[ind].labels[1]], ["a", "c"])
            right[ind].move_index("c", 2)

            working_tensor.replace_label("svd_in", "b")

        else:

            U, S, right[ind] = tn.tensor_svd(working_tensor, [working_tensor.labels[k] for k in
                                                              range(np.size(working_tensor.labels) - 2)])
            right[ind].replace_label(["svd_out", right[ind].labels[1]], ["a", "c"])
            right[ind].move_index("c", 2)

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return left, right, core


def mixed_canonical_full_core_truncation_only_ret_sv(data_tensor, core_bond_dimension, batch_size_position):
    """
    Performs a full mixed canonical MPS decomposition of a pre-partitioned data tensor.
    Only truncation of the core tensor bonds is performed.
    Also provides the singular values (both retained and full) from the decomposition procedure.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param core_bond_dimension: The bond dimension of the output core tensor.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return left: A list of left canonical 3-tensors from the left hand edge up to the core tensor.
    :return right: A list of right canonical 3-tensors from the right hand edge up to the core tensor.
    :return core: The core tensor of the mixed canonical MPS.
    :return full_singular_values: A list of lists of singular values per bond, before truncation.
    :return retained_singular_values: A list of lists of singular values per bond, after truncation
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    left = [None] * (batch_size_position)
    right_count = num_cores - batch_size_position - 1
    right = [None] * (right_count)

    full_singular_values = [[] for k in range(num_cores - 1)]
    retained_singular_values = [[] for k in range(num_cores - 1)]

    for j in range(batch_size_position):

        if j == 0 and j != (batch_size_position - 1):

            left[j], S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0]])
            left[j].add_dummy_index("a", position=0)
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            full_singular_values[j] = np.diag(S.data)
            retained_singular_values[j] = np.diag(S.data)

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

        elif j == 0 and j == (batch_size_position - 1):

            left[j], working_tensor, full_singular_values[j], retained_singular_values[j] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[0]], chi=core_bond_dimension)

            left[j].add_dummy_index("a", position=0)
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")

        elif j > 0 and j == (batch_size_position - 1):

            left[j], working_tensor, full_singular_values[j], retained_singular_values[j] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[0], working_tensor.labels[1]],
                chi=core_bond_dimension)

            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")

        else:

            left[j], S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0], working_tensor.labels[1]])
            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            full_singular_values[j] = np.diag(S.data)
            retained_singular_values[j] = np.diag(S.data)

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

    for j in range(right_count):
        ind = right_count - j - 1
        ind_2 = num_cores - j - 2

        if j == 0 and j != (right_count - 1):

            U, S, right[ind] = tn.tensor_svd(working_tensor, [working_tensor.labels[k] for k in
                                                              range(np.size(working_tensor.labels) - 1)])
            right[ind].add_dummy_index("b", position=1)
            right[ind].replace_label(["svd_out", right[ind].labels[2]], ["a", "c"])

            full_singular_values[ind_2] = np.diag(S.data)
            retained_singular_values[ind_2] = np.diag(S.data)

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

        elif j == 0 and j == (right_count - 1):

            working_tensor, right[ind], full_singular_values[ind_2], retained_singular_values[ind_2] = \
                truncated_svd_ret_sv(working_tensor,
                                     [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 1)],
                                     chi=core_bond_dimension, absorb_singular_values='left')

            right[ind].add_dummy_index("b", position=1)
            right[ind].replace_label(["svd_out", right[ind].labels[2]], ["a", "c"])

            working_tensor.replace_label("svd_in", "b")

        elif j > 0 and j == (right_count - 1):

            working_tensor, right[ind], full_singular_values[ind_2], retained_singular_values[ind_2] = \
                truncated_svd_ret_sv(working_tensor,
                                     [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 2)],
                                     chi=core_bond_dimension, absorb_singular_values='left')
            right[ind].replace_label(["svd_out", right[ind].labels[1]], ["a", "c"])
            right[ind].move_index("c", 2)

            working_tensor.replace_label("svd_in", "b")

        else:

            U, S, right[ind] = tn.tensor_svd(working_tensor, [working_tensor.labels[k] for k in
                                                              range(np.size(working_tensor.labels) - 2)])
            right[ind].replace_label(["svd_out", right[ind].labels[1]], ["a", "c"])
            right[ind].move_index("c", 2)

            full_singular_values[ind_2] = np.diag(S.data)
            retained_singular_values[ind_2] = np.diag(S.data)

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return left, right, core, full_singular_values, retained_singular_values


def mixed_canonical_core_only_core_truncation_only(data_tensor, core_bond_dimension, batch_size_position):
    """
    Performs a mixed canonical MPS decomposition of a pre-partitioned data tensor, and only stores the core tensor.
    Only the bonds directly attached to the core tensor are truncated.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param core_bond_dimension: The bond dimension of the output core tensor.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return core: The core tensor of the mixed canonical MPS.
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    right_count = num_cores - batch_size_position - 1

    for j in range(batch_size_position):

        if j == 0 and j != (batch_size_position - 1):

            U, S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0]])

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

        elif j == 0 and j == (batch_size_position - 1):

            U, working_tensor = truncated_svd_eff(working_tensor, [working_tensor.labels[0]], chi=core_bond_dimension)

            working_tensor.replace_label("svd_out", "a")

        elif j > 0 and j == (batch_size_position - 1):

            U, working_tensor = truncated_svd_eff(working_tensor, [
                working_tensor.labels[0], working_tensor.labels[1]], chi=core_bond_dimension)

            working_tensor.replace_label("svd_out", "a")

        else:

            U, S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0], working_tensor.labels[1]])

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

    for j in range(right_count):

        if j == 0 and j != (right_count - 1):

            U, S, V = tn.tensor_svd(working_tensor,
                                    [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 1)])

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

        elif j == 0 and j == (right_count - 1):

            working_tensor, V = truncated_svd_eff(working_tensor, [working_tensor.labels[k] for k in range(
                np.size(working_tensor.labels) - 1)],
                                                  chi=core_bond_dimension,
                                                  absorb_singular_values='left')

            working_tensor.replace_label("svd_in", "b")

        elif j > 0 and j == (right_count - 1):

            working_tensor, V = truncated_svd_eff(working_tensor, [working_tensor.labels[k] for k in range(
                np.size(working_tensor.labels) - 2)],
                                                  chi=core_bond_dimension,
                                                  absorb_singular_values='left')

            working_tensor.replace_label("svd_in", "b")

        else:

            U, S, V = tn.tensor_svd(working_tensor,
                                    [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 2)])

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return core


def mixed_canonical_core_only_core_truncation_only_ret_sv(data_tensor, core_bond_dimension, batch_size_position):
    """
    Performs a mixed canonical MPS decomposition of a pre-partitioned data tensor, and only stores the core tensor.
    Only the bonds directly attached to the core tensor are truncated.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param core_bond_dimension: The bond dimension of the output core tensor.
    :param batch_size_position: The index number corresponding to the batch (training sample number) index.
    :return core: The core tensor of the mixed canonical MPS.
    :return full_singular_values: A list of lists of singular values per bond, before truncation.
    :return retained_singular_values: A list of lists of singular values per bond, after truncation
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    right_count = num_cores - batch_size_position - 1

    full_singular_values = [[] for k in range(num_cores - 1)]
    retained_singular_values = [[] for k in range(num_cores - 1)]

    for j in range(batch_size_position):

        if j == 0 and j != (batch_size_position - 1):

            U, S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0]])

            full_singular_values[j] = np.diag(S.data)
            retained_singular_values[j] = np.diag(S.data)

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

        elif j == 0 and j == (batch_size_position - 1):

            U, working_tensor, full_singular_values[j], retained_singular_values[j] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[0]], chi=core_bond_dimension)

            working_tensor.replace_label("svd_out", "a")

        elif j > 0 and j == (batch_size_position - 1):

            U, working_tensor, full_singular_values[j], retained_singular_values[j] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[0], working_tensor.labels[1]], chi=core_bond_dimension)

            working_tensor.replace_label("svd_out", "a")

        else:

            U, S, V = tn.tensor_svd(working_tensor, [working_tensor.labels[0], working_tensor.labels[1]])

            full_singular_values[j] = np.diag(S.data)
            retained_singular_values[j] = np.diag(S.data)

            working_tensor = tn.contract(S, V, "svd_in", "svd_out")
            working_tensor.replace_label("svd_out", "a")

    for j in range(right_count):
        ind_2 = num_cores - j - 2

        if j == 0 and j != (right_count - 1):

            U, S, V = tn.tensor_svd(working_tensor,
                                    [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 1)])

            full_singular_values[ind_2] = np.diag(S.data)
            retained_singular_values[ind_2] = np.diag(S.data)

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

        elif j == 0 and j == (right_count - 1):

            working_tensor, V, full_singular_values[ind_2], retained_singular_values[ind_2] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 1)],
                chi=core_bond_dimension,
                absorb_singular_values='left')

            working_tensor.replace_label("svd_in", "b")

        elif j > 0 and j == (right_count - 1):

            working_tensor, V, full_singular_values[ind_2], retained_singular_values[ind_2] = truncated_svd_ret_sv(
                working_tensor, [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 2)],
                chi=core_bond_dimension,
                absorb_singular_values='left')

            working_tensor.replace_label("svd_in", "b")

        else:

            U, S, V = tn.tensor_svd(working_tensor,
                                    [working_tensor.labels[k] for k in range(np.size(working_tensor.labels) - 2)])

            full_singular_values[ind_2] = np.diag(S.data)
            retained_singular_values[ind_2] = np.diag(S.data)

            working_tensor = tn.contract(U, S, "svd_in", "svd_out")
            working_tensor.replace_label("svd_in", "b")

    core = working_tensor.copy()
    core.replace_label(core.labels[1], "c")
    core.move_index("c", 2)

    return core, full_singular_values, retained_singular_values


def core_compression(data_matrix, maximum_bond_dimension):
    """
    Performs dimensionality reduction by extracting the core of a mixed canonical representation of the data tensor.
    In this implementation the data tensor is partitioned via the longest possible prime partition.
    The new representation has maximum_bond_dimension^2 number of features.

    :param data_matrix: A data array with rows as instances and columns as features
    :param maximum_bond_dimension: The maximum bond dimension of the output MPS.
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

    core_tensor = mixed_canonical_core_only(data_tensor, maximum_bond_dimension, batch_size_position)
    data_compressed = tn.tensor_to_matrix(core_tensor, "c")

    return data_compressed


def core_compression_core_truncation_only(data_matrix, maximum_bond_dimension):
    """
    Performs dimensionality reduction by extracting the core of a mixed canonical representation of the data tensor.
    In this implementation the data tensor is partitioned via the longest possible prime partition.
    The new representation has at most maximum_bond_dimension^2 number of features.
    This "core truncation only" version of "core_compression" truncates only the bonds attached to the core tensor

    :param data_matrix: A data array with rows as instances and columns as features
    :param core_bond_dimension: The  bond dimension of the output core tensor.
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

    core_tensor = mixed_canonical_core_only_core_truncation_only(data_tensor,
                                                                 maximum_bond_dimension,
                                                                 batch_size_position)
    data_compressed = tn.tensor_to_matrix(core_tensor, "c")

    return data_compressed


def core_compression_core_truncation_only_with_partition(data_matrix, maximum_bond_dimension, feature_partition):
    """
    Performs dimensionality reduction by extracting the core of a mixed canonical representation of the data tensor.
    In this implementation one has to supply the partition. NB: This has to be a factorization of num_features!
    The new representation has at most maximum_bond_dimension^2 number of features.
    This "core truncation only" version of "core_compression" truncates only the bonds attached to the core tensor

    :param data_matrix: A data array with rows as instances and columns as features
    :param maximum_bond_dimension: The  bond dimension of the output core tensor.
    :param feature_partition: A list. The partition according to which the feature dimension data matrix will be tensorized.
    :return data_compressed: A compressed representation of the initial data array.
    """

    batch_size = np.shape(data_matrix)[0]  # This is not actually the batch size (TrainingSize)
    partition = [batch_size]
    partition.extend(feature_partition)

    tensor_labels = ["batchsize"]
    tensor_labels.extend([str(j + 1) for j in range(np.size(feature_partition))])

    num_cores = np.size(partition)
    batch_size_position = int(round((num_cores - 1) / 2))

    data_tensor = tn.matrix_to_tensor(data_matrix, partition, labels=tensor_labels)
    data_tensor.move_index("batchsize", batch_size_position)

    core_tensor = mixed_canonical_core_only_core_truncation_only(data_tensor,
                                                                 maximum_bond_dimension,
                                                                 batch_size_position)
    data_compressed = tn.tensor_to_matrix(core_tensor, "c")

    return data_compressed


def left_canonical_decompose_no_diagnostics(data_tensor, max_bond_dimension):
    """
    Performs a full left canonical MPS decomposition of a pre-partitioned data tensor.

    :param data_tensor: The multi-dimensional (tncontract) tensor to be decomposed.
    :param max_bond_dimension: The maximum bond dimension of the output MPS.
    :return left: A left canonical matrix product state
    """

    working_tensor = data_tensor.copy()
    num_cores = np.size(working_tensor.labels)
    left = [None] * (num_cores)

    for j in range(num_cores - 1):

        if j == 0:
            left[j], working_tensor = truncated_svd_eff(working_tensor, [working_tensor.labels[0]],
                                                        chi=max_bond_dimension)
            left[j].add_dummy_index("a", position=0)

            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            working_tensor.replace_label("svd_out", "a")

        else:

            left[j], working_tensor = truncated_svd_eff(working_tensor,
                                                        [working_tensor.labels[0], working_tensor.labels[1]],
                                                        chi=max_bond_dimension)

            left[j].replace_label([left[j].labels[1], "svd_in"], ["c", "b"])
            left[j].move_index("c", 2)

            if j == (num_cores - 2):  # I might want to update this for true LN!

                left[j + 1] = working_tensor
                left[j + 1].replace_label(["svd_out", left[j + 1].labels[1]], ["a", "c"])
                left[j + 1].add_dummy_index("b", position=1)

            else:

                working_tensor.replace_label("svd_out", "a")

    return left


def single_vector_individual_left_canonical_mps_compression(feature_vector, partition, labels, max_bond_dimension):
    """
    Performs a full left canonical MPS decomposition of a vector.

    :param feature_vector: The vector to be decomposed.
    :param partition: The partition according to which the decompostion should be performed.
    :param labels: The labels of the partition. This is required for tensorization.
    :param max_bond_dimension: maximum bond dimension of the decomposition
    :return feature_mps: The MPS representing the feature vector
    """

    feature_tensor = tn.matrix_to_tensor(feature_vector, partition, labels=labels)
    feature_mps = left_canonical_decompose_no_diagnostics(feature_tensor, max_bond_dimension)

    return feature_mps


def single_vector_individual_left_canonical_mps_compression_with_reconstruction(feature_vector, partition, labels,
                                                                                max_bond_dimension):
    """
    Performs a full left canonical MPS decomposition of a vector, and then reconstructs a vector from the MPS.

    :param feature_vector: The vector to be decomposed.
    :param partition: The partition according to which the decompostion should be performed.
    :param labels: The labels of the partition. This is required for tensorization.
    :param max_bond_dimension: maximum bond dimension of the decomposition
    :return feature_compressed_vector: The mps representing the feature vector
    """

    feature_mps = single_vector_individual_left_canonical_mps_compression(feature_vector, partition, labels,
                                                                          max_bond_dimension)
    feature_consolidated_tensor = reconstruct_to_tensor(feature_mps)
    feature_compressed_vector = np.reshape(feature_consolidated_tensor.data, [np.size(feature_vector)])

    return feature_compressed_vector


def full_dataset_individual_left_canonical_mps_compression_with_reconstruction(all_data, partition, max_bond_dimension):
    """
    Performs a full left canonical MPS decomposition, and subsequent vector reconstruction,
     of every vector within a dataset.

    :param all_data: A matrix with each row a different instance/vector of feature values
    :param partition: The partition according to which the decomposition should be performed.
    :param max_bond_dimension: maximum bond dimension of the decomposition
    :return compressed_data: A matrix with each row corresponding to the reconstructed compressed row of all_data
    """

    labels = [str(j + 1) for j in range(np.size(partition))]
    compressed_data = np.zeros(np.shape(all_data))

    for j in range(np.shape(all_data)[0]):
        compressed_data[j, :] = single_vector_individual_left_canonical_mps_compression_with_reconstruction(
            all_data[j, :],
            partition,
            labels,
            max_bond_dimension)

    return compressed_data


def full_dataset_individual_left_canonical_mps_compression(all_data, partition, max_bond_dimension):
    """
    Performs a full left canonical MPS decomposition of every vector within a dataset.

    :param all_data: A matrix with each row a different instance/vector of feature values
    :param partition: The partition according to which the decomposition should be performed.
    :param max_bond_dimension: maximum bond dimension of the decomposition
    :return compressed_data: A list in which element j is the MPS representation of row j of all_data
    """

    labels = [str(j + 1) for j in range(np.size(partition))]
    compressed_data = [[] for j in range(np.shape(all_data)[0])]

    for j in range(np.shape(all_data)[0]):
        compressed_data[j] = single_vector_individual_left_canonical_mps_compression(
            all_data[j, :],
            partition,
            labels,
            max_bond_dimension)

    return compressed_data
