import numpy as np
import numpy.testing as npt
import tncontract as tn
from tanalysis import *

import os
import unittest
from unittest import TestCase


class TestMixedCanonicalDecompositions(TestCase):
    def setUp(self):

        # First we set and load common variables for all test cases
        location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) + "/unit_test_data/"
        self.test_data = np.loadtxt(location + "test_data.gz")

        self.partition_list = [[2000, 28, 28],
                               [2000, 7, 16, 7],
                               [2000, 2, 2, 7, 7, 2, 2],
                               [2000, 2, 2, 7, 7, 2, 2],
                               [2000, 2, 2, 7, 7, 2, 2]]

        self.labels_list = [["batchsize", "1", "2"],
                            ["batchsize", "1", "2", "3"],
                            ["batchsize", "1", "2", "3", "4", "5", "6"],
                            ["batchsize", "1", "2", "3", "4", "5", "6"],
                            ["batchsize", "1", "2", "3", "4", "5", "6"]]

        self.num_bonds = [len(partition)-1 for partition in self.partition_list]
        self.batch_size_pos_list = [1, 1, 3, 1, 5]
        self.num_partitions = 5
        self.bond_dimension = 3

        # Load test benchmarks for full truncation test
        self.test_full_sv_a1 = [[] for j in range(self.num_partitions)]
        self.test_ret_sv_a1 = [[] for j in range(self.num_partitions)]
        self.test_cores_a1 = [[] for j in range(self.num_partitions)]

        for j in range(self.num_partitions):
            self.test_cores_a1[j] = np.loadtxt(location+"test_cores_a1_"+str(j)+".gz")
            for k in range(self.num_bonds[j]):
                self.test_full_sv_a1[j].append(np.loadtxt(location+"test_full_sv_a1_"+str(j)+"_"+str(k)+".gz"))
                self.test_ret_sv_a1[j].append(np.loadtxt(location+"test_ret_sv_a1_"+str(j)+"_"+str(k)+".gz"))

        # Load test benchmarks for core truncation test
        self.test_full_sv_b1 = [[] for j in range(self.num_partitions)]
        self.test_ret_sv_b1 = [[] for j in range(self.num_partitions)]
        self.test_cores_b1 = [[] for j in range(self.num_partitions)]

        for j in range(self.num_partitions):
            self.test_cores_b1[j] = np.loadtxt(location+"test_cores_b1_"+str(j)+".gz")
            for k in range(self.num_bonds[j]):
                self.test_full_sv_b1[j].append(np.loadtxt(location+"test_full_sv_b1_"+str(j)+"_"+str(k)+".gz"))
                self.test_ret_sv_b1[j].append(np.loadtxt(location+"test_ret_sv_b1_"+str(j)+"_"+str(k)+".gz"))

        # Use the package to calculate results for both the full and core truncation test
        self.full_sv_a1 = [[] for j in range(self.num_partitions)]
        self.ret_sv_a1 = [[] for j in range(self.num_partitions)]
        self.cores_a1 = [[] for j in range(self.num_partitions)]

        self.full_sv_b1 = [[] for j in range(self.num_partitions)]
        self.ret_sv_b1 = [[] for j in range(self.num_partitions)]
        self.cores_b1 = [[] for j in range(self.num_partitions)]

        for j in range(self.num_partitions):
            temp_labels = list(np.copy(self.labels_list[j]))
            data_tensor = tn.matrix_to_tensor(self.test_data, self.partition_list[j], labels=temp_labels)
            data_tensor.move_index("batchsize", self.batch_size_pos_list[j])

            l, r, core_a, self.full_sv_a1[j], self.ret_sv_a1[j] = mixed_canonical_full_ret_sv(
                data_tensor, self.bond_dimension, self.batch_size_pos_list[j])
            self.cores_a1[j] = tn.tensor_to_matrix(core_a, "c")

            l, r, core_b, self.full_sv_b1[j], self.ret_sv_b1[j] = mixed_canonical_full_core_truncation_only_ret_sv(
                data_tensor, self.bond_dimension, self.batch_size_pos_list[j])
            self.cores_b1[j] = tn.tensor_to_matrix(core_b, "c")

    def test_full_truncation(self):
        for j in range(self.num_partitions):
            npt.assert_allclose(self.cores_a1[j],
                                self.test_cores_a1[j],
                                err_msg="core mismatch at "+str(j)+" in mc full truncation test")
            for k in range(self.num_bonds[j]):

                # The full set of sv's may have extremely small values with low level sign problems
                npt.assert_allclose(self.full_sv_a1[j][k],
                                    self.test_full_sv_a1[j][k],
                                    atol=1e-10,
                                    err_msg="full sv mismatch at "+str(j)+" "+str(k)+"in mc full truncation test")
                npt.assert_allclose(self.ret_sv_a1[j][k],
                                    self.test_ret_sv_a1[j][k],
                                    atol=1e-10,
                                    err_msg="ret sv mismatch at "+str(j)+" "+str(k)+"in mc full truncation test")

    def test_core_truncation(self):
        for j in range(self.num_partitions):
            npt.assert_allclose(self.cores_b1[j],
                                self.test_cores_b1[j],
                                err_msg="core mismatch at "+str(j)+" in mc core truncation test")
            for k in range(self.num_bonds[j]):
                npt.assert_allclose(self.full_sv_b1[j][k],
                                    self.test_full_sv_b1[j][k],
                                    atol=1e-10,
                                    err_msg="full sv mismatch at "+str(j)+" "+str(k)+"in mc core truncation test")
                npt.assert_allclose(self.ret_sv_b1[j][k],
                                    self.test_ret_sv_b1[j][k],
                                    atol=1e-10,
                                    err_msg="ret sv mismatch at "+str(j)+" "+str(k)+"in mc core truncation test")


class TestFeatureExtraction(TestCase):
    def setUp(self):

        # First we set and load common variables for all test cases
        location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) + "/unit_test_data/"
        self.test_data = np.loadtxt(location + "test_data.gz")

        self.partition = [2, 2, 7, 7, 2, 2]
        self.batch_size_position = 3
        self.bond_dimension = 3

        # Load test benchmarks for all tests
        self.test_features_a2 = np.loadtxt(location+"test_features_a2.gz")
        self.test_features_b2 = np.loadtxt(location + "test_features_b2.gz")
        self.test_features_c2 = np.loadtxt(location + "test_features_c2.gz")
        self.test_features_d2 = np.loadtxt(location + "test_features_d2.gz")

        # Calculate results via package for comparison
        self.features_a2 = core_compression(self.test_data, self.bond_dimension)
        self.features_b2 = core_compression_with_partition(self.test_data,
                                                           self.bond_dimension,
                                                           self.partition,
                                                           self.batch_size_position)
        self.features_c2 = core_compression_core_truncation_only(self.test_data, self.bond_dimension)
        self.features_d2 = core_compression_core_truncation_only_with_partition(self.test_data,
                                                                                self.bond_dimension,
                                                                                self.partition,
                                                                                self.batch_size_position)

    def test_core_compression(self):
        npt.assert_allclose(self.features_a2,
                            self.test_features_a2,
                            err_msg="feature set mismatch via core compression method")

    def test_core_compression_with_partition(self):
        npt.assert_allclose(self.features_b2,
                            self.test_features_b2,
                            err_msg="feature set mismatch via core compression with partition method")

    def test_core_compression_core_truncation_only(self):
        npt.assert_allclose(self.features_c2,
                            self.test_features_c2,
                            err_msg="feature set mismatch via core compression (core truncation only) method")

    def test_core_compression_core_truncation_only_with_partition(self):
        npt.assert_allclose(self.features_d2,
                            self.test_features_d2,
                            err_msg="feature set mismatch via core compression (ct only) with partition method")


class TestMultiComponentAnalysisTools(TestCase):
    def setUp(self):

        # First we set and load common variables for all test cases
        location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) + "/unit_test_data/"
        self.test_data = np.loadtxt(location + "test_data.gz")

        self.partition_list = [[2000, 28, 28],
                               [2000, 7, 16, 7],
                               [2000, 2, 2, 7, 7, 2, 2],
                               [2000, 2, 2, 7, 7, 2, 2],
                               [2000, 2, 2, 7, 7, 2, 2]]

        self.labels_list = [["batchsize", "1", "2"],
                            ["batchsize", "1", "2", "3"],
                            ["batchsize", "1", "2", "3", "4", "5", "6"],
                            ["batchsize", "1", "2", "3", "4", "5", "6"],
                            ["batchsize", "1", "2", "3", "4", "5", "6"]]
        self.num_partitions = 5
        self.bond_dimension = 3
        self.batch_size_pos_list = [1, 1, 3, 1, 5]

        # Load test benchmarks
        self.test_extracted_features_a3 = [[] for j in range(self.num_partitions)]

        for j in range(self.num_partitions):
            self.test_extracted_features_a3[j] = np.loadtxt(location+"test_extracted_features_a3_"+str(j)+".gz")

        # Use the package to calculate results for both the full and core truncation test
        self.extracted_features_a3 = [[] for j in range(self.num_partitions)]

        for j in range(self.num_partitions):
            temp_labels = list(np.copy(self.labels_list[j]))
            data_tensor = tn.matrix_to_tensor(self.test_data, self.partition_list[j], labels=temp_labels)
            data_tensor.move_index("batchsize", self.batch_size_pos_list[j])

            l, r, c = mixed_canonical_full_core_truncation_only(data_tensor,
                                                                self.bond_dimension,
                                                                self.batch_size_pos_list[j])
            self.extracted_features_a3[j] = extract_core_tensor_via_common_features_from_matrix(self.test_data, l, r)

    def test_mca(self):
        for j in range(self.num_partitions):
            npt.assert_allclose(self.extracted_features_a3[j][j],
                                self.test_extracted_features_a3[j][j],
                                err_msg="feature mismatch at "+str(j)+" in mca test")

if __name__ == '__main__':
    unittest.main()

