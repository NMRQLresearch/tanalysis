import numpy as np
import tncontract as tn
from tanalysis import decompositions as dc
from tanalysis import partitioning as prt

# Create a data array

data = np.random.normal(size = [30,64])

# Choose the maximum bond-dimension

max_bond_dimension = 3

# Tensorize the data array via the symmetric raw product partition.
# Place the batch size index in the centre

batch_size = np.shape(data)[0]
pre_partition = prt.symmetrize(prt.raw_partition(np.shape(data)[1]))
partition = [batch_size]
partition.extend(pre_partition)

tensor_labels = ["batchsize"]
tensor_labels.extend([str(j+1) for j in range(np.size(pre_partition))])

num_cores = np.size(partition)
batch_size_position = int(round((num_cores-1)/2))

data_partitioned = tn.matrix_to_tensor(data,partition, labels = tensor_labels)
data_partitioned.move_index("batchsize", batch_size_position)

# Perform the full mixed canonical decomposition (with diagnostic tools)

left,right,core_1,threshold_ratios_1,original_bonds_1,bond_ratios_1 = dc.mixed_canonical_full_withdiagnostics(data_partitioned,max_bond_dimension,batch_size_position)
compressed_data_1 = tn.tensor_to_matrix(core_1,"c")

# Perform the core only mixed canonical decomposition (with diagnostic tools)

core_2,threshold_ratios_2,original_bonds_2,bond_ratios_2 = dc.mixed_canonical_core_only_withdiagnostics(data_partitioned,max_bond_dimension,batch_size_position)
compressed_data_2 = tn.tensor_to_matrix(core_2,"c")

# Perform the core only mixed canonical decomposition (without diagnostic tools)

core_3 = dc.mixed_canonical_core_only_nodiagnostics(data_partitioned,max_bond_dimension,batch_size_position)
compressed_data_3 = tn.tensor_to_matrix(core_3,"c")

# Perform the automated compression via symmetrized raw partitioning

compressed_data_4 = dc.core_compression(data,max_bond_dimension)

# We can verify that these all provide the same reduced data array

dist = np.linalg.norm(compressed_data_1 - compressed_data_1,'fro')
dist2 = np.linalg.norm(compressed_data_1 - compressed_data_2,'fro')
dist3 = np.linalg.norm(compressed_data_1 - compressed_data_3,'fro')

print dist
print dist2
print dist3

print np.shape(compressed_data_1)
