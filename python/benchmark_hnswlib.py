import hnswlib
import numpy as np
import time

N = 10000
DIM = 64
K = 10

data = np.random.rand(N, DIM).astype(np.float32)

index = hnswlib.Index(space='l2', dim=DIM)
index.init_index(max_elements=N, ef_construction=200, M=16)

t1 = time.time()
index.add_items(data)
t2 = time.time()

print("Build time:", t2 - t1)

index.set_ef(200)

q = data[1234:1235]

t1 = time.time()
labels, distances = index.knn_query(q, k=K)
t2 = time.time()

print("Query time (us) : ", (t2 - t1) * 1e6)