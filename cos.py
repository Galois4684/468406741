import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cos(v1, v2):
    v1 = numpy.array(v1)
    v2 = numpy.array(v2)
    return np.dot(v1, v2)/(np.dot(v1,v1)*np.dot(v2, v2))**(1/2)

path = r"C:\Users\26271\Desktop\sample_vector.xlsx"
data = pd.read_excel(path)
sample_vector_space = []

for i in range(359):
    sample_vector_space.append(data.iloc[i].tolist())
sum_sample_vector = [0, 0, 0, 0, 0, 0, 0]
for item in sample_vector_space:
    sum_sample_vector = list(numpy.array(sum_sample_vector) + numpy.array(item))
ave_sample_vector = numpy.array(sum_sample_vector)/len(sample_vector_space)

delta_cos = [cos(ave_sample_vector, sample_vector_space[i]) for i in range(len(sample_vector_space))]
day = list(range(len(sample_vector_space)))

delta_cos = np.arccos(delta_cos)
plt.plot(range(len(delta_cos[::8])), delta_cos[::8])
plt.show()


