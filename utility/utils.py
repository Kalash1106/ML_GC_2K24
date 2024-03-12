import numpy as np
from utility.data_gen import simplify_ground_truth


def get_weigths(DataClass):
    labels = np.array(
        [
            l["class_id"]
            for l in simplify_ground_truth(DataClass.gt_file, DataClass.mapping_file)
        ]
    )
    # Find unique elements and their frequencies
    unique_elements, counts = np.unique(labels, return_counts=True)

    # Calculate inverse frequencies
    inverse_frequencies = sum(counts) / counts
    print(unique_elements, inverse_frequencies)
    return inverse_frequencies
