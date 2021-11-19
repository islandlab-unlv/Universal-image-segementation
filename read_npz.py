"""Script to read npz data"""

import numpy as np

def npz2dict(file_location):
    """Converts data stored in an npz zip file into
    a dictionary."""
    dict = {}
    with open(file_location, 'rb') as f:
        file_zip = np.load(f)
        for key in file_zip:
            dict[key] = file_zip[key]
    return dict
