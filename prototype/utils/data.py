#!/usr/bin/env python3
import numpy as np


def mat2csv(output_file, data):
    np.savetxt(output_file, data, delimiter=",")
