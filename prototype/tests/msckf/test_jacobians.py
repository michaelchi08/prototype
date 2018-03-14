import unittest
from prototype.msckf.jacobians import measurement_model


class JacobiansTest(unittest.TestCase):
    def test_measurement_model(self):
        measurement_model()
