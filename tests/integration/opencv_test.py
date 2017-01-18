from integration_tests import EngineCase
from integration_tests.urls_helpers import single_dataset


class OpenCVTest(EngineCase):
    engine = 'opencv_engine'

    def test_single_params(self):
        single_dataset(self.retrieve, with_gif=False)
