from thumbor.integration_tests import EngineTestCase
from thumbor.integration_tests.urls_helpers import single_dataset


class GraphicsmagickTest(EngineTestCase):
    engine = 'opencv_engine'

    def test_single_params(self):
        single_dataset(self.retrieve, with_gif=False)
