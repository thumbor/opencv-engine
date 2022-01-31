from os.path import join

from integration_tests import EngineCase
from integration_tests.urls_helpers import UrlsTester, single_dataset
from tornado.testing import gen_test


class OpenCVTest(EngineCase):
    engine = "opencv_engine"

    @gen_test(timeout=60)
    async def test_single_params(self):
        if not self._app:
            return True
        group = list(single_dataset(False))  # FIXME: remove False
        count = len(group)
        tester = UrlsTester(self.http_client)

        print("Requests count: %d" % count)
        for options in group:
            joined_parts = join(*options)
            url = "unsafe/%s" % joined_parts
            await tester.try_url(self.get_url(f"/{url}"))

        tester.report()
