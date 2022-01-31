black:
	@black .

isort:
	@isort .

test: unit integration

unit:
	@-pytest -sv --cov=opencv_engine tests/unit/
	@-coverage report -m

coverage-html: unit
	@coverage html -d cover

integration:
	@pytest -sv tests/integration/

setup:
	@pip install -U -e .\[tests\]

run:
	@thumbor -c thumbor.conf -l debug
