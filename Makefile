test: unit integration

unit:
	@coverage run --branch `which nosetests` -vvvv --with-yanc -s tests/unit/
	@coverage report -m

coverage-html: unit
	@coverage html -d cover

integration:
	@`which nosetests` -vvvv --with-yanc -s tests/integration/

setup:
	@pip install -U -e .\[tests\]

run:
	@thumbor -c thumbor.conf -l debug
