all: clean pylint test

clean:
	find . -name '*.pyc' -delete

pylint: pylint.nilm pylint.test

pylint.nilm:
	pylint2 --rcfile=.pylintrc nilm/

pylint.test:
	pylint2 --rcfile=.pylintrc test/

test: FORCE
	nosetests2 test/

FORCE:
