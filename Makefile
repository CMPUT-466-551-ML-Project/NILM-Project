clean:
	find . -name '*.pyc' -delete

pylint:
	pylint2 --rcfile=.pylintrc nilm/
