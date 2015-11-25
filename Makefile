all: clean pylint test

clean: clean.python clean.latex

clean.python:
	find . -name '*.pyc' -delete

clean.latex:
	cd report && latexmk -bibtex -c

pylint: pylint.nilm pylint.test

pylint.nilm:
	pylint2 --rcfile=.pylintrc nilm/

pylint.test:
	pylint2 --rcfile=.pylintrc test/

test: FORCE
	nosetests2 test/

FORCE:

report.pdf:
	cd report && \
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" report
