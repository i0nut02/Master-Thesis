.PHONY: report clean

report: src/main.tex
	cd src && latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=thesis main.tex

clean:
	cd src && latexmk -C -jobname=thesis main.tex
