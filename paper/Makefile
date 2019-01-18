#
#  Provides 'make all' to build and 'make clean'
#

FNAME=report

all: $(FNAME).pdf

$(FNAME).pdf: $(FNAME).tex
	rm -rf auto *.aux *.nav *.log *.lof *.lot *.out *.snm *.blg *.bbl *.toc *.vrb *.dvi *.spl *.ps *.xml *.upa *.upb *.acn *.glo  *.ist *.thm;
	pdflatex $(FNAME);
	pdflatex $(FNAME);
	bibtex $(FNAME);
	pdflatex $(FNAME);
	pdflatex $(FNAME);

clean:
	-rm -rf auto *.aux *.nav *.log *.lof *.lot *.out *.snm *.blg *.bbl *.toc *.vrb *.dvi *.spl *.ps *.xml *.upa *.upb *.acn *.glo  *.ist *.thm $(FNAME).pdf;
