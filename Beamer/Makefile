# Makefile for LaTeX
##################################################################
# Use:
# make
# make clean
# options for ps2pdf: http://pages.cs.wisc.edu/~ghost/doc/AFPL/6.50/Ps2pdf.htm

TEX := $(shell ls *.tex)
FOLDER = /home/saul/sauld@cimat.mx/UNISON/Ponencias/SIAM-ENSENADA/Beamer
OTHER = *~ *.aux *.dvi *.toc *.bbl *.blg *.out *.thm *.ps *.idx *.ilg *.ind *.tdo *.pdf *.tar.gz *.log *.spl *.synctex.gz
LATEX	= latex
BIBTEX	= bibtex
MAKEINDEX = makeindex
XDVI	= xdvi -gamma 4
DVIPS	= dvips
DVIPDF  = dvipdft
L2H	= latex2html
GH	= gv
FILES = *.tex *.sty *.bbl *.bib *.pdf Makefile  
FOLDERS = \Imagenes \images \assets 
NAMETAR1 := $(shell date '+%Y%b%d_%H_%M')
NAMETAR = "$(NAMETAR1)_Beamer_siam.tar.gz"
NAMEZIP = "$(NAMETAR1)_Beamer_siam.zip"
#main.aux
pdflatex:main.tex
	/usr/local/texlive/2019/bin/x86_64-linux/pdflatex --synctex=1 main.tex
	# /usr/local/texlive/2019/bin/x86_64-linux/bibtex first_day_main.aux
	./biblio.sh
	/usr/local/texlive/2019/bin/x86_64-linux/pdflatex --synctex=1 main.tex
	/usr/local/texlive/2019/bin/x86_64-linux/pdflatex --synctex=1 main.tex

clean:
	rm -f $(OTHER) 

tar: $(FILES)
	tar -cvf $(NAMETAR) $(FOLDER)

zip: $(FILES)
	zip -r $(NAMEZIP) $(FOLDER)
