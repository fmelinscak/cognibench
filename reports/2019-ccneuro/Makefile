paper = ccneuro2019-main-text
bibliography = ccneuro19-bibliography.bib
csl = apa.csl

.PHONY : all
all : $(paper).md ccn_template.latex
	pandoc -s -f markdown+smart --template=ccn_template.latex --filter=pandoc-fignos --natbib -o $(paper).tex $(paper).md
	latexmk -pdf -quiet $(paper).tex


## tex - render the paper TeX using Pandoc
.PHONY : tex
tex : $(paper).tex
$(paper).tex : $(paper).md ccn_template.latex
	pandoc -s -f markdown+smart --template=ccn_template.latex --filter=pandoc-fignos --natbib -o $(paper).tex $(paper).md


## pdf - render the paper in PDF using Pandoc (and latex)
.PHONY : pdf
pdf : $(paper).pdf
$(paper).pdf : $(paper).tex
	latexmk -pdf -quiet $(paper).tex


## word - render the paper in Microsoft Word using Pandoc
.PHONY : word
word : $(paper).docx
$(paper).docx : $(paper).md
	pandoc -s -f markdown+smart --filter=pandoc-fignos --bibliography=$(bibliography) --csl=$(csl) -o $(paper).docx $(paper).md

## clean - remove generated files
.PHONY : clean
clean :
	latexmk -c
	rm -f $(paper).tex
	rm -f $(paper).bbl

.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<