# Manuscript sources

`main.tex` is the manuscript; sections are in `sections/`, tables in `tables/`, figures in `figures/`.
Every number in the text is a macro defined in `numbers.tex`, and the tables are generated, both by
`build_numbers.py` from the repository results:

    python build_numbers.py --repo ..
    latexmk -pdf main.tex

The document uses the standard `article` class with Times fonts. To submit to a Springer Nature
journal, copy the contents of `main.tex` between `\begin{document}` and `\end{document}` into the
Springer Nature LaTeX template (`sn-jnl.cls`, available on Overleaf) and keep `\input{numbers}`,
`sections/`, `tables/`, `figures/` and `references.bib` unchanged.
