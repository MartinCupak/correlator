#!/usr/bin/bash

path_src="../apps"
exe_src="correlator.cpp"

# generate plain text call graph
cflow ${path_src}/${exe_src} -o ${exe_src%.*}_callgraph.txt

# generate vector graphics (svg) call graph
cflow --format=dot ${path_src}/${exe_src} | dot -Grankdir=LR -Tsvg -o ${exe_src%.*}_callgraph.svg

# print svg into pdf file
inkscape --export-type=pdf --export-filename=${exe_src%.*}_callgraph.pdf --export-area-drawing ${exe_src%.*}_callgraph.svg
