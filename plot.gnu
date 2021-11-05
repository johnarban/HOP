# usage gnuplot -c ../plot.gnu sulafat.Stxt sulafat.tsv  > sulafat_compare.pdf ; open sulafat_compare.pdf
set terminal pdfcairo
unset errorbars
set style data lines

script_name=ARG0
sbig=ARG1
mine=ARG2

stats sbig using 2:3 name "B"
stats mine using 1:2 name "A"

plot mine using 1:2:3 with yerrorlines ls 3 pt -1 notitle, \
    mine using 1:2:3 with lines ls 22 t "Mine", \
    sbig using 2:($3/B_sum_y*A_sum_y) with lines ls 4 t "SBIG", \