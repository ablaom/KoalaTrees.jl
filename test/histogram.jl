using Random
using KoalaTrees

x = 0:10:300
nbins = 6
bag = shuffle(1:31)
log2_nbins = log(2, nbins)

bins, boundaries = KoalaTrees.histogram(bag, x, log2_nbins)
@test reduce(vcat, sort.(bins)) == 1:31
@test boundaries == 0:50:300

bins, boundaries = KoalaTrees.histogram(bag, fill(3.141, 31), log2_nbins)
@test length(bins) == 1
@test typeof(bins) == Vector{Vector{Int}}
@test boundaries == [3.141,]

@test KoalaTrees.histogram(21:41, 11:1000, 3)[1] ==
[[21, 22, 23],
 [24, 25],
 [26, 27, 28],
 [29, 30],
 [31, 32, 33],
 [34, 35],
 [36, 37, 38],
 [39, 40, 41]]

true
