using Koala
using Revise
using KoalaTrees
using Test

# test the contents of transformer.jl:
const X, y = load_ames();
t = KoalaTrees.FrameToTableauTransformer()
tM = Machine(t, X)
dt = transform(tM, X)

# test the histogram maker:
@test KoalaTrees.histogram(21:41, 11:1000, 3)[1] == 
[[21, 22, 23],
 [24, 25],    
 [26, 27, 28],
 [29, 30],    
 [31, 32, 33],
 [34, 35],    
 [36, 37, 38],
 [39, 40, 41]]

# test clean
TreeRegressor(regularization=1.0)
TreeRegressor(min_patterns_split=1)
TreeRegressor(max_bin=1)


const all = eachindex(y) # iterator for all rows

const train, test = partition(all, 0.8); # 80:20 split

const rgs = TreeRegressor(penalty=0.5)

const mach = Machine(rgs, X, y, train, features=names(X)[1:end-1])
fit!(mach, train)
showall(mach)
score = err(mach, test)
println("error = $score")
@test score > 4e4 && score < 5e4
score = mean(cv(mach, all))
println("error = $score")

rgs.extreme = true
fit!(mach)
@test err(mach, test) < 5e4

rgs.penalty = 0
rgs.extreme = false
rgs.max_bin = 16
rgs.bin_factor = 1
fit!(mach)
err(mach, test)
@test err(mach, test) < 5e4  



