using Koala
using KoalaTrees
using Base.Test

const X, y = load_ames();
all = eachindex(y) # iterator for all rows
const train, test = split(all, 0.8); # 80:20 split
rgs = TreeRegressor(penalty=0.5)
mach = Machine(rgs, X, y, train, features=names(X)[1:end-1])
fit!(mach, train)
showall(mach)
score = err(mach, test)
println("error = $score")
@test score > 4e4 && score < 5e4
score = mean(cv(mach, all))
println("error = $score")

