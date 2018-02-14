using Koala
using KoalaTrees
using Base.Test

const X, y = load_ames();
const train, test = splitrows(1:length(y), 0.8); # 80:20 split
rgs = TreeRegressor(penalty=0.5)
mach = SupervisedMachine(rgs, X, y, train)
fit!(mach, train)
showall(mach)
score = err(mach, test)
println("error = $score")
@test score > 4e4 && score < 5e4
