# Load some data and rows for the train/test sets:
using Koala
X, y = load_ames();
const train, test = splitrows(eachindex(y), 0.8); # 80:20 split

# Instantiate a tree model:
using KoalaTrees
t = TreeRegressor(regularization=0.5)
showall(t)

# Build and train a machine:
tree = SupervisedMachine(t, X, y, train)
fit!(tree, train)
showall(tree)

# Compute the RMS error on the test set:
err(tree, test)

# Tune the regularization parameter:
u, v = @curve r logspace(-3,2,100) begin
    t.regularization = r
    fit!(tree, train)
    err(tree, test)
end

t.regularization = u[indmin(v)]
fit!(tree, train)
err(tree, test)
