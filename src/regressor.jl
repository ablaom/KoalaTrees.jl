## REGRESSION DECISION TREE
# With with nearest neighbor regularization, and penalty for
# introducing new features in node splitting criteria.

struct NodeData
    feature::Int
    kind::Int8   # 0: root, 1: ordinal, 2: categorical, 3: leaf
    r::Float64   # A *threshold*, *float repr. of integer subset*, or
                 # *prediction*, according to kind above
end

const RegressorNode = Node{NodeData}

function should_split_left(pattern, node::RegressorNode)
    j, r, kind = node.data.feature, node.data.r, node.data.kind
    if kind == 1     # ordinal
        return pattern[j] <= r
    elseif kind == 2 # categorical
        small = round(Small, pattern[j])
        return small in round(IntegerSet, r)
    else
        @error "Expecting an ordinal or categorical node here."
    end
end

mutable struct TreeRegressor <: Regressor{RegressorNode}
    max_features::Int
    min_patterns_split::Int
    penalty::Float64
    extreme::Bool
    regularization::Float64
    max_height::Int
    max_bin::Int
    bin_factor::Int
end

function clean!(model::TreeRegressor)
    messg = ""
    if model.min_patterns_split <= 1
        model.min_patterns_split = 2
        messg = messg * "`min_patterns_split` must be at least 2. Resetting to 2. "
    end
    if model.regularization >= 1.0
        model.regularization = 0.0
        messg = messg * "`regularization` must be less than 1. Resetting to 0. "
    end
    if model.max_bin == 1
        model.max_bin = 2
        messg = messg * "`max_bin` must be 0 (exact splits) " *
              "or more than 1. Resetting to 2. "
    end
    return messg
end

function TreeRegressor(;max_features=0, min_patterns_split=2,
                       penalty=0.0, extreme=false,
                       regularization=0.0, max_height=1000, max_bin=0, bin_factor=90)
    model = TreeRegressor(max_features, min_patterns_split,
                          penalty,extreme,
                          regularization, max_height, max_bin, bin_factor)
    softwarn(clean!(model)) # only issues warning if `clean!` changes `model`
    return model
end

function feature_importance_curve(popularity_given_feature, names)

    importance_given_name = Dict{Symbol, Float64}()
    kys = keys(popularity_given_feature)
    N = sum(popularity_given_feature[j] for j in kys)
    for j in kys
        importance_given_name[names[j]] = popularity_given_feature[j]/N
    end
    x = Symbol[]
    y = Float64[]
    for name in reverse(keys_ordered_by_values(importance_given_name))
        push!(x, name)
        push!(y, round(Int, 1000*importance_given_name[name])/1000)
    end
    return x, y
end

# transformers:
default_transformer_X(model::TreeRegressor) = TableToTableauTransformer()
default_transformer_y(model::TreeRegressor) =
    RegressionTargetTransformer(standardize=false)


#####################################################################
# For readability we use `X` and `y` for `Xt` and `yt` from now on. #
#####################################################################

mutable struct Cache

    # independent of model parameters:
    X::DataTableau
    y::Vector{Float64}
    n_patterns::Int
    n_features::Int

    # dependent on model parameters:
    max_features::Int
    log2_max_bin::Int
    max_bin::Int # true value to be used (a power of 2)
    popularity::Vector{Float64} # representing popularity of each feature

    Cache(X, y, n_patterns, n_features) = new(X, y, n_patterns, n_features)
end

function setup(rgs::TreeRegressor, X, y, scheme_X, parallel, verbosity)

    n_features = size(X, 2) # = length(scheme_X.encoding.names)
    n_patterns = length(y)

    return Cache(X, y, n_patterns, n_features)

end

"""
Following function is for splitting on feature with index `j` when
the feature is *categorical*.

Returns `(gain, left_values)` where gain is the lowering of the error
(sum of square deviations) obtained for the best split based on sample
inputs `cache.X[j]` with corresponding target values `cache.y`;
`left_values` is a floating point representation of the set of values
of x for splitting left in the optimal split. If the feature is
constant (takes on a single value) then no split can improve the error
and `gain=0`.

Note that only inputs with indices in the iterator `bag` are
considered.

"""
function split_on_categorical(rgs, j, bag, no_split_error, cache)

    if rgs.extreme

        vals2 = unique(Small[round(Small, cache.X.raw[i,j]) for i in bag])
        n_vals = length(vals2)
        if n_vals == 1
            return 0.0, 0.0 # cache.X.raw[j] is constant in this bag, so no gain
        end
        n_select = rand(1:(n_vals - 1))
        left_selection = sample(vals2, n_select; replace=false)
        left_values_encoded = IntegerSet(left_selection)
        left_count = 0
        right_count = 0
        left_mean = 0.0
        right_mean = 0.0
        for i in bag
            v = round(Small, cache.X.raw[i,j])
            if v in left_selection
                left_count += 1
                left_mean += cache.y[i]
            else
                right_count += 1
                right_mean += cache.y[i]
            end
        end
        left_mean = left_mean/left_count
        right_mean = right_mean/right_count

        # Calcluate the split error, denoted `err` (the sum of squares of
        # deviations of target values from the relevant mean - namely
        # left or right
        err = 0.0
        for i in bag
            v = round(Small, cache.X.raw[i,j])
            if v in left_selection
                err += (left_mean - cache.y[i])^2
            else
                err += (right_mean - cache.y[i])^2
            end
        end
        gain = no_split_error - err
        left_values = Float64(IntegerSet(left_selection))
        return gain, left_values

    end

    # Non-extreme case:

    # 1. Determine a Vector{Small} of values taken by `cache.raw.X[j]`,
    # called `values` and order it according to the mean values of
    # cache.y:

    count = Dict{Small,Int}()   # counts of `cache.X[j]` values, keyed on
    # values of `cache.X[j]`.
    mu = Dict{Small,Float64}() # mean values of target `cache.y` keyed on
    # values of `cache.X[j]`, initially just the
    # unnormalized sum of `cache.y` values.
    for i in bag
        value = round(Small, cache.X.raw[i,j])
        if !haskey(count, value)
            count[value] = 1
            mu[value] = cache.y[i]
        else
            count[value] += 1
            mu[value] += cache.y[i]
        end
    end
    if length(count) == 1          # feature is constant for data in bag
        return 0.0, 0.0            # so no point in splitting;
        # second return value is irrelevant
    end

    # normalize sums to get the means:
    for v in keys(count)
        mu[v] = mu[v]/count[v]
    end
    vals = keys_ordered_by_values(mu) # vals is a Vector{Small}

    # 2. Let the "kth split" correspond to the left-values vals[1],
    # vals[2], ... vals[k] (so that the last split is no
    # split).

    n_vals = length(vals)

    # do first split outside loop handling others:
    left_count = 0
    right_count = 0
    left_mean = 0.0
    right_mean = 0.0
    for i in bag
        v = round(Small, cache.X.raw[i,j])
        if v == vals[1]
            left_count += 1
            left_mean += cache.y[i]
        else
            right_count += 1
            right_mean += cache.y[i]
        end
    end
    left_mean = left_mean/left_count
    right_mean = right_mean/right_count
    left_ss = 0.0
    right_ss = 0.0
    for i in bag
        v = round(Small, cache.X.raw[i,j])
        if v == vals[1]
            left_ss += (left_mean - cache.y[i])^2
        else
            right_ss += (right_mean - cache.y[i])^2
        end
    end

    error = left_ss + right_ss
    position = 1 # updated if error is improved in loop below

    for k in 2:(n_vals - 1)

        # Update the means and sum-of-square deviations:
        for i in bag
            if round(Small, cache.X.raw[i,j]) == vals[k]
                left_mean, left_ss = mean_and_ss_after_add(left_mean, left_ss, left_count, cache.y[i])
                left_count += 1
                right_mean, right_ss = mean_and_ss_after_omit(right_mean, right_ss, right_count, cache.y[i])
                right_count += -1
            end
        end

        # Calcluate the kth split error:
        err = left_ss + right_ss

        if err < error
            error = err
            position = k
        end
    end

    gain = no_split_error - error

    # println("pos error ",position," ", err)
    left_values = Float64(IntegerSet(vals[1:position]))
    return gain, left_values

end # of function `split_on_categorical`

"""
For splitting on feature with index `j` when this is an *ordinal*
feature.

Returns (gain, threshold) where `gain` is the lowering of the error
(sum of square deviations) obtained for the best split based on sample
inputs `cache.X[j]` with corresponding target values 'cache.y`;
`threshold` is the maximum value the feature for a left split in the
best case. If the feature is constant (takes on a single value) then
no split can improve the error and `gain = 0.0`.

Note that only inputs with indices in the iterator `bag` are
considered.

"""
function split_on_ordinal(rgs, j, bag, no_split_error, cache)

    if rgs.extreme
        # determine a Vector{Float64} of values taken by cache.X[j], called `vals` below:
        vals = unique(cache.X.raw[bag,j])
        min_val = minimum(vals)
        max_val = maximum(vals)
        if min_val == max_val      # feature is constant for data in bag
            return 0.0, 0.0        # so no point in splitting; second
            # value irrelevant
        end
        threshold = min_val + rand()*(max_val - min_val)

        # Calculate the left and right mean values of target
        left_count = 0
        right_count = 0
        left_mean = 0.0
        right_mean = 0.0
        for i in bag
            v = cache.X.raw[i,j]
            if v <= threshold
                left_count += 1
                left_mean += cache.y[i]
            else
                right_count += 1
                right_mean += cache.y[i]
            end
        end
        left_mean = left_mean/left_count
        right_mean = right_mean/right_count

        # Calcluate the split error, denoted `err` (the sum of
        # squares of deviations of target values from the relevant
        # mean - namely left or right)
        err = 0.0
        for i in bag
            v = cache.X.raw[i,j]
            if v <= threshold
                err += (left_mean - cache.y[i])^2
            else
                err += (right_mean - cache.y[i])^2
            end
        end
        gain = no_split_error - err
        return  gain, threshold
    end

    # Non-extreme case:

    nbins = cache.max_bin

    if nbins > 0 && length(bag) > rgs.bin_factor*nbins

        # Use histogram splitting.

        # calcluate the histogram:

        bins, boundaries = histogram(bag, view(cache.X.raw, :, j), cache.log2_max_bin)

        # deal with special case that feature takes on constant value:
        if length(boundaries) == 1
            return 0.0, 0.0 # no point in splitting
        end

        # We consider splits at each boundary point except the first
        # and last. We do the first split by hand and use recursive
        # formulas to update means and sum-of-square deviations for
        # the others (for speed enhancement)

        # mean and ss (sum-of-square deviations) for first split (k = 1):
        left_targets = cache.y[bins[1]]
        left_mean = mean(left_targets)
        left_count = length(left_targets)
        left_ss = sum((η - left_mean)^2 for η in left_targets)
        right_targets = cache.y[vcat((bins[k] for k in 2:nbins)...)]
        right_mean = mean(right_targets)
        right_count = length(right_targets)
        right_ss = sum((η - right_mean)^2 for η in right_targets)

        # error for first split:
        error = left_ss + right_ss
        best_k = 1 # to be updated if better split found

        for k in 2:(nbins - 1)

            # update the means and ss values recursively:
            for i in bins[k]
                left_mean, left_ss =
                    mean_and_ss_after_add(left_mean, left_ss, left_count, cache.y[i])
                left_count += 1
                right_mean, right_ss =
                    mean_and_ss_after_omit(right_mean, right_ss, right_count, cache.y[i])
                right_count += -1
            end

            # Calcluate the kth split error:
            err = left_ss + right_ss

            # Note value and split position if there is improvement
            if err < error
                error = err
                best_k = k
            end
        end

        gain = no_split_error - error
        threshold = boundaries[best_k + 1]

        return gain, threshold

    else

        # Use exact splitting.

        # 1. Determine a Vector{Float64} of values taken by cache.X[j],
        # called `vals` below:
        vals = unique(cache.X.raw[bag,j])
        n_vals = length(vals)
        if n_vals == 1             # feature is constant for data in bag
            return 0.0, 0.0        # so no point in splitting; second
                                   # value irrelevant
        end
        sort!(vals)

        # 2. Let the "jth split" correspond to threshold = vals[j], (so
        # that the last split is no split). The error for the jth split
        # will be error[j]. We calculate these errors now:


        # println("len = $(length(vals))")

        # we do the first split outside of the loop that considering the
        # others because we will use recursive formulas to update
        # means and sum-of-square deviations (for speed enhancement)

        # mean and ss (sum-of-square deviations) for first split:
        left_mean = 0.0
        left_count = 0
        right_mean = 0.0
        right_count = 0
        for i in bag
            if cache.X.raw[i,j] == vals[1]
                left_mean += cache.y[i]
                left_count += 1
            else
                right_mean += cache.y[i]
                right_count += 1
            end
        end
        left_mean = left_mean/left_count
        right_mean = right_mean/right_count
        left_ss = 0.0
        right_ss = 0.0
        for i in bag
            if cache.X.raw[i,j] == vals[1]
                left_ss += (cache.y[i] - left_mean)^2
            else
                right_ss += (cache.y[i] - right_mean)^2
            end
        end

        # error for first split:
        error = left_ss + right_ss
        position = 1 # its position, to be updated if better split found

        for k in 2:(n_vals - 1)

            # Update the means and sum-of-square deviations:
            for i in bag # Todo: really need to go through entire bag every time?
                x = cache.X.raw[i,j]
                if x == vals[k] # (ie, x > vals[k-1]) && (x < vals[k])
                    left_mean, left_ss =
                        mean_and_ss_after_add(left_mean, left_ss, left_count, cache.y[i])
                    left_count += 1
                    right_mean, right_ss =
                        mean_and_ss_after_omit(right_mean, right_ss, right_count, cache.y[i])
                    right_count += -1
                end
            end

            # Calcluate the kth split error:
            err = left_ss + right_ss

            # Note value and split position if there is improvement
            if err < error
                error = err
                position = k
            end
        end

        gain = no_split_error - error

        threshold = 0.5*(vals[position] + vals[position + 1])
        return gain, threshold

    end

end # of function `split_on_ordinal`

"""
The following returns `split_failed`, `bag_left`, `bag_right` as
follows:

Computes the error for the best split on each feature within a random
sample of features of size `max_features` (set in `TreeRegressor()`)
and, if splitting improves the error (postive `gain`), create a new
splitting node of appropriate type and connect to `parent`. In
evaluating what is the *best* feature, those features not previously
selected have their gains penalized by muliplying by `rgs.penalty`.

`split_failed` is true if no new node is created.

`bag_left` and `bag_right` are splits of `bag` based on the optimum
splitting found, or are empty if `split_failed`.

Note that splitting is based only on patterns with indices in `bag`.

"""
function attempt_split(rgs, bag, F, parent, gender, no_split_error, cache)

    n_features, max_features = cache.n_features, cache.max_features

    if max_features == n_features
        feature_sample_indices = collect(1:n_features)
    else
        feature_sample_indices = sample(1:n_features, max_features; replace=false)
    end

    max_gain = -Inf # max gain so far
    opt_index = 0 # index of `feature_sample_indices`
    # delivering feature index for max_gain
    # (mythical at present)
    opt_crit = 0.0       # criterion delivering that max_gain

    # println("sample indices = $feature_sample_indices")

    for i in 1:max_features
        j = feature_sample_indices[i]

        if cache.X.is_ordinal[j]
            gain, crit = split_on_ordinal(rgs, j, bag, no_split_error, cache)
        else
            gain, crit = split_on_categorical(rgs, j, bag, no_split_error, cache)
        end

        if !(j in F) && rgs.penalty != 0.0
            gain = (1 - rgs.penalty)*gain
        end

        if gain > max_gain
            max_gain = gain
            opt_crit = crit
            opt_index = i
        end
    end

    # If no gain, return to calling function with `split_failed=true`:
    if max_gain == 0.0
        yvals = [cache.y[i] for i in bag]
        return true, Int[], Int[]
    end

    # Otherwise, create a new node with a splitting criterion based on
    # the optimal feature and unite with `parent`:
    j = feature_sample_indices[opt_index] # feature with minimum error

    if cache.X.is_ordinal[j]
        data = NodeData(j, 1, opt_crit)
    else
        data = NodeData(j, 2, opt_crit)
    end
    baby = RegressorNode(data) # new decision node
    unite!(baby, parent, gender)

    # Update the set of unpenalised features and the feature popularity vector:
    push!(F,j)
    cache.popularity[j] += length(bag)

    # Split `bag` accordingly:
    bag_left  = Int[]
    bag_right = Int[]
    for i in bag
        if should_split_left(cache.X[i,:], baby)
            push!(bag_left, i)
        else
            push!(bag_right, i)
        end
    end

    # Return the bag splits with decleration `split_failed=false`:
    return false, bag_left, bag_right

end # of function `attempt_split`

"""
Recursive function to grow the decision tree. Has no return value
but generally alters 2nd and 3rd arguments.

"""
function grow(rgs,
              bag,                   # Patterns to be considerd for splitting
              F,                     # set of penalty feature indices
              parent::RegressorNode, # Node to which any child will be
                                     # connected in successful call to
                                     # `attempt_split` above.
              gender,                # Determines how any child will
                                     # be connected to parent (as
                                     # left, right, or if this is the
                                     # first call to grow,
                                     # androgynous)
              cache)

    n_patterns = length(bag)

    # Compute mean of targets in bag:
    target_mean = 0.0
    for i in bag
        target_mean += cache.y[i]
    end
    target_mean = target_mean/n_patterns

    # Do not split node if insufficient samples, but create and
    # connect leaf node with above target_mean as prediction:
    if n_patterns < rgs.min_patterns_split
        leaf = RegressorNode(NodeData(0, 3, target_mean))
        unite!(leaf, parent, gender)
        return nothing
    end

    # Find sum of square deviations for targets in bag:
    no_split_error = 0.0
    for i in bag
        no_split_error += (cache.y[i] - target_mean)^2
    end

    # If the following is succesful, it creates a child and connects
    # it to `parent`.  In that case it also returns new bags according
    # to the optimal splitting criterion computed there (empty bags
    # otherwise) and updates `F` and `popularity` fields of `cache`:
    split_failed, bag_left, bag_right = attempt_split(rgs, bag, F, parent, gender,
                                                      no_split_error, cache)

    # If split makes no (significant) difference, then create and
    # connect prediction node:
    if split_failed
        leaf = RegressorNode(NodeData(0, 3, target_mean))
        unite!(leaf, parent, gender)
        # println("Leaf born, n_samples = $n_patterns, target prediction = $target_mean")
        return nothing
    end

    # Otherwise continue growing branches left and right:
    baby = child(parent, gender)
    F_left =  copy(F)
    F_right = copy(F)
    grow(rgs, bag_left, F_left, baby, 1, cache)   # grow a left branch
    grow(rgs, bag_right, F_right, baby, 2, cache) # grow a right branch
    # F = union(F_left, F_right)

    return nothing
end

function fit(rgs::TreeRegressor, cache, add, parallel, verbosity)

    # Note: add, parallel are ignored in this method

    cache.max_features = (rgs.max_features == 0 ? cache.n_features : rgs.max_features)
    cache.log2_max_bin = rgs.max_bin != 0 ? round(Int, floor(log2(rgs.max_bin))) : 0
    cache.max_bin = rgs.max_bin !=0 ? 2^cache.log2_max_bin : 0
    cache.popularity = zeros(Float64, cache.n_features)

    # Create a root node to get started. Its unique child is the true
    # stump of the decision tree:
    root = RegressorNode(NodeData(0, 0, 0.0))

    F = Set{Int}() # set of feature indices not penalized (initially empty)

    grow(rgs, 1:cache.n_patterns, F, root, 0, cache) # 0 means `root` is to
                                              # be androgynous,
                                              # meaning `root.left =
                                              # root.right`.

    report = Dict{Symbol, Any}()
    popularity_given_feature = Dict{Int,Int}()
    for j in 1:cache.n_features
        pop = cache.popularity[j]
        if pop != 0.0
            popularity_given_feature[j] = pop
        end
    end

    features, importance = feature_importance_curve(popularity_given_feature,
                             cache.X.features)
    report[:feature_importance_curve] = (features, importance)
    title="Feature importance at penalty=$(rgs.penalty)"
    plt =  UnicodePlots.barplot(features, importance, title=title)
    report[:feature_importance_plot] = plt

    predictor = root.left

    return predictor, report, cache

end

function predict(node::RegressorNode, pattern::Vector)
    while !is_leaf(node)
        if should_split_left(pattern, node)
            node = node.left
        else
            node = node.right
        end
    end
    return node.data.r
end

function predict(rgs::TreeRegressor, predictor, X::DataTableau, parallel, verbosity)

    # Note: `parallel` and `verbosity` currently ignored

    if rgs.regularization == 0.0
        return [predict(predictor, X[i,:]) for i in 1:size(X,1)]
    else
        tree = predictor
        lambda = rgs.regularization

        ret = Array{Float64}(undef, size(X,1))
        k = 1 # counter for index of `ret` (different from bag index)
        for i in 1:size(X,1)

            pattern = X[i,:]

            # Pass pattern down tree from top (stump), recording branching on way
            branchings = Char[] # ['l','r','r', etc]
            node = tree
            while !is_leaf(node)
                if should_split_left(pattern, node)
                    push!(branchings, 'l')
                    node = node.left
                else
                    push!(branchings, 'r')
                    node = node.right
                end
            end
            depth = length(branchings)

            # Passing (partway) back up the tree, collect and sum predictions of
            # nearby leaves each weighted by "distance" away:
            prediction = node.data.r
            height = min(depth, rgs.max_height)
            for h in 1:height
                node = node.parent
                if branchings[depth + 1 - h] == 'l'
                    node = node.right
                else
                    node = node.left
                end
                prediction += (lambda^h)*predict(node, pattern)
                node = node.parent
            end

            # normalize the summed prediction:
            ret[k] = prediction*(1-lambda)/(1-lambda^(height+1))
            k += 1

        end
        return ret

    end

end
