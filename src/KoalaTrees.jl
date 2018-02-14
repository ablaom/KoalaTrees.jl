__precompile__()
module KoalaTrees

export TreeRegressor

# development only:
# import ADBUtilities: @dbg, @colon
# export Node, is_stump, is_leaf, has_left, has_right, make_leftchild!, make_rightchild!
# export is_left, is_right
# export unite!, child

import Koala: Regressor, SupervisedMachine
import Koala: params, keys_ordered_by_values
import DataTableaux
import DataTableaux:  DataTableau, FrameToTableauScheme, IntegerSet
import StatsBase: sample
import DataFrames: AbstractDataFrame
import UnicodePlots

# to be extended:
import Base: show, showall

# to be extended (but not explicitly rexported):
import Koala: get_scheme_X, get_scheme_y, transform, inverse_transform
import Koala: setup, fit, predict

# constants:
const Small = UInt8


## Helpers:
"""
# `function mean_and_ss_after_add(mean, ss, n, x)`

Returns the mean, and the sum-of-square deviations from the mean, of
`n+1` numbers, given the corresponding two quantities for the first
`n` numbers (the inputs `mean` and `ss`) and the value of the `n+1`th
number, `x`.
"""
function mean_and_ss_after_add(mean, ss, n, x)
    n<0 ? throw(DomainError) : nothing
    mean_new = (n*mean + x)/(n + 1)
    ss_new = ss + (x - mean_new)*(x - mean)
    return mean_new, ss_new
end

"""
# `function mean_and_ss_after_omit(mean, ss, n, x)`

Given `n` numbers, their mean `mean` and sum-of-square deviations from
the mean `ss`, this function returns the new mean and corresponding
sum-of-square deviations of the same numbers when one of the numbers,
`x` is omitted from the list.

"""
function mean_and_ss_after_omit(mean, ss, n, x)
    n <= 1 ? throw(DomainError) : nothing
    mean_new = (n*mean - x)/(n-1)
    ss_new = ss - (x - mean)*(x - mean_new)
    return mean_new, ss_new
end


## `Node` type - data structure for building binary trees

# Binary trees are identified with their top (stump) nodes, so only a
# `Node` type, with the appropriate possiblities for connection, is
# defined. Connections are established with the methods
# `make_leftchild!` and `make_rightchild!`. Nodes have a `depth`
# field; when a new connection is made, the child's depth is declared
# to be one more than the parent. When a node `N` is created it is
# initially its own parent (equivalently, `is_stump(N) = true`) and is
# its own left and right child; its depth is initially zero.

mutable struct Node{T}
    parent::Node{T}
    left::Node{T}
    right::Node{T}
    data::T
    depth::Int
    function Node{T}(datum) where T
        node = new{T}()
        node.parent = node
        node.left = node
        node.right = node
        node.data = datum
        node.depth = 0
        return node
    end
end

Node(data::T) where T = Node{T}(data)

# Testing connectivity:

is_stump(node) = node.parent == node
is_left(node) =  (node.parent != node) && (node.parent.left == node) 
is_right(node) = (node.parent != node) && (node.parent.right == node)
has_left(node) =  (node.left  != node)
has_right(node) = (node.right != node)
is_leaf(node) = node.left == node && node.right == node

# Connecting nodes:

""" 
`make_leftchild!(child, parent)` makes `child` the left child of
`parent` and returns the depth of `child`

"""
function make_leftchild!(child, parent)
    parent.left = child
    child.parent = parent
    child.depth = parent.depth + 1
end

""" 
`make_rightchild!(child, parent)` makes `child` the right child of
`parent` and returns the depth of `child`

"""
function make_rightchild!(child, parent)
    parent.right = child
    child.parent = parent
    child.depth = parent.depth + 1
end

# Locating children

"""
## `child(parent, gender)`

Returns the left child of `parent` of a `Node` object if `gender` is 1
and right child if `gender is 2. If `gender` is `0` the routine throws
an error if the left and right children are different and otherwise
returns their common value.  For all other values of gender an error
is thrown. 

"""
function child(parent, gender)
    if gender == 1
        return parent.left
    elseif gender == 2
        return parent.right
    elseif gender == 0
        if parent.left != parent.right
            throw(Base.error("Left and right children different."))
        else
            return parent.left
        end
    end
    throw(Base.error("Only genders 0, 1 or 2 allowed."))
end

"""
# `unite!(child, parent, gender)`

Makes `child` the `left` or `right` child of a `Node` object `parent`
in case `gender` is `1` or `2` respectively; and makes `parent` the
parent of `child`. For any other values of `gender` the routine makes
`child` simultaneously the left and right child of `parent`, and
`parent` the parent of `child`. Returns `nothing`.

"""
function unite!(child, parent, gender)
    if gender == 1
        make_leftchild!(child, parent)
    elseif gender == 2
        make_rightchild!(child, parent)
    else
        make_leftchild!(child, parent)
        make_rightchild!(child, parent)
    end
end
    

# Display functionality:

function spaces(n)
    s = ""
    for i in 1:n
        s = string(s, " ")
    end
    return s
end

function get_gender(node)
    if is_stump(node)
        return 'A' # androgenous
    elseif is_left(node)
        return 'L'
    else
        return 'R'
    end
end

tail(n) = "..."*string(n)[end-3:end]

function Base.show(stream::IO, node::Node)
    print(stream, "Node{$(typeof(node).parameters[1])}@$(tail(hash(node)))")
end

function Base.showall(stream::IO, node::Node)
    gap = spaces(node.depth + 1)
    println(stream, string(get_gender(node), gap, node.data))
    if has_left(node)
        showall(stream, node.left)
    end
    if has_right(node)
        showall(stream, node.right)
    end
    return
end

showall(node::Node)=showall(STDOUT, node)

# for testing purposes:
# Node(data) = Node{typeof(data)}(data)

function Node(data, parent::Node)
    child = Node(data)
    make_leftchild!(child, parent)
    return child
end

function Node(parent::Node, data)
    child = Node(data)
    make_rightchild!(child, parent)
    return child
end


## `TreeRegressor` - regression decision tree with nearest neighbor regularization

immutable NodeData
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
        throw(Base.error("Expecting an ordinal or categorical node here."))
    end
end

mutable struct TreeRegressor <: Regressor{RegressorNode}
    
    max_features::Int
    min_patterns_split::Int
    penalty::Float64
    extreme::Bool
    regularization::Float64
    max_height::Int 

    function TreeRegressor(max_features::Int, min_patterns_split::Int,
                           penalty::Float64, extreme::Bool, regularization,
                           max_height)
        min_patterns_split > 1 || error("min_patterns_split must be at least 2.")
        return new(max_features, min_patterns_split, penalty,
                   extreme, regularization, max_height)

    end

end

TreeRegressor(;max_features::Int=0, min_patterns_split::Int=2,
              penalty=0.0, extreme::Bool=false,
              regularization=0.0, max_height=1000) =
                  TreeRegressor(max_features, min_patterns_split, penalty,
                                extreme, regularization, max_height)

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

get_metadata(model::TreeRegressor, X::AbstractDataFrame, y, rows, features) =
    features

get_scheme_X(model::TreeRegressor, X, train_rows, features) =
    FrameToTableauScheme(X[train_rows,features])

function transform(model::TreeRegressor, frame_to_tableau_scheme, X::AbstractDataFrame)
    features = frame_to_tableau_scheme.encoding.names
    issubset(Set(features), Set(names(X))) ||
        error("DataFrame feature incompatibility encountered.")
    return DataTableaux.transform(frame_to_tableau_scheme, X[features])
end

get_scheme_y(model::TreeRegressor, y, train_rows) = nothing
transform(model::TreeRegressor, no_thing::Void, y::AbstractVector{T} where T<:Real) = y
inverse_transform(model::TreeRegressor, no_thing::Void, yt) = yt

#####################################################################
# For readability we use `X` and `y` for `Xt` and `yt` from now on. #
#####################################################################

mutable struct Cache
    X::DataTableau
    y::Vector{Float64}
    n_patterns::Int
    n_features::Int
    max_features::Int
    popularity::Vector{Float64} # representing popularity of each feature
end

function setup(rgs::TreeRegressor, X, y, metadata, parallel, verbosity)

    n_patterns = length(y)
    n_features = length(metadata)
    max_features = (rgs.max_features == 0 ? n_features : rgs.max_features)
    popularity = zeros(Float64, n_features)
    
    return Cache(X, y, n_patterns, n_features, max_features, popularity)

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
            return 0.0, 0.0 # cache.X[j] is constant in this bag, so no gain
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
    
    # 1. Determine a Vector{Small} of values taken by `cache.X[j]`,
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

    # 1. Determine a Vector{Float64} of values taken by cache.X[j], called `vals` below:
    vals = unique(cache.X.raw[bag,j])
    sort!(vals)
    n_vals = length(vals)
    if n_vals == 1     # feature is constant for data in bag
        return 0.0, 0.0        # so no point in splitting; second
        # value irrelevant
    end
    
    if rgs.extreme
        min_val = minimum(vals)
        max_val = maximum(vals)
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
        if cache.X.raw[i,j] <= vals[1]
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
        for i in bag
            x = cache.X.raw[i,j]
            if x == vals[k] # (x > vals[k-1]) && (x <vals[k])
                left_mean, left_ss = mean_and_ss_after_add(left_mean, left_ss, left_count, cache.y[i])
                left_count += 1
                right_mean, right_ss = mean_and_ss_after_omit(right_mean, right_ss, right_count, cache.y[i])
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
        
        if cache.X.encoding.is_ordinal[j]
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
    
    if cache.X.encoding.is_ordinal[j]
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

    # add, parallel are ignored in this method

    n_patterns, n_features = cache.n_patterns, cache.n_features    
    max_features = cache.max_features

    # Create a root node to get started. Its unique child is the true
    # stump of the decision tree:
    root = RegressorNode(NodeData(0, 0, 0.0)) 

    F = Set{Int}() # set of feature indices not penalized (initially empty)
    
    grow(rgs, 1:n_patterns, F, root, 0, cache) # 0 means `root` is to
                                              # be androgynous,
                                              # meaning `root.left =
                                              # root.right`.

    report = Dict{ Symbol, Tuple{Vector{Symbol},Vector{Float64}} }()
    popularity_given_feature = Dict{Int,Int}()
    for j in 1:n_features
        pop = cache.popularity[j]
        if pop != 0.0
            popularity_given_feature[j] = pop
        end
    end
    report[:feature_importance_curve] =
            feature_importance_curve(popularity_given_feature, cache.X.encoding.names)
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

        ret = Array{Float64}(size(X,1))
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

## Displaying machines with `TreeRegressor` models

function Base.showall(stream::IO, mach::SupervisedMachine{RegressorNode , TreeRegressor})
    show(stream, mach)
    println(stream)
    dict = params(mach)
    dict[:Xt] = string(typeof(mach.Xt), " of shape ", size(mach.Xt))
    dict[:yt] = string(typeof(mach.yt), " of shape ", size(mach.yt))
    dict[:metadata] = string("Object of type $(typeof(mach.metadata))")
    delete!(dict, :cache)
    if isdefined(mach,:report) && :feature_importance_curve in keys(mach.report)
        features, importance = mach.report[:feature_importance_curve]
        plt = UnicodePlots.barplot(features, importance,
              title="Feature importance at penalty=$(mach.model.penalty)")
    end
    delete!(dict, :report)
    showall(stream, dict)
    println(stream, "Model hyperparameters:")
    showall(stream, mach.model)
    show(stream, plt)
end


end # module
