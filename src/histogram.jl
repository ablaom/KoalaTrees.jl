"""
    histogram(bag, x, log2_nbins)

Assumes `bag` is an `AbstractVector` of integers such that `x[i]` is
`Real` for all `i` in `bag`, and sorts `bag` into `n` subvectors
`bin[1], bin[2], ..., bin[n]` based on the values of `x`, using a
uniform subdivision of the interval `(minimum(x[bag]),
maximum(x[bag])` (whose boundaries are placed in a vector called
`boundaries` below). Here `n=2^log2_nbins`. The bins are
"left-closed", except the last which is closed at both ends.

### Return value

The normal return value is

`(bins, boundaries)`

where `bins = [bin[k] for k in 1:n]`. However, in the special case
that `x[bag]` takes on a *single* value `x0`, `bins` is an unassigned
vector of length one, and boundaries is replaced with `[x0]`.

, [x0]`.

"""
function histogram(bag::AbstractVector{Int},
                   x::AbstractVector{T} where T<:Real,
                   log2_nbins)
    nbins = round(Int, 2^log2_nbins)
    nbins > 1 || throw(DomainError)
    boundaries = range(minimum(x[bag]),
                       stop=maximum(x[bag]),
                       length=nbins + 1) |> collect

    if boundaries[1] == boundaries[end]
        nbins = 1
    end

    # create bins
    bin = Vector{Vector{Int}}(undef, nbins)

    # return unassigned bins if x takes constant value:
    if nbins == 1
        return bin, boundaries[1:1]
    end

    # initialize bins as empty:
    for k in 1:nbins
        bin[k] = Int[]
    end

    # For each `i` in `bag` we add `i` to vector `bin[k]` if `x[i]`
    # is between `boundaries[k]` and `boundaries[k+1]` (left included,
    # right excluded unless `k=n_bins`). We sort using binary
    # search. In particular, the bin number `k` is represented as a
    # sum of powers of 2 in the inner loop below.

    for i in bag
        k = 1 # initialize bin number
        for power in reverse(0:(log2_nbins - 1))
            δk = round(Int, 2^power)
            if x[i] >= boundaries[k + δk]
                k += δk
            end
        end
        push!(bin[k], i)
    end

    return bin, boundaries

end
