## CONSTANTS

const Small=UInt8
const Big=UInt64
const SMALL_MAX = Small(52) 
const BIG_MAX = Big(2^(Int(SMALL_MAX) + 1) - 1)


## TYPE FOR SETS OF SMALL INTEGERS

"""
    mutable struct `IntegerSet`

A type of collection for storing subsets of {0, 1, 2, ... 52}. Every
such subset can be stored as an `Float64` object (although not all
`Float64`'s arise in this way). To convert an `IntegerSet` object `s` to
a floating point number, use `Float64(s)`. To recover the original
object from a float `f`, use `round(IntegerSet, f)`.

To instantiate an empty collection use, `IntegerSet()`. To add an
element `i::Integer` use `push!(s, i)` which is quickest if `i` is
type `Small=UInt8`. Membership is tested as usual with `in`. One can also
instantiate an object with multiple elements as in the following example:

    julia> 15 in IntegerSet([1, 24, 16])
    false

"""
mutable struct IntegerSet
    coded::Big
end

IntegerSet() = IntegerSet(Big(0))

function IntegerSet(v::AbstractVector{T} where T <: Integer)
    s = IntegerSet()
    for k in v
        push!(s, k) # push! defined below
    end
    return s
end

IntegerSet(x::Integer...) = IntegerSet([x...])

function Base.in(k::Small, s::IntegerSet)
    large = Big(2)^k
    return large & s.coded == large
end

function Base.in(k::Integer, s::IntegerSet)
    large = Big(2)^Small(k)
    return large & s.coded == large
end

function Base.push!(s::IntegerSet, k::Small)
    if k > SMALL_MAX
        @error "Cannot push! an integer larger "*
           "than $(Int(SMALL_MAX)) into an IntegerSet object."
    end
    if !(k in s)
        s.coded = s.coded | Big(2)^k
    end
    return s
end

function Base.push!(s::IntegerSet, k::Integer)
    if k > SMALL_MAX
        @error "Cannot push! an integer larger "*
           "than $(Int(SMALL_MAX)) into an IntegerSet object."
    end
    push!(s, Small(k))
end

isempty(s::IntegerSet) = s.coded == Big(0)

function Base.show(stream::IO, s::IntegerSet)
    # initiate string to represent s:
    str = "IntegerSet(" # 

    if !isempty(s)

        # get vector of all integers:
        integers = Small[]
        for i in 0:62
            if i in s
                push!(integers, i)
            end
        end

        # add them to representation:
        str *= string(integers[1])
        for i in integers[2:end]
            str *= ", $i"
        end
    end

    # finish up:
    str *= ")"
    print(stream, str)

end

Base.Float64(s::IntegerSet) = Float64(s.coded)

function Base.round(T::Type{IntegerSet}, f::Float64)
    if f < 0 || f > BIG_MAX
        @error "Float64 numbers outside the "*
           "range [0,BIG_MAX] cannot be rounded to IntegerSet values"
    end
    return IntegerSet(round(Big, f))
end


## THE `DataTableau` TYPE

struct DataTableau <: BaseType

    raw::Array{Float64,2}
    is_ordinal::Vector{Bool}
    features::Vector{Symbol}
    nrows::Int
    ncols::Int

end

struct FrameToTableauTransformer <: Transformer end

struct FrameToTableauScheme <: BaseType
    
    schemes::Vector{ToIntScheme}
    is_ordinal::Vector{Bool}
    features::Vector{Symbol}
    
end

# Note: we later define `function FrameToTableauScheme(X::DataTableau)`

function fit(transformer::FrameToTableauTransformer,
             df::AbstractDataFrame, parallel, verbosity)
    ncols = size(df, 2)
    schemes = Array{ToIntScheme}(undef, ncols)
    is_ordinal = Array{Bool}(undef, ncols)
    to_int_transformer = ToIntTransformer(sorted=true)
    for j in 1:ncols
        col = [df[j]...] # shrink eltype 
        col_type = eltype(col)
        if col_type <: AbstractFloat
            is_ordinal[j] = true
            schemes[j] = ToIntScheme(Nothing)
        else
            is_ordinal[j] = false
            length(unique(col)) <= 52 || @error "KoalaTrees cannot accept DataFrames "*
            "Encountered categorical feature with more than $SMALL_MAX classes. "*
            "KoalaTrees cannot accept DataFrames with such features. Consider "*
            "Consider, eg, categorical feature embedding using KoalaFlux."
            schemes[j] = fit(to_int_transformer, col, parallel, verbosity - 1) 
        end
    end
    return FrameToTableauScheme(schemes, is_ordinal, names(df))
end

function transform(transformer::FrameToTableauTransformer,
                   scheme::FrameToTableauScheme, df::AbstractDataFrame)
    
    scheme.features == names(df) || @error "Attempt to transform AbstractDataFrame "*
                                         "with features different from fitted frame. "
    nrows, ncols = size(df)
    raw    = Array{Float64}(undef, nrows, ncols)
    to_int_transformer = ToIntTransformer(sorted=true)
    
    for j in 1:ncols
        if scheme.is_ordinal[j]
            for i in 1:nrows
                raw[i,j] = Float64(df[i,j])
            end
        else
            for i in 1:nrows
                raw[i,j] = transform(to_int_transformer, scheme.schemes[j], df[i,j])
            end
        end
    end

    return DataTableau(raw, scheme.is_ordinal, scheme.features, nrows, ncols)

end

row(dt::DataTableau, i::Int) = dt.raw[i,:]
size(dt::DataTableau) = (dt.nrows, dt.ncols)

function size(dt::DataTableau, i::Integer)
    if  i == 1
        return dt.nrows
    elseif i == 2
        return dt.ncols
    else
        throw(BoundsError)
    end
end

function Base.getindex(dt::DataTableau, a::AbstractVector{Int}, c::Colon)
    raw = dt.raw[a,:]
    features = dt.features
    nrows = length(a)
    ncols = dt.ncols
    is_ordinal = dt.is_ordinal
    return DataTableau(raw, is_ordinal, features, nrows, ncols)
end

# unlike a DataFrame, the following spits out a raw row:
Base.getindex(dt::DataTableau, i, c::Colon) = dt.raw[i,:]




