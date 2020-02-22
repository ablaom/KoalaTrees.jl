## CONSTANTS

const Small=UInt8
const Big=UInt64
const SMALL_MAX = Small(52) 
const BIG_MAX = Big(2^(Int(SMALL_MAX) + 1) - 1)


## TYPE FOR SETS OF SMALL INTEGERS

"""
    IntegerSet

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

==(s1::IntegerSet, s2::IntegerSet) = s1.coded == s1.coded

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
    k < 0 && return false
    k > SMALL_MAX && return false
    return in(Small(k), s)
end

function Base.push!(s::IntegerSet, k::Small)
    k > SMALL_MAX &&
        throw(DomainError(k ,"Cannot `push!` an integer larger than "*
                          "$(Int(SMALL_MAX)) into an `IntegerSet` object. "))
    if !(k in s)
        s.coded = s.coded | Big(2)^k
    end
    return s
end

function Base.push!(s::IntegerSet, k::Integer)
    k < 0 && throw(DomainError(k, "Cannot add negative integers to an "*
                               "`IntegerSet` object." ))
    return  push!(s, Small(k))
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

