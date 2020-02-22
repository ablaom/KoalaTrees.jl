## THE `DataTableau` TYPE

struct DataTableau{nrows,ncols,features,is_ordinal}
    raw::Array{Float64,2}
end

# need these?
row(dt::DataTableau, i::Int) = dt.raw[i,:]
Base.size(dt::DataTableau{nrows,ncols}) where {nrows,ncols} = (nrows, ncols)
Base.size(dt::DataTableau, j::Integer) = size(dt)[j]

function Base.getindex(dt::DataTableau{nrows,ncols,features,is_ordinal},
                       a::AbstractVector{Int},
                       c::Colon) where {nrows,ncols,features,is_ordinal}
    raw = dt.raw[a,:]
    nrows2 = length(a)
    return DataTableau{nrows2,ncols,features,is_ordinal}(raw)
end

# needed?
# unlike a DataFrame, the following spits out a raw row:
Base.getindex(dt::DataTableau, i, c::Colon) = dt.raw[i,:]


## TABLE TO TABLEAU TRANFORMER

struct TableToTableauTransformer <: MLJModelInterface.Unsupervised end

struct Schema{features,is_ordinal} end

function MLJModelInterface.fit(transformer::TableToTableauTransformer,
                               verbosity::Integer,
                               df)
    s = Tables.schema(df)
    features = s.names
    cols = Tables.columns(df)
    is_ordinal = map(features) do ftr
        col = getproperty(cols, ftr)
        elscitype(col) <: Union{Continuous,OrderedFactor,Count}
    end

    schema = Schema{features,is_ordinal}()
    cache = nothing
    report = nothing

    return schema, nothing, nothing
end

function MLJModelInterface.transform(transformer::TableToTableauTransformer,
                                     ::Schema{features,is_ordinal},
                                     df) where {features, is_ordinal}

    s = Tables.schema(df)

    features == s.names ||
        throw(DomainError("Attempt to transform a table "*
                          "with features different from fitted table. "))

    ncols = length(features)
    coerce(col) = coerce(col, elscitype(col))
    coerce(col, ::Type{<:Continuous}) = Float64.(col)
    coerce(col, ::Type{<:Count}) = Float64.(col)
    coerce(col, ::Type{<:Finite}) = Float64.(MLJModelInterface.int(col))
    cols = Tables.columns(df)
    coerce(j::Integer) = coerce(getproperty(cols, features[j]))

    raw = hcat(coerce.(1:ncols)...)
    nrows = size(raw, 1)

    return DataTableau{nrows,ncols,features,is_ordinal}(raw)

end


