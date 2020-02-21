
## THE `DataTableau` TYPE

struct DataTableau <: BaseType

    raw::Array{Float64,2}
    is_ordinal::Vector{Bool}
    features::Vector{Symbol}
    nrows::Int
    ncols::Int

end

struct TableToTableauTransformer <: Transformer end

struct TableToTableauScheme <: BaseType
    
    schemes::Vector{ToIntScheme}
    is_ordinal::Vector{Bool}
    features::Vector{Symbol}
    
end

# Note: we later define `function TableToTableauScheme(X::DataTableau)`

function fit(transformer::TableToTableauTransformer,
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
    return TableToTableauScheme(schemes, is_ordinal, names(df))
end

function transform(transformer::TableToTableauTransformer,
                   scheme::TableToTableauScheme, df::AbstractDataFrame)
    
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




