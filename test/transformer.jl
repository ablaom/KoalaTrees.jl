module TestTransformer

using Test
using KoalaTrees
using MLJBase
using Tables

multi=categorical([:john, :mary, :anna])
levels!(multi, [:anna, :john, :mary])

X = (count=1:3,
     continuous=[11.0, 12.0, 13.0],
     ordered=categorical([10, 20, 30], ordered=true),
     multi=multi) |> Tables.rowtable

# julia> pretty(X)
# ┌───────┬───────────────┬────────────────────────────────┬──────────────
# │ count │ continuous    │ ordered                        │ multi
# │ Int64 │ Float64       │ CategoricalValue{Int64,UInt32} │ CategoricalVa
# │ Count │ Continuous    │ OrderedFactor{3}               │ Multiclass{3}
# ├───────┼───────────────┼────────────────────────────────┼──────────────
# │ 1     │ 11.0          │ 10                             │ john
# │ 2     │ 12.0          │ 20                             │ mary
# │ 3     │ 13.0          │ 30                             │ anna
# └───────┴───────────────┴────────────────────────────────┴──────────────


t = KoalaTrees.TableToTableauTransformer()
fitresult, _, _, = fit(t, 1, X)
dt = transform(t, fitresult, X[[1, 3]])
@test dt.raw == hcat([1.0, 3.0],
                     [11.0, 13.0],
                     [1.0, 3.0],
                     [2.0, 1.0])
dt = transform(t, fitresult, X)
@test size(dt) == (3, 4)
@test size(dt, 2) == 4

@test dt[[3, 1],:].raw == dt.raw[[3, 1],:] 

X = load_reduced_ames();
dt = transform(fit!(machine(t, X)), X)
is_ordinal = map(schema(X).scitypes) do st
    st <: Union{Count,Continuous,OrderedFactor}
end
@test is_ordinal == typeof(dt).parameters[4]

end

true
