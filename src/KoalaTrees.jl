module KoalaTrees

export TreeRegressor

# needed for this module:
# import Koala: BaseType, Transformer, Regressor,
#    SupervisedMachine, softwarn, clean!
# import Koala: params, keys_ordered_by_values
# import DataFrames: AbstractDataFrame
# import KoalaTransforms: ToIntTransformer,
#   ToIntScheme, RegressionTargetTransformer

import StatsBase: sample, countmap
using Statistics
import UnicodePlots
import AbstractTrees

# to be extended:
import Base: show, round, isempty, size, in, push!, Float64, getindex

# to be extended but not explicitly rexported:
# import Koala: default_transformer_X, default_transformer_y
# import Koala: fit, transform, inverse_transform
# import Koala: setup, predict

# constants:
const Small = UInt8


# include `IntegerSet` type and custom transformer
# `FrameToTableauTransformer`:
include("utilities.jl")
include("histogram.jl")
include("nodes.jl")
# include("transformer.jl")
# include("regressor.jl")

end # module
