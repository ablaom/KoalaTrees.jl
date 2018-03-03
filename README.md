# KoalaTrees ![logo](logo.png) 

> *Decision tree machine learning algorithms for use with Koala*

### Basic usage

Load some data and rows for the train/test sets:

````julia
    julia> using Koala
    julia> X, y = load_ames()
    julia> train, test = split(eachindex(y), 0.8); # 80:20 split
````

This data consists of a mix of numerical and categorical features. By
convention, any column of `X` whose eltype is a subtype of `Real` is
treated as numerical; any column of eltype `Char` or `AbstractString`
is treated categorical. Any other type will throw an exception.

Let us instantiate a tree model:

````julia
    julia> using KoalaTrees
    julia> tree = TreeRegressor(regularization=0.5)
    TreeRegressor@...095

    julia> showall(tree)
    TreeRegressor@...095

    key                     | value
    ------------------------|------------------------
    extreme                 |false
    max_features            |0
    max_height              |1000
    min_patterns_split      |2
    penalty                 |0.0
    regularization          |0.5
````

Here `max_features=0` means that all features are considered in
computing splits at a node. We reset this as follows:

````julia
    tree.max_features = 3
````

Now we build and train a machine. The machine essentially wraps the
model `tree` in the learning data supplied to the `Machine` constructor,
transformed into a form appropriate for our tree building algorithm
(but using only the `train` rows to calculate the transformation
parameters):
    
````julia
    julia> treeM = Machine(tree, X, y, train)
    julia> fit!(treeM, train)
    julia> showall(treeM)
    
    SupervisedMachine{TreeRegressor@...095}@...548

	key                     | value
	------------------------|------------------------
	Xt                      |DataTableaux.DataTableau of   shape (1456, 12)
	metadata                |Object of type Array{Symbol,1}
	model                   |TreeRegressor@...095
	n_iter                  |1
	predictor               |Node{KoalaTrees.NodeData}@...3798
	scheme_X                |FrameToTableauScheme@...8626
	scheme_y                |nothing
	yt                      |Array{Float64,1} of shape (1456,)

	Model detail:
	TreeRegressor@...095

	key                     | value
	------------------------|------------------------
	extreme                 |false
	max_features            |3
	max_height              |1000
	min_patterns_split      |2
	penalty                 |0.0
	regularization          |0.8497534359086443
	
                        Feature importance at penalty=0.0:
                    ┌────────────────────────────────────────┐ 
          GrLivArea │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.225 │ 
       Neighborhood │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.153            │ 
        OverallQual │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.141             │ 
         BsmtFinSF1 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.108                  │ 
         GarageArea │▪▪▪▪▪▪▪▪▪▪▪▪ 0.085                      │ 
        TotalBsmtSF │▪▪▪▪▪▪▪▪▪ 0.062                         │ 
            LotArea │▪▪▪▪▪▪▪▪ 0.056                          │ 
         MSSubClass │▪▪▪▪▪▪▪▪ 0.055                          │ 
          YearBuilt │▪▪▪▪▪▪ 0.041                            │ 
       YearRemodAdd │▪▪▪▪▪▪ 0.038                            │ 
          x1stFlrSF │▪▪▪▪ 0.026                              │ 
         GarageCars │▪▪ 0.012                                │ 
                    └────────────────────────────────────────┘ 
````

Compute the RMS error on the test set:
    
````julia
    julia> err(treeM, test)
    42581.70098526429
````

Tune the regularization parameter:
    
````julia
    julia> u, v = @curve r logspace(-3,2,100) begin
           t.regularization = r
           fit!(treeM, train)
           err(treeM, test)
       end
    julia> t.regularization = u[indmin(v)]
    0.8497534359086443

    julia> fit!(treeM, train)
    SupervisedMachine{TreeRegressor@...095}@...548

    julia> err(treeM, test)
    39313.459637964435
````

### Model parameters

- `max_features=0`: Number of features randomly selected at each node to
                                  determine splitting criterion selection (integer).
                                  If 0 (default) all features are used`
                                  
- `min_patterns_split=2`: Minimum number of patterns at node to consider split (integer). 

- `penalty=0` (range, [0,1]): Float between 0 and 1. The gain afforded
      by new features is penalized by mulitplying by the factor `1 -
      penalty` before being compared with the gain afforded by
      previously selected features. Useful for feature selection, as
      introduced in "Feature Selection via Regularized Trees", H. Deng
      and G Runger, *International Joint Conference on Neural Networks
      (IJCNN)*, IEEE, 2012.

- `extreme=false`: If true then the split of each feature considered
  is uniformly random rather than optimal. Mainly used to build
  extreme random forests using KoalaEnsembles.
                            
- `regularization=0.0` (range, [0,1)): regularization in which predictions 
    are a weighted sum of predictions at the leaf and its "nearest" neighbours. For details, sqee this [post](https://ablaom.github.io/regression/2017/10/17/nearest-neighbor-regularization-for-decision-trees.html).
    
- `max_height=1000` (range, [0, Inf]) as defined by the
     pattern. Roughly, how high the predictors looks for "nearby"
     leaves in regularized predictions.








