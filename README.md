# KoalaTrees

## static hyper parameters

- `max_features=0`: Number of features randomly selected at each node to
                                  determine splitting criterion selection (integer).
                                  If 0 (default) then redefined as `n_features=length(X)`
- `min_patterns_split=2`: Minimum number of patterns at node to consider split (integer). 

- `penalty=0` (range, [0,1]): Float between 0 and 1. The gain afforded by new features
      is penalized by mulitplying by the factor `1 - penalty` before being
      compared with the gain afforded by previously selected features.

- `extreme=false`: If true then the split of each feature considered is uniformly random rather than optimal.                              
- `regularization=0.0` (range, [0,1)): regularization in which predictions 
    are a weighted sum of predictions at the leaf and its "nearest neighbours`
     as defined by the pattern. 

- `cutoff=0` (range, [`max_features`, `size(X, 2)`]): features with
       indices above `cutoff` are ignored completely. If zero then set to
       maximum feature index.

### Hyperparameters (dynamic, ie affected by fitting):

- `popularity_given_feature=Dict{Int,Int}()`: A dictionary keyed on
          feature index. Each feature whose index is a key is not
          penalised when considering new node splits (see `penalty`
          above). Whenever a feature with index `j` is chosen as the
          basis of a decision in a split, then the number of patterns
          affected by the decision is added to
          `popularity_given_feature[j]`. If `j` is not already a key,
          then it is first added and `popularity_given_feature[j]`
          initialized to zero.

### Post-fitted parameters:

- `names`: the names of the features

