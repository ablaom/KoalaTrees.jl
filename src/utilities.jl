"""
# `function mean_and_ss_after_add(mean, ss, n, x)`

Returns the mean, and the sum-of-square deviations from the mean, of
`n+1` numbers, given the corresponding two quantities for the first
`n` numbers (the inputs `mean` and `ss`) and the value of the `n+1`th
number, `x`.
"""
function mean_and_ss_after_add(mean, ss, n, x)
    n >= 0 || throw(DomainError)
    mean_new = (n*mean + x)/(n + 1)
    ss_new = ss + (x - mean_new)*(x - mean)
    return mean_new, ss_new
end

"""
    function mean_and_ss_after_omit(mean, ss, n, x)

Given `n` numbers, their mean `mean` and sum-of-square deviations from
the mean `ss`, this function returns the new mean and corresponding
sum-of-square deviations of the same numbers when one of the numbers,
`x` is omitted from the list.

"""
function mean_and_ss_after_omit(mean, ss, n, x)
    n > 1 || throw(DomainError)
    mean_new = (n*mean - x)/(n-1)
    ss_new = ss - (x - mean)*(x - mean_new)
    return mean_new, ss_new
end
