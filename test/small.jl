using Test
import KoalaTrees: IntegerSet, SMALL_MAX

s = IntegerSet(1, 5, 16)
@test 5 in s
@test !(3 in s)
push!(s, 3)
@test 3 in s
@test 5 in s
@test !(-4 in s)
@test !(SMALL_MAX + 1 in s)

@test_throws DomainError IntegerSet(SMALL_MAX + 1)
@test_throws DomainError IntegerSet(-1, 3)
@test_throws DomainError push!(s, -2)
@test_throws DomainError push!(s, SMALL_MAX + 1)

@test round(IntegerSet, Float64(s)) == s
@test isempty(IntegerSet())
@test !isempty(IntegerSet(1))
