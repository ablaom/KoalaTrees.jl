using KoalaTrees
using Random
using Statistics
using Test

Random.seed!(1)

n = 10
v = rand(n)
μ = mean(v)
ss = sum((v .- μ).^2)

x = rand()
v1 = push!(v, x)
μ1 = mean(v1)
ss1 = sum((v1 .- μ1).^2)

@testset "mean and ss shortcut - adding element" begin
    μ1_, ss1_ = KoalaTrees.mean_and_ss_after_add(μ, ss, n, x)
    @test μ1 ≈ μ1_
    @test ss1 ≈ ss1_
end

@testset "mean and ss shortcut - omitting element" begin
    μ_, ss_ = KoalaTrees.mean_and_ss_after_omit(μ1, ss1, n + 1, x)
    @test μ ≈ μ_
    @test ss ≈ ss_
end

true
