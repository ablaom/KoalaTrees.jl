using Test
import AbstractTrees
import KoalaTrees
import KoalaTrees: create_stump, create_left, create_right,
    left, right, parent, node, Tree, data, has_left, has_right,
    is_left, is_right, is_stump, Prenode
using Random

@testset "prenodes" begin

    r = create_stump("r")
    r1 = create_left("r1", r)
    r2 = create_right("r2", r)
    r21 = create_left("r21", r2)
    r22 = create_right("r22", r2)

    @test KoalaTrees.is_stump(r)
    @test KoalaTrees.is_left(r1)
    @test KoalaTrees.is_left(r21)
    @test KoalaTrees.is_right(r2)
    @test KoalaTrees.is_right(r22)

end

@test "nodes" begin

    prenodes = [r22, r1, r, r22, r2, r21] # no particular order

    s = node(prenodes)
    s1 = left(s)
    s2 = right(s)
    s21 = left(s2)
    s22 = right(s2)

    @test collect(AbstractTrees.PreOrderDFS(s)) == [s, s1, s2, s21, s22]

    @test parent(s) == s

    # I AM HERE

    @test KoalaTrees.is_stump(s)
    @test KoalaTrees.is_left(s1)
    @test KoalaTrees.is_left(s21)
    @test KoalaTrees.is_right(s2)
    @test KoalaTrees.is_right(s22)
    @test KoalaTrees.has_left(s)
    @test !KoalaTrees.has_right(s1)
    @test !KoalaTrees.is_leaf(s)
    @test KoalaTrees.is_leaf(s21)
    @test !KoalaTrees.has_sibling(s)
    @test KoalaTrees.has_sibling(s1)

end

make_tree(5)
make_tree2(5)

@test collect(AbstractTrees.PreOrderDFS(root)) ==
    [root, r1, r11, r11.left, r11.right, r2, r21, r22]

@test KoalaTrees.child(r1, 1) == r11
@test KoalaTrees.child(r2, 2) == r22


r112 = node(r11.left, "r112")
@test KoalaTrees.is_right(r112)
@test !KoalaTrees.has_sibling(r112)

AbstractTrees.print_tree(root)
root # same thing

# for use in benchmarking:





function make_tree(depth)
    prenodes = [create_stump("S"),]
    
    depth > 0 || return node(prenodes)
    
    d = 0 # depth so far
    
    function build!(prenodes, d, p)
        d < depth || return nothing
        l = create_left(p.data*"L", p)
        r = create_right(p.data*"R", p)
        push!(prenodes, l)
        push!(prenodes, r)
        build!(prenodes, d + 1, l)
        build!(prenodes, d + 1, r)
        return nothing
    end

    build!(prenodes, d, prenodes[1])
    
    return node(prenodes)
end

function make_tree2(depth)
    prenodes = [create_stump(rand(4)),]
    
    depth > 0 || return node(prenodes)
    
    d = 0 # depth so far
    
    function build!(prenodes, d, p)
        d < depth || return nothing
        l = create_left(rand(4), p)
        r = create_right(rand(4), p)
        push!(prenodes, l)
        push!(prenodes, r)
        build!(prenodes, d + 1, l)
        build!(prenodes, d + 1, r)
        return nothing
    end

    build!(prenodes, d, prenodes[1])
    
    return node(prenodes)
end

function make_pretree2(depth)
    prenodes = [create_stump(rand(4)),]
    
    depth > 0 || return node(prenodes)
    
    d = 0 # depth so far
    
    function build!(prenodes, d, p)
        d < depth || return nothing
        l = create_left(rand(4), p)
        r = create_right(rand(4), p)
        push!(prenodes, l)
        push!(prenodes, r)
        build!(prenodes, d + 1, l)
        build!(prenodes, d + 1, r)
        return nothing
    end

    build!(prenodes, d, prenodes[1])
    
    return prenodes
end

# using BenchmarkTools
# @btime make_tree(6);
#   181.587 μs (299 allocations: 39.38 KiB)

#@btime make_pretree2(6);
#  10.248 μs (263 allocations: 20.19 KiB)


true
