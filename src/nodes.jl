const U = UInt
using AbstractTrees

# Nodes with both parent and children fields cannot be immutable
# because they cannot be instantiated simulaneously. We construct a
# tree with "prenodes" instead.  A prenode represents a node of some
# tree but the children are defined only implicitly by the prenode
# `gender` and `parent` fields.

# When all prenodes are defined, a collection of immutable objects of
# type `Node` are constructed. These nodes admit `left`, `right` and
# `parent` *methods*.


## PRENODES

struct Prenode{T} <: MLJModelInterface.MLJType
    data::T
    gender::Bool
    parent::Union{Prenode{T},Prenode{T}}
    Prenode(data::T, gender, parent) where T = new{T}(data, gender, parent)
    Prenode(data::T) where T = new{T}(data)
end

""" 

    create_stump(data)

Create a stump prenode for a binary tree labelled with specified `data`.

See also [`node`](@ref)

"""
create_stump(data) = Prenode(data)

"""
    create_left(data, parent)

Create a prenode for a binary tree that is the left child of the
specified `parent` prenode, labelling it with the specified `data`.

See also [`node`](@ref)

"""
create_left(data, parent::Prenode) = Prenode(data, false, parent)

"""
    create_left(data, parent)

Create a prenode for a binary tree that is the left child of the
specified `parent` prenode, labelling it with the specified `data`.

See also [`node`](@ref)

"""
create_right(data, parent::Prenode) = Prenode(data, true, parent)

is_stump(prenode) = !isdefined(prenode, :parent)
is_left(prenode::Prenode) = !is_stump(prenode) && !prenode.gender
is_right(prenode::Prenode) = !is_stump(prenode) && prenode.gender

function Base.show(stream::IO, ::MIME"text/plain", prenode::Prenode)
    if isdefined(prenode, :parent)
        print(stream,
              "data: $(prenode.data)\n",
              "gender: $(prenode.gender)\n",
              "parent data: $(prenode.parent.data)")
    else
        print(stream,
              "data: $(prenode.data)\n",
              "gender: $(prenode.gender)\n",
              "parent: #undef")
    end
end


## TREES

"""
    Tree{T,N}

An immutable struct for explicitly encoding the binary tree defined
*implicitly* by `N` instances of `Prenode{T}`.

"""
struct Tree{T,N} <: MLJModelInterface.MLJType
    stump::U
    prenodes::NTuple{N,Prenode{T}}
    lefts::NTuple{N,U}
    rights::NTuple{N,U}
    parents::NTuple{N,U}
end

"""
    Tree(prenodes)

Construct a `Tree` instance from a vector of identically typed
`Prenode` instances. 

"""
function Tree(prenodes::AbstractVector{Prenode{T}}) where T

    N = length(prenodes)

    _left = Dict{Prenode{T},U}()
    _right = Dict{Prenode{T},U}()
    _parents = Dict{Prenode{T},U}()
    j = zero(U)
    stump = 0
    for n in prenodes
        j  += 1
        if isdefined(n, :parent)
            if n.gender
                _right[n.parent] = j
            else
                _left[n.parent] = j
            end
        else
            stump == 0 ||
                error("More than one stump encountered. ")
           stump = j
        end
    end

    stump == 0 &&
        error("Cannot construct a tree from prenodes without stump. ")

    internal_prenodes = keys(_left) # excludes leaves

    lefts = Vector{U}(undef, N)
    rights = Vector{U}(undef, N)
    parents = Vector{U}(undef, N)

    j = zero(U)
    for n in prenodes
        j += 1
        if j == stump
            parents[j] = 0
        end
        if n in internal_prenodes
            parents[_left[n]] = j
            parents[_right[n]] = j
            lefts[j] = _left[n]
            rights[j] = _right[n]
        else
            lefts[j] = 0
            rights[j] = 0
        end
    end

    return Tree(stump,
                Tuple(prenodes),
                Tuple(lefts),
                Tuple(rights),
                Tuple(parents))

end

Base.length(::Tree{N} where N) = N

Base.parent(j, tree::Tree) = tree.parents[U(j)]
left(j, tree::Tree) = tree.lefts[U(j)]
right(j, tree::Tree) = tree.rights[U(j)]
data(j, tree::Tree) = tree.prenodes[j].data
stump(tree::Tree) = tree.stump


## NODE

"""
    Node{T,N}

Type for representing nodes in a binary tree constructed using `N`
prenodes.

A node representing a stump prenode is its own parent. A node
representing a prenode without children has itself as both children.

See also [`node`](@ref)

"""
struct Node{T,N} <: MLJModelInterface.MLJType
    id::U
    tree:: Tree{T,N}
end

"""

   node(prenodes)

Return the stump `Node` object associated with the binary tree
constructed using the vector of identically typed `prenodes`.

See also [`create_stump`](@ref), [`create_left`](@ref),
[`create_right`](@ref).

"""
function node(prenodes)
    tree = Tree(prenodes)
    return Node(stump(tree), tree)
end

prenode(n::Node) = n.tree.prenodes[n.id]

function Base.parent(n::Node)
    tree = n.tree
    p = parent(n.id, tree)
    p == 0 && return n
    return Node(p, tree)
end

left(n::Node) = Node(left(n.id, n.tree), n.tree)
right(n::Node) = Node(right(n.id, n.tree), n.tree)

has_left(n::Node) = left(n).id != 0
has_right(n::Node) = right(n).id != 0
is_stump(n::Node) = parent(n) == 0

data(n::Node) = prenode(n).data

# implement AbstractTrees.jl interface:
const AT = AbstractTrees
function AT.children(n::Node)
    if has_left(n)
        if has_right(n)
            return (left(n), right(n))
        end
        return (left(n),)
    end
    has_right(n) && return (right(n),)
    return ()
end
AT.printnode(io::IO, n::Node) = print(io, data(n))

Base.eltype(::Type{<:AT.TreeIterator{S}}) where {T,N,S<:Node{T,N}} = Node{T,N}
Base.IteratorEltype(::Type{<:AT.TreeIterator{S}}) where {T,N,S<:Node{T,N}} =
    Base.HasEltype()

Base.show(stream::IO, ::MIME"text/plain", node::Node) =
    AT.print_tree(node)
