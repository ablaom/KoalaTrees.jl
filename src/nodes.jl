## `Node` type - data structure for building binary trees

# Binary trees are identified with their top (stump) nodes, so only a
# `Node` type, with the appropriate possiblities for connection, is
# defined. Connections are established with the methods
# `make_leftchild!` and `make_rightchild!`. Nodes have a `depth`
# field; when a new connection is made, the child's depth is declared
# to be one more than the parent. When a node `N` is created it is
# initially its own parent (equivalently, `is_stump(N) = true`) and is
# its own left and right child; its depth is initially zero.

"""

    Node{T}

Node type for building binary trees with nodes labelled with data of
type `T`. A `Node` instance has fields `parent`, `left`, `right`,
`data` and `depth`.

    Node(data)

Construct a node labelled with `data`. It has itself as children and
its `depth` is `0`.

Nodes are connected using `make_leftchild!(child, parent)` or
`make_rightchild(child, parent)`, which reset the children's depth to
one more than the parent.

See also [`unite!`](@ref), [`child](@ref), [`is_stump`](@ref),
[`is_left`](@ref), [`is_right`](@ref), [`has_left`](@ref),
[`has_right`](@ref), [`is_leaf`](@ref).

"""
mutable struct Node{T}
    parent::Node{T}
    left::Node{T}
    right::Node{T}
    data::T
    depth::Int
    function Node{T}(datum) where T
        node = new{T}()
        node.parent = node
        node.left = node
        node.right = node
        node.data = datum
        node.depth = 0
        return node
    end
end

Node(data::T) where T = Node{T}(data)


# Testing connectivity:

is_stump(node) = node.parent == node
is_left(node) =  (node.parent != node) && (node.parent.left == node)
is_right(node) = (node.parent != node) && (node.parent.right == node)
has_left(node) =  (node.left  != node)
has_right(node) = (node.right != node)
is_leaf(node) = node.left == node && node.right == node


# Connecting nodes:

"""
    `make_leftchild!(child, parent)` 

Make `child` the left child of `parent` and return the depth of
`child`.

"""
function make_leftchild!(child, parent)
    parent.left = child
    child.parent = parent
    child.depth = parent.depth + 1
end

"""
    `make_rightchild!(child, parent)` 

Make `child` the right child of `parent` and return the depth of
`child`

"""
function make_rightchild!(child, parent)
    parent.right = child
    child.parent = parent
    child.depth = parent.depth + 1
end

# Shortcut constructors:

function Node(data, parent::Node)
    child = Node(data)
    make_leftchild!(child, parent)
    return child
end
function Node(parent::Node, data)
    child = Node(data)
    make_rightchild!(child, parent)
    return child
end

# Locating children

"""
    child(parent, gender)

Return the `left` child of `parent` of a `Node` object if `gender` is 1
and `right` child if `gender is 2. If `gender` is `0` the routine throws
an error if the left and right children are different and otherwise
returns their common value.  For all other values of gender an error
is thrown.

"""
function child(parent, gender)
    if gender == 1
        return parent.left
    elseif gender == 2
        return parent.right
    elseif gender == 0
        if parent.left != parent.right
            error("Left and right children different. ")
        else
            return parent.left
        end
    end
    error("Only genders 0, 1 or 2 allowed. ")
end

"""
    unite!(child, parent, gender)

Make `child` the `left` or `right` child of a `Node` object's `parent`
in case `gender` is `1` or `2` respectively; or make `parent` the
parent of `child`. For any other values of `gender` the routine makes
`child` simultaneously the left and right child of `parent`, and
`parent` the parent of `child`. Returns `nothing`.

"""
function unite!(child, parent, gender)
    if gender == 1
        make_leftchild!(child, parent)
    elseif gender == 2
        make_rightchild!(child, parent)
    else
        make_leftchild!(child, parent)
        make_rightchild!(child, parent)
    end
end

# Implement AbstractTrees.jl interface:

const AT = AbstractTrees

function AT.children(node::Node)
    if has_left(node)
        if has_right(node)
            return (node.left, node.right)
        end
        return (node.left,)
    end
    has_right(node) && return (node.right,)
    return ()
end

has_sibling(node::Node) =
    (is_left(node) && has_right(node.parent)) ||
    (is_right(node) && has_left(node.parent))

function AT.printnode(io::IO, node::Node)
    print(io, node.data)
    if !has_sibling(node)
        if is_left(node)
            print(io, " (left)")
        elseif is_right(node)
            print(io, " (right)")
        end
    end
end

Base.eltype(::Type{<:AT.TreeIterator{Node{T}}}) where T = Node{T}
Base.IteratorEltype(::Type{<:AT.TreeIterator{Node{T}}}) where T =
    Base.HasEltype()

# Display functionality:

function spaces(n)
    s = ""
    for i in 1:n
        s = string(s, " ")
    end
    return s
end

tail(n) = "..."*string(n)[end-3:end]

function Base.show(stream::IO, node::Node)
    print(stream, "Node{$(typeof(node).parameters[1])}@$(tail(hash(node)))")
end

Base.show(stream::IO, ::MIME"text/plain", node::Node) = AT.print_tree(node)

