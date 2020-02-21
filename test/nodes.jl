using Test
import AbstractTrees
import KoalaTrees: Node, make_leftchild!, make_rightchild!

root = Node("root")
r1 = Node("r1")
r2 = Node("r2")
@test make_leftchild!(r1, root) == 1
@test make_rightchild!(r2, root) == 1
r11 = Node("r11")
@test make_leftchild!(r11, r1) == 2
r21 = Node("r21")
@test make_leftchild!(r21, r2) == 2
r22 = Node("r22")
@test make_rightchild!(r22, r2) == 2

@test KoalaTrees.is_stump(root) 
@test KoalaTrees.is_left(r1)
@test KoalaTrees.is_left(r11)
@test KoalaTrees.is_left(r21)
@test KoalaTrees.is_right(r2)
@test KoalaTrees.is_right(r22)
@test KoalaTrees.has_left(root)
@test !KoalaTrees.has_right(r1)
@test !KoalaTrees.is_leaf(root)
@test KoalaTrees.is_leaf(r21)

@test collect(AbstractTrees.PreOrderDFS(root)) == [root, r1, r11, r2, r21, r22]

Node(r11, "r112")
Node("r112", r11)

@test collect(AbstractTrees.PreOrderDFS(root)) ==
    [root, r1, r11, r11.left, r11.right, r2, r21, r22]

@test KoalaTrees.child(r1, 1) == r11
@test KoalaTrees.child(r2, 2) == r22

lonely = Node("lonely")
KoalaTrees.child(lonely, 0) == lonely
@test_throws Exception KoalaTrees(root, 0)

L = Node("L")
R = Node("R")
@test KoalaTrees.unite!(L, lonely, 1) ==  1 # depth
@test KoalaTrees.unite!(R, lonely, 2) == 1
@test collect(AbstractTrees.PreOrderDFS(lonely)) == [lonely, L, R]

@test !KoalaTrees.has_sibling(r11)
@test !KoalaTrees.has_sibling(root)

r112 = Node(r11.left, "r112")
@test KoalaTrees.is_right(r112)
@test !KoalaTrees.has_sibling(r112)

AbstractTrees.print_tree(root)
root # same thing
