import sys

class Node:
    def __init__(self, data, left, right):
        self.data = data
        self.left = left
        self.right = right

def in_order(node):
    if node.left != None:
        in_order(tree[node.left])
    print(node.data, end='')
    if node.right != None:
        in_order(tree[node.right])

N = int(sys.stdin.readline().rstrip())
tree = {}

for i in range(N):
    data, left, right = sys.stdin.readline().split()
    if left == '.':
        left = None
    if right == '.':
        right = None
    tree[data] = Node(data, left, right)

in_order(tree['A'])
print()