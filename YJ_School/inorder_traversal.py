import sys

# class Node:
#     def __init__(self, data, left, right):
#         self.data = data
#         self.left = left
#         self.right = right

def in_order(node):
    print("node2", node_list[i])
    print("node2", node_list[i][2])
    if node_list[i][2] != None:
        in_order(node_list[i][2])
    print(node_list[i][1], end='')
    if node_list[i][3] != None:
        in_order(node_list[i][3])

for _ in range(10):
    N = int(input())
    node_list = [[i+1, '', None, None] for i in range(N)]

    for i in range(N):
        node = input()
        node = node.split(' ')
        node_list[i][0] = node[0]
        node_list[i][1] = node[1]
        if len(node) >= 3:
            node_list[i][2] = node[2]
        if len(node) >= 4:
            node_list[i][3] = node[3]
        print(node_list[2])

        in_order(node_list[N])