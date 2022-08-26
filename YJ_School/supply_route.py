from collections import deque
# 이동할 네 가지 방향 정의 (상, 하, 좌, 우)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

# BFS 소스코드 구현
def bfs(_graph, _visited_list):
    # 큐(Queue) 구현을 위해 deque 라이브러리 사용
    x, y = 0, 0
    queue = deque()
    queue.append((x, y))
    _visited_list[x][y] = True
    price = 0
    price_list = []
    dest = [width, width]
    # 큐가 빌 때까지 반복하기
    while queue:
        x, y = queue.popleft()
        # 현재 위치에서 4가지 방향으로의 위치 확인
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            # 미로 찾기 공간을 벗어난 경우 무시
            if nx < 0 or nx >= width or ny < 0 or ny >= width:
                continue

            # if not visited_list[dest]:
            #     visited_list[dest] = True
            #     queue.append(dest)

            if nx == width and ny == width:
                break

            if _visited_list[nx][ny] == False:
                _visited_list[nx][ny] = True
                queue.append((nx, ny))
                price += _graph[nx][ny]
                print("price:", price)

            price_list.append(price)

        print("price_list:", price_list)
    # 가장 오른쪽 아래까지의 최단 거리 반환


# BFS를 수행한 결과 출력
if __name__ == "__main__":
    test_case = int(input())
    # 2차원 리스트의 맵 정보 입력 받기
    for _ in range(test_case):
        width = int(input())
        graph = []
        visited = []
        visited_list = []
        for _ in range(width):
            graph.append(list(map(int, input())))
        for _ in range(width):
            visited.append(False)
        for _ in range(width):
            visited_list.append(visited)

        print("graph, visited_list:", graph, visited_list)
        print(bfs(graph, visited_list))