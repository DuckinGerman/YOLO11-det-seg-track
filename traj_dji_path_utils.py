import numpy as np
import cv2
import heapq

def distance_transform_path(mask_2d, start, goal):
    """
    在二值图 mask_2d (0或255) 上做距离变换(DIST_L2),
    然后用Dijkstra在前景里搜索从 start -> goal 的最优路径 (越靠近中心代价越低).
    mask_2d: shape=(H,W), 0表示背景, 255表示前景
    start  : (sx, sy)  起点(列,行)
    goal   : (gx, gy)  终点(列,行)

    返回:
      path      : list[(x,y)] 搜索到的路径像素序列(若无可行路径则为空[])
      dist_map  : np.ndarray(float32), 距离变换结果(可用于调试或可视化)
    """
    # 1) 先把mask转换成 0/1
    binary = (mask_2d > 127).astype(np.uint8)
    # 2) 距离变换
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)  # 或其他距离类型

    h, w = dist_map.shape
    visited = np.zeros((h, w), dtype=bool)
    cost_map = np.full((h, w), np.inf, dtype=float)
    parent = -1 * np.ones((h, w, 2), dtype=int)

    # 检查起点终点是否在图内
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        # 超出边界，直接返回
        return [], dist_map

    # 如果起点或终点不在前景，可直接返回空
    if mask_2d[sy, sx] <= 127 or mask_2d[gy, gx] <= 127:
        return [], dist_map

    cost_map[sy, sx] = 0.0
    heap = []
    # 放入优先队列
    heapq.heappush(heap, (0.0, (sx, sy)))

    # 8邻居(也可改4邻居)
    directions = [(-1,-1), (0,-1), (1,-1),
                  (-1, 0),         (1, 0),
                  (-1, 1), (0, 1), (1, 1)]

    # --- Dijkstra 搜索 ---
    while heap:
        cur_cost, (cx, cy) = heapq.heappop(heap)
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True

        # 若到达goal
        if (cx, cy) == (gx, gy):
            break

        for dx, dy in directions:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < w and 0 <= ny < h:
                # 只走前景
                if mask_2d[ny, nx] > 127:
                    step_cost = -dist_map[ny, nx]   # 越靠中心(值大)则代价越小
                    new_cost = cur_cost + step_cost
                    if new_cost < cost_map[ny, nx]:
                        cost_map[ny, nx] = new_cost
                        parent[ny, nx] = [cy, cx]
                        heapq.heappush(heap, (new_cost, (nx, ny)))

    # --- 回溯路径 ---
    path = []
    if not visited[gy, gx]:
        # 没找到可行路径
        return [], dist_map

    px, py = gx, gy
    while not (px == sx and py == sy):
        path.append((px, py))
        py_prev, px_prev = parent[py, px]
        px, py = px_prev, py_prev
    path.append((sx, sy))
    path.reverse()

    return path, dist_map


def distance_transform_path_local(mask_2d, start, goal, roi, margin=0):
    """
    只在 roi=(rx1, ry1, rx2, ry2) 范围内搜索，避免整张图的高开销.
    mask_2d: 0/255, shape=(H,W)
    start, goal: 全局坐标(x, y)
    roi         : (rx1, ry1, rx2, ry2) 外接矩形(左上,右下)
    margin      : 额外保留的边界像素, 默认0

    返回:
      global_path: list of (x,y) 全局路径坐标
      local_dist_map: 局部子图的distance map(可选查看)
    """
    H, W = mask_2d.shape[:2]
    rx1, ry1, rx2, ry2 = roi
    # 给 ROI 留点 margin
    rx1 = max(rx1 - margin, 0)
    ry1 = max(ry1 - margin, 0)
    rx2 = min(rx2 + margin, W-1)
    ry2 = min(ry2 + margin, H-1)

    # 截取子图
    local_mask = mask_2d[ry1:ry2+1, rx1:rx2+1]  # shape=(roi_h, roi_w)
    # 转换start, goal到局部坐标
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    # clip一下，避免超出roi
    sx = np.clip(sx, rx1, rx2)
    sy = np.clip(sy, ry1, ry2)
    gx = np.clip(gx, rx1, rx2)
    gy = np.clip(gy, ry1, ry2)

    sx_local = sx - rx1
    sy_local = sy - ry1
    gx_local = gx - rx1
    gy_local = gy - ry1

    # 在局部子图调用 distance_transform_path
    path_local, local_dist_map = distance_transform_path(
        local_mask, (sx_local, sy_local), (gx_local, gy_local)
    )

    # 映射回全局坐标
    global_path = []
    for (lx, ly) in path_local:
        gx_ = lx + rx1
        gy_ = ly + ry1
        global_path.append((gx_, gy_))

    return global_path, local_dist_map
