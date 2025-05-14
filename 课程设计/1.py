def _construct_10_cycle_with_hex_coloring(self):
    """构造长度为10的环，从中心六边形开始往左走一步（两个六边形）"""
    # 从中心六边形开始
    center_hex = (0, 0)
    
    # 如果中心六边形不存在，尝试找其他合适的六边形作为中心
    if center_hex not in self.hexagons:
        if not self.hexagons:
            return None, None
        center_hex = next(iter(self.hexagons))
    
    # 往左走一步（西方向，q-1, r不变）
    left_hex = (center_hex[0] - 1, center_hex[1])
    
    # 如果左边六边形不存在，尝试其他方向
    if left_hex not in self.hexagons:
        # 按顺时针顺序尝试六个方向
        for dq, dr in self.axial_directions_clockwise:
            neighbor = (center_hex[0] + dq, center_hex[1] + dr)
            if neighbor in self.hexagons:
                left_hex = neighbor
                break
        
        # 如果仍然找不到合适的邻居
        if left_hex not in self.hexagons:
            messagebox.showerror("构造错误", "无法找到两个相邻的六边形来构造10环")
            return None, None
    
    # 构建环
    hex_region_to_color = {center_hex, left_hex}
    boundary_edges = self._get_boundary_edges_of_hex_region(hex_region_to_color)
    
    # 验证结果
    if len(boundary_edges) != 10:
        messagebox.showerror("构造错误", f"构造的环边数为 {len(boundary_edges)}，而不是预期的10")
        return None, None
    
    if not self._verify_cycle_connectivity(boundary_edges):
        messagebox.showerror("构造错误", "构造的环不连通")
        return None, None
    
    return boundary_edges, list(hex_region_to_color)

def _construct_12_cycle_with_triangle(self):
    """
    构造长度为12的环，从中心六边形开始，往左走一步再往右下走一步（三角状的三个六边形）
    """
    # 从中心六边形开始
    center_hex = (0, 0)
    
    # 如果中心不存在，尝试找其他合适的六边形作为中心
    if center_hex not in self.hexagons:
        if not self.hexagons:
            return None, None
        center_hex = next(iter(self.hexagons))
    
    # 往左走一步（西方向，q-1, r不变）
    left_hex = (center_hex[0] - 1, center_hex[1])
    
    # 如果左边六边形不存在，尝试用不同的六边形作为中心
    if left_hex not in self.hexagons:
        for hex_coord in self.hexagons:
            if hex_coord == center_hex:
                continue
                
            test_center = hex_coord
            # 尝试以这个六边形为中心
            for dq, dr in self.axial_directions_clockwise:
                test_left = (test_center[0] + dq, test_center[1] + dr)
                if test_left in self.hexagons:
                    center_hex = test_center
                    left_hex = test_left
                    break
            
            if left_hex in self.hexagons:
                break
                
        # 如果仍然找不到合适的中心和左邻居
        if left_hex not in self.hexagons:
            messagebox.showerror("构造错误", "无法找到相邻的六边形来构造12环")
            return None, None
    
    # 从left_hex往右下走一步（东南方向，r+1）
    bottom_hex = (left_hex[0], left_hex[1] + 1)
    
    # 如果右下六边形不存在，尝试其他方向
    if bottom_hex not in self.hexagons:
        # 尝试从left_hex出发的其他方向
        for dq, dr in self.axial_directions_clockwise:
            if dq == 0 and dr == 0:  # 跳过自身
                continue
                
            test_bottom = (left_hex[0] + dq, left_hex[1] + dr)
            # 确保新的六边形不是center_hex
            if test_bottom in self.hexagons and test_bottom != center_hex:
                # 检查是否与center_hex相邻，形成三角形
                dist_to_center = max(
                    abs(test_bottom[0] - center_hex[0]),
                    abs(test_bottom[1] - center_hex[1]),
                    abs((test_bottom[0] + test_bottom[1]) - (center_hex[0] + center_hex[1]))
                )
                if dist_to_center == 1:  # 相邻
                    bottom_hex = test_bottom
                    break
    
    # 如果仍然找不到合适的第三个六边形
    if bottom_hex not in self.hexagons or bottom_hex == center_hex:
        messagebox.showerror("构造错误", "无法构造三角形状的三个六边形")
        return None, None
    
    # 构建环
    hex_region_to_color = {center_hex, left_hex, bottom_hex}
    boundary_edges = self._get_boundary_edges_of_hex_region(hex_region_to_color)
    
    # 验证结果
    if len(boundary_edges) != 12:
        messagebox.showerror("构造错误", f"构造的环边数为 {len(boundary_edges)}，而不是预期的12")
        return None, None
    
    if not self._verify_cycle_connectivity(boundary_edges):
        messagebox.showerror("构造错误", "构造的环不连通")
        return None, None
    
    return boundary_edges, list(hex_region_to_color)
def _generate_case4_uncoloring_traversal_path(self):
    """
    生成用于Case 4 (-4k) 操作的六边形遍历路径，与Case 2的路径生成完全独立。
    针对Case 4有特殊处理：
    1. 对于奇数t：当到达第三层的西南标记点时触发特殊处理
    2. 对于偶数t：
       - 最外层特殊点不加入路径，跳转到第二层西南角点
       - 从第二层西南角点先往东走，然后逆时针绕行
       - 内层西北部标记点遇到时，向左走两格再往左下走
    确保能够生成足够长的路径以支持大k值去色
    """
    if self.t < 1: 
        return []
    
    traversal_path = []
    visited_on_traversal = set()

    # 获取各层角点
    corner_points_by_layer = {}  # 存储各层角点: layer -> {label -> (q,r)}
    for layer in range(1, self.t + 1):
        corners = self._get_corner_hexagons(layer)
        if corners:
            corner_points_by_layer[layer] = corners
    
    # 检查最外层西北角点是否存在
    if self.t not in corner_points_by_layer or "NW" not in corner_points_by_layer[self.t]:
        # 找不到最外层西北角点，使用(0,0)作为退路
        start_hex = (0,0) if (0,0) in self.hexagons else None
        if not start_hex:
            return []  # 没有有效起点
    else:
        # 无论t的奇偶性，都从最外层西北角开始
        start_hex = corner_points_by_layer[self.t]["NW"]
    
    # 获取特殊点
    special_points = set()
    special_points_by_layer = {}  # 存储各层特殊点: layer -> (q,r)
    
    relevant_tiers = []
    if self.t % 2 == 0: 
        # t为偶数，特殊点在偶数层
        relevant_tiers = [m for m in range(2, self.t + 1, 2)]
    else: 
        # t为奇数，特殊点在奇数层
        relevant_tiers = [m for m in range(1, self.t + 1, 2)]
        
    for m in relevant_tiers:
        if m < 1: continue
        
        # 获取该层的西北角点坐标
        nw_corners_m = self._get_corner_hexagons(m)
        if "NW" not in nw_corners_m:
            continue
            
        nw_corner = nw_corners_m["NW"]
        
        # 从西北角点往左下走两格获取特殊点
        step1_q = nw_corner[0] - 1
        step1_r = nw_corner[1] + 1
        step2_q = step1_q - 1
        step2_r = step1_r + 1
        special_point = (step2_q, step2_r)
        
        # 检查特殊点是否在网格内
        s_coord = -special_point[0] - special_point[1]
        if max(abs(special_point[0]), abs(special_point[1]), abs(s_coord)) < self.t:
            if special_point in self.hexagons:
                special_points.add(special_point)
                special_points_by_layer[m] = special_point
    
    # 初始化路径
    current_hex = start_hex
    traversal_path.append(current_hex)
    visited_on_traversal.add(current_hex)
    
    # 顺时针方向向量（东，东南，西南，西，西北，东北）
    clockwise_directions = [
        (1, 0),    # 东
        (0, 1),    # 东南
        (-1, 1),   # 西南
        (-1, 0),   # 西 
        (0, -1),   # 西北
        (1, -1)    # 东北
    ]
    
    # 逆时针方向向量（东，东北，西北，西，西南，东南）
    counterclockwise_directions = [
        (1, 0),    # 东
        (1, -1),   # 东北
        (0, -1),   # 西北
        (-1, 0),   # 西
        (-1, 1),   # 西南
        (0, 1)     # 东南
    ]
    
    # 默认情况下从东北方向开始
    current_dir_index = 5  # 东北方向（顺时针顺序）
    using_counterclockwise = False  # 标记是否使用逆时针方向
    directions = clockwise_directions  # 默认使用顺时针方向
    
    # 标记是否已经找到并处理了关键点
    found_l3_sw_point = False  # 第三层西南角点（奇数t）
    found_outer_special_point = False  # 最外层特殊点（偶数t）
    
    # 收集所有角点和关键标记点
    all_marker_points = set()
    l3_sw_point = None  # 第三层西南角点
    l2_sw_point = None  # 第二层西南角点（用于偶数t的跳转）
    
    # 找到第三层西南角点（对奇数t）
    if self.t % 2 == 1 and 3 in corner_points_by_layer and "SW" in corner_points_by_layer[3]:
        l3_sw_point = corner_points_by_layer[3]["SW"]
    
    # 找到第二层西南角点（对偶数t）
    if self.t % 2 == 0 and 2 in corner_points_by_layer and "SW" in corner_points_by_layer[2]:
        l2_sw_point = corner_points_by_layer[2]["SW"]
    
    # 收集所有角点（包括边缘点，除了内层的西北角点）
    nw_corner_points = set()  # 存储各层的西北角点，需要特殊处理
    
    for layer, corners in corner_points_by_layer.items():
        for label, point in corners.items():
            if label == "NW":
                nw_corner_points.add(point)  # 记录西北角点
                if layer < self.t:  # 除最外层外的西北角点不作为标记点
                    continue
            all_marker_points.add(point)
    
    # 对于偶数t，不将最外层特殊点加入标记点集合和路径
    if self.t % 2 == 0 and self.t in special_points_by_layer:
        outer_special_point = special_points_by_layer[self.t]
        special_points.discard(outer_special_point)
    
    # 其他特殊点正常加入标记点集合
    all_marker_points.update(special_points)
    
    # 开始主要的遍历循环
    max_iterations = self.t * 200  # 增加最大迭代次数以确保能生成足够长的路径
    iterations = 0
    
    # 确保我们尝试访问每个六边形
    remaining_hexes = set(self.hexagons)
    
    while iterations < max_iterations and remaining_hexes:
        iterations += 1
        
        # 检查当前位置是否为西北角点，需要特殊处理（除最外层外）
        if current_hex in nw_corner_points and current_hex != corner_points_by_layer[self.t]["NW"]:
            # 执行向左走两格再往左下走的特殊处理
            
            # 向左走一格（西方向，q-1, r不变）
            step1 = (current_hex[0] - 1, current_hex[1])
            if step1 in self.hexagons and step1 not in visited_on_traversal:
                traversal_path.append(step1)
                visited_on_traversal.add(step1)
                remaining_hexes.discard(step1)
                current_hex = step1
            
            # 向左再走一格（继续西方向）
            step2 = (current_hex[0] - 1, current_hex[1])
            if step2 in self.hexagons and step2 not in visited_on_traversal:
                traversal_path.append(step2)
                visited_on_traversal.add(step2)
                remaining_hexes.discard(step2)
                current_hex = step2
            
            # 向左下走一格（西南方向，q-1, r+1）
            step3 = (current_hex[0] - 1, current_hex[1] + 1)
            if step3 in self.hexagons and step3 not in visited_on_traversal:
                traversal_path.append(step3)
                visited_on_traversal.add(step3)
                remaining_hexes.discard(step3)
                current_hex = step3
            
            # 设置下一个方向
            if using_counterclockwise:
                current_dir_index = 3  # 西方向（逆时针顺序）
            else:
                current_dir_index = 3  # 西方向（顺时针顺序）
            continue
        
        # 检查当前位置是否为标记点
        if current_hex in all_marker_points:
            # 奇数t，检查是否到达了第三层西南角点
            if not found_l3_sw_point and self.t % 2 == 1 and l3_sw_point and current_hex == l3_sw_point:
                found_l3_sw_point = True
                
                # 特殊处理：从SW标记点往左上走一步（西北方向，q不变, r-1）
                nw_step = (current_hex[0], current_hex[1] - 1)
                if nw_step in self.hexagons and nw_step not in visited_on_traversal:
                    traversal_path.append(nw_step)
                    visited_on_traversal.add(nw_step)
                    remaining_hexes.discard(nw_step)
                    current_hex = nw_step
                
                # 特殊处理：右上走一步（东北方向，q+1, r-1）
                ne_step = (current_hex[0] + 1, current_hex[1] - 1)
                if ne_step in self.hexagons and ne_step not in visited_on_traversal:
                    traversal_path.append(ne_step)
                    visited_on_traversal.add(ne_step)
                    remaining_hexes.discard(ne_step)
                    current_hex = ne_step
                
                # 特殊处理：往右走一步（东方向，q+1, r不变）
                e_step = (current_hex[0] + 1, current_hex[1])
                if e_step in self.hexagons and e_step not in visited_on_traversal:
                    traversal_path.append(e_step)
                    visited_on_traversal.add(e_step)
                    remaining_hexes.discard(e_step)
                    current_hex = e_step
                
                # 设置下一个方向为西方向
                current_dir_index = 3  # 西方向（顺时针顺序）
                continue
            else:
                # 其他标记点正常转向
                if using_counterclockwise:
                    # 逆时针方向：向右转（即索引+1）
                    current_dir_index = (current_dir_index + 1) % 6
                else:
                    # 顺时针方向：向左转（即索引+1）
                    current_dir_index = (current_dir_index + 1) % 6
        
        # 偶数t，检查是否到达了最外层特殊标记点（在all_marker_points之外检查）
        elif not found_outer_special_point and self.t % 2 == 0 and self.t in special_points_by_layer and current_hex == special_points_by_layer[self.t]:
            found_outer_special_point = True
            
            # 特殊处理：直接跳转到第二层西南角点（如果存在）
            if l2_sw_point:
                current_hex = l2_sw_point
                if current_hex not in visited_on_traversal:
                    traversal_path.append(current_hex)
                    visited_on_traversal.add(current_hex)
                    remaining_hexes.discard(current_hex)
                
                # 从这里开始使用逆时针方向和转向规则
                using_counterclockwise = True
                directions = counterclockwise_directions
                
                # 必须先往东走一步（q+1, r不变）
                e_step = (current_hex[0] + 1, current_hex[1])
                if e_step in self.hexagons and e_step not in visited_on_traversal:
                    traversal_path.append(e_step)
                    visited_on_traversal.add(e_step)
                    remaining_hexes.discard(e_step)
                    current_hex = e_step
                
                # 设置初始方向为东（在逆时针顺序中）
                current_dir_index = 0  # 东方向（逆时针顺序）
                continue
        
        # 计算下一步的坐标
        next_dir = directions[current_dir_index]
        next_hex = (current_hex[0] + next_dir[0], current_hex[1] + next_dir[1])
        
        # 检查下一步是否有效
        if next_hex in self.hexagons and next_hex not in visited_on_traversal:
            # 可以移动到下一个六边形
            current_hex = next_hex
            traversal_path.append(current_hex)
            visited_on_traversal.add(current_hex)
            remaining_hexes.discard(current_hex)
        else:
            # 如果下一步无效，尝试转向
            direction_tried = 1
            found_valid_move = False
            
            while direction_tried <= 6:
                if using_counterclockwise:
                    # 逆时针方向：向右转（即索引+1）
                    current_dir_index = (current_dir_index + 1) % 6
                else:
                    # 顺时针方向：向左转（即索引+1）
                    current_dir_index = (current_dir_index + 1) % 6
                
                next_dir = directions[current_dir_index]
                next_hex = (current_hex[0] + next_dir[0], current_hex[1] + next_dir[1])
                
                if next_hex in self.hexagons and next_hex not in visited_on_traversal:
                    current_hex = next_hex
                    traversal_path.append(current_hex)
                    visited_on_traversal.add(current_hex)
                    remaining_hexes.discard(current_hex)
                    found_valid_move = True
                    break
                
                direction_tried += 1
            
            # 如果所有方向都不可行，跳到一个未访问的六边形继续（如果有）
            if not found_valid_move:
                remaining_unvisited = [h for h in self.hexagons if h not in visited_on_traversal]
                if remaining_unvisited:
                    # 找到最近的未访问六边形（基于简单距离）
                    closest_hex = min(remaining_unvisited, 
                                    key=lambda h: abs(h[0]-current_hex[0]) + abs(h[1]-current_hex[1]))
                    current_hex = closest_hex
                    traversal_path.append(current_hex)
                    visited_on_traversal.add(current_hex)
                    remaining_hexes.discard(current_hex)
                else:
                    # 所有六边形都已访问，结束遍历
                    break
    
    # 确保每个六边形都被至少考虑一次
    # 如果有六边形没有被访问，将它们添加到路径末尾
    for hex_qr in self.hexagons:
        if hex_qr not in visited_on_traversal:
            traversal_path.append(hex_qr)
    
    return traversal_path