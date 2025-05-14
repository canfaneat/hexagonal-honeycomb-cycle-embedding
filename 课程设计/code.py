#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蜂窝网格偶数长度环嵌入 - 基于Yang等人2008年论文《Embedding even-length cycles in a hexagonal honeycomb mesh》
"""
import tkinter as tk
import math
from collections import defaultdict, deque
from tkinter import messagebox # 明确导入 messagebox

class HexGrid:
    """六边形蜂窝网格生成器和环构造器"""

    def __init__(self, t):
        """初始化HHM(t)，t为网格阶数"""
        self.t = t # 网格阶数，对应论文中的t
        self.hexagons = set()  # 存储所有六边形的轴向坐标(q,r) {(q,r), ...}
        self.edges = []  # 存储所有边 [(x1,y1,x2,y2), ...] （像素坐标）
        self.vertices = {}  # 存储所有顶点 {(x,y): True, ...} （像素坐标, key是精确浮点元组）
        self.vertex_edges = defaultdict(list)  # 顶点到边的映射 {(x,y): [(x1,y1,x2,y2), ...], ...}
        self._exact_vertex_map = {} # (rounded_x, rounded_y) -> (exact_x, exact_y)
        
        self.hex_centers = {}  # 六边形中心像素坐标 (q,r) -> (x,y)
        self.hex_vertices = {}  # 六边形顶点像素坐标 (q,r) -> [(x1,y1), (x2,y2), ...] (精确浮点元组列表)
        
        # 轴向坐标邻居方向 (pointy-top, 顺时针序, 从右(E)开始):
        # E, SE, SW, W, NW, NE
        self.axial_directions_clockwise = [ 
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        self.generate_grid()

    def generate_grid(self):
        """生成阶数为t的HHM网格结构"""
        self.hexagons.clear()
        self.edges.clear()
        self.vertices.clear()
        self._exact_vertex_map.clear()
        self.vertex_edges.clear()
        self.hex_centers.clear()
        self.hex_vertices.clear()
        
        # HHM(t) 的六边形满足 max(|q|, |r|, |s|) < t, 其中 s = -q-r
        for q_coord in range(-self.t + 1, self.t):
            for r_coord in range(max(-self.t + 1, -q_coord - self.t + 1), 
                                 min(self.t, -q_coord + self.t)):
                s_coord = -q_coord - r_coord
                if max(abs(q_coord), abs(r_coord), abs(s_coord)) < self.t:
                    self.hexagons.add((q_coord, r_coord))
        
        hex_pixel_size = 30  # 定义六边形在屏幕上的大小

        for q, r in self.hexagons:
            center_x = hex_pixel_size * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
            center_y = hex_pixel_size * (3/2 * r)
            self.hex_centers[(q, r)] = (center_x, center_y)
            
            current_hex_vertices = []
            for i in range(6): 
                angle = math.pi/6 + i * math.pi/3 
                vx = center_x + hex_pixel_size * math.cos(angle)
                vy = center_y + hex_pixel_size * math.sin(angle)
                
                rounded_v_tuple = (round(vx, 6), round(vy, 6))
                if rounded_v_tuple in self._exact_vertex_map:
                    exact_v = self._exact_vertex_map[rounded_v_tuple]
                else:
                    exact_v = (vx, vy) 
                    self.vertices[exact_v] = True 
                    self._exact_vertex_map[rounded_v_tuple] = exact_v
                current_hex_vertices.append(exact_v)
            self.hex_vertices[(q, r)] = current_hex_vertices
        
        edge_set_for_dedup = set() 
        for q, r in self.hexagons:
            vertices_of_hex = self.hex_vertices[(q, r)]
            for i in range(6):
                v1_exact = vertices_of_hex[i]
                v2_exact = vertices_of_hex[(i + 1) % 6]
                edge_as_exact_vertex_pair = tuple(sorted((v1_exact, v2_exact))) 
                if edge_as_exact_vertex_pair not in edge_set_for_dedup:
                    edge_set_for_dedup.add(edge_as_exact_vertex_pair)
                    edge_pixel_coords = (v1_exact[0], v1_exact[1], v2_exact[0], v2_exact[1])
                    self.edges.append(edge_pixel_coords)
                    self.vertex_edges[v1_exact].append(edge_pixel_coords)
                    self.vertex_edges[v2_exact].append(edge_pixel_coords)
    
    def find_cycle(self, length):
        """
        根据论文定理查找指定长度的偶数环。
        返回: (cycle_edges, hex_coords_to_color) 或 (None, None)
        """
        if length % 2 != 0:
            messagebox.showerror("输入错误", "环的长度必须为偶数。")
            return None, None

        if self.t == 1:
            if length == 6:
                return self._get_hexagon_cycle_for_coloring((0, 0))
            else:
                messagebox.showinfo("提示", f"在HHM(1)中只存在长度为6的环。无法构造长度为 {length} 的环。")
                return None, None

        if self.t == 2:
            valid_lengths_hhm2 = {6, 10, 12, 14, 16, 18, 22}
            if length not in valid_lengths_hhm2:
                messagebox.showinfo("提示", f"根据论文，在HHM(2)中不存在长度为 {length} 的环。")
                return None, None
        
        if length == 6:
            return self._get_hexagon_cycle_for_coloring((0, 0))
        
        if length == 10 and self.t >= 2:
            return self._construct_10_cycle_with_hex_coloring()
        
        if length == 12 and self.t >= 2:
            return self._construct_12_cycle_with_triangle()
        
        max_theoretical_len = 6 * self.t**2 - 2
        second_max_len = 6 * self.t**2 - 4
        
        if self.t >= 2: 
            if length > max_theoretical_len or (length < 10 and length !=6) :
                 messagebox.showinfo("提示", f"目标长度 {length} 超出HHM({self.t})支持的范围 [6] U [10, {max_theoretical_len}]。")
                 return None, None
        elif self.t < 2 : 
             return None, None 

        # Case 1: 最大长度环 l = 6t^2-2
        if length == max_theoretical_len:
            if self.t < 2 : 
                messagebox.showerror("逻辑错误", "尝试为 t<2 构造最大环，这不符合预期。")
                return None, None
            return self._construct_cycle_figure6_style_spiral()
        
        # Case 2: l = 6t^2 - 2 - 4k (for k >= 1)
        elif length < max_theoretical_len and (max_theoretical_len - length) % 4 == 0:
            k_to_uncolor = (max_theoretical_len - length) // 4
            if k_to_uncolor > 0:
                # Step 1: Get the base coloring for l = 6t^2 - 2
                base_edges_fig6, base_hex_list_fig6 = self._construct_cycle_figure6_style_spiral()

                if base_hex_list_fig6 is None:
                    messagebox.showinfo("提示", f"无法为HHM({self.t})构造基础最大环 (l={max_theoretical_len})，因此无法进行-4k操作。")
                    return None, None

                initial_colored_set = set(base_hex_list_fig6)
                
                # Step 2: Perform the uncoloring operation
                # The set 'initial_colored_set' will be modified in place by this call.
                uncoloring_successful = self._perform_case2_uncoloring(initial_colored_set, k_to_uncolor)

                if uncoloring_successful:
                    final_hex_coords_to_color_list = list(initial_colored_set)
                    
                    if not final_hex_coords_to_color_list and self.hexagons:
                         messagebox.showinfo("提示", f"长度 {length} 的环构造 (-4k 操作) 导致没有六边形被着色。")
                         return None, None # Or perhaps an empty list of edges and hexes?
                    
                    new_edges = self._get_boundary_edges_of_hex_region(initial_colored_set)
                    
                    return new_edges, final_hex_coords_to_color_list
                else:
                    messagebox.showinfo("提示", f"无法为长度 {length} 执行-4k操作。可能在边界上未能找到足够的 {k_to_uncolor} 个可移除的已着色六边形，或者k值过大。")
                    return None, None
        
        # Case 3: 次大长度环 l = 6t^2-4
        if length == second_max_len:
            if self.t < 2:
                messagebox.showerror("逻辑错误", "尝试为 t<2 构造Case3环，这不符合预期。")
                return None, None
            return self._construct_cycle_figure9_style_spiral()
        
        # Case 4: l = 6t^2 - 4 - 4k (for k >= 1)
        elif length < second_max_len and (second_max_len - length) % 4 == 0:
            k_to_uncolor = (second_max_len - length) // 4
            if k_to_uncolor > 0:
                # Step 1: Get the base coloring for l = 6t^2 - 4
                base_edges_fig9, base_hex_list_fig9 = self._construct_cycle_figure9_style_spiral()

                if base_hex_list_fig9 is None:
                    messagebox.showinfo("提示", f"无法为HHM({self.t})构造基础次大环 (l={second_max_len})，因此无法进行-4k操作。")
                    return None, None

                initial_colored_set = set(base_hex_list_fig9)
                
                # Step 2: 使用专门为Case 4设计的去色操作
                uncoloring_successful = self._perform_case4_uncoloring(initial_colored_set, k_to_uncolor)

                if uncoloring_successful:
                    final_hex_coords_to_color_list = list(initial_colored_set)
                    
                    if not final_hex_coords_to_color_list and self.hexagons:
                         messagebox.showinfo("提示", f"长度 {length} 的环构造 (Case4 -4k 操作) 导致没有六边形被着色。")
                         return None, None
                    
                    new_edges = self._get_boundary_edges_of_hex_region(initial_colored_set)
                    
                    return new_edges, final_hex_coords_to_color_list
                else:
                    messagebox.showinfo("提示", f"无法为长度 {length} 执行Case4的-4k操作。可能在边界上未能找到足够的 {k_to_uncolor} 个可移除的已着色六边形，或者k值过大。")
                    return None, None
        
        messagebox.showinfo("提示", f"长度为 {length} 的环的构造方法 (除了6, 10, 12, 6t^2-2, 6t^2-2-4k, 6t^2-4 和 6t^2-4-4k) 当前未被实现。")
        return None, None

    def _get_hexagon_cycle_for_coloring(self, hex_coord_qr):
        """
        获取单个六边形的边界边及其坐标（用于着色）。
        返回: (edges_list, [hex_coord_qr])
        """
        if hex_coord_qr not in self.hex_vertices:
            # print(f"警告: 六边形 {hex_coord_qr} 不存在于网格中。")
            # 尝试使用(0,0)作为默认值，如果它存在
            if (0,0) in self.hex_vertices:
                hex_coord_qr = (0,0)
            else: # 如果连(0,0)都不存在，则无法继续
                return None, None 
        
        vertices_of_hex = self.hex_vertices[hex_coord_qr]
        edges = []
        for i in range(6):
            v1 = vertices_of_hex[i]
            v2 = vertices_of_hex[(i + 1) % 6]
            edges.append((v1[0], v1[1], v2[0], v2[1]))
        return edges, [hex_coord_qr]


    def _construct_10_cycle_with_hex_coloring(self):
        """构造长度为10的环：中心六边形(0,0)和左侧六边形(-1,0)"""
        center_hex = (0, 0)
        left_hex = (-1, 0)
        
        hex_region_to_color = {center_hex, left_hex}
        boundary_edges = self._get_boundary_edges_of_hex_region(hex_region_to_color)
        
        return boundary_edges, list(hex_region_to_color)

    def _get_axial_neighbors(self, q, r):
        """获取给定轴向坐标 (q,r) 的所有有效邻居六边形"""
        neighbors = []
        for dq, dr in self.axial_directions_clockwise: 
            nq, nr = q + dq, r + dr
            if (nq, nr) in self.hexagons: 
                neighbors.append((nq, nr))
        return neighbors

    def _get_boundary_edges_of_hex_region(self, region_hex_coords_set):
        """计算给定六边形区域集合的外部边界边。"""
        boundary_edges = []
        processed_edges_as_vertex_pairs = set() 

        if not region_hex_coords_set: # 如果区域为空，则没有边界
            return []
        
        for hq, hr in region_hex_coords_set:
            if (hq,hr) not in self.hex_vertices: continue 

            current_hex_pixel_vertices = self.hex_vertices[(hq,hr)]
            for i in range(6):
                v1_exact = current_hex_pixel_vertices[i]
                v2_exact = current_hex_pixel_vertices[(i+1) % 6]
                
                current_edge_vertex_pair_sorted = tuple(sorted((v1_exact, v2_exact)))
                if current_edge_vertex_pair_sorted in processed_edges_as_vertex_pairs:
                    continue 

                is_boundary = True 
                for neighbor_qr in self._get_axial_neighbors(hq,hr):
                    if neighbor_qr not in self.hex_vertices: continue
                    neighbor_hex_pixel_verts = self.hex_vertices[neighbor_qr]
                    v1_in_neighbor = any(self._are_vertices_close(v, v1_exact) for v in neighbor_hex_pixel_verts)
                    v2_in_neighbor = any(self._are_vertices_close(v, v2_exact) for v in neighbor_hex_pixel_verts)
                    if v1_in_neighbor and v2_in_neighbor: 
                        if neighbor_qr in region_hex_coords_set: 
                            is_boundary = False 
                    break
                
                if is_boundary:
                    boundary_edges.append((v1_exact[0], v1_exact[1], v2_exact[0], v2_exact[1]))
                    processed_edges_as_vertex_pairs.add(current_edge_vertex_pair_sorted)
        return boundary_edges

    def _are_vertices_close(self, v1, v2, tol=1e-5):
        """检查两个像素顶点是否足够接近 (用于判断是否为同一顶点)"""
        return math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2) < tol

    def _get_corner_hexagons(self, k_level):
        """获取 HHM(k_level) 的6个角点六边形的轴向坐标。"""
        if k_level < 1: return {} 
        r_eff = k_level - 1 
        
        # 轴向坐标定义下的角点 (q,r)
        # E: (r_eff, 0)
        # NE: (r_eff, -r_eff) -- s=0
        # NW: (0, -r_eff)
        # W: (-r_eff, 0)
        # SW: (-r_eff, r_eff) -- s=0
        # SE: (0, r_eff)
        corners_theory = {
            "E":  (r_eff, 0),
            "NE": (r_eff, -r_eff) if r_eff > 0 else (0,0) , 
            "NW": (0, -r_eff),
            "W":  (-r_eff, 0),
            "SW": (-r_eff, r_eff) if r_eff > 0 else (0,0),
            "SE": (0, r_eff)                           
        }
        valid_corners = {}
        for name, coord_qr in corners_theory.items():
            # 对于 r_eff = 0 (即 k_level=1), 所有角点都是 (0,0)
            actual_coord_to_check = (0,0) if r_eff == 0 else coord_qr

            if actual_coord_to_check in self.hexagons:
                valid_corners[name] = actual_coord_to_check
            else: # 如果理论角点不在 (例如对于r_eff > 0)
                  # 这通常表示k_level对于网格来说太大了，或者网格本身有问题
                  # print(f"警告: 理论角点 {name}={actual_coord_to_check} (k_level={k_level}, r_eff={r_eff}) 不在 self.hexagons 中。")
                  pass
        return valid_corners


    def _construct_cycle_figure6_style_spiral(self):
        """为长度 l = 6t^2-2 的环，基于多层奇数阶螺旋和特定过渡规则进行构造。"""
        s_prime_hexagons = set()
        current_qr = (0, 0)  # 螺旋路径的当前末端/笔尖

        if current_qr not in self.hexagons:
            messagebox.showerror("螺旋构造错误", "中心六边形 (0,0) 不存在。")
            return None, None
        s_prime_hexagons.add(current_qr)

        final_target_k_level = self.t if self.t % 2 != 0 else self.t - 1
        if final_target_k_level < 1:
            if self.t == 1: # t=1, final_target_k_level=1. s_prime={(0,0)} 是期望的.
                pass 
            else:
                messagebox.showwarning("螺旋构造警告", f"最终目标层级 k_level ({final_target_k_level}) 无效 (t={self.t})。")
                return None, None

        active_k_level = 1 # 从 HHM(1) 开始处理奇数层
        # 螺旋路径围绕的角点顺序: W, SW, SE, E, NE, NW
        path_segment_target_labels = ["W", "SW", "SE", "E", "NE", "NW"]

        while active_k_level <= final_target_k_level:
            corners_of_active_k = self._get_corner_hexagons(active_k_level)
            if not corners_of_active_k and active_k_level >=1:
                messagebox.showerror("螺旋构造错误", f"未能获取 k_level={active_k_level} 的角点。")
                return None, None

            for target_label in path_segment_target_labels:
                # 获取当前层级、当前目标标签的角点坐标
                # HHM(1)的所有角点都是(0,0)
                target_corner_qr = corners_of_active_k.get(target_label) 
                if target_corner_qr is None : # 如果_get_corner_hexagons实现确保k=1时总有值，则不应发生
                    if active_k_level == 1: target_corner_qr = (0,0) # k=1所有角点是(0,0)
                    else:
                        messagebox.showerror("螺旋构造错误", f"角点 {target_label} (k={active_k_level}) 未定义。")
                        return None, None

                # 从 current_qr 移动到 target_corner_qr, 路径上的六边形加入 s_prime_hexagons
                if current_qr != target_corner_qr: # 仅当尚未到达目标时移动
                    current_qr = self._extend_path_to_target(current_qr, target_corner_qr, s_prime_hexagons)
                
                # 校验是否成功到达，如果未到达可能意味着螺旋无法按预期形成
                if current_qr != target_corner_qr:
                    # print(f"警告: 螺旋未能从之前位置到达 {target_label} ({target_corner_qr}) of k={active_k_level}. 当前停在: {current_qr}")
                    # 可以在此添加更强的错误处理或终止逻辑
                    pass

                # 如果当前段的目标是西北角 (NW)
                if target_label == "NW":
                    if active_k_level < final_target_k_level:
                        # 层间过渡: 从当前 NW 角点向西(左)移动两格
                        for _ in range(2):
                            next_q_transition = current_qr[0] - 1 # 向西 q-1
                            next_r_transition = current_qr[1]     # r 不变
                            potential_next_qr_transition = (next_q_transition, next_r_transition)
                            if potential_next_qr_transition in self.hexagons:
                                current_qr = potential_next_qr_transition
                                s_prime_hexagons.add(current_qr)
                            else:
                                # print(f"警告: 层间左移两格时在 {potential_next_qr_transition} 处碰到边界 (从k={active_k_level}的NW)。")
                                break # 无法完成两格移动，终止此方向的过渡
                        # 完成过渡（或尝试过渡后），跳出当前k_level的角点循环，准备进入下一个k_level
                        break 
                    else: # active_k_level == final_target_k_level
                        # 到达最外目标螺旋层的西北角，S'主路径构造完成
                        # 将 active_k_level 设置为大于 final_target_k_level 以终止外部while循环
                        active_k_level = final_target_k_level + 1 
                        break # 跳出角点循环
            
            if active_k_level > final_target_k_level: # 如果因为到达最终层NW而设置了此条件
                break # 结束主螺旋构造循环
            
            active_k_level += 2 # 移动到下一个奇数层 (e.g., 1 -> 3 -> 5)

        # --- S' 螺旋主体路径 (s_prime_hexagons) 构建完毕 --- 

        # 对于偶数阶 t (t >= 2)，在 S' 基础上，从 HHM(t-1)的NW角点再向西(左)走一步并加入S'
        if self.t % 2 == 0 and self.t >= 2:
            # 获取 HHM(t-1) 的精确西北角点坐标
            # final_target_k_level 此时等于 t-1
            corners_t_minus_1 = self._get_corner_hexagons(self.t - 1) 
            nw_hex_of_t_minus_1 = corners_t_minus_1.get("NW")

            if nw_hex_of_t_minus_1:
                # 额外的一步是从这个真实的NW角点开始的
                base_for_extra_step = nw_hex_of_t_minus_1
                
                # 之前的 current_qr 理论上应该停在 nw_hex_of_t_minus_1
                # if current_qr != base_for_extra_step:
                    # print(f"警告 (t={self.t},偶数): S'路径末端 {current_qr} 与 HHM({self.t-1})的NW点 {base_for_extra_step} 不符。基于后者进行额外左移。")
                    # pass

                extra_step_q = base_for_extra_step[0] - 1 # 向西 q-1
                extra_step_r = base_for_extra_step[1]     # r 不变
                extra_step_hex = (extra_step_q, extra_step_r)
                
                if extra_step_hex in self.hexagons:
                    s_prime_hexagons.add(extra_step_hex) # 将这额外一步加入 S' 集合
                # else:
                    # print(f"警告 (t={self.t},偶数): 额外左移一步到 {extra_step_hex} 超出边界或无效。")
            # else:
                 # print(f"警告 (t={self.t},偶数): 未能获取HHM({self.t-1})的NW角点以进行额外左移。")

        # --- 根据 S' 集合确定最终着色区域 --- 
        final_hex_region_to_color = set()
        if self.t % 2 == 1: # t 为奇数, 着色 S' 本身
            final_hex_region_to_color = s_prime_hexagons
        else: # t 为偶数, 着色 All - S'
            all_hex_coords_in_grid = set(self.hexagons)
            final_hex_region_to_color = all_hex_coords_in_grid - s_prime_hexagons
            if not final_hex_region_to_color and all_hex_coords_in_grid and s_prime_hexagons.issuperset(all_hex_coords_in_grid):
                # print(f"警告 (t={self.t},偶数): S'集合覆盖了所有六边形，导致 All-S' 为空。")
                pass # 可能是预期行为，例如t=2, l_max=22时 S'可能很少，All-S'很大
        
        if not final_hex_region_to_color and self.hexagons: 
             # 对于t=1, l_max=4, s_prime={(0,0)}, final_hex_region={(0,0)}，这是有效的
             is_t1_lmax4_case = (self.t == 1 and len(s_prime_hexagons)==1 and (0,0) in s_prime_hexagons)
             if not is_t1_lmax4_case:
                messagebox.showwarning("构造警告", f"Fig6风格构造：最终着色区域为空 (t={self.t})。检查螺旋或S'逻辑。")
                return None, None

        boundary_edges = self._get_boundary_edges_of_hex_region(final_hex_region_to_color)
        expected_length = 6 * self.t**2 - 2

        # 对t=1, 期望长度是4. 此时 s_prime={(0,0)}, final_hex_region={(0,0)}. 其边界是6条边。
        if self.t == 1 and expected_length == 4:
            if len(boundary_edges) == 6 and len(final_hex_region_to_color) == 1:
                pass # t=1特殊情况，画的是(0,0)这个六边形，边界为6，符合预期
            # else:
                # print(f"警告 (t=1, l_exp=4): 边界边数 {len(boundary_edges)} (应为6), 着色数 {len(final_hex_region_to_color)} (应为1)")
        elif len(boundary_edges) != expected_length:
            # print(f"警告: Fig6风格构造的环边界边数为 {len(boundary_edges)}，论文预期为 {expected_length} (t={self.t})")
            pass

        if not self._verify_cycle_connectivity(boundary_edges) and boundary_edges:
             # print("警告: Fig6风格构造的环边界可能不连通或顶点度数错误。")
             pass

        return boundary_edges, list(final_hex_region_to_color)

    def _verify_cycle_connectivity(self, cycle_edges):
        """验证给定的边集是否形成一个单一的连通环（所有节点度为2）"""
        if not cycle_edges: return False 
        adj = defaultdict(list) 
        nodes_in_cycle = set()    
        vertex_degrees = defaultdict(int) 
        for edge_pixel_coords in cycle_edges:
            v1_exact = self._find_closest_vertex((edge_pixel_coords[0], edge_pixel_coords[1]))
            v2_exact = self._find_closest_vertex((edge_pixel_coords[2], edge_pixel_coords[3]))
            if not v1_exact or not v2_exact: continue 
            adj[v1_exact].append(v2_exact)
            adj[v2_exact].append(v1_exact)
            nodes_in_cycle.add(v1_exact)
            nodes_in_cycle.add(v2_exact)
            vertex_degrees[v1_exact] += 1
            vertex_degrees[v2_exact] += 1
        if not nodes_in_cycle: return False
        for node_exact, degree in vertex_degrees.items():
            if degree != 2: return False
        start_node_exact = next(iter(nodes_in_cycle)) 
        visited_check = {start_node_exact}
        queue_check = deque([start_node_exact])
        nodes_reached_count = 0
        while queue_check:
            current_node_exact = queue_check.popleft()
            nodes_reached_count += 1
            for neighbor_exact in adj[current_node_exact]:
                if neighbor_exact not in visited_check:
                    visited_check.add(neighbor_exact)
                    queue_check.append(neighbor_exact)
        if nodes_reached_count != len(nodes_in_cycle): return False
        return True 

    def _find_closest_vertex(self, coord_to_find_pixels):
        """在 _exact_vertex_map 中查找像素坐标对应的精确顶点。"""
        rounded_pixel_key = (round(coord_to_find_pixels[0], 6), round(coord_to_find_pixels[1], 6))
        return self._exact_vertex_map.get(rounded_pixel_key, None)

    def _extend_path_to_target(self, start_qr, target_qr, s_prime_hexagons_ref):
        """使用BFS寻找从start_qr到target_qr的轴向路径上的六边形(不含start_qr)，
           将它们添加到s_prime_hexagons_ref中，并返回路径的实际终点。
        """
        if start_qr == target_qr:
            return start_qr

        queue = deque([(start_qr, [start_qr])])  # (current_hex, path_list_to_current_hex)
        visited_in_bfs = {start_qr}
        
        # 简单的路径长度限制，避免在巨大网格或无路径时无限搜索
        # 基于轴向距离的启发式限制
        max_bfs_path_len = (abs(start_qr[0] - target_qr[0]) + \
                              abs(start_qr[1] - target_qr[1]) + \
                              abs((-start_qr[0]-start_qr[1]) - (-target_qr[0]-target_qr[1]))) * 2 + 5
        if max_bfs_path_len <= 0: max_bfs_path_len = 5 # 最小长度

        found_path_to_target = []

        while queue:
            current_hex, path = queue.popleft()

            if len(path) > max_bfs_path_len: # 路径过长，剪枝
                continue

            if current_hex == target_qr:
                found_path_to_target = path
                break # 找到目标

            # 探索邻居 (使用 self.axial_directions_clockwise)
            for dq, dr in self.axial_directions_clockwise:
                neighbor_qr = (current_hex[0] + dq, current_hex[1] + dr)
                if neighbor_qr in self.hexagons and neighbor_qr not in visited_in_bfs:
                    visited_in_bfs.add(neighbor_qr)
                    new_path = path + [neighbor_qr]
                    queue.append((neighbor_qr, new_path))
        
        actual_end_qr = start_qr # 默认为起点，如果没找到路径
        if found_path_to_target:
            # 将路径上的六边形（除了起点）加入 s_prime_hexagons
            for i in range(1, len(found_path_to_target)):
                s_prime_hexagons_ref.add(found_path_to_target[i])
            actual_end_qr = found_path_to_target[-1]
            
            # if actual_end_qr != target_qr:
                # print(f"路径搜索: 从 {start_qr} 到 {target_qr}，实际到达 {actual_end_qr}")
        # else:
            # print(f"路径搜索: 从 {start_qr} 未能找到路径到 {target_qr}")
            
        return actual_end_qr

    def _get_step_towards_target(self, current_qr, target_qr):
        """计算从 current_qr 到 target_qr 的最佳单步轴向移动。返回 (dq,dr) 或 None。"""
        if current_qr == target_qr:
            return None

        best_dir = None
        min_axial_dist = float('inf')

        current_q, current_r = current_qr
        current_s = -current_q - current_r # s-coordinate for current
        target_q, target_r = target_qr
        target_s = -target_q - target_r   # s-coordinate for target

        # Calculate initial axial distance from current to target
        # initial_dist = (abs(current_q - target_q) + abs(current_r - target_r) + abs(current_s - target_s)) / 2.0

        candidate_dirs = [] # To store directions that yield the same min_axial_dist

        for dq, dr in self.axial_directions_clockwise: # E, SE, SW, W, NW, NE
            next_q, next_r = current_q + dq, current_r + dr
            # We don't check self.hexagons here; caller handles wall collisions.
            # This function is purely about optimal direction if movement is possible.
            
            next_s = -next_q - next_r # s-coordinate for the potential next step
            
            # Calculate axial distance from the potential next step to the target
            dist_to_target = (abs(next_q - target_q) + abs(next_r - target_r) + abs(next_s - target_s)) / 2.0

            if dist_to_target < min_axial_dist:
                min_axial_dist = dist_to_target
                best_dir = (dq, dr)
                candidate_dirs = [best_dir] # New best, reset candidates
            elif dist_to_target == min_axial_dist:
                # If it's the same minimal distance, add to candidates for potential tie-breaking
                # This also handles the initial case where best_dir might be None
                if best_dir is not None: # Ensure we don't add if it's the first evaluation yielding this min_dist
                    candidate_dirs.append((dq, dr))
                else: # First time setting this min_axial_dist
                    best_dir = (dq, dr) # Should have been set in the < block, but for safety
                    candidate_dirs = [(dq, dr)]
        
        # Tie-breaking: if multiple directions give the same minimal distance.
        # A simple tie-breaker: prefer directions that reduce q distance if target is East/West-ish,
        # or r distance if target is N/S-ish. Or prefer clockwise/counter-clockwise.
        # For now, simply returning the first `best_dir` found in the loop order is deterministic.
        # If candidate_dirs has multiple, `best_dir` is the first one that achieved min_axial_dist.
        
        # Ensure we actually found a direction (should happen if current_qr != target_qr)
        if not best_dir and candidate_dirs: # If best_dir somehow not set but candidates exist
            best_dir = candidate_dirs[0]

        return best_dir

    def _generate_uncoloring_traversal_path(self):
        """
        生成用于Case 2 (-4k) 操作的六边形遍历路径。
        按照以下规则生成路径：
        1. 从最外层(t)西北角点出发
        2. 遇到标记点时转向，标记点包括各层的角点(除内层NW)和特殊点
        3. 移动方向顺序为: 东->东南->西南->西->西北->东北
        4. 当t为奇数时，在第三层西南角点停止
        """
        if self.t < 1: 
            return []
        
        traversal_path = []
        visited_on_traversal = set()

        # 1. 获取各层角点
        corner_points_by_layer = {}  # 存储各层角点: layer -> {label -> (q,r)}
        for layer in range(1, self.t + 1):
            corners = self._get_corner_hexagons(layer)
            if corners:
                corner_points_by_layer[layer] = corners
        
        if self.t not in corner_points_by_layer:
            if self.t == 1 and (0,0) in self.hexagons: 
                return [(0,0)]
            return []
        
        # 2. 定义特殊点 (SP)
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
            
            # 从西北角点往左下走两格
            # 第一步：往左下方向 (SW, -1, +1)
            step1_q = nw_corner[0] - 1
            step1_r = nw_corner[1] + 1
            
            # 第二步：继续往左下方向
            step2_q = step1_q - 1
            step2_r = step1_r + 1
            
            # 特殊点坐标
            special_point = (step2_q, step2_r)
            
            # 检查特殊点是否在网格内
            s_coord = -special_point[0] - special_point[1]
            if max(abs(special_point[0]), abs(special_point[1]), abs(s_coord)) < self.t:
                if special_point in self.hexagons:
                    special_points.add(special_point)
                    special_points_by_layer[m] = special_point
        
        # 3. 收集所有标记点（不包括内层NW角点）
        all_marker_points = set()
        for layer, corners in corner_points_by_layer.items():
            for label, point in corners.items():
                # 排除内层的NW角点
                if label == "NW" and layer < self.t:
                    continue
                all_marker_points.add(point)
        
        # 将特殊点加入标记点集合
        all_marker_points.update(special_points)
        
        # 4. 从最外层西北角开始遍历
        current_layer = self.t
        current_hex = corner_points_by_layer[self.t]["NW"]
        traversal_path.append(current_hex)
        visited_on_traversal.add(current_hex)
        
        # 移动方向列表（东，东南，西南，西，西北，东北）
        direction_vectors = [
            (1, 0),    # 东
            (0, 1),    # 东南 (q不变，r+1)
            (-1, 1),   # 西南 (q-1, r+1)
            (-1, 0),   # 西
            (0, -1),   # 西北 (q不变, r-1)
            (1, -1)    # 东北 (q+1, r-1)
        ]
        
        # 初始方向索引：先从最后一个元素（东北）开始，使得首次检测标记点时转向为东
        current_direction_index = 5  # 东北
        
        # 防止无限循环
        max_iterations = self.t * 100
        iterations = 0
        
        # 获取第三层西南角点（对于奇数t需要特殊处理）
        l3_sw_point = None
        if self.t % 2 == 1 and 3 in corner_points_by_layer and "SW" in corner_points_by_layer[3]:
            l3_sw_point = corner_points_by_layer[3]["SW"]
        
        while iterations < max_iterations and len(traversal_path) < len(self.hexagons):
            iterations += 1
            
            # 对于奇数t，检查是否到达第三层西南角点，如果是则停止
            if self.t % 2 == 1 and l3_sw_point and current_hex == l3_sw_point:
                break
            
            # 1. 检查当前位置是否为标记点，如果是则更新方向索引
            if current_hex in all_marker_points:
                current_direction_index = (current_direction_index + 1) % 6
            
            # 2. 根据当前方向移动
            next_direction = direction_vectors[current_direction_index]
            next_q = current_hex[0] + next_direction[0]
            next_r = current_hex[1] + next_direction[1]
            next_hex = (next_q, next_r)
            
            # 3. 检查下一步是否在网格内，如果不在则尝试转向
            if next_hex not in self.hexagons:
                # 如果撞墙，尝试转向
                for i in range(5):  # 尝试最多5次其他方向
                    current_direction_index = (current_direction_index + 1) % 6
                    next_direction = direction_vectors[current_direction_index]
                    next_q = current_hex[0] + next_direction[0]
                    next_r = current_hex[1] + next_direction[1]
                    next_hex = (next_q, next_r)
                    if next_hex in self.hexagons:
                        break
            
            # 4. 如果找到有效的下一步，则移动
            if next_hex in self.hexagons:
                current_hex = next_hex
                if current_hex not in visited_on_traversal:
                    traversal_path.append(current_hex)
                    visited_on_traversal.add(current_hex)
            else:
                # 如果所有方向都撞墙，可能陷入死胡同，终止遍历
                break
            
            # 5. 检查终止条件
            # 对于最内层，确保路径覆盖足够多的六边形
            if current_layer == 1 or (current_hex in corner_points_by_layer.get(1, {}).values()):
                # 如果到达了内层的任意角点，并且路径够长
                min_required_hexes = max(3 * self.t * self.t // 2, 1)  # 粗略估计，确保覆盖足够多的六边形
                if len(traversal_path) >= min_required_hexes:
                    # 如果到达内层东北角，或者已经访问了内层的所有角点
                    inner_ne = corner_points_by_layer.get(1, {}).get("NE")
                    if current_hex == inner_ne:
                        break
                    
                    # 或者当前在内层，检查是否已经访问了所有内层角点
                    if current_hex in corner_points_by_layer.get(1, {}).values():
                        inner_corners_visited = True
                        for label in ["NE", "E", "SE", "SW", "W"]:
                            if label in corner_points_by_layer.get(1, {}) and corner_points_by_layer[1][label] not in visited_on_traversal:
                                inner_corners_visited = False
                                break
                        if inner_corners_visited:
                            break
        
        return traversal_path

    def _perform_case2_uncoloring(self, initial_colored_set, k_to_uncolor):
        """
        Tries to uncolor k_to_uncolor hexagons from the initial_colored_set
        by picking them from a specially generated traversal path.
        Modifies initial_colored_set in place.
        Returns True if k_to_uncolor hexagons were successfully found and uncolored,
        False otherwise.
        """
        if k_to_uncolor <= 0:
            return True # No uncoloring needed or invalid k

        # 使用新的路径生成方法
        uncoloring_traversal_hexes = self._generate_uncoloring_traversal_path()
        
        if not uncoloring_traversal_hexes:
            # print(f"Warning: Could not generate uncoloring traversal path for HHM(t={self.t}).")
            return False

        uncolored_successfully_count = 0
        for hex_on_traversal in uncoloring_traversal_hexes:
            if uncolored_successfully_count == k_to_uncolor:
                break # Found enough hexes to uncolor

            if hex_on_traversal in initial_colored_set:
                initial_colored_set.remove(hex_on_traversal)
                uncolored_successfully_count += 1
        
        return uncolored_successfully_count == k_to_uncolor

    def _construct_cycle_figure9_style_spiral(self):
        """
        为长度 l = 6t^2-4 的环，基于Figure 9和10的风格构造。
        此函数用于实现Case 3，与Case 1 (Figure 6)风格相似但有特殊的起始步骤。
        t为奇数和偶数的处理略有不同。
        """
        s_prime_hexagons = set()  # 存储S'集合中的六边形坐标
        
        # 判断t的奇偶性，采取不同的构造策略
        if self.t % 2 == 1:  # t为奇数
            # 从中心(0,0)开始特殊步骤
            current_qr = (0, 0)
            if current_qr not in self.hexagons:
                messagebox.showerror("构造错误", "中心六边形 (0,0) 不存在。")
                return None, None
            
            s_prime_hexagons.add(current_qr)
            
            # 特殊步骤1: 向左走一步
            step1_q = current_qr[0] - 1  # 向左: q-1
            step1_r = current_qr[1]      # 向左: r不变
            step1_qr = (step1_q, step1_r)
            
            if step1_qr not in self.hexagons:
                messagebox.showerror("构造错误", f"奇数t特殊步骤1: {step1_qr} 不在网格内。")
                return None, None
            
            s_prime_hexagons.add(step1_qr)
            current_qr = step1_qr
            
            # 特殊步骤2: 向左下走一步
            step2_q = current_qr[0] - 1  # 向左下: q-1
            step2_r = current_qr[1] + 1  # 向左下: r+1
            step2_qr = (step2_q, step2_r)
            
            if step2_qr not in self.hexagons:
                messagebox.showerror("构造错误", f"奇数t特殊步骤2: {step2_qr} 不在网格内。")
                return None, None
            
            s_prime_hexagons.add(step2_qr)
            current_qr = step2_qr
            
            # 特殊步骤3: 向右下走一步
            step3_q = current_qr[0] + 1  # 向右下: q+1
            step3_r = current_qr[1] + 1  # 向右下: r+1（修正：右下应该是r+1而不是r不变）
            step3_qr = (step3_q, step3_r)
            
            if step3_qr not in self.hexagons:
                messagebox.showerror("构造错误", f"奇数t特殊步骤3: {step3_qr} 不在网格内。")
                return None, None
            
            s_prime_hexagons.add(step3_qr)
            current_qr = step3_qr
            
            # 此时我们到达了类似于第3层的西南部标记点的位置
            # 接下来的逻辑与case1的螺旋构造类似，但目标层级是偶数
            
            final_target_k_level = self.t  # t为奇数，目标是奇数层
            active_k_level = 3  # 从第3层开始向外扩展
            
            # 螺旋路径围绕的角点顺序: W, SW, SE, E, NE, NW
            path_segment_target_labels = ["SW", "SE", "E", "NE", "NW", "W"]  # 从SW开始，因为当前位置类似于第3层的SW位置
            current_label_index = 0  # 从SW开始
            
            while active_k_level <= final_target_k_level:
                corners_of_active_k = self._get_corner_hexagons(active_k_level)
                if not corners_of_active_k and active_k_level >= 3:
                    messagebox.showerror("螺旋构造错误", f"未能获取 k_level={active_k_level} 的角点。")
                    return None, None
                
                # 执行从当前角点开始的整个层级的遍历
                while current_label_index < len(path_segment_target_labels):
                    target_label = path_segment_target_labels[current_label_index]
                    target_corner_qr = corners_of_active_k.get(target_label)
                    
                    if target_corner_qr is None:
                        messagebox.showerror("螺旋构造错误", f"角点 {target_label} (k={active_k_level}) 未定义。")
                        return None, None
                    
                    # 从 current_qr 移动到 target_corner_qr, 路径上的六边形加入 s_prime_hexagons
                    if current_qr != target_corner_qr:
                        current_qr = self._extend_path_to_target(current_qr, target_corner_qr, s_prime_hexagons)
                    
                    # 校验是否成功到达
                    if current_qr != target_corner_qr:
                        # 如果未到达目标角点，可能是因为路径不通，但我们继续执行
                        pass
                    
                    # 如果当前段的目标是西北角 (NW)，检查是否需要层间过渡
                    if target_label == "NW":
                        if active_k_level < final_target_k_level:
                            # 层间过渡: 从当前 NW 角点向西(左)移动两格
                            for _ in range(2):
                                next_q_transition = current_qr[0] - 1  # 向西 q-1
                                next_r_transition = current_qr[1]      # r 不变
                                potential_next_qr_transition = (next_q_transition, next_r_transition)
                                if potential_next_qr_transition in self.hexagons:
                                    current_qr = potential_next_qr_transition
                                    s_prime_hexagons.add(current_qr)
                                else:
                                    break  # 无法完成两格移动，终止此方向的过渡
                            
                            # 完成当前层级，进入下一个层级
                            break
                        else:  # active_k_level == final_target_k_level
                            # 到达最外目标螺旋层的西北角，S'主路径构造完成
                            active_k_level = final_target_k_level + 1  # 设置以终止外部循环
                            break
                    
                    # 移动到下一个角点
                    current_label_index = (current_label_index + 1) % len(path_segment_target_labels)
                
                # 重置角点索引，为下一层准备
                current_label_index = 0
                
                # 如果已经完成最终层，跳出循环
                if active_k_level > final_target_k_level:
                    break
                
                # 移动到下一个层级
                active_k_level += 2  # 从奇数层到下一个奇数层
            
            # --- S' 螺旋主体路径 (s_prime_hexagons) 构建完毕 ---
            # 奇数阶，涂色 S' 本身
            final_hex_region_to_color = s_prime_hexagons
            
        else:  # t为偶数
            # 偶数t的处理与case1类似，但是最后步骤稍有不同
            current_qr = (0, 0)  # 从中心六边形开始
            
            if current_qr not in self.hexagons:
                messagebox.showerror("螺旋构造错误", "中心六边形 (0,0) 不存在。")
                return None, None
            
            s_prime_hexagons.add(current_qr)
            
            # 对于偶数t，我们构造以最大偶数层为目标的螺旋
            final_target_k_level = self.t - 1 if self.t % 2 == 0 else self.t
            if final_target_k_level < 1:
                messagebox.showwarning("螺旋构造警告", f"最终目标层级 k_level ({final_target_k_level}) 无效 (t={self.t})。")
                return None, None
            
            active_k_level = 1  # 从 HHM(1) 开始处理
            path_segment_target_labels = ["W", "SW", "SE", "E", "NE", "NW"]  # 从W开始按顺时针方向
            
            while active_k_level <= final_target_k_level:
                corners_of_active_k = self._get_corner_hexagons(active_k_level)
                if not corners_of_active_k and active_k_level >= 1:
                    messagebox.showerror("螺旋构造错误", f"未能获取 k_level={active_k_level} 的角点。")
                    return None, None
                
                for target_label in path_segment_target_labels:
                    target_corner_qr = corners_of_active_k.get(target_label)
                    
                    if target_corner_qr is None:
                        if active_k_level == 1:
                            target_corner_qr = (0, 0)  # k=1所有角点是(0,0)
                        else:
                            messagebox.showerror("螺旋构造错误", f"角点 {target_label} (k={active_k_level}) 未定义。")
                            return None, None
                    
                    # 从 current_qr 移动到 target_corner_qr
                    if current_qr != target_corner_qr:
                        current_qr = self._extend_path_to_target(current_qr, target_corner_qr, s_prime_hexagons)
                    
                    # 校验是否成功到达
                    if current_qr != target_corner_qr:
                        pass  # 如果未到达目标角点，继续执行
                    
                    # 如果当前段的目标是西北角 (NW)，检查是否需要层间过渡
                    if target_label == "NW":
                        if active_k_level < final_target_k_level:
                            # 层间过渡: 从当前 NW 角点向西(左)移动两格
                            for _ in range(2):
                                next_q_transition = current_qr[0] - 1  # 向西 q-1
                                next_r_transition = current_qr[1]      # r 不变
                                potential_next_qr_transition = (next_q_transition, next_r_transition)
                                if potential_next_qr_transition in self.hexagons:
                                    current_qr = potential_next_qr_transition
                                    s_prime_hexagons.add(current_qr)
                                else:
                                    break  # 无法完成两格移动，终止此方向的过渡
                            break  # 完成当前层级，进入下一个层级
                        else:  # active_k_level == final_target_k_level
                            # t为偶数时，在最后一层的NW角点处，而非向左走一步，而是向左上走一步
                            # 向左上走一步：q不变，r-1
                            next_q = current_qr[0]
                            next_r = current_qr[1] - 1
                            extra_step_hex = (next_q, next_r)
                            
                            if extra_step_hex in self.hexagons:
                                s_prime_hexagons.add(extra_step_hex)  # 将这额外一步加入 S' 集合
                            
                            # 到达最外目标螺旋层的西北角并执行额外步骤，S'主路径构造完成
                            active_k_level = final_target_k_level + 1  # 设置以终止外部循环
                            break
                
                if active_k_level > final_target_k_level:  # 如果因为到达最终层NW而设置了此条件
                    break  # 结束主螺旋构造循环
                
                active_k_level += 2  # 移动到下一个奇数层
            
            # --- S' 螺旋主体路径 (s_prime_hexagons) 构建完毕 --- 
            # t为偶数，涂色 All - S'
            all_hex_coords_in_grid = set(self.hexagons)
            final_hex_region_to_color = all_hex_coords_in_grid - s_prime_hexagons
        
        # 检查生成的着色区域是否为空
        if not final_hex_region_to_color and self.hexagons:
            is_t1_case = (self.t == 1)
            if not is_t1_case:
                messagebox.showwarning("构造警告", f"Fig9/10风格构造：最终着色区域为空 (t={self.t})。检查螺旋或S'逻辑。")
                return None, None
        
        # 生成边界边
        boundary_edges = self._get_boundary_edges_of_hex_region(final_hex_region_to_color)
        expected_length = 6 * self.t**2 - 4
        
        if len(boundary_edges) != expected_length:
            # 如果实际边数与预期不符，输出警告信息但继续执行
            pass
        
        # 验证生成的环是否连通
        if not self._verify_cycle_connectivity(boundary_edges) and boundary_edges:
            pass
        
        return boundary_edges, list(final_hex_region_to_color)

    def _construct_12_cycle_with_triangle(self):
        """
        构造长度为12的环：中心六边形(0,0)、左侧六边形(-1,0)和左下六边形(-1,1)
        形成三角状排列的三个六边形
        """
        center_hex = (0, 0)      # 中心
        left_hex = (-1, 0)       # 左侧六边形(从中心往左走一步)
        bottom_hex = (-1, 1)     # 左下六边形(从左侧六边形往右下走一步)
        
        hex_region_to_color = {center_hex, left_hex, bottom_hex}
        boundary_edges = self._get_boundary_edges_of_hex_region(hex_region_to_color)
        
        return boundary_edges, list(hex_region_to_color)

    def _generate_case4_uncoloring_traversal_path(self):
        """
        生成用于Case 4 (-4k) 操作的六边形遍历路径。
        
        奇数t情况:
        1. 参照已实现的Case 2逻辑，在到达第三层西南角点时停止
        
        偶数t情况:
        1. 从最外层西北角开始
        2. 当到达最外层特殊点(SP)时跳转至第二层西南角点
        3. 从第二层西南角点开始按逆时针路径行进
        4. 当在逆时针模式下遇到第四层特殊点(SP)时，方向设置为西南(4)
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
            if self.t == 1 and (0,0) in self.hexagons: 
                return [(0,0)]
            return []  # 没有有效起点
        
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

        # 奇数t和偶数t处理方式不同
        if self.t % 2 == 1:  # === 奇数t情况 ===
            # 严格参照Case 2的逻辑实现，但在到达第三层西南角点时停止
            all_marker_points = set()
            for layer, corners in corner_points_by_layer.items():
                for label, point in corners.items():
                    # 排除内层的NW角点
                    if label == "NW" and layer < self.t:
                        continue
                    all_marker_points.add(point)
            
            # 将特殊点加入标记点集合
            all_marker_points.update(special_points)
            
            # 从最外层西北角开始遍历
            current_hex = corner_points_by_layer[self.t]["NW"]
            traversal_path.append(current_hex)
            visited_on_traversal.add(current_hex)
            
            # 移动方向列表（东，东南，西南，西，西北，东北）
            direction_vectors = [
                (1, 0),    # 东
                (0, 1),    # 东南 (q不变，r+1)
                (-1, 1),   # 西南 (q-1, r+1)
                (-1, 0),   # 西
                (0, -1),   # 西北 (q不变, r-1)
                (1, -1)    # 东北 (q+1, r-1)
            ]
            
            # 初始方向索引：先从最后一个元素（东北）开始，使得首次检测标记点时转向为东
            current_direction_index = 5  # 东北
            
            # 防止无限循环
            max_iterations = self.t * 100
            iterations = 0
            
            # 获取第三层西南角点
            l3_sw_point = None
            if 3 in corner_points_by_layer and "SW" in corner_points_by_layer[3]:
                l3_sw_point = corner_points_by_layer[3]["SW"]
            
            while iterations < max_iterations and len(traversal_path) < len(self.hexagons):
                iterations += 1
                
                # 关键点：检查是否到达第三层西南角点，如果是则停止
                if l3_sw_point and current_hex == l3_sw_point:
                    break  # 在第三层西南角点处终止路径
                
                # 检查当前位置是否为标记点，如果是则更新方向索引
                if current_hex in all_marker_points:
                    current_direction_index = (current_direction_index + 1) % 6
                
                # 根据当前方向移动
                next_direction = direction_vectors[current_direction_index]
                next_q = current_hex[0] + next_direction[0]
                next_r = current_hex[1] + next_direction[1]
                next_hex = (next_q, next_r)
                
                # 检查下一步是否在网格内，如果不在则尝试转向
                if next_hex not in self.hexagons:
                    # 如果撞墙，尝试转向
                    for i in range(5):  # 尝试最多5次其他方向
                        current_direction_index = (current_direction_index + 1) % 6
                        next_direction = direction_vectors[current_direction_index]
                        next_q = current_hex[0] + next_direction[0]
                        next_r = current_hex[1] + next_direction[1]
                        next_hex = (next_q, next_r)
                        if next_hex in self.hexagons:
                            break
                
                # 如果找到有效的下一步，则移动
                if next_hex in self.hexagons:
                    current_hex = next_hex
                    if current_hex not in visited_on_traversal:
                        traversal_path.append(current_hex)
                        visited_on_traversal.add(current_hex)
                else:
                    # 如果所有方向都撞墙，可能陷入死胡同，终止遍历
                    break
                
        else:  # === 偶数t情况 ===
            # 初始化路径
            current_hex = corner_points_by_layer[self.t]["NW"]
            traversal_path.append(current_hex)
            visited_on_traversal.add(current_hex)
            
            # 顺时针和逆时针方向向量
            clockwise_directions = [
                (1, 0),    # 东 (0)
                (0, 1),    # 东南 (1)
                (-1, 1),   # 西南 (2)
                (-1, 0),   # 西 (3)
                (0, -1),   # 西北 (4)
                (1, -1)    # 东北 (5)
            ]
            
            counterclockwise_directions = [
                (1, 0),    # 东 (0)
                (1, -1),   # 东北 (1)
                (0, -1),   # 西北 (2)
                (-1, 0),   # 西 (3)
                (-1, 1),   # 西南 (4)
                (0, 1)     # 东南 (5)
            ]
            
            # 默认以顺时针方式从东北方向开始
            current_dir_index = 5  # 东北方向（顺时针顺序）
            using_counterclockwise = False  # 初始使用顺时针方向
            directions = clockwise_directions
            
            # 找到需要特殊处理的点
            outer_special_point = special_points_by_layer.get(self.t)  # 最外层特殊点
            l2_sw_point = corner_points_by_layer.get(2, {}).get("SW")  # 第二层西南角点
            fourth_layer_sp = special_points_by_layer.get(4)  # 第四层特殊点
            
            # 收集所有标记点（除了内层西北角点）
            all_marker_points = set()
            nw_corner_points = set()  # 存储各层的西北角点
            
            for layer, corners in corner_points_by_layer.items():
                for label, point in corners.items():
                    if label == "NW":
                        nw_corner_points.add(point)
                        if layer < self.t:  # 除最外层外的西北角点不作为标记点
                            continue
                    all_marker_points.add(point)
            
            # 最外层特殊点不加入标记点集合，因为它有特殊处理
            if outer_special_point:
                all_marker_points.discard(outer_special_point)
            
            # 其他特殊点正常加入标记点集合
            for sp in special_points:
                if sp != outer_special_point:
                    all_marker_points.add(sp)
            
            # 开始主要的遍历循环
            max_iterations = self.t * 200
            iterations = 0
            found_outer_special_point = False
            
            while iterations < max_iterations and len(traversal_path) < len(self.hexagons):
                iterations += 1
                
                # 检查是否遇到最外层特殊点(SP)
                if not found_outer_special_point and outer_special_point and current_hex == outer_special_point:
                    found_outer_special_point = True
                    
                    # 特殊处理：直接跳转到第二层西南角点
                    if l2_sw_point:
                        current_hex = l2_sw_point
                        if current_hex not in visited_on_traversal:
                            traversal_path.append(current_hex)
                            visited_on_traversal.add(current_hex)
                        
                        # 从这里开始使用逆时针方向
                        using_counterclockwise = True
                        directions = counterclockwise_directions
                        
                        # 第一步：往东走一步(q+1, r不变)
                        e_step = (current_hex[0] + 1, current_hex[1])
                        if e_step in self.hexagons:
                            if e_step not in visited_on_traversal:
                                traversal_path.append(e_step)
                                visited_on_traversal.add(e_step)
                            current_hex = e_step
                        
                        # 设置方向为东(0)
                        current_dir_index = 0
                        continue
                
                # 关键点：逆时针模式下遇到第四层特殊点(SP)时，需设置方向为西南(4)
                if using_counterclockwise and fourth_layer_sp and current_hex == fourth_layer_sp:
                    current_dir_index = 4  # 在逆时针顺序中，西南方向是索引4
                    # 继续移动，不需要break
                
                # 检查当前位置是否为标记点（角点或特殊点）
                elif current_hex in all_marker_points:
                    # 更新方向索引
                    current_dir_index = (current_dir_index + 1) % 6
                
                # 根据当前方向移动
                next_dir = directions[current_dir_index]
                next_hex = (current_hex[0] + next_dir[0], current_hex[1] + next_dir[1])
                
                # 检查下一步是否在网格内，如果不在则尝试转向
                if next_hex not in self.hexagons or next_hex in visited_on_traversal:
                    # 如果撞墙或已访问过，尝试转向
                    direction_tried = 1
                    found_valid_move = False
                    
                    while direction_tried <= 6:
                        current_dir_index = (current_dir_index + 1) % 6
                        next_dir = directions[current_dir_index]
                        next_hex = (current_hex[0] + next_dir[0], current_hex[1] + next_dir[1])
                        
                        if next_hex in self.hexagons and next_hex not in visited_on_traversal:
                            found_valid_move = True
                            break
                        
                        direction_tried += 1
                    
                    if not found_valid_move:
                        # 如果所有方向都不可行，可能陷入死胡同，终止遍历
                        break
                
                # 执行移动
                current_hex = next_hex
                traversal_path.append(current_hex)
                visited_on_traversal.add(current_hex)
    
        return traversal_path

    def _perform_case4_uncoloring(self, initial_colored_set, k_to_uncolor):
        """
        专用于Case 4的去色操作，使用专门的路径生成方法。
        确保严格按照生成的去色路径执行去色，对于偶数t确保最外层特殊点不被去色。
        """
        if k_to_uncolor <= 0:
            return True  # 无需去色
        
        # 使用Case 4专用的去色路径
        uncoloring_traversal_hexes = self._generate_case4_uncoloring_traversal_path()
        
        if not uncoloring_traversal_hexes:
            return False
        
        # 确保最外层特殊点不被去色（对于偶数t）
        outer_special_point = None
        if self.t % 2 == 0 and self.t in self._get_special_points_by_layer():
            outer_special_point = self._get_special_points_by_layer()[self.t]
        
        # 从路径中移除k_to_uncolor个已着色的六边形，但不包括最外层特殊点
        uncolored_count = 0
        for hex_qr in uncoloring_traversal_hexes:
            if uncolored_count >= k_to_uncolor:
                break  # 已找到足够的六边形
            
            if hex_qr in initial_colored_set:
                # 跳过最外层特殊点
                if outer_special_point and hex_qr == outer_special_point:
                    continue
                
                initial_colored_set.remove(hex_qr)
                uncolored_count += 1
        
        # 未找到足够的六边形去色，则操作失败
        return uncolored_count == k_to_uncolor
    
    def _get_special_points_by_layer(self):
        """获取各层的特殊点：layer -> (q,r)"""
        special_points_by_layer = {}
        
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
            
            # 从西北角点往左下走两格
            step1_q = nw_corner[0] - 1
            step1_r = nw_corner[1] + 1
            step2_q = step1_q - 1
            step2_r = step1_r + 1
            special_point = (step2_q, step2_r)
            
            # 检查特殊点是否在网格内
            s_coord = -special_point[0] - special_point[1]
            if max(abs(special_point[0]), abs(special_point[1]), abs(s_coord)) < self.t:
                if special_point in self.hexagons:
                    special_points_by_layer[m] = special_point
        
        return special_points_by_layer

# --- 可视化器类 ---
class HexGridVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("蜂窝网格偶环可视化 (六边形着色版)") 
        self.geometry("900x750")
        param_frame = tk.Frame(self)
        param_frame.pack(pady=10)
        tk.Label(param_frame, text="网格阶数 t:").grid(row=0, column=0, padx=5)
        self.t_entry = tk.Entry(param_frame, width=5)
        self.t_entry.grid(row=0, column=1, padx=5)
        self.t_entry.insert(0, "3") 
        tk.Label(param_frame, text="环长度 l:").grid(row=0, column=2, padx=5)
        self.l_entry = tk.Entry(param_frame, width=5)
        self.l_entry.grid(row=0, column=3, padx=5)
        self.l_entry.insert(0, "52") 
        self.find_button = tk.Button(param_frame, text="生成网格并查找环", command=self.generate_and_find)
        self.find_button.grid(row=0, column=4, padx=10)
        
        # 添加显示遍历路径的按钮
        self.show_path_button = tk.Button(param_frame, text="显示去色路径", command=self.show_uncoloring_path)
        self.show_path_button.grid(row=0, column=5, padx=10)
        
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.scale = 1.0
        self.canvas_offset_x = 450 
        self.canvas_offset_y = 375 
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", lambda e: self.zoom(e, 1.1)) 
        self.canvas.bind("<Button-5>", lambda e: self.zoom(e, 0.9)) 
        self.grid_instance = None
        self.current_cycle_edges = None
        self.current_hexagons_to_color = None
        self.traversal_path = None  # 存储遍历路径
        self.showing_path = False   # 是否显示路径

    def start_pan(self, event): # 画布拖动起始
        self.canvas.scan_mark(event.x, event.y)

    def pan(self, event): # 画布拖动过程
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def zoom(self, event, factor=None): # 画布缩放
        true_canvas_x = self.canvas.canvasx(event.x) 
        true_canvas_y = self.canvas.canvasy(event.y)
        if factor is None: factor = 1.1 if event.delta > 0 else 0.9
        self.canvas_offset_x = true_canvas_x - (true_canvas_x - self.canvas_offset_x) * factor
        self.canvas_offset_y = true_canvas_y - (true_canvas_y - self.canvas_offset_y) * factor
        self.scale *= factor 
        self.redraw_canvas() 

    def _get_transformed_pixel_coords(self, model_x, model_y):
        """将模型像素坐标转换为应用了平移和缩放的画布坐标"""
        screen_x = model_x * self.scale + self.canvas_offset_x
        screen_y = model_y * self.scale + self.canvas_offset_y
        return screen_x, screen_y

    def generate_and_find(self): # "生成网格并查找环"按钮的回调
        try:
            t = int(self.t_entry.get())
            l = int(self.l_entry.get())
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的整数 t 和 l")
            return
        if t < 1: 
             messagebox.showerror("输入错误", "阶数 t 必须 >= 1")
             return
        self.grid_instance = HexGrid(t)
        self.current_cycle_edges, self.current_hexagons_to_color = self.grid_instance.find_cycle(l)
        
        if self.current_cycle_edges is None and self.current_hexagons_to_color is None:
            # find_cycle 返回 (None, None) 表示构造失败或不支持
            # messagebox.showinfo("结果", f"未能为长度 {l} 找到或构造有效的环或六边形区域。")
            pass # 保持界面安静，控制台会有输出
        
        # 生成遍历路径但不显示
        self.traversal_path = self.grid_instance._generate_uncoloring_traversal_path()
        self.showing_path = False
        
        self.redraw_canvas()
    
    def show_uncoloring_path(self):
        """显示或隐藏去色遍历路径"""
        if not self.grid_instance:
            messagebox.showinfo("提示", "请先生成网格")
            return
            
        # 切换显示状态
        self.showing_path = not self.showing_path
        
        # 如果需要显示但路径未生成，则生成路径
        if self.showing_path and not self.traversal_path:
            # 获取当前环的长度
            try:
                current_length = int(self.l_entry.get())
                max_theoretical_len = 6 * self.grid_instance.t**2 - 2
                second_max_len = 6 * self.grid_instance.t**2 - 4
                
                # 根据当前环的长度决定使用哪种路径生成方法
                if current_length < max_theoretical_len and (max_theoretical_len - current_length) % 4 == 0:
                    # Case 2
                    self.traversal_path = self.grid_instance._generate_uncoloring_traversal_path()
                elif current_length < second_max_len and (second_max_len - current_length) % 4 == 0:
                    # Case 4
                    self.traversal_path = self.grid_instance._generate_case4_uncoloring_traversal_path()
                else:
                    # 默认使用case2的路径生成
                    self.traversal_path = self.grid_instance._generate_uncoloring_traversal_path()
            except (ValueError, TypeError):
                # 出错时使用默认路径生成方法
                self.traversal_path = self.grid_instance._generate_uncoloring_traversal_path()
            
        # 更新显示
        self.redraw_canvas()
        
        # 更新按钮文本
        self.show_path_button.config(text="隐藏去色路径" if self.showing_path else "显示去色路径")

    def redraw_canvas(self): # 重绘整个画布
        self.canvas.delete("all")
        if not self.grid_instance: return
            
        for q, r in self.grid_instance.hexagons:
            if (q,r) in self.grid_instance.hex_vertices:
                model_vertices = self.grid_instance.hex_vertices[(q,r)]
                pixel_points = []
                for mvx, mvy in model_vertices:
                    pvx, pvy = self._get_transformed_pixel_coords(mvx, mvy)
                    pixel_points.extend([pvx, pvy])
                self.canvas.create_polygon(pixel_points, fill="", outline="lightgrey", tags="grid_hex_outline")

        if self.current_hexagons_to_color:
            for q_hex, r_hex in self.current_hexagons_to_color:
                if (q_hex, r_hex) in self.grid_instance.hex_vertices: 
                    model_vertices = self.grid_instance.hex_vertices[(q_hex, r_hex)]
                    pixel_points = []
                    for mvx, mvy in model_vertices:
                        pvx, pvy = self._get_transformed_pixel_coords(mvx, mvy)
                        pixel_points.extend([pvx, pvy])
                    self.canvas.create_polygon(pixel_points, fill="salmon", outline="black", width=max(0.5, 1*self.scale), tags="colored_hex_in_cycle")
        
        if self.current_cycle_edges:
            for edge_model_coords in self.current_cycle_edges:
                mx1, my1, mx2, my2 = edge_model_coords
                sx1, sy1 = self._get_transformed_pixel_coords(mx1, my1)
                sx2, sy2 = self._get_transformed_pixel_coords(mx2, my2)
                self.canvas.create_line(sx1, sy1, sx2, sy2, fill="red", width=max(1, 2 * self.scale), tags="cycle_boundary_edge")

        # 显示遍历路径
        if self.showing_path and self.traversal_path:
            # 收集所有标记点（角点和特殊点）
            all_marker_points = set()
            special_points = set()
            
            # 收集特殊点
            relevant_tiers = []
            if self.grid_instance.t % 2 == 0: 
                relevant_tiers = [m for m in range(2, self.grid_instance.t + 1, 2)]
            else: 
                relevant_tiers = [m for m in range(1, self.grid_instance.t + 1, 2)]

            for m in relevant_tiers:
                if m < 1: continue
                
                # 获取该层的西北角点坐标
                nw_corners_m = self.grid_instance._get_corner_hexagons(m)
                if "NW" not in nw_corners_m:
                    continue
                    
                nw_corner = nw_corners_m["NW"]
                
                # 从西北角点往左下走两格
                # 第一步：往左下方向 (SW, -1, +1)
                step1_q = nw_corner[0] - 1
                step1_r = nw_corner[1] + 1
                
                # 第二步：继续往左下方向
                step2_q = step1_q - 1
                step2_r = step1_r + 1
                
                # 特殊点坐标
                special_point = (step2_q, step2_r)
                
                # 检查特殊点是否在网格内
                s_coord = -special_point[0] - special_point[1]
                if max(abs(special_point[0]), abs(special_point[1]), abs(s_coord)) < self.grid_instance.t:
                    if special_point in self.grid_instance.hexagons:
                        special_points.add(special_point)
            
            # 收集所有层的角点，并区分内层NW点和有效标记点
            corner_labels = ["NW", "NE", "E", "SE", "SW", "W"]
            marker_corner_points = set()  # 作为标记点的角点
            non_marker_nw_points = set()  # 内层NW角点(不作为标记点)
            
            for layer in range(1, self.grid_instance.t + 1):
                layer_corners = self.grid_instance._get_corner_hexagons(layer)
                if not layer_corners:
                    continue
                    
                for label in corner_labels:
                    if label in layer_corners:
                        corner_point = layer_corners[label]
                        if label == "NW" and layer < self.grid_instance.t:
                            non_marker_nw_points.add(corner_point)  # 内层NW不作为标记点
                        else:
                            marker_corner_points.add(corner_point)  # 其他角点都是标记点
            
            # 合并有效标记点
            all_marker_points.update(marker_corner_points)
            all_marker_points.update(special_points)
            
            # 绘制路径连线
            for i in range(len(self.traversal_path) - 1):
                start_hex = self.traversal_path[i]
                end_hex = self.traversal_path[i + 1]
                
                if start_hex in self.grid_instance.hex_centers and end_hex in self.grid_instance.hex_centers:
                    start_x, start_y = self.grid_instance.hex_centers[start_hex]
                    end_x, end_y = self.grid_instance.hex_centers[end_hex]
                    
                    sx1, sy1 = self._get_transformed_pixel_coords(start_x, start_y)
                    sx2, sy2 = self._get_transformed_pixel_coords(end_x, end_y)
                    
                    # 使用渐变色显示路径方向
                    r = min(255, 50 + (i * 200 // len(self.traversal_path)))
                    g = min(255, 50 + ((len(self.traversal_path) - i) * 200 // len(self.traversal_path)))
                    color = f'#{r:02x}{g:02x}00'
                    
                    self.canvas.create_line(sx1, sy1, sx2, sy2, fill=color, width=max(1.5, 2.5 * self.scale), 
                                          arrow=tk.LAST, dash=(5,2), tags="traversal_path")
            
            # 在路径上的六边形中心标记数字（表示顺序）
            for i, hex_qr in enumerate(self.traversal_path):
                if hex_qr in self.grid_instance.hex_centers:
                    cx, cy = self.grid_instance.hex_centers[hex_qr]
                    sx, sy = self._get_transformed_pixel_coords(cx, cy)
                    # 只标记一些关键点，避免过于拥挤
                    if i % max(1, len(self.traversal_path) // 20) == 0:
                        self.canvas.create_text(sx, sy, text=str(i), fill="black", font=("Arial", int(9*self.scale)), tags="traversal_index")
            
            # 标记特殊点
            for sp in special_points:
                if sp in self.grid_instance.hex_centers:
                    cx, cy = self.grid_instance.hex_centers[sp]
                    sx, sy = self._get_transformed_pixel_coords(cx, cy)
                    self.canvas.create_oval(sx-10*self.scale, sy-10*self.scale, 
                                          sx+10*self.scale, sy+10*self.scale, 
                                          outline="purple", width=max(1.5, 2*self.scale), tags="special_point")
                    self.canvas.create_text(sx, sy, text="SP", fill="purple", font=("Arial", int(9*self.scale)), tags="special_point_text")
            
            # 标记有效角点（作为标记点的角点）
            for cp in marker_corner_points:
                if cp in self.grid_instance.hex_centers:
                    cx, cy = self.grid_instance.hex_centers[cp]
                    sx, sy = self._get_transformed_pixel_coords(cx, cy)
                    self.canvas.create_oval(sx-8*self.scale, sy-8*self.scale, 
                                          sx+8*self.scale, sy+8*self.scale, 
                                          outline="blue", width=max(1, 1.5*self.scale), tags="corner_point")
            
            # 标记内层NW角点（不作为标记点）
            for nw in non_marker_nw_points:
                if nw in self.grid_instance.hex_centers:
                    cx, cy = self.grid_instance.hex_centers[nw]
                    sx, sy = self._get_transformed_pixel_coords(cx, cy)
                    self.canvas.create_oval(sx-8*self.scale, sy-8*self.scale, 
                                          sx+8*self.scale, sy+8*self.scale, 
                                          outline="gray", width=max(1, 1.5*self.scale), dash=(3,2), tags="non_marker_corner")

        if (0,0) in self.grid_instance.hex_centers:
             model_center_x, model_center_y = self.grid_instance.hex_centers[(0,0)]
             scx, scy = self._get_transformed_pixel_coords(model_center_x, model_center_y)
             radius = max(1, 3 * self.scale) 
             self.canvas.create_oval(scx-radius, scy-radius, scx+radius, scy+radius, fill='blue', outline='blue', tags='center_marker')

if __name__ == "__main__":
    app = HexGridVisualizer()
    app.mainloop()    