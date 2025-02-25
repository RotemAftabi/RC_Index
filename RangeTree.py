from CoverTree import CoverTree
import numpy as np
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor
import math

class RangeTreeNode:
    def __init__(self, data, b=2.0, d=0, max_attrib=1):
        self.data = np.array(data)
        self.d = d
        self.cover_tree = CoverTree(self.data, b, d, max_attrib) if len(self.data) > 1 else None
        self.left = None
        self.right = None
        self.next = None

    def max_and_min_value_for_column(self, index):
        column_values = self.data[:, index]
        return [np.max(column_values), np.min(column_values)]
class RangeTree:
    def __init__(self, data, b=2.0, adist=None, max_attrib=1):
        print(f"Building RangeTree for {len(data)} rows...")
        self.max_attrib = max_attrib
        self.data=np.array(data)
        self.root = self.build_tree_parallel(self.data[np.argsort(self.data[:, adist[0]])], b, adist,0, max_attrib)
        print(f"RangeTree built successfully.")
    def build_tree(self, sorted_data, b, adist, d=0, max_attrib=1):
        if len(sorted_data) == 0:
            return None
        if len(sorted_data) == 1:
            node = RangeTreeNode(sorted_data, b,adist[d],max_attrib)
            return node
        if len(sorted_data) < 50:
            node = RangeTreeNode(sorted_data, b,adist[d],max_attrib)
            return node
        target=float(sorted_data[len(sorted_data)-1][adist[d]])+float(sorted_data[0][adist[d]])
        median_index=-1
        if float(sorted_data[len(sorted_data)-1][adist[d]])-float(sorted_data[0][adist[d]])<=2:
            node = RangeTreeNode(sorted_data, b,adist[d],max_attrib)
            node.next = self.build_tree(sorted_data[np.argsort(sorted_data[:, adist[d+1]])], b, adist,d + 1, max_attrib) if d < len(adist) - 1 else None
            return node
        left, right = 0, len(sorted_data) - 1
        median_index=0
        while left <= right:
            mid = (left + right) // 2
            if float(sorted_data[mid,adist[d]]) > target/2:
                median_index = mid 
                right = mid - 1  
            else:
                left = mid + 1
        node = RangeTreeNode(sorted_data, b, adist[d], max_attrib)
        left_data = sorted_data[:median_index]
        right_data = sorted_data[median_index:]
        if len(left_data) == len(sorted_data) or len(right_data) == len(sorted_data):
            return node  # Prevent infinite recursion
        node.left = self.build_tree(left_data, b, adist,d, max_attrib)
        node.right = self.build_tree(right_data, b, adist,d, max_attrib)
        node.next = self.build_tree(sorted_data[np.argsort(sorted_data[:, adist[d+1]])], b, adist,d + 1, max_attrib) if d < len(adist) - 1 else None
        return node

    def build_tree_parallel(self, sorted_data, b, adist=None,d=0, max_attrib=1):
        """Parallelized version of build_tree using ThreadPoolExecutor."""
        if len(sorted_data) == 0:
            return None
        if len(sorted_data) == 1:
            return RangeTreeNode(sorted_data, b, adist[d], max_attrib)
        if float(sorted_data[len(sorted_data)-1][adist[d]])-float(sorted_data[0][adist[d]])<=2:
            node = RangeTreeNode(sorted_data, b,adist[d],max_attrib)
            node.next = self.build_tree_parallel(sorted_data[np.argsort(sorted_data[:, adist[d+1]])], b, adist,d + 1, max_attrib) if d < len(adist)-1 else None
            return node
        node = RangeTreeNode(sorted_data, b, adist[d], max_attrib)
        target=float(sorted_data[len(sorted_data)-1][adist[d]])+float(sorted_data[0][adist[d]])
        left, right = 0, len(sorted_data) - 1
        median_index=0
        while left <= right:
            mid = (left + right) // 2
            if float(sorted_data[mid,adist[d]]) > target/2:
                median_index = mid 
                right = mid - 1  
            else:
                left = mid + 1
        left_data = sorted_data[:median_index] 
        right_data = sorted_data[median_index:]
        if len(left_data) == len(sorted_data) or len(right_data) == len(sorted_data):
            return node 
        with ThreadPoolExecutor() as executor:
            left_future = executor.submit(self.build_tree, left_data, b, adist, d, max_attrib)
            right_future = executor.submit(self.build_tree, right_data, b, adist, d, max_attrib)
            node.left = left_future.result()
            node.right = right_future.result()
        node.next = self.build_tree(sorted_data[np.argsort(sorted_data[:, adist[d+1]])], b, adist,d + 1, max_attrib) if d < len(adist) - 1 else None
        return node
    def range_query(self, node,query_range,query_index):
        if node is None:
            return []
        min_val, max_val = query_range
        maximum,minimum=node.max_and_min_value_for_column(query_index)
        if max_val<minimum or min_val>maximum:
            return []
        elif maximum<=max_val and minimum>=min_val:
            return [node.cover_tree] if node.cover_tree else []
        results = []
        results=np.concatenate((results,self.range_query(node.left, query_range,query_index))) if node.left else results
        results=np.concatenate((results,self.range_query(node.right, query_range,query_index))) if node.right else results
        results=np.concatenate((results,self.range_query(node.next, query_range,query_index))) if node.next else results
        return results

    def greedy_selection(self, candidates, k):
        if not candidates:
            print("No candidates found for greedy selection.")
            return []
        diverse_set = []
        candidates = list(candidates)
        while len(diverse_set) < k and candidates:
            best_candidate = max(candidates, key=lambda x: min(
                distance.euclidean(x, y) for y in diverse_set) if diverse_set else float('inf'))
            diverse_set.append(best_candidate)
            candidates.remove(best_candidate)
        return diverse_set
