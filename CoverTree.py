import numpy as np
import math
from scipy.spatial import distance

class CoverTree:
    def __init__(self, data, b=2.0, adist = 0,max_attrib=1):
        """
        Initializes the Cover Tree with hierarchical levels based on distances.
        Args:
            data (np.array): Dataset to build the Cover Tree from.
            b (float): Base factor that determines the distance scale between levels.
        """
        self.data = np.array(data)
        self.b = b
        self.adist =adist
        self.layers = self.build_tree(max_attrib)
        if not self.layers:
            print("⚠️ Warning: CoverTree built with empty layers!")
        else:
            print(f" CoverTree built with {len(self.layers)} levels")

    def build_tree(self,max_attrib):
        """
        Build hierarchical layers for the Cover Tree with proper nesting and parent-child relationships.
        Returns:
            dict: Nested levels from l=0 down to l_min.
        """
        if len(self.data) == 0:
            return {}

        centroid = np.mean(self.data[:, self.adist], axis=0)
        temp_centroid=[]
        for i in range(0,max_attrib):
            if i!=self.adist:
                temp_centroid.append(0)
            else:
                temp_centroid.append(centroid)
        centroid=np.array(temp_centroid)
        # Compute min distance to determine l_min
        max_dist=distance.euclidean(self.data[len(self.data)-1,self.adist].flatten(),self.data[0,self.adist].flatten()) if len(self.data) > 1 else 1.0
        """ max_dist = max(
            distance.euclidean(self.data[i, self.adist].flatten(), self.data[j, self.adist].flatten())
            for i in range(len(self.data)) for j in range(i + 1, len(self.data))
        ) if len(self.data) > 1 else 1.0"""
        if(max_dist==0):
            layers={}
            layers[0]=self.data
            return layers
        # Compute l_max and l_min
        l_max = int(math.ceil(math.log(max_dist,self.b)))
        # Initialize levels
        layers = {l: [] for l in range(0, -(l_max)-2, -1)}  # יורד מ-l_max ל-l_min
        layers[0] = [centroid]
        current_layer=layers[0]
        # Build hierarchical layers (top-down)
        for l in range(-1, -(l_max)-1, -1):  # יורדים ברמות
            if(l==-1):
                    threshold = self.b ** (l_max+l-1)
            else:
                 threshold = self.b ** (l_max+l)
            new_layer=[]
            for item in current_layer:
                new_layer.append(item)
            for i in range(math.ceil(len(self.data)/2)):
                last=self.data[len(self.data)-i-1]
                first=self.data[i]
                is_larger_beginning=True
                for item in new_layer:
                    if distance.euclidean(item[self.adist].flatten(),first[self.adist].flatten())<threshold:
                        is_larger_beginning=False
                        break
                is_larger_end=True
                for item in new_layer:
                    if distance.euclidean(item[self.adist].flatten(),last[self.adist].flatten())<threshold:
                        is_larger_end=False
                        break
                new_layer.append(first) if is_larger_beginning else None        
                new_layer.append(last) if is_larger_end else None 
            if l==-1:
                new_layer.remove(centroid)
            new_data = [list(y) for y in set([tuple(x) for x in new_layer])]
            layers[l] = new_data
            current_layer = layers[l]
            if(len(current_layer)==len(self.data)):
                for j in range(l, -(l_max)-2, -1):
                    layers[j]=current_layer
                break
        layers[-(l_max)-1]=self.data
        return layers

        

    def extract_candidates(self, k, delta):
        """
        Extract candidates based on the k-level rule: Find the first level with at least k points,
        then descend delta levels below it while ensuring no duplicates.
        Args:
            k (int): Minimum number of candidates required.
            delta (int): Number of levels to explore below the found k-level.
        Returns:
            np.array: Candidate set (without duplicates).
        """
        print(f"Extracting candidates from CoverTree with {len(self.data)} elements")
        if len(self.data) == 0:
            print("⚠️ Warning: CoverTree has no data!")
            return []
            # Check if the layers dictionary is empty

        if not self.layers:
            print("⚠️ Warning: CoverTree has no layers!")
            return []

        # Find the first level where there are at least k candidates
        k_level = None
        for l in self.layers.keys():
            print(f"🔹 Level {l}: {len(self.layers[l])} elements ")
            if len(self.layers[l]) >= k:
                k_level = l
                break

        #  אם לא נמצאה רמה עם לפחות k מועמדים, נבחר את הרמה עם הכי הרבה מועמדים
        if k_level is None:
            k_level = max(self.layers.keys(), key=lambda l: len(self.layers[l]))
            print(f"⚠️ Warning: No level had at least {k} candidates. Using most populated level {k_level} instead.")

        # מציאת מועמדים מהשכבות שמתחת לרמת k-level
        seen = set()
        candidates = []
        l=max(k_level - delta, min(self.layers.keys()))
        for point in self.layers[l]:
            point_tuple = tuple(point)
            if point_tuple not in seen:
                seen.add(point_tuple)
                candidates.append(point)

        print(f" Found {len(candidates)} candidates from CoverTree")
        return np.array(candidates)