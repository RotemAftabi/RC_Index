import numpy as np
from scipy.spatial import distance

class CoverTree:
    def __init__(self, data, b=2.0, Adist = None):
        """
        Initializes the Cover Tree with hierarchical levels based on distances.
        Args:
            data (np.array): Dataset to build the Cover Tree from.
            b (float): Base factor that determines the distance scale between levels.
        """
        self.data = np.array(data)
        self.b = b
        self.Adist =Adist
        self.layers = self._build_tree()
        self.parent_child_map = {}  # Explicit parent-child mapping

        if not self.layers:
            print("⚠️ Warning: CoverTree built with empty layers!")
        else:
            print(f"✅ CoverTree built with {len(self.layers)} levels")

    def _build_tree(self):
        """
        Build hierarchical layers for the Cover Tree with proper nesting and parent-child relationships.
        Returns:
            dict: Nested levels from l=0 down to l_min.
        """
        if len(self.data) == 0:
            return {}, {}

        centroid = np.mean(self.data[:, self.Adist], axis=0)

        # Compute min distance to determine l_min

        min_dist = min(
            distance.euclidean(self.data[i, self.Adist].flatten(), self.data[j, self.Adist].flatten())
            for i in range(len(self.data)) for j in range(i + 1, len(self.data))
        ) if len(self.data) > 1 else 1.0
        # Compute l_max and l_min
        l_min = int(np.floor(np.log(min_dist) / np.log(self.b)))
        # Initialize levels
        layers = {l: [] for l in range(0, -(l_min - 1), -1)}  # יורד מ-l_max ל-l_min
        self.parent_child_map = {}

        distances = np.array([distance.euclidean(row[self.Adist].flatten(), centroid.flatten()) for row in self.data])
        sorted_indices = np.argsort(distances)
        sorted_data = self.data[sorted_indices]
        max_distance_idx = np.argmax(distances)
        layers[0] = [centroid]
        current_layer = sorted_data  # המתחילים הם רק האלמנט הזה

        # Build hierarchical layers (top-down)
        for l in range(0, -(l_min - 1), -1):  # יורדים ברמות
            threshold = self.b ** l
            new_layer = []

            for row in layers[l]:
                added = False
                for child in current_layer:
                    print(child)
                    print(distance.euclidean(row.flatten(), child.flatten()))
                    if distance.euclidean(row[self.Adist].flatten(), child[self.Adist].flatten()) <= threshold:
                        self.parent_child_map.setdefault(tuple(child), []).append(tuple(row))  # קשר הורה-ילד
                        new_layer.append(row)
                        is_added = True
                        break


                layers[l] = new_layer
                current_layer = new_layer  # ממשיכים לרמה הבאה

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

        for l in range(k_level, max(k_level - delta - 1, min(self.layers.keys()) - 1), -1):
            for point in self.layers[l]:
                point_tuple = tuple(point)
                if point_tuple not in seen:
                    seen.add(point_tuple)
                    candidates.append(point)

        print(f" Found {len(candidates)} candidates from CoverTree")
        return np.array(candidates)