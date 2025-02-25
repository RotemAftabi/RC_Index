from RangeTree import RangeTree
from CoverTree import CoverTree
import numpy as np
import pandas as pd
import time
import random
from scipy.spatial import distance
import sys

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]
data = pd.read_csv(url, header=None, names=columns, skipinitialspace=True)
df = pd.DataFrame(data)

# Select only numeric columns
numeric_columns = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
numeric_data = data[numeric_columns].values
randomly_chosen_columns = np.random.choice(numeric_columns, size=2, replace=False)
adist = [numeric_columns.index(randomly_chosen_columns[i]) for i in range(len(randomly_chosen_columns))]
"distances = distance.pdist(adist, metric='euclidean')"

def greedy_without_filtering(query=None,query_index=0,k=0):
    candidates=whole_data(query,query_index)
    if not candidates:
        return np.array([])
    if len(candidates) < k:
        print(f"Warning: Only {len(candidates)} candidates found, but requested {k}. Returning all available.")
        return np.array(candidates)
    candidates = list(candidates)
    rand_inx=random.randint(0,len(candidates)-1)
    random_candidate=candidates[rand_inx]
    candidates.pop(rand_inx)
    diverse_set = [random_candidate]
    distances=[np.linalg.norm(np.array(random_candidate)-np.array(item)) for item in candidates]
    while len(candidates)>0 and len(diverse_set)<k:
        max_int=distances.index(max(distances))
        candidates[max_int]
        best_candidate=candidates[max_int]
        diverse_set.append(best_candidate)
        distances=[min(distances[i],np.linalg.norm(np.array(best_candidate)-np.array(candidates[i]))) for i in range(len(candidates))]
    return np.array(diverse_set)
def whole_data(query=None,query_index=0):
    fitting=[]
    minimum,maximum=query
    for item in numeric_data:
        if(item[query_index]>=minimum and item[query_index]<=maximum):
            fitting.append(item)
    return fitting
class RCIndex:
    def __init__(self, data, b=2.0):
        """
        RC-Index that integrates Range Tree and Cover Trees.
        Args:
            data (np.array): The dataset on which the index is built.
            b (float): Base distance parameter for the Cover Trees.
        """
        start_time = time.time()
        print("Initializing RC-Index...")
        self.data = data  # שמירת הנתונים
        print(f"Dataset contains {len(data)} rows")

        # בניית ה-RangeTree
        print("Building RangeTree...")

        self.range_tree = RangeTree(data, b,adist,len(numeric_columns))
        print(f"RangeTree built in {time.time() - start_time:.2f} seconds")

    def query(self, query_column, query_range, k, delta):
        """
        Executes a range query and returns k diverse results using Cover Trees.
        Args:
            query_column (str): The column to apply the range filter on.
            query_range (tuple): (min_val, max_val) range filter.
            k (int): Number of diverse results to return.
            delta (int): Depth parameter for extracting candidates.
        Returns:
            np.array: k diverse results.
        """
        
        column_index = numeric_columns.index(query_column)  # Find index of the column
        # Build a temporary RCIndex for the filtered data
        relevant_cover_trees = self.range_tree.range_query(self.range_tree.root, query_range,column_index)
        candidates = []

        # Extract diverse candidates from each relevant Cover Tree
        for cover_tree in relevant_cover_trees:
            if cover_tree:
                candidates.extend(cover_tree.extract_candidates(k, delta))
        candidates=set(tuple(i) for i in candidates)
        # Apply Greedy Tree++ selection algorithm
        return self.greedy_selection(candidates, k)

    def greedy_selection(self, candidates, k):
        if not candidates:
            return np.array([])

        if len(candidates) < k:
            print(f"Warning: Only {len(candidates)} candidates found, but requested {k}. Returning all available.")
            return np.array(candidates)

        candidates = list(candidates)
        rand_inx=random.randint(0,len(candidates)-1)
        random_candidate=candidates[rand_inx]
        candidates.pop(rand_inx)
        diverse_set = [random_candidate]
        distances=[np.linalg.norm(np.array(random_candidate)-np.array(item)) for item in candidates]
        while len(candidates)>0 and len(diverse_set)<k:
            max_int=distances.index(max(distances))
            candidates[max_int]
            best_candidate=candidates[max_int]
            diverse_set.append(best_candidate)        
            distances=[min(distances[i],np.linalg.norm(np.array(best_candidate)-np.array(candidates[i]))) for i in range(len(candidates))]
        return np.array(diverse_set)


# Example usage:
if __name__ == "__main__":
    # Initialize RC-Index with numeric data
    sample_data = numeric_data
    rc_index = RCIndex(sample_data)

    # Get user input for query
    print("Available numeric columns:", numeric_columns)
    query_column = input("Enter the column name to apply range filter: ")
    if query_column not in numeric_columns:
        print("Invalid column. Exiting.")
        exit()

    min_val = float(input(f"Enter minimum value for {query_column}: "))
    max_val = float(input(f"Enter maximum value for {query_column}: "))
    query_range = (min_val, max_val)

    k = int(input("Enter number of diverse results to return (k): "))
    delta = int(input("Enter depth parameter for extracting candidates (delta): "))
    start_time = time.time()
    results = rc_index.query(query_column, query_range, k, delta)
    end_time=time.time()-start_time
    print("Diversification ended after " + str(end_time)+ " seconds")
    print("Diverse results:", results)
    print("/////////////////////////////////////////////////////////////////////////////////////////")
    start_time = time.time()
    print(greedy_without_filtering(query_range,numeric_columns.index(query_column),k))
    end_time=time.time()-start_time
    print("Greedy without diversification ended after " + str(end_time)+ " seconds")
    print("/////////////////////////////////////////////////////////////////////////////////////////")
    print("Printing the whole relevant data")
    print(whole_data(query_range,numeric_columns.index(query_column)))


