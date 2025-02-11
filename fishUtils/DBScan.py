'''
How does DBSCAN work: https://en.wikipedia.org/wiki/DBSCAN
'''
import numpy as np
from copy import deepcopy
# deque provides an O(1) time complexity for append and pop operations instead of O(n) for lists.
from collections import deque
# dataset to toy around with.
from sklearn.datasets import make_moons


def pairwise_sq_distance(X1, X2):
    # Calculate the pairwise distance between all pairs of points from X1 and X2.
    return np.sum(X1**2, axis=1, keepdims=True) - 2*np.matmul(X1, X2.T) + np.sum(X2**2, axis=1, keepdims=True).T

class DBSCAN:
    
    def __init__(self, eps=0.5, minpts=5):
        self.eps = eps
        self.minpts = minpts
        
    def fit(self, X):
        dist = pairwise_sq_distance(X, X)
        neighbours = list(map(lambda d: np.arange(d.shape[0])[d < self.eps**2], dist))
        
        # Label all points as outliers initially.
        self.assignment = np.full((X.shape[0],), -1, dtype=int)
        # Find core points.
        ## Determine the number of neighbors of each point.
        N_neighbors = np.sum(dist < self.eps**2, axis=1)
        self.assignment[N_neighbors >= self.minpts] = -2
        
        # Create clusters.
        cluster = 0
        stack = deque()
        for p in range(X.shape[0]):
            if self.assignment[p] != -2:
                continue
                
            self.assignment[p] = cluster
            
            stack.extend(neighbours[p])
            # Expand cluster outwards. 
            while len(stack) > 0:
                n = stack.pop()
                label = self.assignment[n]
                # If core point include all points in ε-neighborhood.
                if label == -2:
                    stack.extend(neighbours[n])
                # If not core point (edge of cluster).
                if label < 0:
                    self.assignment[n] = cluster
            
            cluster += 1
            
    def fit_predict(self, X):
        self.fit(X)
        return self.assignment
    
    def predict(self,X):
        return self.assignment

class RadarDBSCAN(DBSCAN):
    def __init__(self, eps=0.5, min_samples=5, vel_th=1.0):
        super().__init__(eps, min_samples)
        self.vel_th = vel_th

    def fitRadar(self, radar_data):
        ''' Return the cluster assignment for each point in the radar data.'''
        # Get radar points x,y
        radar_data = np.array(radar_data)
        radar_points = radar_data[:, :2]
        return self.fit_predict(radar_points)

    def stackRadar(self, radar_data_list):
        '''
        Stack Radar data from multiple frames.
        Add frame id to radar data so that we can know which frame the stack radar data belongs to.
        Args:
            radar_data_list: list of np.array, shape: (n, 5), [x, y, z, vx, vy]
        return:
            radar_data_list: list of np.array, shape: (n, 6), [x, y, z, vx, vy, frame_id]
        
        radar_data_list[-1] is the latest frame. radar_data_list[0] is the oldest frame.
        '''
        for i, radar_data in enumerate(radar_data_list):
            radar_data_list[i] = np.hstack([radar_data, np.ones((len(radar_data), 1)) * i])

        return np.vstack(radar_data_list)
        
    def segmentRadar(self, radar_data):
        '''
        Return the segmented radar data.
        Shape of radar_data: (n, 6), [x, y, z, vx, vy, frame_id]
        Shape of segmented radar data: (n, 7), [x, y, z, vx, vy, frame_id, cluster_id]
        '''
        radar_data = np.array(radar_data)
        # filter out points that has too small velocity.
        self.vel_th = 1.0
        filtered_radar_data = deepcopy(radar_data[np.linalg.norm(radar_data[:, 3:5], axis=1) > self.vel_th])
        assignment = self.fitRadar(filtered_radar_data)
        return np.hstack((filtered_radar_data, assignment.reshape(-1, 1))), assignment

    def avgSegmentation(self, radar_seg):
        '''
        Average the segmentation results. (Same segments)
        Args:
            radar_seg: list of np.array, shape: (n, 7), [x, y, z, vx, vy, frame_id, cluster_id]
        return:
            avg_radar_seg: np.array, shape: (n, 7), [x, y, z, vx, vy, frame_id, cluster_id]
        '''
        if len(radar_seg) == 0:
            return np.array([])
        if len(radar_seg) == 1:
            return radar_seg
        # Get the unique cluster ids.
        cluster_ids = np.unique(np.hstack([seg[-1] for seg in radar_seg]))
        avg_radar_seg = []
        for cluster_id in cluster_ids:
            cluster_points = np.vstack([seg[seg[-1] == cluster_id] for seg in radar_seg])
            avg_values = np.mean(cluster_points[:, :5], axis=0)
            frame_and_cluster_id = cluster_points[0, 5:]
            avg_radar_seg.append(np.hstack([avg_values, frame_and_cluster_id]))
        
        return np.array(avg_radar_seg)
        
    
if __name__ == '__main__':
    X,y = make_moons(100)
    model = DBSCAN()
    preds = model.fit_predict(X)
    # Either low or high values are good since DBSCAN might switch class labels.
    print(f"Accuracy: {round((sum(preds == y)/len(preds))*100,2)}%")