import numpy as np
from scipy.spatial import distance as dist
import math

# A global counter to assign unique IDs to new objects
nextObjectID = 0

class TrackedObject:
    """Stores the history and calculates velocity for a single object ID."""
    
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]  # List of historical center points
        self.max_history = 5         # Keep track of last 5 frames for stability
        self.disappeared = 0         # Counter for how many frames the object was missed
        self.total_frames = 1        # Total frames tracked
        self.current_centroid = centroid
        self.velocity = 0.0          # Current normalized speed

    def update_position(self, new_centroid):
        """Updates the centroid list and calculates the new velocity."""
        global RESIZED_W # Assuming RESIZED_W is available globally (e.g., 640)
        
        # Calculate displacement (distance) from the last position
        if len(self.centroids) > 0:
            last_centroid = self.centroids[-1]
            # Euclidean distance in pixels
            pixel_distance = dist.euclidean(last_centroid, new_centroid)
            
            # Normalize velocity (simple way: distance relative to frame width)
            # This gives a normalized speed (0.0 to ~1.0)
            self.velocity = pixel_distance / RESIZED_W 
        
        # Update history
        self.centroids.append(new_centroid)
        if len(self.centroids) > self.max_history:
            self.centroids.pop(0)
            
        self.current_centroid = new_centroid
        self.disappeared = 0
        self.total_frames += 1

class CentroidTracker:
    """Manages all tracked objects and performs ID association."""
    
    def __init__(self, maxDisappeared=10):
        self.objects = {}               # Dictionary of {ID: TrackedObject}
        self.nextObjectID = 0
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, bbox):
        """Registers a new object that was just detected."""
        objID = self.nextObjectID
        self.nextObjectID += 1
        self.objects[objID] = TrackedObject(objID, centroid)
        return objID

    def deregister(self, objectID):
        """Removes a lost object."""
        del self.objects[objectID]

    def update(self, rects):
        """
        Takes the new bounding boxes (rects) and matches them to existing objects.
        
        Args:
            rects (list): A list of tuples/lists, where each element is a bounding box [x1, y1, x2, y2].
            
        Returns:
            list: A list of dicts: [{'id': ID, 'bbox': [x1, y1, x2, y2], 'velocity': speed}]
        """
        
        # 1. Handle no detections
        if len(rects) == 0:
            lostIDs = list(self.objects.keys())
            for objectID in lostIDs:
                self.objects[objectID].disappeared += 1
                if self.objects[objectID].disappeared > self.maxDisappeared:
                    self.deregister(objectID)
            return []

        # 2. Compute new centroids
        inputCentroids = []
        inputCentroids_map = {} # Maps centroid index back to original bbox
        
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX, cY = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
            inputCentroids.append((cX, cY))
            inputCentroids_map[i] = [x1, y1, x2, y2]

        # 3. Handle first frame / initialization
        if len(self.objects) == 0:
            for (i, c) in enumerate(inputCentroids):
                self.register(c, inputCentroids_map[i])
        
        # 4. Match existing objects to new detections
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [obj.current_centroid for obj in self.objects.values()]

            # Compute Euclidean distance between all old and new centroids
            D = dist.cdist(np.array(objectCentroids), np.array(inputCentroids))

            # Perform assignment using the minimum distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            
            # --- Primary Assignment ---
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                objectID = objectIDs[row]
                new_centroid = inputCentroids[col]
                
                # Update the position and calculate velocity
                self.objects[objectID].update_position(new_centroid)
                
                usedRows.add(row)
                usedCols.add(col)

            # --- Handle Lost and New Objects ---
            inputIdxs = set(range(len(inputCentroids)))
            unassignedInputCentroids = list(inputIdxs - usedCols)

            # Deregister lost objects
            objectIdxs = set(range(len(objectCentroids)))
            unassignedObjectCentroids = list(objectIdxs - usedRows)

            for row in unassignedObjectCentroids:
                objectID = objectIDs[row]
                self.objects[objectID].disappeared += 1
                if self.objects[objectID].disappeared > self.maxDisappeared:
                    self.deregister(objectID)

            # Register new objects
            for col in unassignedInputCentroids:
                self.register(inputCentroids[col], inputCentroids_map[col])
        
        # 5. Compile final output with velocity
        final_output = []
        for objID, obj in self.objects.items():
            # NOTE: Bbox coordinates are lost in this simplified tracker, 
            # so we use a dummy size for demonstration simplicity.
            x1, y1 = int(obj.current_centroid[0] - 15), int(obj.current_centroid[1] - 30)
            x2, y2 = int(obj.current_centroid[0] + 15), int(obj.current_centroid[1] + 5)
            
            final_output.append({
                'id': objID,
                'bbox': [x1, y1, x2, y2],
                'velocity': obj.velocity,
                'centroid': obj.current_centroid
            })

        return final_output