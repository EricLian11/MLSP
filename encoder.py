import numpy as np
import cv2
import pickle
from scipy.spatial.distance import cdist

class AdaptiveRVQEncoder:
    def __init__(self, codebooks_path='data/codebooks.pkl'):
        with open(codebooks_path, 'rb') as f:
            self.codebooks = pickle.load(f)
    
    def classify_block(self, block):
        return 'mid' if np.var(block) > 150 else 'low'
    
    def get_num_stages(self, classification):
        return {'low': 2, 'mid': 4, 'high': 3}[classification]
    
    def encode_block(self, block, num_stages):
        vector = block.flatten().astype(np.float64)
        residual = vector.copy()
        indices = []
        
        for stage in range(1, num_stages + 1):
            codebook = self.codebooks[f'stage{stage}']
            distances = cdist([residual], codebook, metric='euclidean')
            idx = np.argmin(distances)
            indices.append(idx)
            residual -= codebook[idx]
        
        return indices
    
    def encode_frame(self, frame):
        h, w = frame.shape
        encoded_data = []
        
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = frame[i:i+8, j:j+8]
                classification = self.classify_block(block)
                num_stages = self.get_num_stages(classification)
                indices = self.encode_block(block, num_stages)
                
                encoded_data.append({
                    'classification': classification,
                    'num_stages': num_stages,
                    'indices': indices
                })
        
        return encoded_data
    
    def encode_video(self, video_path, max_frames=None):
        print(f"Encoding: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        encoded_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            encoded_frames.append(self.encode_frame(gray))
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"  {frame_count} frames encoded")
        
        cap.release()
        
        # Statistics
        all_blocks = [b for frame in encoded_frames for b in frame]
        classifications = [b['classification'] for b in all_blocks]
        unique, counts = np.unique(classifications, return_counts=True)
        
        for cls, count in zip(unique, counts):
            print(f"  {cls}: {count} ({100*count/len(all_blocks):.1f}%)")
        
        metadata = {
            'frame_shape': gray.shape,
            'num_frames': frame_count,
            'total_blocks': len(all_blocks)
        }
        
        return encoded_frames, metadata

if __name__ == "__main__":
    encoder = AdaptiveRVQEncoder('data/codebooks.pkl')
    encoded_frames, metadata = encoder.encode_video(
        'data/cif_videos/coastguard_cif.y4m', 
        max_frames=10
    )
    
    np.save('encoded_test.npy', {
        'frames': encoded_frames,
        'metadata': metadata
    }, allow_pickle=True)
    
