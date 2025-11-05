# decoder.py
import numpy as np
import cv2
import pickle

class AdaptiveRVQDecoder:
    def __init__(self, codebooks_path='data/codebooks.pkl'):
        with open(codebooks_path, 'rb') as f:
            self.codebooks = pickle.load(f)
    
    def decode_block(self, indices):
        reconstructed = np.zeros(64, dtype=np.float64)
        
        for stage, idx in enumerate(indices, 1):
            reconstructed += self.codebooks[f'stage{stage}'][idx]
        
        return np.clip(reconstructed, 0, 255).reshape(8, 8).astype(np.uint8)
    
    def decode_frame(self, encoded_data, frame_shape):
        h, w = frame_shape
        frame = np.zeros((h, w), dtype=np.uint8)
        
        block_idx = 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = self.decode_block(encoded_data[block_idx]['indices'])
                frame[i:i+8, j:j+8] = block
                block_idx += 1
        
        return frame
    
    def decode_video(self, encoded_frames, metadata, output_path='output.avi'):
        
        h, w = metadata['frame_shape']
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h), isColor=False)
        
        for idx, encoded_data in enumerate(encoded_frames):
            frame = self.decode_frame(encoded_data, (h, w))
            out.write(frame)
            if (idx + 1) % 10 == 0:
                print(f"  {idx + 1} frames decoded")
        
        out.release()
if __name__ == "__main__":
    decoder = AdaptiveRVQDecoder('data/codebooks.pkl')
    encoded_data = np.load('encoded_test.npy', allow_pickle=True).item()
    
    decoder.decode_video(
        encoded_data['frames'],
        encoded_data['metadata'],
        'decoded_test.avi'
    )
    