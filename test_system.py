# test_system.py
import numpy as np
from encoder import AdaptiveRVQEncoder
from decoder import AdaptiveRVQDecoder
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def test_encoder_decoder():
    print("=== Testing Adaptive RVQ System ===\n")
    
    encoder = AdaptiveRVQEncoder('data/codebooks.pkl')
    decoder = AdaptiveRVQDecoder('data/codebooks.pkl')
    video_path = 'data/cif_videos/coastguard_cif.y4m'
    encoded_frames, metadata = encoder.encode_video(video_path, max_frames=30)
    decoder.decode_video(encoded_frames, metadata, 'output_reconstructed.avi')
    cap_original = cv2.VideoCapture(video_path)
    cap_decoded = cv2.VideoCapture('output_reconstructed.avi')
    psnr_values = []
    ssim_values = []
    frame_count = 0

    while frame_count < 30:
        ret1, frame1 = cap_original.read()
        ret2, frame2 = cap_decoded.read()
        
        if not ret1 or not ret2:
            print(f"Stopped at frame {frame_count}")
            break
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        if len(frame2.shape) == 3:
            gray2 = frame2[:, :, 0]  
        else:
            gray2 = frame2
        
        h, w = metadata['frame_shape']
        gray1 = gray1[:h, :w]
        gray2 = gray2[:h, :w]
        
        psnr = peak_signal_noise_ratio(gray1, gray2, data_range=255)
        ssim = structural_similarity(gray1, gray2, data_range=255)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        frame_count += 1
    
    cap_original.release()
    cap_decoded.release()
    
    total_bits = sum(
        2 + 8 * block['num_stages']
        for frame in encoded_frames
        for block in frame
    )
    h, w = metadata['frame_shape']
    total_pixels = metadata['num_frames'] * h * w
    bits_per_pixel = total_bits / total_pixels
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Frames processed: {frame_count}")
    print(f"Average PSNR: {np.mean(psnr_values):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")
    print(f"Bitrate: {bits_per_pixel:.3f} bpp ({total_bits} total bits)")
    print(f"Frame size: {h}x{w}")
    print(f"\n✓ System test complete!")

if __name__ == "__main__":
    test_encoder_decoder()