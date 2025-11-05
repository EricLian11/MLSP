import numpy as np
import cv2
def compute_dct(block):
    block_float = np.float32(block)
    dct_block = cv2.dct(block_float)
    return dct_block
def classify_block_frequency(block,block_size = 8):
    dct_block = compute_dct(block)
    low_mask =np.zeros((block_size,block_size),dtype=bool)
    low_size = block_size //3
    low_mask[:low_size,:low_size] = True
    high_mask = np.zeros((block_size,block_size),dtype=bool)
    high_start = (2*block_size)//3
    high_mask[high_start:, high_start:] = True
    mid_mask = ~(low_mask | high_mask)
    dct_squared = dct_block ** 2
    energy_low = np.sum(dct_squared[low_mask])
    energy_high = np.sum(dct_squared[high_mask])
    energy_mid = np.sum(dct_squared[mid_mask])
    max_energy = max(energy_low,energy_mid,energy_high)
    if max_energy == energy_low:
        return "low"
    elif max_energy == energy_mid:
        return "mid"
    else:
        return "high"
def classify_all_blocks(blocks):
    classifications = []
    num_blocks = len(blocks)
    print(f"Classifying {num_blocks} blocks..")
    for i,block in enumerate(blocks):
        if i % 100000 == 0 and i > 0:
            print(f" Progress: {i}/{num_blocks} ({100*i/num_blocks:.1f}%)")
        classification = classify_block_frequency(block)
        classifications.append(classification)
    classifications = np.array(classifications)
    unique,counts = np.unique(classifications,return_counts = True)
    print("Classification Results")
    for class_name,count in zip(unique,counts):
        percentage = 100 * count / num_blocks
        print(f"{class_name}: {count} ({percentage:.1f}%)")
    return classifications
if __name__ == "__main__":
    print("Loading training blocks")
    training_blocks = np.load("training_blocks.npy")
    classifications = classify_all_blocks(training_blocks)
    np.save("block_classifications.npy",classifications)
    print("Done")