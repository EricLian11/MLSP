import numpy as np
from scipy.cluster.vq import kmeans2
import pickle
def train_codebook(vectors,n_codewords = 256,n_iter=20):
    vectors_float = vectors.astype(np.float64)
    
    codebook, labels = kmeans2(vectors_float, n_codewords, iter=n_iter, minit='points')
    
    return codebook
def quantize_with_codebook(vectors,codebook):
    from scipy.spatial.distance import cdist
    distances = cdist(vectors,codebook,metric='euclidean')
    indices = np.argmin(distances,axis=1)
    quantized = codebook[indices]
    residuals = vectors-quantized
    return indices,residuals
if __name__ == "__main__":
    training_vectors = np.load("training_vectors.npy")
    codebooks = {}
    print("Stage 1")
    codebooks['stage1']=train_codebook(training_vectors)
    print("Stage 1 residuals")
    _,residuals_1 = quantize_with_codebook(training_vectors,codebooks['stage1'])
    print(f"Residual range: [{residuals_1.min():.2f}, {residuals_1.max():.2f}]")
    
    print("\nStage 2: Training on Stage 1 residuals")
    codebooks['stage2'] = train_codebook(residuals_1)
    
    print("\nComputing Stage 2 residuals")
    _, residuals_2 = quantize_with_codebook(residuals_1, codebooks['stage2'])
    
    print("\nStage 3: Training on Stage 2 residuals")
    codebooks['stage3'] = train_codebook(residuals_2)
    
    print("\nComputing Stage 3 residuals")
    _, residuals_3 = quantize_with_codebook(residuals_2, codebooks['stage3'])
    
    print("\nStage 4: Training on Stage 3 residuals")
    codebooks['stage4'] = train_codebook(residuals_3)
    
    with open('codebooks.pkl', 'wb') as f:
        pickle.dump(codebooks, f)
    
    print(f"Total codebooks trained: {len(codebooks)}")
    for stage, codebook in codebooks.items():
        print(f"  {stage}: {codebook.shape}")