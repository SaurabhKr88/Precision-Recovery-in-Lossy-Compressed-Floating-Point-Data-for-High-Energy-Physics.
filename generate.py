import numpy as np
import struct
import matplotlib.pyplot as plt



# Generate large samples
N = 100000
uniform_data = np.random.uniform(0, 1, N)
gaussian_data = np.random.normal(0, 1, N)
exponential_data = np.random.exponential(1.0, N)

def float_to_bin(f):
    """Convert float to IEEE 754 binary representation (as an integer)."""
    return struct.unpack('!I', struct.pack('!f', f))[0]

def bin_to_float(b):
    """Convert IEEE 754 binary representation (integer) back to float."""
    return struct.unpack('!f', struct.pack('!I', b))[0]

def compress_float(f, bits_to_zero=8):
    """Zero out the least significant bits of the mantissa."""
    b = float_to_bin(f)
    mask = ~((1 << bits_to_zero) - 1)  # Create mask to zero out bits
    compressed_b = b & mask
    return bin_to_float(compressed_b)

def apply_compression(data, bits=8):
    return np.array([compress_float(f, bits) for f in data], dtype=np.float32)

# Apply compression
compressed_uniform = apply_compression(uniform_data, 8)
compressed_gaussian = apply_compression(gaussian_data, 8)
compressed_exponential = apply_compression(exponential_data, 8)

def save_binary(filename, data):
    data.tofile(filename)

# Save original data
save_binary("uniform_original.bin", uniform_data)
save_binary("gaussian_original.bin", gaussian_data)
save_binary("exponential_original.bin", exponential_data)

# Save compressed data
save_binary("uniform_compressed.bin", compressed_uniform)
save_binary("gaussian_compressed.bin", compressed_gaussian)
save_binary("exponential_compressed.bin", compressed_exponential)

def compute_statistics(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    stats = {
        "Mean (Original)": np.mean(original),
        "Mean (Compressed)": np.mean(compressed),
        "Variance (Original)": np.var(original),
        "Variance (Compressed)": np.var(compressed),
        "Mean Squared Error": mse
    }
    return stats

# Compare statistics
stats_uniform = compute_statistics(uniform_data, compressed_uniform)
stats_gaussian = compute_statistics(gaussian_data, compressed_gaussian)
stats_exponential = compute_statistics(exponential_data, compressed_exponential)
print("Uniform:", stats_uniform)
print("Gaussian:", stats_gaussian)
print("Exponential:", stats_exponential)

def plot_distributions(original, compressed, title):
    plt.figure(figsize=(8, 5))
    plt.hist(original, bins=100, alpha=0.6, label="Original", density=True)
    plt.hist(compressed, bins=100, alpha=0.6, label="Compressed", density=True)
    plt.legend()
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()

plot_distributions(uniform_data, compressed_uniform, "Uniform Distribution")
plot_distributions(gaussian_data, compressed_gaussian, "Gaussian Distribution")
plot_distributions(exponential_data, compressed_exponential, "Exponential Distribution")
