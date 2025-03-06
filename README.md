#Precision Recovery in Lossy-Compressed Floating-Point Data for High Energy Physics
<br>
Overview
This project explores lossy floating-point compression by manipulating the least significant bits of the mantissa in IEEE 754 floating-point representation. The goal is to analyze how compression affects data storage and statistical properties, ensuring an optimal balance between precision and storage efficiency.
Libraries Used
 1.NumPy: To generate large floating-point datasets from different probability distributions.
 2.Struct: To convert floating-point numbers to and from their IEEE 754 binary representation for compression.
 3.Matplotlib: Plot original and compressed data to observe distribution changes.
 <br>
Steps Performed
    Step 1: Generate Floating-Point Data.
    We generate 100,000 floating-point numbers from three distributions:
        Uniform Distribution: Values between 0 and 1.
        Gaussian Distribution: Mean = 0, Standard Deviation = 1.
        Exponential Distribution: Mean = 1.0.
    <br>
       ''' N = 100000
        uniform_data = np.random.uniform(0, 1, N)
        gaussian_data = np.random.normal(0, 1, N)
        exponential_data = np.random.exponential(1.0, N)'''
    <br>
    Step 2: Implement Lossy Compression
    To compress the data, we zero out between 8 and 16 bits of the mantissa while preserving the exponent and sign bit.
       def float_to_bin(f):
        """Convert float to IEEE 754 binary representation (as an integer)."""
        return struct.unpack('!I', struct.pack('!f', f))[0]
        def bin_to_float(b):
        """Convert IEEE 754 binary representation (integer) back to float."""
        return struct.unpack('!f', struct.pack('!I', b))[0]
    Zero Out Least Significant Mantissa Bits
    def compress_float(f, bits_to_zero=8):
    """Zero out the least significant bits of the mantissa."""
    b = float_to_bin(f)  # Convert float to IEEE 754 binary
    mask = ~((1 << bits_to_zero) - 1)  # Create a mask to zero out bits
    compressed_b = b & mask  # Apply mask
    return bin_to_float(compressed_b)  # Convert back to float
<br>
    Apply Compression to Entire Dataset
        def apply_compression(data, bits=8):
            return np.array([compress_float(f, bits) for f in data], dtype=np.float32)
        compressed_uniform = apply_compression(uniform_data, 8)
        compressed_gaussian = apply_compression(gaussian_data, 8)
        compressed_exponential = apply_compression(exponential_data, 8)
<br>
    Step 3: Save Data to Binary Files
    Both the original and compressed datasets are stored in binary format for efficient storage.
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
<br>
    To check storage savings, run:
    ls -lh *.bin
    Step 4: Compute Statistical Differences
    We compare key statistical parameters before and after compression:
        Mean
        Variance
        Mean Squared Error (MSE)
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
    # Compute statistics
    stats_uniform = compute_statistics(uniform_data, compressed_uniform)
    stats_gaussian = compute_statistics(gaussian_data, compressed_gaussian)
    stats_exponential = compute_statistics(exponential_data, compressed_exponential)
    print("Uniform:", stats_uniform)
    print("Gaussian:", stats_gaussian)
    print("Exponential:", stats_exponential)
    <br>
    Step 5: Visualize Distribution Changes
    We plot histograms of the original and compressed datasets to analyze how compression affects their distributions.
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
Results & Observations
Storage Savings
    1.The compressed binary files are smaller than the original ones.
    2.More bits zeroed out → Higher compression, but more loss in precision.


   




    

