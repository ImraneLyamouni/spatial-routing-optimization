# High-Performance Spatial Sorting & Routing Optimization

An advanced, low-latency algorithm designed to optimize large-scale node routing and solve computational geometry bottlenecks using Python and NumPy. By leveraging localized spatial sorting and eliminating nested loops, this implementation scales efficiently to massive datasets without performance degradation or memory crashes.

## 🚀 Key Innovation
Traditional routing and geometric optimization algorithms often suffer from $O(N^2)$ or higher complexities due to nested iterations, leading to memory crashes on large datasets. This project introduces a novel strategy that utilizes vectorized matrix operations in NumPy and directional sorting to maintain a dense, high-efficiency computation path.

## 📊 Empirical Execution Results
The algorithm was benchmarked across various dataset scales, demonstrating exceptional scalability:


| Dataset Size (Nodes) | Execution Time (Seconds) | Total Traveled Distance |
| :--- | :--- | :--- |
| **700 Nodes** | 4.06 s | 21,592.43 |
| **1,000 Nodes** | 8.69 s | 26,650.15 |
| **100,000 Nodes** | 0.52 s | 34,345,303.98 |

*Note: The highly optimized dense path execution for 100,000 nodes takes advantage of advanced vectorization, significantly outperforming iterative structures on sparse datasets.*

## 📂 Repository Structure
* `main.py` / `notebook.ipynb`: Core Python execution files featuring vectorized NumPy logic.
* `plots/`: High-density fractal path visualizations and dataset distribution graphs.
* `data/`: Sample node coordinate matrices used for benchmarking.

## 🧑‍🔬 About the Author
I am **Imrane Lyamouni**, a 13-year-old independent researcher from Morocco. Passionate about algorithm optimization, computational geometry, and theoretical computer science. 

Continuous research and academic critique are welcomed. For academic inquiries or mentorship opportunities, please contact: **imranelyamouni0@gmail.com**.
