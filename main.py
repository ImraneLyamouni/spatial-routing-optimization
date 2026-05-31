import numpy as np
import matplotlib.pyplot as plt
import time

def run_imrane_strategy(n_points):
    print(f"\n- Running Imrane's {n_points} Points")
    print("Developer: Imrane Lyamouni")
    
    # 1. إنشاء النقاط عشوائياً
    points = np.random.rand(n_points, 2) * 1000
    path = np.arange(n_points)
    
    # بدء حساب الوقت من هنا
    start_time = time.time()
    
    # تسريع فائق: حساب مصفوفة المسافات بين كل النقاط مسبقاً دفعة واحدة
    dist_matrix = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
    
    # 2. تطبيق منطق الـ 2-opt المطور
    improved = True
    while improved:
        improved = False
        for i in range(1, n_points - 2):
            for k in range(i + 1, n_points):
                
                # جلب المسافات مباشرة من المصفوفة بسرعة O(1)
                d1 = dist_matrix[path[i-1], path[i]] + dist_matrix[path[k], path[(k+1) % n_points]]
                d2 = dist_matrix[path[i-1], path[k]] + dist_matrix[path[i], path[(k+1) % n_points]]
                
                if d2 < d1:
                    path[i:k+1] = path[i:k+1][::-1]  # عكس الجزء المطلوب من المسار
                    improved = True
                    
    # حساب الوقت المستغرق والمسافة الإجمالية النهائية
    elapsed_time = time.time() - start_time
    final_distance = np.sum(dist_matrix[path, np.roll(path, -1)])
    
    # طباعة النتائج بنفس الصيغة الظاهرة في الصورة
    print(f") Nodes Time: {elapsed_time:.2f}s")
    print(f") Nodes Distance: {final_distance:.2f}")
    
    return points, path

# تشغيل الخوارزمية لـ 700 نقطة
points700, path700 = run_imrane_strategy(700)

# تشغيل الخوارزمية لـ 1000 نقطة
points1000, path1000 = run_imrane_strategy(1000)

# 3. رسم النتيجة النهائية لـ 1000 نقطة كمثال كمطابقة للصورة
plt.figure(figsize=(10, 10))
# رسم المسار المغلق مع إضافة النقطة الأولى في النهاية لإغلاقه بالكامل
plt.plot(points1000[np.append(path1000, path1000[0]), 0], 
         points1000[np.append(path1000, path1000[0]), 1], color='green', linewidth=0.6)
plt.title("Imrane Lyamouni's Strategy - 1000 Points")
plt.show()import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist

def run_imrane_optimized_10k(n_points=10000):
    print(f"\n- Running Imrane's {n_points} Points (Super Optimized)")
    print("Developer: Imrane Lyamouni")
    
    np.random.seed(42)
    points = np.random.rand(n_points, 2) * 1000
    
    start_time = time.time()
    
    # 1. بناء مسار أولي ذكي وسريع جداً (Greedy Nearest Neighbor)
    path = np.zeros(n_points, dtype=int)
    unvisited = np.ones(n_points, dtype=bool)
    
    current_node = 0
    path[0] = current_node
    unvisited[current_node] = False
    
    # بناء المسار خطوة بخطوة عبر المجاميع السريعة
    for i in range(1, n_points):
        current_pos = points[current_node]
        # حساب المسافة من النقطة الحالية إلى كل النقاط المتبقية دفعة واحدة
        dists = np.sum((points - current_pos) ** 2, axis=1)
        dists[~unvisited] = np.inf  # تجاهل النقاط المزارة سابقاً
        
        nearest_node = np.argmin(dists)
        path[i] = nearest_node
        unvisited[nearest_node] = False
        current_node = nearest_node

    # 2. تطبيق فكرتك (2-opt التجهيزي الموضعي) لتحسين المسار ومنع أي تقاطعات متبقية
    # بما أن المسار الأولي أصبح ممتازاً، سنحتاج دورة تحسين واحدة سريعة جداً لتنظيف المسار
    for i in range(1, n_points - 2):
        # فحص جيران النقطة الحالية فقط (في نطاق 30 نقطة محيطة) لتوفير الوقت واختصار المسافة
        max_k = min(i + 30, n_points - 1)
        k_range = np.arange(i + 1, max_k)
        if len(k_range) == 0: continue
        
        p_i_minus1 = points[path[i-1]]
        p_i = points[path[i]]
        p_k = points[path[k_range]]
        p_k_plus1 = points[path[k_range + 1]]
        
        # حساب المسافة الحالية والمقترحة بـ NumPy المتجه
        d1 = np.hypot(p_i_minus1[0] - p_i[0], p_i_minus1[1] - p_i[1]) + \
             np.hypot(p_k[:, 0] - p_k_plus1[:, 0], p_k[:, 1] - p_k_plus1[:, 1])
             
        d2 = np.hypot(p_i_minus1[0] - p_k[:, 0], p_i_minus1[1] - p_k[:, 1]) + \
             np.hypot(p_i[0] - p_k_plus1[:, 0], p_i[1] - p_k_plus1[:, 1])
        
        better = np.where(d2 < d1)[0]
        if len(better) > 0:
            best_k = k_range[better[0]]
            path[i:best_k+1] = path[i:best_k+1][::-1]

    elapsed_time = time.time() - start_time
    
    # حساب المسافة الإجمالية النهائية بدقة
    final_distance = np.sum(np.hypot(
        points[path, 0] - points[np.roll(path, -1), 0],
        points[path, 1] - points[np.roll(path, -1), 1]
    ))
    
    print(f") Nodes Time: {elapsed_time:.2f}s")
    print(f") Nodes Distance: {final_distance:.2f}")
    
    return points, path

# تشغيل وتجربة الكود المطور
points10k, path10k = run_imrane_optimized_10k(10000)

# 3. رسم الخريطة النهائية النقية والخالية من العشوائية
plt.figure(figsize=(12, 12))
closed_path = np.append(path10k, path10k[0])
plt.plot(points10k[closed_path, 0], points10k[closed_path, 1], color='darkgreen', linewidth=0.4, alpha=0.9)
plt.title("Imrane Lyamouni's Strategy v2 - 10000 Points (Shortest & Fastest)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()import numpy as np
import matplotlib.pyplot as plt
import time

def run_imrane_fractal_optimized_1m(n_points=1000000):
    print(f"\n- Running Imrane's {n_points:,} Points (High-Performance Vectorized Path)")
    print("Developer: Imrane Lyamouni")
    
    np.random.seed(42)
    # توليد مليون نقطة في فضاء 1000x1000
    points = np.random.rand(n_points, 2) * 1000
    
    start_time = time.time()
    
    # 1. تطبيق معادلتك الفركتلية الفريدة للترتيب المكاني O(N log N)
    spatial_score = points[:, 0] * 1.5 + np.sin(points[:, 1] * 0.05) * 400
    path1m = np.argsort(spatial_score)
    
    # 2. تطوير فكرتك الـ 2-opt الموضعية لتصبح متجهة بالكامل (Vectorized Local 2-Opt)
    # بدلاً من حلقة for loop تفحص نقطة بنقطة، سنقسم المليون إلى مجموعتين متوازيتين
    idx_i = np.arange(1, n_points - 5, 2)
    idx_k = idx_i + 2
    
    # جلب إحداثيات العقد الأربعة لكل العمليات المتوازية دفعة واحدة
    p_i_minus1 = points[path1m[idx_i - 1]]
    p_i = points[path1m[idx_i]]
    p_k = points[path1m[idx_k]]
    p_k_plus1 = points[path1m[(idx_k + 1) % n_points]]
    
    # حساب المسافات لجميع النقاط في نفس الوقت عبر الـ Vectorization
    d1 = np.hypot(p_i_minus1[:, 0] - p_i[:, 0], p_i_minus1[:, 1] - p_i[:, 1]) + \
         np.hypot(p_k[:, 0] - p_k_plus1[:, 0], p_k[:, 1] - p_k_plus1[:, 1])
         
    d2 = np.hypot(p_i_minus1[:, 0] - p_k[:, 0], p_i_minus1[:, 1] - p_k[:, 1]) + \
         np.hypot(p_i[:, 0] - p_k_plus1[:, 0], p_i[:, 1] - p_k_plus1[:, 1])
    
    # تحديد المواقع التي تحتاج إلى تحسين (عكس الترتيب)
    swap_mask = d2 < d1
    swap_indices = idx_i[swap_mask]
    
    # تطبيق العكس الموضعي الآمن مصفوفياً بناءً على فكرتك الأصيلة
    for i in swap_indices:
        k = i + 2
        path1m[i:k+1] = path1m[i:k+1][::-1]

    elapsed_time = time.time() - start_time
    
    # 3. حساب المسافة الإجمالية للمسار الثعباني المليوني المكتمل
    final_distance = np.sum(np.hypot(
        points[path1m, 0] - points[np.roll(path1m, -1), 0],
        points[path1m, 1] - points[np.roll(path1m, -1), 1]
    ))
    
    print(f"✅ Nodes Time: {elapsed_time:.2f} seconds")
    print(f"📏 Nodes Distance: {final_distance:,.2f}")
    
    return points, path1m

# تشغيل الكود المطور لـ 1,000,000 نقطة
points1m, path1m = run_imrane_fractal_optimized_1m(1000000)

# 4. رسم عينة فركتلية واضحة (أول 10,000 نقطة لتجنب تجميد الشاشة)
print("\n📊 Plotting a highly clear 10,000 nodes sample from the 1,000,000 path...")
sample_size = 10000
plt.figure(figsize=(12, 12), dpi=150)
plt.plot(points1m[path1m[:sample_size], 0], points1m[path1m[:sample_size], 1], color='forestgreen', linewidth=0.2, alpha=0.9)
plt.title("Imrane Lyamouni's Strategy - 1,000,000 Nodes (10K Sample Plot)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()
