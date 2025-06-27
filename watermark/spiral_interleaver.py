"""
螺旋交织编解码模块
使用numpy实现高效的螺旋交织算法，用于数据错误分散

依赖库：
- numpy: 矩阵运算和数组处理
- math: 数学计算
- itertools: 迭代器工具（用于错误分析）

使用方法：
    from spiral_interleaver import spiral_encode, spiral_decode
    
    # 编码
    data = b"Hello World!"
    encoded = spiral_encode(data)
    
    # 解码  
    decoded = spiral_decode(encoded, len(data))
    assert decoded == data
"""

import numpy as np
import math
import itertools
from typing import Union, Tuple, Optional, List

# 全局配置
SPIRAL_CACHE = {}  # 路径缓存，提高性能

def calculate_optimal_matrix_size(data_length: int) -> Tuple[int, int]:
    """
    计算最优的矩阵尺寸
    
    优先选择接近正方形的矩阵，以获得最佳的螺旋交织效果
    
    Args:
        data_length: 数据长度
        
    Returns:
        (width, height) 元组
    """
    if data_length <= 0:
        return (1, 1)
        
    # 寻找最接近正方形的因数分解
    sqrt_len = int(math.sqrt(data_length))
    
    # 从sqrt开始向下寻找因数
    for width in range(sqrt_len, 0, -1):
        if data_length % width == 0:
            height = data_length // width
            return (width, height)
    
    # 如果没找到完美因数，使用ceil来创建略大的矩阵
    width = sqrt_len + 1
    height = math.ceil(data_length / width)
    
    return (width, height)
def generate_clockwise_spiral_path(width: int, height: int) -> List[Tuple[int, int]]:
    """
    生成顺时针从外向内的螺旋路径
    
    这是最常用和稳定的螺旋模式
    
    Args:
        width: 矩阵宽度
        height: 矩阵高度
        
    Returns:
        坐标列表 [(row, col), ...]
    """
    # 检查缓存
    cache_key = (width, height, 'clockwise_in')
    if cache_key in SPIRAL_CACHE:
        return SPIRAL_CACHE[cache_key]
    
    path = []
    top, bottom = 0, height - 1
    left, right = 0, width - 1
    
    while top <= bottom and left <= right:
        # 上边：左到右
        for col in range(left, min(right + 1, width)):
            if top < height:
                path.append((top, col))
        top += 1
        
        # 右边：上到下
        for row in range(top, min(bottom + 1, height)):
            if right < width:
                path.append((row, right))
        right -= 1
        
        # 下边：右到左
        if top <= bottom:
            for col in range(right, left - 1, -1):
                if bottom < height and col >= 0:
                    path.append((bottom, col))
            bottom -= 1
        
        # 左边：下到上
        if left <= right:
            for row in range(bottom, top - 1, -1):
                if left < width and row >= 0:
                    path.append((row, left))
            left += 1
    
    # 缓存结果
    SPIRAL_CACHE[cache_key] = path
    return path

def spiral_encode(data: Union[bytes, bytearray, List[int], np.ndarray]) -> bytes:
    """
    螺旋交织编码
    
    Args:
        data: 输入数据（bytes, bytearray, list或numpy数组）
        
    Returns:
        交织后的数据（bytes）
        
    Example:
        >>> data = b"Hello World!"
        >>> encoded = spiral_encode(data)
        >>> decoded = spiral_decode(encoded, len(data))
        >>> assert decoded == data
    """
    if not data:
        return b''
    
    # 转换为numpy数组
    if isinstance(data, (bytes, bytearray)):
        data_array = np.frombuffer(data, dtype=np.uint8)
    elif isinstance(data, list):
        data_array = np.array(data, dtype=np.uint8)
    else:
        data_array = data.astype(np.uint8)
    
    data_length = len(data_array)
    width, height = calculate_optimal_matrix_size(data_length)
    
    # 创建矩阵并按行填充数据
    matrix = np.zeros((height, width), dtype=np.uint8)
    
    for i, value in enumerate(data_array):
        row = i // width
        col = i % width
        if row < height and col < width:
            matrix[row, col] = value
    
    # 生成螺旋路径并按路径读取数据
    spiral_path = generate_clockwise_spiral_path(width, height)
    
    interleaved_data = []
    for i, (row, col) in enumerate(spiral_path):
        if i < data_length and row < height and col < width:
            interleaved_data.append(matrix[row, col])
    
    return bytes(interleaved_data)

def spiral_decode(interleaved_data: Union[bytes, bytearray, List[int], np.ndarray], 
                  original_length: Optional[int] = None) -> bytes:
    """
    螺旋交织解码
    
    Args:
        interleaved_data: 交织后的数据
        original_length: 原始数据长度，None时使用交织数据长度
        
    Returns:
        解码后的原始数据（bytes）
        
    Example:
        >>> encoded = spiral_encode(b"Hello!")
        >>> decoded = spiral_decode(encoded, 6)
        >>> assert decoded == b"Hello!"
    """
    if not interleaved_data:
        return b''
    
    # 转换为numpy数组
    if isinstance(interleaved_data, (bytes, bytearray)):
        data_array = np.frombuffer(interleaved_data, dtype=np.uint8)
    elif isinstance(interleaved_data, list):
        data_array = np.array(interleaved_data, dtype=np.uint8)
    else:
        data_array = interleaved_data.astype(np.uint8)
    
    data_length = len(data_array)
    if original_length is not None:
        data_length = min(data_length, original_length)
    
    # 计算矩阵尺寸（必须与编码时一致）
    width, height = calculate_optimal_matrix_size(data_length)
    
    # 创建矩阵
    matrix = np.zeros((height, width), dtype=np.uint8)
    
    # 生成螺旋路径（与编码时相同）
    spiral_path = generate_clockwise_spiral_path(width, height)
    
    # 按螺旋路径填充数据到矩阵
    for i, (row, col) in enumerate(spiral_path):
        if i < len(data_array) and row < height and col < width:
            matrix[row, col] = data_array[i]
    
    # 按行顺序读取数据（逆向编码过程）
    decoded_data = []
    for i in range(data_length):
        row = i // width
        col = i % width
        if row < height and col < width:
            decoded_data.append(matrix[row, col])
    
    return bytes(decoded_data)

def test_spiral_interleaver():
    """
    测试螺旋交织器的功能
    """
    print("="*80)
    print("螺旋交织器测试")
    print("="*80)
    
    # 基本功能测试
    test_cases = [
        b"Hello World!",
        b"A" * 16,  # 完美正方形
        b"Test" * 10,  # 40字节
        b"X",  # 单字节
        b"",   # 空数据
        bytes(range(256)),  # 完整字节范围
    ]
    
    for i, test_data in enumerate(test_cases):
        print(f"\n{'-'*40}")
        print(f"测试用例 {i+1}: {len(test_data)} 字节")
        if len(test_data) <= 50:  # 只显示短数据
            print(f"数据: {test_data[:50]}")
        
        try:
            # 编码
            encoded = spiral_encode(test_data)
            print(f"编码后长度: {len(encoded)} 字节")
            
            # 验证编码是否改变了数据
            if len(encoded) == len(test_data):
                print("✓ 长度保持一致")
            else:
                print(f"⚠ 长度发生变化: {len(test_data)} -> {len(encoded)}")
            
            if len(test_data) > 0 and set(encoded) == set(test_data):
                print("✓ 字节集合保持一致")
            
            # 解码
            decoded = spiral_decode(encoded, len(test_data))
            print(f"解码后长度: {len(decoded)} 字节")
            
            # 验证解码正确性
            if decoded == test_data:
                print("✅ 编解码测试通过")
            else:
                print("❌ 编解码测试失败")
                if len(decoded) == len(test_data):
                    diff_count = sum(a != b for a, b in zip(decoded, test_data))
                    print(f"差异字节数: {diff_count}")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    # 性能测试
    print(f"\n{'-'*40}")
    print("性能测试")
    
    import time
    large_data = b"Performance test data. " * 1000  # ~23KB
    
    # 编码性能
    start_time = time.time()
    for _ in range(100):
        encoded = spiral_encode(large_data)
    encode_time = time.time() - start_time
    
    # 解码性能
    start_time = time.time()
    for _ in range(100):
        decoded = spiral_decode(encoded, len(large_data))
    decode_time = time.time() - start_time
    
    print(f"数据大小: {len(large_data)} 字节")
    print(f"编码性能: {encode_time:.4f}s / 100次 = {encode_time*10:.2f}ms/次")
    print(f"解码性能: {decode_time:.4f}s / 100次 = {decode_time*10:.2f}ms/次")
    print(f"编码吞吐量: {len(large_data)*100/encode_time/1024/1024:.2f} MB/s")

def demonstrate_error_spreading():
    """
    演示螺旋交织的错误分散效果
    """
    print(f"\n{'='*80}")
    print("错误分散效果演示")
    print("="*80)
    
    # 创建测试数据：重复模式便于观察
    pattern = b"ABCDEFGHIJKLMNOP"
    test_data = pattern * 4  # 64字节
    
    print(f"原始数据: {test_data.decode('ascii', errors='ignore')}")
    
    # 编码
    encoded = spiral_encode(test_data)
    print(f"交织后: {encoded.decode('ascii', errors='ignore')}")
    
    # 模拟连续错误：将第16-31字节全部置为X
    damaged = bytearray(encoded)
    error_start, error_end = 16, min(32, len(damaged))
    for i in range(error_start, error_end):
        damaged[i] = ord('X')
    
    print(f"损坏数据: {bytes(damaged).decode('ascii', errors='ignore')}")
    print(f"错误位置: {error_start}-{error_end-1} (连续{error_end-error_start}字节)")
    
    # 解码看错误分布
    decoded_damaged = spiral_decode(damaged, len(test_data))
    print(f"解码后: {decoded_damaged.decode('ascii', errors='ignore')}")
    
    # 统计错误分布
    error_positions = []
    for i, (orig, recv) in enumerate(zip(test_data, decoded_damaged)):
        if orig != recv:
            error_positions.append(i)
    
    print(f"错误位置: {error_positions}")
    print(f"错误总数: {len(error_positions)}")
    
    # 计算最大连续错误长度
    if error_positions:
        consecutive_errors = []
        current_run = 1
        for i in range(1, len(error_positions)):
            if error_positions[i] == error_positions[i-1] + 1:
                current_run += 1
            else:
                consecutive_errors.append(current_run)
                current_run = 1
        consecutive_errors.append(current_run)
        max_consecutive = max(consecutive_errors)
    else:
        max_consecutive = 0
    
    print(f"最大连续错误长度: {max_consecutive}")
    print(f"错误分散效果: {'良好' if max_consecutive <= len(error_positions)//2 else '一般'}")

def get_matrix_visualization(data: bytes, max_size: int = 16) -> str:
    """
    可视化矩阵填充过程（仅用于小数据）
    
    Args:
        data: 输入数据
        max_size: 最大显示的数据长度
        
    Returns:
        矩阵可视化字符串
    """
    if len(data) > max_size:
        return f"数据太大({len(data)}字节)，跳过可视化"
    
    if len(data) == 0:
        return "空数据"
    
    width, height = calculate_optimal_matrix_size(len(data))
    
    # 创建显示矩阵
    display_matrix = [['.' for _ in range(width)] for _ in range(height)]
    
    # 按行填充
    for i, byte_val in enumerate(data):
        row = i // width
        col = i % width
        if row < height and col < width:
            display_matrix[row][col] = chr(byte_val) if 32 <= byte_val <= 126 else f'{byte_val:02x}'
    
    # 生成螺旋路径并标记
    spiral_path = generate_clockwise_spiral_path(width, height)
    
    result = f"矩阵尺寸: {width}x{height}\n"
    result += "按行填充:\n"
    for row in display_matrix:
        result += " ".join(f"{cell:>2}" for cell in row) + "\n"
    
    # 显示螺旋路径前几个位置
    result += f"\n螺旋路径前10个位置: {spiral_path[:10]}\n"
    
    return result

# 便捷的别名函数
def spiral_interleave(data: bytes) -> bytes:
    """螺旋交织编码的别名"""
    return spiral_encode(data)

def spiral_deinterleave(data: bytes, original_length: int = None) -> bytes:
    """螺旋交织解码的别名"""
    return spiral_decode(data, original_length)

if __name__ == "__main__":
    # 运行所有测试
    test_spiral_interleaver()
    demonstrate_error_spreading()
    
    print(f"\n{'='*80}")
    print("使用示例")
    print("="*80)
    print("""
# 基本使用
from spiral_interleaver import spiral_encode, spiral_decode

# 编码
data = b"Your data here"
encoded = spiral_encode(data)

# 解码
decoded = spiral_decode(encoded, len(data))
assert decoded == data

# 可视化小数据的矩阵排列
from spiral_interleaver import get_matrix_visualization
print(get_matrix_visualization(b"Hello World!"))

# 别名函数
interleaved = spiral_interleave(data)
deinterleaved = spiral_deinterleave(interleaved, len(data))
""")
    
    # 简单可视化演示
    print(f"\n{'='*80}")
    print("矩阵可视化演示")
    print("="*80)
    
    demo_data = b"Hello World!"
    print(get_matrix_visualization(demo_data))