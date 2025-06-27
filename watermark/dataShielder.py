## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms
"""
鲁棒二进制信息嵌入编码器（修复版）

修复了以下问题：
1. CRC16-CCITT实现错误
2. 同步模式匹配问题
3. 数据包提取逻辑错误
"""

import struct
import zlib
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
from collections import defaultdict, Counter
from reedsolo import RSCodec
import numpy as np
import pyldpc
USE_PYLDPC = True


def crc16_ccitt(data: bytes) -> int:
    """
    计算CRC16-CCITT校验码（修复版）
    使用标准的CRC16-CCITT-FALSE（初始值0xFFFF）
    """
    crc = 0xFFFF
    
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    
    return crc

def get_encoding_stats(text: str) -> dict:
    """获取编码统计信息"""
    import zlib
    
    if isinstance(text, str):
        original_bytes = text.encode('utf-8')
    else:
        original_bytes = text
    
    compressed_bytes = zlib.compress(original_bytes, level=9)
    min_capacity_bits, total_chunks = estimate_required_capacity(text)
    
    return {
        'original_size': len(original_bytes),
        'compressed_size': len(compressed_bytes),
        'min_capacity_bits': min_capacity_bits,
        'total_chunks': total_chunks
    }

def adapt_to_watermark_capacity(text: str, available_capacity: int, safety_margin: float = 0.85) -> List[bool]:
    """根据可用容量自动适配编码"""
    target_capacity = int(available_capacity * safety_margin)
    
    # 先尝试标准编码
    min_required, _ = estimate_required_capacity(text)
    
    if min_required <= target_capacity:
        # 容量足够，使用标准编码并重复填充
        return encode_binary_to_watermark(text, max_capacity_bits=target_capacity)
    else:
        # 容量不够，截断文本
        # 二分查找最大可编码的文本长度
        left, right = 1, len(text)
        best_text = text[:1]  # 至少保留一个字符
        
        while left <= right:
            mid = (left + right) // 2
            test_text = text[:mid]
            required, _ = estimate_required_capacity(test_text)
            
            if required <= target_capacity:
                best_text = test_text
                left = mid + 1
            else:
                right = mid - 1
        
        print(f"警告：文本过长，已截断到 {len(best_text)} 字符")
        return encode_binary_to_watermark(best_text, max_capacity_bits=target_capacity)

def print_encoding_report(text: str):
    """打印编码报告"""
    stats = get_encoding_stats(text)
    print(f"\n=== 编码报告 ===")
    print(f"原始文本长度: {len(text)} 字符")
    print(f"UTF-8字节数: {stats['original_size']} 字节")
    print(f"压缩后字节数: {stats['compressed_size']} 字节")
    print(f"压缩率: {(1 - stats['compressed_size']/stats['original_size'])*100:.1f}%")
    print(f"需要最小容量: {stats['min_capacity_bits']} 比特")
    print(f"数据分片数: {stats['total_chunks']}")
    print(f"================")

def analyze_watermark_quality(bitstream: List[bool]) -> Dict:
    """分析水印质量"""
    # 基本质量指标
    total_bits = len(bitstream)
    if total_bits == 0:
        return {
            'quality_metrics': {
                'data_integrity_score': 0.0,
                'rs_success_rate': 0.0,
                'chunk_recovery_rate': 0.0
            }
        }
    
    # 尝试提取数据包
    try:
        packets = extract_packets_from_bitstream(bitstream)
        rs_decoder = ReedSolomonDecoder()
        
        successful_decodes = 0
        total_packets = len(packets)
        
        for packet_bytes, _ in packets:
            try:
                rs_decoder.decode(packet_bytes)
                successful_decodes += 1
            except:
                pass
        
        rs_success_rate = successful_decodes / max(total_packets, 1)
        
        # 尝试解码以获取分片信息
        _, stats = decode_watermark_to_binary(bitstream)
        chunk_recovery_rate = len(stats['chunks_recovered']) / max(1, 
            max(stats['chunks_recovered'].keys(), default=0) + 1 if stats['chunks_recovered'] else 1)
        
        # 综合评分
        data_integrity_score = (rs_success_rate * 0.6 + chunk_recovery_rate * 0.4)
        
    except:
        rs_success_rate = 0.0
        chunk_recovery_rate = 0.0
        data_integrity_score = 0.0
    
    return {
        'quality_metrics': {
            'data_integrity_score': data_integrity_score,
            'rs_success_rate': rs_success_rate,
            'chunk_recovery_rate': chunk_recovery_rate
        }
    }

def print_decode_report(bitstream: List[bool]):
    """打印解码报告"""
    print("\n=== 解码质量报告 ===")
    
    # 基本信息
    print(f"比特流长度: {len(bitstream)} 比特")
    
    # 同步头检测
    sync_positions = find_sync_patterns(bitstream)
    print(f"检测到同步头: {len(sync_positions)} 个")
    
    # 数据包提取
    packets = extract_packets_from_bitstream(bitstream)
    print(f"提取到数据包: {len(packets)} 个")
    
    # RS解码测试
    if packets:
        rs_decoder = ReedSolomonDecoder()
        successful = 0
        for packet_bytes, pos in packets[:3]:  # 只测试前3个
            try:
                decoded_data, errors = rs_decoder.decode(packet_bytes)
                successful += 1
                print(f"  包@{pos}: RS解码成功，纠错{errors}字节")
            except Exception as e:
                print(f"  包@{pos}: RS解码失败 - {e}")
        
        success_rate = successful / len(packets) * 100
        print(f"RS解码成功率: {success_rate:.1f}%")
    
    # 尝试完整解码
    try:
        decoded_text, stats = decode_watermark_to_string(bitstream)
        if decoded_text:
            print(f"完整解码: 成功")
            print(f"有效包: {stats['valid_packets']}/{stats['total_packets_found']}")
            print(f"纠错总数: {stats['total_errors_corrected']} 字节")
        else:
            print(f"完整解码: 失败")
    except Exception as e:
        print(f"完整解码: 异常 - {e}")
    
    print("=====================")


class ReedSolomonEncoder:
    """基于reedsolo库的Reed-Solomon编码器"""
    
    def __init__(self, n: int = 255, k: int = 63):
        self.n = n
        self.k = k
        self.parity_symbols = n - k
        self.rs_codec = RSCodec(self.parity_symbols, nsize=n)
        
    def encode(self, data: bytes) -> bytes:
        if len(data) > self.k:
            raise ValueError(f"数据长度{len(data)}超过RS码的k值{self.k}")
        
        # 填充数据到k长度
        padded_data = data + b'\x00' * (self.k - len(data))
        
        # 使用reedsolo进行编码
        encoded = self.rs_codec.encode(padded_data)
        
        # 确保返回长度为n
        return bytes(encoded[:self.n])
    
    def get_correction_capacity(self) -> int:
        return self.parity_symbols // 2


class ReedSolomonDecoder:
    """基于reedsolo库的Reed-Solomon解码器"""
    
    def __init__(self, n: int = 255, k: int = 63):
        self.n = n
        self.k = k
        self.parity_symbols = n - k
        
        try:
            from reedsolo import RSCodec
            self.rs_codec = RSCodec(self.parity_symbols, nsize=n)
            self.codec_available = True
        except ImportError:
            print("[警告] reedsolo库未安装，使用模拟解码")
            self.codec_available = False
        
    def decode(self, encoded_data: bytes) -> Tuple[bytes, int]:
        print(f"    RS解码输入: {len(encoded_data)} 字节")
        
        # 确保数据长度正确
        if len(encoded_data) != self.n:
            if len(encoded_data) < self.n:
                encoded_data = encoded_data + b'\x00' * (self.n - len(encoded_data))
                print(f"      填充到 {self.n} 字节")
            else:
                encoded_data = encoded_data[:self.n]
                print(f"      截断到 {self.n} 字节")
        
        if not self.codec_available:
            # 模拟解码 - 仅用于测试
            print("      使用模拟解码")
            return encoded_data[:self.k], 0
        
        try:
            # 使用reedsolo进行解码
            decoded = self.rs_codec.decode(encoded_data)
            
            # 处理不同的返回格式
            if isinstance(decoded, tuple):
                decoded_data = decoded[0]
                if len(decoded) > 1:
                    correction_info = decoded[1]
                    if isinstance(correction_info, int):
                        num_corrected = correction_info
                    elif isinstance(correction_info, (list, tuple)):
                        num_corrected = len(correction_info)
                    else:
                        num_corrected = 0
                else:
                    num_corrected = 0
            else:
                decoded_data = decoded
                num_corrected = 0
            
            # 确保返回bytes类型
            if not isinstance(decoded_data, bytes):
                if isinstance(decoded_data, (list, tuple, bytearray)):
                    decoded_data = bytes(decoded_data)
                else:
                    decoded_data = bytes(decoded_data)
            
            print(f"      RS解码成功，纠错 {num_corrected} 个符号")
            return decoded_data[:self.k], num_corrected
            
        except Exception as e:
            print(f"      RS解码失败: {e}")
            
            # 尝试软判决解码或错误恢复
            try:
                # 尝试仅纠错而不检测擦除
                decoded = self.rs_codec.decode(encoded_data, erase_pos=None)
                if isinstance(decoded, tuple):
                    decoded_data = decoded[0]
                else:
                    decoded_data = decoded
                
                if not isinstance(decoded_data, bytes):
                    decoded_data = bytes(decoded_data)
                
                print(f"      RS软判决解码成功")
                return decoded_data[:self.k], -1  # -1表示软判决解码
                
            except Exception as e2:
                print(f"      RS软判决解码也失败: {e2}")
                # 返回原始数据的前k字节作为最后的尝试
                return encoded_data[:self.k], -2  # -2表示解码完全失败


def encode_binary_to_watermark(
    data: Union[bytes, str],
    max_capacity_bits: int = None,
    ldpc_rate: float = 0.25,  # 1/4 码率
    ldpc_block_size: int = 12000,  # LDPC码块长度
    rs_n: int = 255,
    rs_k: int = 63,  # 增强型RS(255,223)
    sync_header: int = 0xB593,
    spiral_interleave: bool = True,
    tmr_redundancy: bool = True,  # 三模冗余
    variable_repetition: bool = True  # 可变重复率
) -> List[bool]:
    """增强版水印编码"""
    
    # 1. 数据预处理
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    # 2. 压缩数据
    compressed_data = zlib.compress(data, level=9)
    
    # 3. 第一层：可变重复编码（针对关键数据段）
    if variable_repetition:
        # 对前10%的数据使用5倍重复，其余使用3倍重复
        critical_len = len(compressed_data) // 10
        critical_data = compressed_data[:critical_len]
        normal_data = compressed_data[critical_len:]
        
        repeated_critical = repeat_encode(critical_data, 5)
        repeated_normal = repeat_encode(normal_data, 3)
        layer1_output = repeated_critical + repeated_normal
    else:
        layer1_output = compressed_data
    
    # 4. 第二层：增强型Reed-Solomon编码
    rs_encoder = ReedSolomonEncoder(rs_n, rs_k)
    rs_packets = []
    
    metadata_size = 12  # 增加了纠错层信息
    k_data = rs_k - metadata_size
    total_chunks = (len(layer1_output) + k_data - 1) // k_data
    
    for chunk_index in range(total_chunks):
        start_idx = chunk_index * k_data
        end_idx = min(start_idx + k_data, len(layer1_output))
        data_chunk = layer1_output[start_idx:end_idx]
        
        # 构建增强元数据
        metadata = struct.pack('>I', len(compressed_data))  # 原始长度
        metadata += struct.pack('>H', chunk_index)
        metadata += struct.pack('>H', total_chunks)
        metadata += struct.pack('>B', 0x01)  # 纠错层版本
        metadata += struct.pack('>B', ldpc_block_size // 1000)  # LDPC参数
        
        header_crc = crc16_ccitt(metadata)
        metadata += struct.pack('>H', header_crc)
        
        packet_data = metadata + data_chunk
        if len(packet_data) < rs_k:
            packet_data += b'\x00' * (rs_k - len(packet_data))
        
        encoded_packet = rs_encoder.encode(packet_data)
        rs_packets.append(encoded_packet)
    
    # 5. 第三层：LDPC编码
    ldpc_output = []
    ldpc_metadata = []  # 新增：存储每个包的元数据
    for packet in rs_packets:
        ldpc_encoded, original_length = ldpc_encode(packet, ldpc_rate, ldpc_block_size)
        ldpc_output.extend(ldpc_encoded)
        ldpc_metadata.append(original_length)  # 保存长度信息以供解码使用
    
    # 6. 螺旋交织
    if spiral_interleave:
        interleaved_data = spiral_interleave_enhanced(ldpc_output)
    else:
        interleaved_data = ldpc_output
    
    # 7. 添加同步头和转换为比特流
    master_bitstream = []
    sync_header_bytes = struct.pack('>H', sync_header)
    
    # 为每个LDPC块添加同步头
    for i in range(0, len(interleaved_data), ldpc_block_size // 8):
        block = interleaved_data[i:i + ldpc_block_size // 8]
        full_block = sync_header_bytes + block
        
        for byte_val in full_block:
            for bit in range(8):
                master_bitstream.append(bool((byte_val >> (7 - bit)) & 1))
    
    # 8. 三模冗余（TMR）
    if tmr_redundancy:
        tmr_bitstream = []
        # 将比特流分成三份，交错排列
        third_len = len(master_bitstream) // 3
        for i in range(third_len):
            # 每个比特重复3次，但位置分散
            tmr_bitstream.append(master_bitstream[i])
            tmr_bitstream.append(master_bitstream[i + third_len])
            tmr_bitstream.append(master_bitstream[i + 2 * third_len])
        master_bitstream = tmr_bitstream
    
    # 9. 扩频编码
    spread_bitstream = apply_dsss(master_bitstream)
    
    # 10. 最终填充
    if max_capacity_bits is None:
        final_bitstream = spread_bitstream
    else:
        final_bitstream = []
        bit_index = 0
        while len(final_bitstream) < max_capacity_bits:
            final_bitstream.append(spread_bitstream[bit_index % len(spread_bitstream)])
            bit_index += 1
    
    return final_bitstream


def interleave_bitstream(bitstream, depth=8):
    """对比特流进行交织"""
    rows = [bitstream[i::depth] for i in range(depth)]
    interleaved = []
    for i in range(max(len(row) for row in rows)):
        for row in rows:
            if i < len(row):
                interleaved.append(row[i])
    return interleaved


def repeat_encode(data: bytes, repetition_factor: int) -> bytes:
    """可变重复编码"""
    result = bytearray()
    for byte_val in data:
        for _ in range(repetition_factor):
            result.append(byte_val)
    return bytes(result)

def spiral_interleave_enhanced(data: bytes) -> bytes:
    """修复的螺旋交织函数 - 保证长度不变"""
    if len(data) == 0:
        return data
    
    # 使用较小的素数深度，避免复杂性导致的错误
    prime_depths = [7, 11, 13]  # 减少深度数量
    result = bytearray(data)
    
    for depth in prime_depths:
        if len(result) <= depth:
            continue  # 跳过太大的深度
        
        # 创建索引映射而不是实际移动数据
        temp = bytearray(len(result))
        
        # 计算每行有多少元素
        rows_per_depth = len(result) // depth
        remainder = len(result) % depth
        
        # 创建行索引
        read_idx = 0
        write_idx = 0
        
        # 按行写入，但使用螺旋偏移
        for row in range(depth):
            # 计算这一行有多少个元素
            row_size = rows_per_depth + (1 if row < remainder else 0)
            
            for col in range(row_size):
                if read_idx < len(result):
                    # 计算螺旋偏移的写入位置
                    spiral_col = (col + row * 3) % row_size  # 简单的螺旋偏移
                    target_idx = row + spiral_col * depth
                    
                    # 确保目标索引在有效范围内
                    if target_idx < len(temp):
                        temp[target_idx] = result[read_idx]
                    else:
                        # 如果目标位置超出范围，顺序放置
                        temp[write_idx] = result[read_idx]
                    
                    read_idx += 1
                    write_idx += 1
        
        # 填充任何遗漏的位置
        for i in range(len(temp)):
            if i < len(result) and temp[i] == 0 and result[i] != 0:
                temp[i] = result[i]
        
        result = temp
    
    # 最终验证长度
    assert len(result) == len(data), f"交织改变了数据长度: {len(data)} -> {len(result)}"
    return bytes(result)

def apply_dsss(bitstream: List[bool], spreading_factor: int = 15) -> List[bool]:
    """修复的直接序列扩频（DSSS）"""
    # 生成伪随机扩频码
    pn_sequence = generate_pn_sequence(spreading_factor)
    spread_bits = []
    
    for bit in bitstream:
        # 每个比特用扩频码调制
        for pn_bit in pn_sequence:
            # 使用BPSK调制：原始比特XOR扩频码
            spread_bits.append(bit ^ pn_bit)
    
    return spread_bits


def generate_pn_sequence(length: int) -> List[bool]:
    """生成伪随机序列"""
    # 使用线性反馈移位寄存器（LFSR）
    lfsr = 0b10011  # 5位初始值
    sequence = []
    
    for _ in range(length):
        bit = (lfsr & 1)
        sequence.append(bool(bit))
        # 反馈多项式: x^5 + x^3 + 1
        new_bit = ((lfsr >> 4) ^ (lfsr >> 2)) & 1
        lfsr = ((lfsr >> 1) | (new_bit << 4)) & 0b11111
    
    return sequence


def ldpc_encode(data: bytes, rate: float, block_size: int) -> tuple:
    """使用标准LDPC库的编码实现"""
    return ldpc_encode_pyldpc(data, rate, block_size)

def ldpc_encode_pyldpc(data: bytes, rate: float, block_size: int) -> tuple:
    """使用pyldpc库的LDPC编码"""
    n = block_size
    
    # 创建LDPC码 - 确保参数满足约束
    d_v = 3
    d_c = int(d_v / rate)
    
    # 确保 d_c 能整除 n（这是 pyldpc 的硬性要求）
    while n % d_c != 0:
        d_c += 1
    
    # 验证约束条件
    assert d_c >= d_v, f"d_c={d_c} 必须大于等于 d_v={d_v}"
    assert n % d_c == 0, f"d_c={d_c} 必须能整除 n={n}"
    
    # 获取校验矩阵和生成矩阵
    H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    
    # 使用实际的 k 值，而不是理论值
    k_actual = G.shape[1]  # G的形状是 (k, n)，所以k是第一个维度 ## 修复：G的形状实际上是 (n, k)，不是 (k, n)
    actual_rate = k_actual / n
    
    print(f"LDPC 参数: n={n}, d_v={d_v}, d_c={d_c}")
    print(f"生成矩阵 G 形状: {G.shape}")
    print(f"校验矩阵 H 形状: {H.shape}")
    print(f"实际信息位长度: k_actual={k_actual}")
    print(f"目标码率: {rate:.4f}, 实际码率: {actual_rate:.4f}")
    
    # 将数据转换为比特
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    data_length = len(data_bits)  # 添加这一行记录原始比特长度
    encoded_bits = []
    
    # 分块编码 - 使用实际的 k_actual
    for i in range(0, len(data_bits), k_actual):
        # 提取信息块
        info_block = data_bits[i:i+k_actual]
        
        # 如果最后一块不足，进行填充
        if len(info_block) < k_actual:
            info_block = np.pad(info_block, (0, k_actual - len(info_block)), 'constant')
        
        # 修复：正确的矩阵乘法应该是 info_block @ G
        # info_block 形状: (k_actual,)
        # G 形状: (k_actual, n)
        # 结果 codeword 形状: (n,)
        codeword = (G @ info_block) % 2 
        
        encoded_bits.extend(codeword.astype(np.uint8))
    
    # 转换回字节
    encoded_bits = np.array(encoded_bits, dtype=np.uint8)
    return np.packbits(encoded_bits).tobytes(), data_length

def create_irregular_ldpc_matrix(n: int, k: int) -> np.ndarray:
    """创建非规则LDPC校验矩阵（优化的度分布）"""
    m = n - k  # 校验位数量
    
    # 优化的度分布（来自Richardson-Urbanke密度演化）
    # 变量节点度分布
    lambda_coeffs = {
        2: 0.2,
        3: 0.3,
        4: 0.2,
        8: 0.2,
        9: 0.1
    }
    
    # 校验节点度分布
    rho_coeffs = {
        5: 0.5,
        6: 0.5
    }
    
    # 创建稀疏校验矩阵
    H = np.zeros((m, n), dtype=np.int8)
    
    # 根据度分布分配边
    edges_per_vnode = []
    for degree, fraction in lambda_coeffs.items():
        count = int(n * fraction)
        edges_per_vnode.extend([degree] * count)
    
    edges_per_cnode = []
    for degree, fraction in rho_coeffs.items():
        count = int(m * fraction)
        edges_per_cnode.extend([degree] * count)
    
    # 随机连接边（避免短环）
    np.random.shuffle(edges_per_vnode)
    np.random.shuffle(edges_per_cnode)
    
    # PEG算法构造（Progressive Edge Growth）
    for v in range(n):
        degree = edges_per_vnode[v % len(edges_per_vnode)]
        # 选择度数最小的校验节点
        cnode_degrees = np.sum(H, axis=1)
        available_cnodes = np.where(cnode_degrees < edges_per_cnode)[0]
        
        if len(available_cnodes) >= degree:
            selected = np.random.choice(available_cnodes, degree, replace=False)
            H[selected, v] = 1
    
    return H

def ldpc_encode_optimized(data: bytes, rate: float = 0.25, block_size: int = 12000) -> bytes:
    """优化的LDPC编码实现，使用非规则码"""
    n = block_size
    k = int(n * rate)
    
    if USE_PYLDPC:
        # 使用优化的非规则LDPC码
        # 创建度分布优化的LDPC码
        d_v = [2, 3, 4, 8, 9]  # 变量节点度
        d_c = [5, 6]  # 校验节点度
        
        # 生成非规则LDPC码
        # 返回 (H, G) 元组
        H, G = pyldpc.make_ldpc(
            n, 
            d_v,
            d_c,
            systematic=True,
            sparse=True,
            seed=42
        )
        
        
        # 数据预处理
        data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        
        # 编码处理
        encoded_all = []
        
        for i in range(0, len(data_bits), k):
            # 提取数据块
            message = data_bits[i:i+k]
            
            if len(message) < k:
                message = np.pad(message, (0, k - len(message)), 'constant')
            
            # LDPC编码
            if hasattr(G, 'dot'):  # 稀疏矩阵
                codeword = (G.dot(message) % 2).astype(np.uint8)
            else:  # 密集矩阵
                codeword = (np.dot(G.T, message) % 2).astype(np.uint8)
            
            encoded_all.extend(codeword)
        
        # 转换为字节
        encoded_bits = np.array(encoded_all, dtype=np.uint8)
        return np.packbits(encoded_bits).tobytes()
    
    else:
        # 使用自定义的非规则LDPC实现
        H = create_irregular_ldpc_matrix(n, k)
        return ldpc_encode_with_matrix(data, H, k)

def ldpc_encode_with_matrix(data: bytes, H: np.ndarray, k: int) -> bytes:
    """使用给定校验矩阵的LDPC编码"""
    n = H.shape[1]
    m = H.shape[0]
    
    # 系统化编码
    # 找到可逆的子矩阵
    H_permuted, perm = systematic_form(H)
    
    # 分离信息位和校验位部分
    H1 = H_permuted[:, :k]
    H2 = H_permuted[:, k:]
    
    # 计算生成矩阵的校验部分
    try:
        H2_inv = np.linalg.inv(H2).astype(int) % 2
        P = (H2_inv @ H1) % 2
    except:
        # 如果H2不可逆，使用高斯消元
        P = gaussian_elimination_gf2(H2, H1)
    
    # 数据编码
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    encoded_all = []
    
    for i in range(0, len(data_bits), k):
        message = data_bits[i:i+k]
        if len(message) < k:
            message = np.pad(message, (0, k - len(message)), 'constant')
        
        # 计算校验位
        parity = (P.T @ message) % 2
        
        # 组合为码字
        codeword = np.concatenate([message, parity])
        
        # 恢复原始顺序
        codeword_original = np.zeros(n, dtype=np.uint8)
        codeword_original[perm] = codeword
        
        encoded_all.extend(codeword_original)
    
    return np.packbits(np.array(encoded_all, dtype=np.uint8)).tobytes()

def systematic_form(H: np.ndarray) -> tuple:
    """将校验矩阵转换为系统形式"""
    m, n = H.shape
    H_copy = H.copy()
    
    # 列置换向量
    perm = np.arange(n)
    
    # 高斯消元
    pivot_row = 0
    for col in range(n):
        # 找到主元
        found = False
        for row in range(pivot_row, m):
            if H_copy[row, col] == 1:
                # 交换行
                H_copy[[pivot_row, row]] = H_copy[[row, pivot_row]]
                found = True
                break
        
        if found:
            # 消元
            for row in range(m):
                if row != pivot_row and H_copy[row, col] == 1:
                    H_copy[row] = (H_copy[row] + H_copy[pivot_row]) % 2
            
            pivot_row += 1
            if pivot_row >= m:
                break
    
    # 重排列使单位矩阵在右边
    identity_cols = []
    other_cols = []
    
    for col in range(n):
        col_sum = np.sum(H_copy[:, col])
        if col_sum == 1:
            row_idx = np.where(H_copy[:, col] == 1)[0][0]
            if row_idx < m:
                identity_cols.append((row_idx, col))
        else:
            other_cols.append(col)
    
    # 排序
    identity_cols.sort()
    new_order = other_cols + [col for _, col in identity_cols]
    
    H_permuted = H_copy[:, new_order]
    perm = perm[new_order]
    
    return H_permuted, perm

def gaussian_elimination_gf2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """GF(2)域上的高斯消元求解 AX = B"""
    m, n = A.shape
    _, k = B.shape
    
    # 增广矩阵
    augmented = np.hstack([A, B]).astype(np.uint8)
    
    # 前向消元
    for i in range(min(m, n)):
        # 找主元
        pivot = None
        for j in range(i, m):
            if augmented[j, i] == 1:
                pivot = j
                break
        
        if pivot is None:
            continue
        
        # 交换行
        augmented[[i, pivot]] = augmented[[pivot, i]]
        
        # 消元
        for j in range(m):
            if j != i and augmented[j, i] == 1:
                augmented[j] = (augmented[j] + augmented[i]) % 2
    
    # 提取解
    X = augmented[:, n:n+k]
    
    return X

def find_sync_patterns(bitstream: List[bool], sync_header: int = 0xB593) -> List[int]:
    """在比特流中查找同步头模式"""
    # 将同步头转换为比特模式
    sync_pattern = []
    for bit in range(16):
        sync_pattern.append(bool((sync_header >> (15 - bit)) & 1))
    
    positions = []
    
    # 使用更有效的搜索方法
    for i in range(len(bitstream) - 15):
        match = True
        for j in range(16):
            if bitstream[i + j] != sync_pattern[j]:
                match = False
                break
        if match:
            positions.append(i)
    
    return positions


def extract_packets_from_bitstream(
    bitstream: List[bool], 
    rs_n: int = 255, 
    sync_header: int = 0xB593
) -> List[Tuple[bytes, int]]:
    """从比特流中提取数据包（修复版）"""
    sync_positions = find_sync_patterns(bitstream, sync_header)
    packets = []
    
    packet_size_bits = (2 + rs_n) * 8  # 同步头(2字节) + RS数据(rs_n字节)
    
    for pos in sync_positions:
        # 检查是否有足够的数据
        if pos + packet_size_bits <= len(bitstream):
            # 提取RS编码的数据（跳过16位同步头）
            rs_data_start = pos + 16
            rs_data_bits = bitstream[rs_data_start:rs_data_start + rs_n * 8]
            
            # 转换为字节
            packet_bytes = []
            for i in range(0, len(rs_data_bits), 8):
                if i + 8 <= len(rs_data_bits):
                    byte_val = 0
                    for j in range(8):
                        if rs_data_bits[i + j]:
                            byte_val |= (1 << (7 - j))
                    packet_bytes.append(byte_val)
            
            if len(packet_bytes) == rs_n:  # 确保数据包完整
                packets.append((bytes(packet_bytes), pos))
    
    return packets


def parse_packet_metadata(packet_data: bytes) -> Tuple[Optional[Dict], bytes]:
    """解析数据包元数据"""
    if len(packet_data) < 10:
        return None, packet_data
    
    try:
        # 解析元数据
        total_payload_length = struct.unpack('>I', packet_data[0:4])[0]
        chunk_index = struct.unpack('>H', packet_data[4:6])[0]
        total_chunks = struct.unpack('>H', packet_data[6:8])[0]
        header_crc = struct.unpack('>H', packet_data[8:10])[0]
        
        # 验证CRC
        header_data = packet_data[0:8]
        calculated_crc = crc16_ccitt(header_data)
        
        if calculated_crc != header_crc:
            return None, packet_data[10:]
        
        # 基本合理性检查
        if (total_payload_length > 10 * 1024 * 1024 or
            total_chunks == 0 or total_chunks > 10000 or
            chunk_index >= total_chunks):
            return None, packet_data[10:]
        
        metadata = {
            'total_payload_length': total_payload_length,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'header_crc': header_crc
        }
        
        return metadata, packet_data[10:]
        
    except (struct.error, IndexError):
        return None, packet_data[10:] if len(packet_data) > 10 else packet_data


def deinterleave_bitstream(bitstream, depth=8):
    """对比特流进行去交织"""
    if depth <= 1:
        return bitstream
    
    total_bits = len(bitstream)
    cols = (total_bits + depth - 1) // depth  # 向上取整
    
    # 重建行
    rows = [[] for _ in range(depth)]
    
    for i, bit in enumerate(bitstream):
        row_idx = i % depth
        rows[row_idx].append(bit)
    
    # 重新排列
    deinterleaved = []
    for col in range(cols):
        for row_idx in range(depth):
            if col < len(rows[row_idx]):
                deinterleaved.append(rows[row_idx][col])
    
    return deinterleaved



def decode_watermark_to_binary(
    bitstream: List[bool],
    rs_n: int = 255,
    rs_k: int = 63,  # 更新为增强型RS(255,223)
    sync_header: int = 0xB593,
    max_attempts: int = 1000,
    ldpc_rate: float = 0.25,
    ldpc_block_size: int = 12000,
    spiral_interleave: bool = True,
    tmr_redundancy: bool = True,
    variable_repetition: bool = True,
    debug: bool = False
) -> Tuple[Optional[bytes], Dict]:
    """从增强水印比特流解码出原始二进制数据（支持20%噪声容忍）"""
    
    # 初始化统计信息
    stats = {
        'total_packets_found': 0,
        'valid_packets': 0,
        'rs_decode_success': 0,
        'rs_decode_failed': 0,
        'ldpc_decode_success': 0,
        'ldpc_decode_failed': 0,
        'metadata_valid': 0,
        'metadata_invalid': 0,
        'total_errors_corrected': 0,
        'chunks_recovered': {},
        'final_payload_length': 0,
        'compression_successful': False,
        'spiral_interleave_applied': spiral_interleave,
        'tmr_applied': tmr_redundancy,
        'original_bitstream_length': len(bitstream),
        'debug_info': []
    }
    
    if debug:
        print(f"[DEBUG] 原始比特流长度: {len(bitstream)}")
        print(f"[DEBUG] LDPC参数: 码率={ldpc_rate}, 块大小={ldpc_block_size}")
    
    # 1. 去扩频处理
    if debug:
        print("[DEBUG] 开始去扩频处理...")
    bitstream = remove_dsss(bitstream)
    
    # 2. TMR投票恢复
    if tmr_redundancy:
        if debug:
            print("[DEBUG] 执行TMR投票恢复...")
        bitstream = tmr_voting_decode(bitstream)
        stats['tmr_applied'] = True
    
    # 3. 转换为字节流准备LDPC解码
    byte_stream = bits_to_bytes(bitstream)
    
    # 4. 提取同步头定位的LDPC块
    ldpc_blocks = extract_ldpc_blocks(byte_stream, sync_header, ldpc_block_size)
    
    if debug:
        print(f"[DEBUG] 找到 {len(ldpc_blocks)} 个LDPC块")
    
    if not ldpc_blocks:
        return None, stats
    
    # 5. LDPC解码
    ldpc_decoded_data = []
    for block_idx, (block_data, position) in enumerate(ldpc_blocks):
        if debug and block_idx < 3:
            print(f"\n[DEBUG] LDPC块 {block_idx} @ 位置 {position}:")
        
        try:
            # 移除同步头
            ldpc_payload = block_data[2:]  # 跳过2字节同步头
            
            # LDPC解码
            estimated_original_length = len(ldpc_payload) * ldpc_rate * 8  # 近似计算
            decoded = ldpc_decode(ldpc_payload,int(estimated_original_length), ldpc_rate, ldpc_block_size)
            ldpc_decoded_data.extend(decoded)
            stats['ldpc_decode_success'] += 1
            
            if debug and block_idx < 3:
                print(f"  LDPC解码成功")
        except Exception as e:
            stats['ldpc_decode_failed'] += 1
            if debug and block_idx < 3:
                print(f"  LDPC解码失败: {e}")
    
    if not ldpc_decoded_data:
        return None, stats
    
    # 6. 螺旋去交织
    if spiral_interleave:
        if debug:
            print("[DEBUG] 执行螺旋去交织...")
        ldpc_decoded_data = spiral_deinterleave_enhanced(bytes(ldpc_decoded_data))
        stats['spiral_interleave_applied'] = True
    
    # 7. RS解码
    rs_decoder = ReedSolomonDecoder(rs_n, rs_k)
    rs_packets = extract_rs_packets(ldpc_decoded_data, rs_n)
    
    if debug:
        print(f"[DEBUG] 找到 {len(rs_packets)} 个RS数据包")
    
    valid_chunks = defaultdict(list)
    total_chunks_expected = None
    total_payload_length = None
    
    for packet_idx, packet_bytes in enumerate(rs_packets):
        if packet_idx >= max_attempts:
            break
        
        try:
            # RS解码
            decoded_data, errors_corrected = rs_decoder.decode(packet_bytes)
            stats['rs_decode_success'] += 1
            stats['total_errors_corrected'] += errors_corrected
            
            # 解析增强元数据（12字节）
            metadata = parse_enhanced_metadata(decoded_data)
            
            if metadata is not None:
                stats['metadata_valid'] += 1
                
                # 提取数据部分
                metadata_size = 12
                chunk_data = decoded_data[metadata_size:]
                
                # 验证纠错层版本
                if metadata['ecc_version'] != 0x01:
                    if debug:
                        print(f"[DEBUG] 警告：未知的纠错层版本 {metadata['ecc_version']}")
                    continue
                
                # 设置或验证全局参数
                if total_chunks_expected is None:
                    total_chunks_expected = metadata['total_chunks']
                    total_payload_length = metadata['total_payload_length']
                
                # 存储有效数据块
                chunk_index = metadata['chunk_index']
                if chunk_index < total_chunks_expected:
                    valid_chunks[chunk_index].append((chunk_data, metadata, errors_corrected))
                    
            else:
                stats['metadata_invalid'] += 1
                
        except Exception as e:
            stats['rs_decode_failed'] += 1
            if debug:
                print(f"[DEBUG] RS解码失败: {e}")
    
    if not valid_chunks:
        return None, stats
    
    # 8. 重组第一层输出
    layer1_output = b''
    metadata_size = 12
    k_data = rs_k - metadata_size
    
    for chunk_index in range(total_chunks_expected):
        if chunk_index in valid_chunks:
            candidates = valid_chunks[chunk_index]
            best_candidate = min(candidates, key=lambda x: x[2])
            chunk_data = best_candidate[0]
            
            # 计算实际数据长度
            start_pos = chunk_index * k_data
            expected_length = min(k_data, len(chunk_data))
            layer1_output += chunk_data[:expected_length]
            
            stats['chunks_recovered'][chunk_index] = {
                'candidates': len(candidates),
                'errors_corrected': best_candidate[2]
            }
    
    # 9. 可变重复解码
    if variable_repetition:
        if debug:
            print("[DEBUG] 执行可变重复解码...")
        
        # 解码时需要知道原始压缩数据长度
        if total_payload_length:
            original_data = variable_repetition_decode(layer1_output, total_payload_length)
        else:
            original_data = layer1_output
    else:
        original_data = layer1_output
    
    stats['final_payload_length'] = len(original_data)
    
    # 10. 解压缩
    try:
        final_data = zlib.decompress(original_data)
        stats['compression_successful'] = True
        
        if debug:
            print(f"[DEBUG] 解压缩成功，最终数据长度: {len(final_data)}")
        
        return final_data, stats
    except Exception as e:
        if debug:
            print(f"[DEBUG] 解压缩失败: {e}")
        return original_data, stats


def decode_watermark_to_string(
    bitstream: List[bool],
    encoding: str = 'utf-8',
    debug: bool = False,
    **kwargs
) -> Tuple[Optional[str], Dict]:
    """从水印比特流解码出字符串"""
    binary_data, stats = decode_watermark_to_binary(bitstream, debug=debug, **kwargs)
    
    if binary_data is None:
        return None, stats
    
    try:
        text = binary_data.decode(encoding)
        return text, stats
    except UnicodeDecodeError as e:
        print(f"字符串解码失败: {e}")
        return None, stats

def remove_dsss(bitstream: List[bool], spreading_factor: int = 15) -> List[bool]:
    """修复的去扩频"""
    pn_sequence = generate_pn_sequence(spreading_factor)
    despread_bits = []
    
    for i in range(0, len(bitstream), spreading_factor):
        if i + spreading_factor <= len(bitstream):
            # 对每个扩频块进行相关运算
            block = bitstream[i:i+spreading_factor]
            
            # 去扩频：用相同的PN序列解调
            despread_block = []
            for j in range(spreading_factor):
                despread_block.append(block[j] ^ pn_sequence[j])
            
            # 多数决策：统计1的个数
            ones_count = sum(1 for bit in despread_block if bit)
            # 如果1的个数超过一半，认为原始比特是1
            despread_bits.append(ones_count > spreading_factor // 2)
    
    return despread_bits

def tmr_voting_decode(bitstream: List[bool]) -> List[bool]:
    """修复的TMR投票解码 - 与编码逻辑匹配"""
    decoded = []
    
    # 每3个连续比特为一组进行投票
    for i in range(0, len(bitstream), 3):
        if i + 2 < len(bitstream):
            # 获取3个连续的冗余比特
            bit1 = bitstream[i]
            bit2 = bitstream[i + 1] 
            bit3 = bitstream[i + 2]
            
            # 多数投票
            vote = int(bit1) + int(bit2) + int(bit3)
            decoded.append(vote >= 2)
        elif i + 1 < len(bitstream):
            # 只有2个比特时，取第一个
            decoded.append(bitstream[i])
        elif i < len(bitstream):
            # 只有1个比特时，直接取
            decoded.append(bitstream[i])
    
    return decoded

def bits_to_bytes(bitstream: List[bool]) -> bytes:
    """将比特流转换为字节"""
    # 填充到8的倍数
    if len(bitstream) % 8 != 0:
        bitstream = bitstream + [False] * (8 - len(bitstream) % 8)
    
    byte_array = bytearray()
    for i in range(0, len(bitstream), 8):
        byte_val = 0
        for j in range(8):
            if bitstream[i + j]:
                byte_val |= (1 << (7 - j))
        byte_array.append(byte_val)
    
    return bytes(byte_array)

def extract_ldpc_blocks(byte_stream: bytes, sync_header: int, block_size: int) -> List[Tuple[bytes, int]]:
    """提取LDPC编码块"""
    sync_bytes = struct.pack('>H', sync_header)
    blocks = []
    
    i = 0
    while i < len(byte_stream) - 1:
        if byte_stream[i:i+2] == sync_bytes:
            # 找到同步头
            block_size_bytes = block_size // 8 + 2  # 加上同步头
            if i + block_size_bytes <= len(byte_stream):
                block = byte_stream[i:i+block_size_bytes]
                blocks.append((block, i))
            i += block_size_bytes
        else:
            i += 1
    
    return blocks

def ldpc_decode(data: bytes, original_bit_length: int, rate: float, block_size: int) -> bytes:
    """LDPC解码（使用标准库）"""
    return ldpc_decode_pyldpc(data, original_bit_length, rate, block_size)

# def ldpc_decode_pyldpc(encoded_data: bytes,original_bit_length: int, rate: float, block_size: int, max_iter: int = 150) -> bytes:
#     """使用 pyldpc 库进行 LDPC 解码"""
#     n = block_size
    
#     # 使用相同的参数创建 LDPC 码
#     d_v = 3
#     d_c = int(d_v / rate)
#     while n % d_c != 0:
#         d_c += 1
    
#     H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
#     k_actual = G.shape[1]  # 修复：使用正确的维度
    
#     # 将编码数据转换为比特
#     encoded_bits = np.unpackbits(np.frombuffer(encoded_data, dtype=np.uint8))
    
#     decoded_bits = []
    
#     # 分块解码
#     for i in range(0, len(encoded_bits), n):
#         codeword = encoded_bits[i:i+n]
        
#         if len(codeword) < n:
#             codeword = np.pad(codeword, (0, n - len(codeword)), 'constant')
        
#         # 转换为 LLR（对数似然比）
#         # 假设 BPSK: 0 -> +1, 1 -> -1
#         y = 1.0 - 2.0 * codeword.astype(np.float64)
        
#         # 解码
#         decoded = pyldpc.decode(H, y, snr=100, maxiter=max_iter)
        
#         # 提取信息位（系统码的前 k 位）
#         info_bits = pyldpc.get_message(G, decoded)
#         decoded_bits.extend(info_bits)
    
#     # 转换回字节
#     decoded_bits_array = np.array(decoded_bits, dtype=np.uint8)
#     final_decoded_bits = decoded_bits_array[:original_bit_length]  # 精确截断到原始长度
#     return np.packbits(final_decoded_bits).tobytes()


def ldpc_decode_pyldpc(encoded_data: bytes, original_bit_length: int, rate: float, block_size: int, max_iter: int = 150) -> bytes:
    """修复的多块LDPC解码函数 - 正确处理分块解码"""
    
    print(f"  LDPC多块解码:")
    print(f"    输入: {len(encoded_data)} 字节 = {len(encoded_data)*8} 比特")
    print(f"    目标比特长度: {original_bit_length}")
    
    n = block_size
    
    # 使用与编码完全相同的参数
    d_v = 3
    d_c = int(d_v / rate)
    while n % d_c != 0:
        d_c += 1
    
    # 创建LDPC码
    H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k_actual = G.shape[1]  # 实际信息位长度
    actual_rate = k_actual / n
    
    print(f"    LDPC参数: n={n}, k={k_actual}, 码率={actual_rate:.3f}")
    
    # 转换为比特
    encoded_bits = np.unpackbits(np.frombuffer(encoded_data, dtype=np.uint8))
    print(f"    编码比特总数: {len(encoded_bits)}")
    
    # 计算期望的块数
    expected_blocks = len(encoded_bits) // n
    info_bits_per_block = k_actual
    total_expected_info_bits = expected_blocks * info_bits_per_block
    
    print(f"    期望解码块数: {expected_blocks}")
    print(f"    期望总信息位: {total_expected_info_bits}")
    print(f"    目标信息位: {original_bit_length}")
    
    # 验证长度匹配
    if len(encoded_bits) % n != 0:
        print(f"    [警告] 编码长度({len(encoded_bits)})不是块大小({n})的整数倍")
        # 填充到完整块
        padding_needed = n - (len(encoded_bits) % n)
        encoded_bits = np.pad(encoded_bits, (0, padding_needed), 'constant')
        print(f"    填充 {padding_needed} 比特")
    
    decoded_bits = []
    successful_blocks = 0
    failed_blocks = 0
    
    # 逐块解码
    for block_idx in range(len(encoded_bits) // n):
        start_idx = block_idx * n
        end_idx = start_idx + n
        codeword = encoded_bits[start_idx:end_idx]
        
        try:
            # 转换为LLR (Log-Likelihood Ratio)
            # 0 -> +1, 1 -> -1
            y = 1.0 - 2.0 * codeword.astype(np.float64)
            
            # LDPC解码
            decoded_block = pyldpc.decode(H, y, snr=100, maxiter=max_iter)
            
            # 提取信息位
            info_bits = pyldpc.get_message(G, decoded_block)
            decoded_bits.extend(info_bits)
            successful_blocks += 1
            
            if block_idx == 0:  # 详细显示第一块
                print(f"      块 {block_idx}: 解码成功")
                print(f"        码字: {len(codeword)} 比特")
                print(f"        信息位: {len(info_bits)} 比特")
                print(f"        前16比特: {''.join(str(int(b)) for b in info_bits[:16])}")
            
        except Exception as e:
            print(f"      块 {block_idx}: 解码失败 - {e}")
            # 使用零填充作为失败块的输出
            failed_info_bits = np.zeros(k_actual, dtype=np.uint8)
            decoded_bits.extend(failed_info_bits)
            failed_blocks += 1
    
    print(f"    解码统计: 成功 {successful_blocks}, 失败 {failed_blocks}")
    print(f"    解码比特总数: {len(decoded_bits)}")
    
    # 截断到原始长度
    if original_bit_length <= len(decoded_bits):
        final_bits = decoded_bits[:original_bit_length]
        print(f"    截断到目标长度: {len(final_bits)} 比特")
    else:
        print(f"    [警告] 目标长度超出解码长度")
        final_bits = decoded_bits
    
    # 字节对齐
    while len(final_bits) % 8 != 0:
        final_bits.append(0)
    
    # 转换为字节
    final_bits_array = np.array(final_bits, dtype=np.uint8)
    result = np.packbits(final_bits_array).tobytes()
    
    print(f"    最终输出: {len(result)} 字节")
    
    # 验证输出长度
    expected_bytes = (original_bit_length + 7) // 8  # 向上取整
    if len(result) != expected_bytes:
        print(f"    [警告] 输出长度({len(result)})与期望长度({expected_bytes})不匹配")
    
    return result


def spiral_deinterleave_enhanced(data: bytes) -> bytes:
    """修复的螺旋去交织函数"""
    if len(data) == 0:
        return data
    # 使用相同的素数深度，但反向顺序
    prime_depths = [13, 11, 7]
    result = bytearray(data)
    for depth in prime_depths:
        if len(result) <= depth:
            continue
        temp = bytearray(len(result))
        # 反向螺旋操作
        rows_per_depth = len(result) // depth
        remainder = len(result) % depth
        read_idx = 0
        write_idx = 0
        for row in range(depth):
            row_size = rows_per_depth + (1 if row < remainder else 0)
            for col in range(row_size):
                if read_idx < len(result):
                    # 反向螺旋偏移
                    spiral_col = (col - row * 3) % row_size
                    source_idx = row + spiral_col * depth
                    if source_idx < len(result):
                        temp[write_idx] = result[source_idx]
                    else:
                        temp[write_idx] = result[read_idx]
                    read_idx += 1
                    write_idx += 1
        result = temp
    
    assert len(result) == len(data), f"去交织改变了数据长度: {len(data)} -> {len(result)}"
    return bytes(result)

# def extract_rs_packets(data: bytes, rs_n: int) -> List[bytes]:
#     """从数据流中提取RS数据包"""
#     packets = []
#     for i in range(0, len(data), rs_n):
#         if i + rs_n <= len(data):
#             packets.append(data[i:i+rs_n])
#     return packets

def extract_rs_packets(data: bytes, rs_n: int) -> List[bytes]:
    """修复的RS数据包提取函数 - 处理不完整包"""
    packets = []
    print(f"  extract_rs_packets_fixed: 输入数据长度 {len(data)} 字节")
    
    # 计算完整包数量和剩余字节
    complete_packets = len(data) // rs_n
    remainder = len(data) % rs_n
    
    print(f"  完整包数量: {complete_packets}, 剩余字节: {remainder}")
    
    # 提取完整包
    for i in range(complete_packets):
        start_idx = i * rs_n
        end_idx = start_idx + rs_n
        packet = data[start_idx:end_idx]
        packets.append(packet)
        print(f"    包 {i}: 字节 {start_idx}-{end_idx-1}")
    
    # 处理剩余不完整包 - 用零填充
    if remainder > 0:
        start_idx = complete_packets * rs_n
        incomplete_packet = data[start_idx:]
        # 填充到完整包大小
        padded_packet = incomplete_packet + b'\x00' * (rs_n - remainder)
        packets.append(padded_packet)
        print(f"    不完整包: 字节 {start_idx}-, 填充 {rs_n - remainder} 字节零")
    
    return packets

def parse_enhanced_metadata(decoded_data: bytes) -> Optional[Dict]:
    """解析增强的元数据（12字节）"""
    if len(decoded_data) < 12:
        return None
    
    try:
        # 解析元数据字段
        total_payload_length = struct.unpack('>I', decoded_data[0:4])[0]
        chunk_index = struct.unpack('>H', decoded_data[4:6])[0]
        total_chunks = struct.unpack('>H', decoded_data[6:8])[0]
        ecc_version = struct.unpack('>B', decoded_data[8:9])[0]
        ldpc_param = struct.unpack('>B', decoded_data[9:10])[0]
        header_crc = struct.unpack('>H', decoded_data[10:12])[0]
        
        # 验证CRC
        metadata_for_crc = decoded_data[0:10]
        calculated_crc = crc16_ccitt(metadata_for_crc)
        
        if calculated_crc != header_crc:
            return None
        
        return {
            'total_payload_length': total_payload_length,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'ecc_version': ecc_version,
            'ldpc_param': ldpc_param,
            'header_crc': header_crc
        }
    except:
        return None

def variable_repetition_decode(data: bytes, expected_length: int) -> bytes:
    """重复解码"""
    decoded = bytearray()
    pos = 0
    
    while len(decoded) < expected_length and pos < len(data):
        votes = []
        for _ in range(3):  # 3倍重复
            if pos < len(data):
                votes.append(data[pos])
                pos += 1
        
        if votes:
            # 多数投票
            decoded.append(max(set(votes), key=votes.count))
    
    return bytes(decoded[:expected_length])

def estimate_required_capacity(data: Union[bytes, str], rs_n: int = 255, rs_k: int = 63) -> Tuple[int, int]:
    """估算编码后需要的最小容量"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    compressed_data = zlib.compress(data, level=9)
    total_payload_length = len(compressed_data)
    
    metadata_size = 10
    k_data = rs_k - metadata_size
    total_chunks = (total_payload_length + k_data - 1) // k_data
    
    # 每个数据包的大小：同步头(2) + RS编码后数据(rs_n)
    packet_size_bytes = 2 + rs_n
    packet_size_bits = packet_size_bytes * 8
    
    # 总共需要的比特数（一个完整序列）
    total_bits = total_chunks * packet_size_bits
    
    return total_bits, total_chunks


# 测试函数
def test_crc16():
    """测试CRC16-CCITT-FALSE实现"""
    print("\n--- CRC16-CCITT-FALSE测试 ---")
    
    # 测试向量
    test_cases = [
        (b"123456789", 0x29B1),
        (b"Hello", 0xDADA),  # 这个是正确的期望值
        (b"", 0xFFFF),
    ]
    
    for data, expected in test_cases:
        calculated = crc16_ccitt(data)
        print(f"数据: {data!r}")
        print(f"  计算的CRC: 0x{calculated:04X}")
        print(f"  预期的CRC: 0x{expected:04X}")
        print(f"  {'✓ 通过' if calculated == expected else '✗ 失败'}")

def calculate_ldpc_output_size(input_bytes, ldpc_rate, ldpc_block_size):
    """
    计算LDPC编码后的输出大小
    
    Args:
        input_bytes: 输入字节数
        ldpc_rate: LDPC码率 (如0.5)
        ldpc_block_size: LDPC块大小 (如256)
        
    Returns:
        预期的LDPC编码输出字节数
    """
    input_bits = input_bytes * 8
    
    # LDPC编码会将数据分成块，每块大小为 ldpc_block_size * ldpc_rate
    info_bits_per_block = int(ldpc_block_size * ldpc_rate)
    
    # 计算需要多少个LDPC块
    num_blocks = (input_bits + info_bits_per_block - 1) // info_bits_per_block
    
    # 每个LDPC块编码后的大小是 ldpc_block_size 比特
    total_output_bits = num_blocks * ldpc_block_size
    
    # 转换为字节
    output_bytes = (total_output_bits + 7) // 8
    
    print(f"LDPC参数分析:")
    print(f"  输入: {input_bytes} 字节 ({input_bits} 比特)")
    print(f"  码率: {ldpc_rate}")
    print(f"  块大小: {ldpc_block_size} 比特")
    print(f"  每块信息位: {info_bits_per_block} 比特")
    print(f"  需要块数: {num_blocks}")
    print(f"  输出: {output_bytes} 字节 ({total_output_bits} 比特)")
    
    return output_bytes

# 诊断/自测试函数
def diagnose_encoding_decoding():
    """改进的编码解码流程诊断（去掉螺旋交织，通用化LDPC块大小计算）"""
    print("\n" + "="*80)
    print("改进的编码/解码流程诊断")
    print("="*80)
    
    # 使用简单的测试数据
    test_data = "Hello This is a Test for FULL!"
    print(f"原始测试数据: '{test_data}'")
    print(f"原始UTF-8字节: {test_data.encode('utf-8')}")
    print(f"原始数据长度: {len(test_data)} 字符, {len(test_data.encode('utf-8'))} 字节")
    
    # 编码参数
    ldpc_rate = 0.5  # LDPC码率
    ldpc_block_size = 256  # LDPC块大小（比特）
    rs_n, rs_k = 255, 63  # Reed-Solomon参数
    sync_header = 0xB593  # 同步头
    
    print(f"\n编码参数:")
    print(f"  LDPC码率: {ldpc_rate}")
    print(f"  LDPC块大小: {ldpc_block_size} 比特")
    print(f"  RS参数: RS({rs_n}, {rs_k})")
    print(f"  同步头: 0x{sync_header:04X}")
    
    # ===== 编码流程诊断 =====
    print(f"\n" + "-"*60)
    print("开始编码流程诊断")
    print("-"*60)
    
    # 步骤1: 数据预处理和压缩
    print(f"\n[步骤1] 数据预处理和压缩")
    if isinstance(test_data, str):
        utf8_data = test_data.encode('utf-8')
    else:
        utf8_data = test_data
    
    compressed_data = zlib.compress(utf8_data, level=9)
    print(f"  UTF-8编码: {len(utf8_data)} 字节")
    print(f"  压缩后: {len(compressed_data)} 字节")
    print(f"  压缩率: {(1-len(compressed_data)/len(utf8_data))*100:.1f}%")
    print(f"  压缩数据hex: {compressed_data.hex()}")
    
    # 步骤2: 可变重复编码（简化版，仅用3倍重复）
    print(f"\n[步骤2] 可变重复编码")
    repeated_data = repeat_encode(compressed_data, 3)
    print(f"  重复因子: 3")
    print(f"  重复后长度: {len(repeated_data)} 字节")
    print(f"  前16字节hex: {repeated_data[:16].hex()}")
    
    # 步骤3: Reed-Solomon编码
    print(f"\n[步骤3] Reed-Solomon编码")
    rs_encoder = ReedSolomonEncoder(rs_n, rs_k)
    rs_packets = []
    
    metadata_size = 12
    k_data = rs_k - metadata_size
    total_chunks = (len(repeated_data) + k_data - 1) // k_data
    
    print(f"  每包数据容量: {k_data} 字节")
    print(f"  总分片数: {total_chunks}")
    
    for chunk_index in range(total_chunks):
        start_idx = chunk_index * k_data
        end_idx = min(start_idx + k_data, len(repeated_data))
        data_chunk = repeated_data[start_idx:end_idx]
        
        # 构建元数据
        metadata = struct.pack('>I', len(compressed_data))
        metadata += struct.pack('>H', chunk_index)
        metadata += struct.pack('>H', total_chunks)
        metadata += struct.pack('>B', 0x01)  # 版本
        metadata += struct.pack('>B', ldpc_block_size // 1000)
        
        header_crc = crc16_ccitt(metadata)
        metadata += struct.pack('>H', header_crc)
        
        packet_data = metadata + data_chunk
        if len(packet_data) < rs_k:
            packet_data += b'\x00' * (rs_k - len(packet_data))
        
        encoded_packet = rs_encoder.encode(packet_data)
        rs_packets.append(encoded_packet)
        
        if chunk_index == 0:  # 详细显示第一个包
            print(f"  分片 {chunk_index}:")
            print(f"    元数据: {metadata.hex()}")
            print(f"    数据: {data_chunk.hex()}")
            print(f"    RS编码前: {len(packet_data)} 字节")
            print(f"    RS编码后: {len(encoded_packet)} 字节")
    
    print(f"  总RS包数: {len(rs_packets)}")
    
    # 步骤4: LDPC编码
    print(f"\n[步骤4] LDPC编码")
    
    # 预计算LDPC编码后的单个包大小（通用化计算）
    expected_ldpc_size = calculate_ldpc_output_size(rs_n, ldpc_rate, ldpc_block_size)
    print(f"  预期单个LDPC包大小: {expected_ldpc_size} 字节")
    
    ldpc_output = []
    ldpc_lengths = []  # 保存每个包的原始长度
    actual_ldpc_sizes = []  # 记录实际的LDPC编码大小
    
    for i, packet in enumerate(rs_packets):
        try:
            # LDPC编码
            ldpc_encoded, original_length = ldpc_encode_pyldpc(packet, ldpc_rate, ldpc_block_size)
            ldpc_output.extend(ldpc_encoded)
            ldpc_lengths.append(original_length)
            actual_ldpc_sizes.append(len(ldpc_encoded))
            
            if i == 0:  # 详细显示第一个包
                print(f"  包 {i} LDPC编码:")
                print(f"    原始长度: {len(packet)} 字节 ({len(packet)*8} 比特)")
                print(f"    编码后长度: {len(ldpc_encoded)} 字节 ({len(ldpc_encoded)*8} 比特)")
                print(f"    扩展率: {len(ldpc_encoded)/len(packet):.2f}")
                print(f"    记录的原始比特长度: {original_length}")
                print(f"    预期vs实际: {expected_ldpc_size} vs {len(ldpc_encoded)} 字节")
        except Exception as e:
            print(f"  包 {i} LDPC编码失败: {e}")
            break
    
    print(f"  LDPC编码总输出: {len(ldpc_output)} 字节")
    
    # 分析实际LDPC大小的一致性
    if actual_ldpc_sizes:
        unique_sizes = set(actual_ldpc_sizes)
        print(f"  实际LDPC包大小分布: {dict(zip(unique_sizes, [actual_ldpc_sizes.count(s) for s in unique_sizes]))}")
        if len(unique_sizes) == 1:
            single_ldpc_size = list(unique_sizes)[0]
            print(f"  ✓ 所有LDPC包大小一致: {single_ldpc_size} 字节")
        else:
            single_ldpc_size = max(unique_sizes)  # 使用最大值作为标准
            print(f"  ⚠ LDPC包大小不一致，使用最大值: {single_ldpc_size} 字节")
    else:
        single_ldpc_size = expected_ldpc_size
        print(f"  使用预期大小: {single_ldpc_size} 字节")
    
    # 步骤5: 直接处理（跳过螺旋交织）
    print(f"\n[步骤5] 跳过螺旋交织，直接处理")
    processed_data = bytes(ldpc_output)
    print(f"  处理后数据长度: {len(processed_data)} 字节")
    print(f"  前16字节: {processed_data[:16].hex()}")
    
    # 步骤6: 添加同步头并转换为比特流
    print(f"\n[步骤6] 添加同步头和比特流转换")
    master_bitstream = []
    sync_header_bytes = struct.pack('>H', sync_header)
    
    # 为每个LDPC块添加同步头
    blocks_with_sync = 0
    
    for i in range(0, len(processed_data), single_ldpc_size):
        block = processed_data[i:i + single_ldpc_size]
        if len(block) > 0:
            # 如果块不完整，进行填充
            if len(block) < single_ldpc_size:
                block = block + b'\x00' * (single_ldpc_size - len(block))
                print(f"  块 {blocks_with_sync}: 填充 {single_ldpc_size - len(block)} 字节")
            
            full_block = sync_header_bytes + block
            blocks_with_sync += 1
            
            for byte_val in full_block:
                for bit in range(8):
                    master_bitstream.append(bool((byte_val >> (7 - bit)) & 1))
    
    print(f"  添加同步头的块数: {blocks_with_sync}")
    print(f"  每块大小: 同步头(2字节) + LDPC数据({single_ldpc_size}字节) = {single_ldpc_size + 2}字节")
    print(f"  比特流长度: {len(master_bitstream)} 比特")
    print(f"  前32比特: {''.join('1' if b else '0' for b in master_bitstream[:32])}")
    
    # 步骤7: TMR (三模冗余)
    print(f"\n[步骤7] TMR (三模冗余)")
    tmr_bitstream = []
    for bit in master_bitstream:
        tmr_bitstream.extend([bit, bit, bit])  # 每个比特重复3次
    
    print(f"  TMR前长度: {len(master_bitstream)} 比特")
    print(f"  TMR后长度: {len(tmr_bitstream)} 比特")
    print(f"  扩展因子: {len(tmr_bitstream)/len(master_bitstream):.1f} (期望: 3.0)")
    
    # 步骤8: 扩频编码
    print(f"\n[步骤8] 扩频编码")
    spread_bitstream = apply_dsss(tmr_bitstream)
    print(f"  扩频前: {len(tmr_bitstream)} 比特")
    print(f"  扩频后: {len(spread_bitstream)} 比特")
    print(f"  扩频因子: {len(spread_bitstream)/len(tmr_bitstream):.1f}")
    
    # ===== 解码流程诊断 =====
    print(f"\n" + "-"*60)
    print("开始解码流程诊断")
    print("-"*60)
    
    # 使用编码结果进行解码测试
    encoded_bitstream = spread_bitstream
    
    # 步骤1: 去扩频
    print(f"\n[解码步骤1] 去扩频")
    despread_bits = remove_dsss(encoded_bitstream)
    print(f"  去扩频前: {len(encoded_bitstream)} 比特")
    print(f"  去扩频后: {len(despread_bits)} 比特")
    
    # 步骤2: TMR投票解码
    print(f"\n[解码步骤2] TMR投票解码")
    print(f"  TMR解码前: {len(despread_bits)} 比特")
    print(f"  前32比特: {''.join('1' if b else '0' for b in despread_bits[:32])}")
    
    tmr_decoded = tmr_voting_decode(despread_bits)
    print(f"  TMR解码后: {len(tmr_decoded)} 比特")
    print(f"  前32比特: {''.join('1' if b else '0' for b in tmr_decoded[:32])}")
    print(f"  长度比例: {len(despread_bits)/len(tmr_decoded):.2f} (期望: 3.0)")
    
    # 验证TMR解码质量：检查同步头模式
    sync_pattern = []
    for bit in range(16):
        sync_pattern.append(bool((sync_header >> (15 - bit)) & 1))
    
    print(f"  期望同步头模式: {''.join('1' if b else '0' for b in sync_pattern)}")
    
    # 在TMR解码结果中查找同步头模式
    tmr_sync_positions = []
    for i in range(len(tmr_decoded) - 15):
        match = True
        for j in range(16):
            if tmr_decoded[i + j] != sync_pattern[j]:
                match = False
                break
        if match:
            tmr_sync_positions.append(i)
    
    print(f"  TMR解码后找到同步头位置: {tmr_sync_positions[:5]} (前5个)")
    
    if len(tmr_sync_positions) == 0:
        print(f"  [警告] TMR解码后未找到同步头！数据可能已损坏。")
    else:
        print(f"  ✓ TMR解码质量良好，找到 {len(tmr_sync_positions)} 个同步头")
    
    # 步骤3: 转换为字节流
    print(f"\n[解码步骤3] 转换为字节流")
    byte_stream = bits_to_bytes(tmr_decoded)
    print(f"  字节流长度: {len(byte_stream)} 字节")
    
    # 步骤4: 提取LDPC块（通用化）
    print(f"\n[解码步骤4] 提取LDPC块")
    print(f"  字节流前16字节: {byte_stream[:16].hex()}")
    print(f"  寻找同步头: 0x{sync_header:04X} ({struct.pack('>H', sync_header).hex()})")
    print(f"  LDPC块大小: {single_ldpc_size} 字节（通用计算）")
    
    # 手动搜索同步头
    sync_bytes = struct.pack('>H', sync_header)
    sync_positions = []
    for i in range(len(byte_stream) - 1):
        if byte_stream[i:i+2] == sync_bytes:
            sync_positions.append(i)
    
    print(f"  手动搜索同步头位置: {sync_positions[:10]}")  # 只显示前10个
    
    ldpc_blocks = []
    expected_block_size = single_ldpc_size + 2  # 同步头2字节 + LDPC数据
    
    for pos in sync_positions:
        if pos + expected_block_size <= len(byte_stream):
            block_data = byte_stream[pos:pos + expected_block_size]
            ldpc_blocks.append((block_data, pos))
            print(f"    块 {len(ldpc_blocks)-1}: 位置 {pos}, 长度 {len(block_data)} 字节")
            print(f"      前8字节: {block_data[:8].hex()}")
            print(f"      同步头验证: {block_data[:2].hex() == sync_bytes.hex()}")
        else:
            print(f"    位置 {pos}: 数据不足 ({len(byte_stream) - pos} < {expected_block_size}), 跳过")
    
    print(f"  找到LDPC块: {len(ldpc_blocks)} 个")
    
    # 步骤5: LDPC解码
    print(f"\n[解码步骤5] LDPC解码")
    ldpc_decoded_data = []
    
    for block_idx, (block_data, position) in enumerate(ldpc_blocks):
        try:
            # 移除同步头
            ldpc_payload = block_data[2:]
            
            # 使用保存的原始长度（如果可用）
            if block_idx < len(ldpc_lengths):
                original_length = ldpc_lengths[block_idx]
            else:
                # 估算长度
                original_length = rs_n * 8  # RS包的比特长度
            
            decoded = ldpc_decode_pyldpc(ldpc_payload, original_length, ldpc_rate, ldpc_block_size)
            ldpc_decoded_data.extend(decoded)
            
            if block_idx == 0:
                print(f"    块 {block_idx} LDPC解码:")
                print(f"      输入: {len(ldpc_payload)} 字节")
                print(f"      原始长度: {original_length} 比特")
                print(f"      解码输出: {len(decoded)} 字节")
                print(f"      解码数据hex: {decoded.hex()}")
        except Exception as e:
            print(f"    块 {block_idx} LDPC解码失败: {e}")
    
    print(f"  LDPC解码总输出: {len(ldpc_decoded_data)} 字节")
    
    # 步骤6: 跳过螺旋去交织
    print(f"\n[解码步骤6] 跳过螺旋去交织")
    deinterleaved_data = bytes(ldpc_decoded_data)
    print(f"  数据长度保持: {len(deinterleaved_data)} 字节")
    print(f"  前16字节: {deinterleaved_data[:16].hex()}")
    
    # 步骤7: RS解码
    print(f"\n[解码步骤7] RS解码")
    recovered_chunks = {}
    
    rs_decoder = ReedSolomonDecoder(rs_n, rs_k)
    rs_packets_decoded = extract_rs_packets(deinterleaved_data, rs_n)
    print(f"  提取RS包: {len(rs_packets_decoded)} 个")
    
    for i, packet_bytes in enumerate(rs_packets_decoded[:3]):  # 只处理前3个
        try:
            decoded_data, errors_corrected = rs_decoder.decode(packet_bytes)
            print(f"    包 {i}: RS解码成功，纠错 {errors_corrected} 字节")
            
            # 解析元数据
            metadata = parse_enhanced_metadata(decoded_data)
            if metadata:
                chunk_index = metadata['chunk_index']
                print(f"      分片索引: {chunk_index}")
                print(f"      总分片数: {metadata['total_chunks']}")
                
                # 提取数据部分
                chunk_data = decoded_data[12:]  # 跳过12字节元数据
                recovered_chunks[chunk_index] = chunk_data
                
        except Exception as e:
            print(f"    包 {i}: RS解码失败 - {e}")
    
    # 步骤8: 重组和解压缩
    print(f"\n[解码步骤8] 重组和解压缩")
    if recovered_chunks:
        # 重组数据
        reassembled = b''
        for i in range(max(recovered_chunks.keys()) + 1):
            if i in recovered_chunks:
                chunk_data = recovered_chunks[i]
                # 去除填充的零字节
                useful_length = min(k_data, len(chunk_data))
                reassembled += chunk_data[:useful_length]
        
        print(f"  重组数据长度: {len(reassembled)} 字节")
        
        # 可变重复解码（简化为3倍重复解码）
        original_compressed = variable_repetition_decode(reassembled, len(compressed_data))
        print(f"  重复解码后: {len(original_compressed)} 字节")
        print(f"  与原压缩数据匹配: {'是' if original_compressed == compressed_data else '否'}")
        
        # 解压缩
        try:
            final_decoded = zlib.decompress(original_compressed)
            final_string = final_decoded.decode('utf-8')
            print(f"  解压缩成功: '{final_string}'")
            print(f"  与原始数据匹配: {'是' if final_string == test_data else '否'}")
        except Exception as e:
            print(f"  解压缩失败: {e}")
    
    print(f"\n" + "="*80)
    print("诊断完成")
    print("="*80)


def diagnose_encoding_decoding_with_spiral():
    """集成螺旋交织的编码解码流程诊断"""
    print("\n" + "="*80)
    print("集成螺旋交织的编码/解码流程诊断")
    print("="*80)
    
    # 导入螺旋交织模块
    try:
        from spiral_interleaver import spiral_encode, spiral_decode
        print("✓ 螺旋交织模块导入成功")
    except ImportError:
        print("❌ 无法导入螺旋交织模块，请确保spiral_interleaver.py在同目录下")
        return
    
    # 使用简单的测试数据
    test_data = "Hello This is a Test for FULL!"
    print(f"原始测试数据: '{test_data}'")
    print(f"原始UTF-8字节: {test_data.encode('utf-8')}")
    print(f"原始数据长度: {len(test_data)} 字符, {len(test_data.encode('utf-8'))} 字节")
    
    # 编码参数
    ldpc_rate = 0.5  # LDPC码率
    ldpc_block_size = 256  # LDPC块大小（比特）
    rs_n, rs_k = 255, 63  # Reed-Solomon参数
    sync_header = 0xB593  # 同步头
    
    print(f"\n编码参数:")
    print(f"  LDPC码率: {ldpc_rate}")
    print(f"  LDPC块大小: {ldpc_block_size} 比特")
    print(f"  RS参数: RS({rs_n}, {rs_k})")
    print(f"  同步头: 0x{sync_header:04X}")
    print(f"  ✨ 启用螺旋交织")
    
    # ===== 编码流程诊断 =====
    print(f"\n" + "-"*60)
    print("开始编码流程诊断")
    print("-"*60)
    
    # 步骤1: 数据预处理和压缩
    print(f"\n[步骤1] 数据预处理和压缩")
    if isinstance(test_data, str):
        utf8_data = test_data.encode('utf-8')
    else:
        utf8_data = test_data
    
    import zlib
    compressed_data = zlib.compress(utf8_data, level=9)
    print(f"  UTF-8编码: {len(utf8_data)} 字节")
    print(f"  压缩后: {len(compressed_data)} 字节")
    print(f"  压缩率: {(1-len(compressed_data)/len(utf8_data))*100:.1f}%")
    print(f"  压缩数据hex: {compressed_data.hex()}")
    
    # 步骤2: 可变重复编码（简化版，仅用3倍重复）
    print(f"\n[步骤2] 可变重复编码")
    repeated_data = repeat_encode(compressed_data, 3)
    print(f"  重复因子: 3")
    print(f"  重复后长度: {len(repeated_data)} 字节")
    print(f"  前16字节hex: {repeated_data[:16].hex()}")
    
    # 步骤3: Reed-Solomon编码
    print(f"\n[步骤3] Reed-Solomon编码")
    rs_encoder = ReedSolomonEncoder(rs_n, rs_k)
    rs_packets = []
    
    metadata_size = 12
    k_data = rs_k - metadata_size
    total_chunks = (len(repeated_data) + k_data - 1) // k_data
    
    print(f"  每包数据容量: {k_data} 字节")
    print(f"  总分片数: {total_chunks}")
    
    import struct
    for chunk_index in range(total_chunks):
        start_idx = chunk_index * k_data
        end_idx = min(start_idx + k_data, len(repeated_data))
        data_chunk = repeated_data[start_idx:end_idx]
        
        # 构建元数据
        metadata = struct.pack('>I', len(compressed_data))
        metadata += struct.pack('>H', chunk_index)
        metadata += struct.pack('>H', total_chunks)
        metadata += struct.pack('>B', 0x01)  # 版本
        metadata += struct.pack('>B', ldpc_block_size // 1000)
        
        header_crc = crc16_ccitt(metadata)
        metadata += struct.pack('>H', header_crc)
        
        packet_data = metadata + data_chunk
        if len(packet_data) < rs_k:
            packet_data += b'\x00' * (rs_k - len(packet_data))
        
        encoded_packet = rs_encoder.encode(packet_data)
        rs_packets.append(encoded_packet)
        
        if chunk_index == 0:  # 详细显示第一个包
            print(f"  分片 {chunk_index}:")
            print(f"    元数据: {metadata.hex()}")
            print(f"    数据: {data_chunk.hex()}")
            print(f"    RS编码前: {len(packet_data)} 字节")
            print(f"    RS编码后: {len(encoded_packet)} 字节")
    
    print(f"  总RS包数: {len(rs_packets)}")
    
    # 步骤4: LDPC编码
    print(f"\n[步骤4] LDPC编码")
    
    # 预计算LDPC编码后的单个包大小（通用化计算）
    expected_ldpc_size = calculate_ldpc_output_size(rs_n, ldpc_rate, ldpc_block_size)
    print(f"  预期单个LDPC包大小: {expected_ldpc_size} 字节")
    
    ldpc_output = []
    ldpc_lengths = []  # 保存每个包的原始长度
    actual_ldpc_sizes = []  # 记录实际的LDPC编码大小
    
    for i, packet in enumerate(rs_packets):
        try:
            # LDPC编码
            ldpc_encoded, original_length = ldpc_encode_pyldpc(packet, ldpc_rate, ldpc_block_size)
            ldpc_output.extend(ldpc_encoded)
            ldpc_lengths.append(original_length)
            actual_ldpc_sizes.append(len(ldpc_encoded))
            
            if i == 0:  # 详细显示第一个包
                print(f"  包 {i} LDPC编码:")
                print(f"    原始长度: {len(packet)} 字节 ({len(packet)*8} 比特)")
                print(f"    编码后长度: {len(ldpc_encoded)} 字节 ({len(ldpc_encoded)*8} 比特)")
                print(f"    扩展率: {len(ldpc_encoded)/len(packet):.2f}")
                print(f"    记录的原始比特长度: {original_length}")
                print(f"    预期vs实际: {expected_ldpc_size} vs {len(ldpc_encoded)} 字节")
        except Exception as e:
            print(f"  包 {i} LDPC编码失败: {e}")
            break
    
    print(f"  LDPC编码总输出: {len(ldpc_output)} 字节")
    
    # 分析实际LDPC大小的一致性
    if actual_ldpc_sizes:
        unique_sizes = set(actual_ldpc_sizes)
        print(f"  实际LDPC包大小分布: {dict(zip(unique_sizes, [actual_ldpc_sizes.count(s) for s in unique_sizes]))}")
        if len(unique_sizes) == 1:
            single_ldpc_size = list(unique_sizes)[0]
            print(f"  ✓ 所有LDPC包大小一致: {single_ldpc_size} 字节")
        else:
            single_ldpc_size = max(unique_sizes)  # 使用最大值作为标准
            print(f"  ⚠ LDPC包大小不一致，使用最大值: {single_ldpc_size} 字节")
    else:
        single_ldpc_size = expected_ldpc_size
        print(f"  使用预期大小: {single_ldpc_size} 字节")
    
    # 步骤5: 螺旋交织
    print(f"\n[步骤5] 螺旋交织")
    ldpc_data_bytes = bytes(ldpc_output)
    print(f"  交织前数据长度: {len(ldpc_data_bytes)} 字节")
    print(f"  交织前前16字节: {ldpc_data_bytes[:16].hex()}")
    
    try:
        interleaved_data = spiral_encode(ldpc_data_bytes)
        print(f"  ✓ 螺旋交织成功")
        print(f"  交织后数据长度: {len(interleaved_data)} 字节")
        print(f"  交织后前16字节: {interleaved_data[:16].hex()}")
        
        # 验证数据完整性（字节集合应该相同）
        if set(ldpc_data_bytes) == set(interleaved_data) and len(ldpc_data_bytes) == len(interleaved_data):
            print(f"  ✓ 数据完整性验证通过（长度和字节集合保持一致）")
        else:
            print(f"  ⚠ 数据完整性检查异常")
    except Exception as e:
        print(f"  ❌ 螺旋交织失败: {e}")
        interleaved_data = ldpc_data_bytes  # 失败时跳过交织
    
    # 步骤6: 添加同步头并转换为比特流
    print(f"\n[步骤6] 添加同步头和比特流转换")
    master_bitstream = []
    sync_header_bytes = struct.pack('>H', sync_header)
    
    # 为每个LDPC块添加同步头
    blocks_with_sync = 0
    
    for i in range(0, len(interleaved_data), single_ldpc_size):
        block = interleaved_data[i:i + single_ldpc_size]
        if len(block) > 0:
            # 如果块不完整，进行填充
            if len(block) < single_ldpc_size:
                block = block + b'\x00' * (single_ldpc_size - len(block))
                print(f"  块 {blocks_with_sync}: 填充 {single_ldpc_size - len(block)} 字节")
            
            full_block = sync_header_bytes + block
            blocks_with_sync += 1
            
            for byte_val in full_block:
                for bit in range(8):
                    master_bitstream.append(bool((byte_val >> (7 - bit)) & 1))
    
    print(f"  添加同步头的块数: {blocks_with_sync}")
    print(f"  每块大小: 同步头(2字节) + LDPC数据({single_ldpc_size}字节) = {single_ldpc_size + 2}字节")
    print(f"  比特流长度: {len(master_bitstream)} 比特")
    print(f"  前32比特: {''.join('1' if b else '0' for b in master_bitstream[:32])}")
    
    # 步骤7: TMR (三模冗余)
    print(f"\n[步骤7] TMR (三模冗余)")
    tmr_bitstream = []
    for bit in master_bitstream:
        tmr_bitstream.extend([bit, bit, bit])  # 每个比特重复3次
    
    print(f"  TMR前长度: {len(master_bitstream)} 比特")
    print(f"  TMR后长度: {len(tmr_bitstream)} 比特")
    print(f"  扩展因子: {len(tmr_bitstream)/len(master_bitstream):.1f} (期望: 3.0)")
    
    # 步骤8: 扩频编码
    print(f"\n[步骤8] 扩频编码")
    spread_bitstream = apply_dsss(tmr_bitstream)
    print(f"  扩频前: {len(tmr_bitstream)} 比特")
    print(f"  扩频后: {len(spread_bitstream)} 比特")
    print(f"  扩频因子: {len(spread_bitstream)/len(tmr_bitstream):.1f}")
    
    # ===== 解码流程诊断 =====
    print(f"\n" + "-"*60)
    print("开始解码流程诊断")
    print("-"*60)
    
    # 使用编码结果进行解码测试
    encoded_bitstream = spread_bitstream
    
    # 步骤1: 去扩频
    print(f"\n[解码步骤1] 去扩频")
    despread_bits = remove_dsss(encoded_bitstream)
    print(f"  去扩频前: {len(encoded_bitstream)} 比特")
    print(f"  去扩频后: {len(despread_bits)} 比特")
    
    # 步骤2: TMR投票解码
    print(f"\n[解码步骤2] TMR投票解码")
    print(f"  TMR解码前: {len(despread_bits)} 比特")
    print(f"  前32比特: {''.join('1' if b else '0' for b in despread_bits[:32])}")
    
    tmr_decoded = tmr_voting_decode(despread_bits)
    print(f"  TMR解码后: {len(tmr_decoded)} 比特")
    print(f"  前32比特: {''.join('1' if b else '0' for b in tmr_decoded[:32])}")
    print(f"  长度比例: {len(despread_bits)/len(tmr_decoded):.2f} (期望: 3.0)")
    
    # 验证TMR解码质量：检查同步头模式
    sync_pattern = []
    for bit in range(16):
        sync_pattern.append(bool((sync_header >> (15 - bit)) & 1))
    
    print(f"  期望同步头模式: {''.join('1' if b else '0' for b in sync_pattern)}")
    
    # 在TMR解码结果中查找同步头模式
    tmr_sync_positions = []
    for i in range(len(tmr_decoded) - 15):
        match = True
        for j in range(16):
            if tmr_decoded[i + j] != sync_pattern[j]:
                match = False
                break
        if match:
            tmr_sync_positions.append(i)
    
    print(f"  TMR解码后找到同步头位置: {tmr_sync_positions[:5]} (前5个)")
    
    if len(tmr_sync_positions) == 0:
        print(f"  [警告] TMR解码后未找到同步头！数据可能已损坏。")
    else:
        print(f"  ✓ TMR解码质量良好，找到 {len(tmr_sync_positions)} 个同步头")
    
    # 步骤3: 转换为字节流
    print(f"\n[解码步骤3] 转换为字节流")
    byte_stream = bits_to_bytes(tmr_decoded)
    print(f"  字节流长度: {len(byte_stream)} 字节")
    
    # 步骤4: 提取LDPC块（通用化）
    print(f"\n[解码步骤4] 提取LDPC块")
    print(f"  字节流前16字节: {byte_stream[:16].hex()}")
    print(f"  寻找同步头: 0x{sync_header:04X} ({struct.pack('>H', sync_header).hex()})")
    print(f"  LDPC块大小: {single_ldpc_size} 字节（通用计算）")
    
    # 手动搜索同步头
    sync_bytes = struct.pack('>H', sync_header)
    sync_positions = []
    for i in range(len(byte_stream) - 1):
        if byte_stream[i:i+2] == sync_bytes:
            sync_positions.append(i)
    
    print(f"  手动搜索同步头位置: {sync_positions[:10]}")  # 只显示前10个
    
    ldpc_blocks = []
    expected_block_size = single_ldpc_size + 2  # 同步头2字节 + LDPC数据
    
    for pos in sync_positions:
        if pos + expected_block_size <= len(byte_stream):
            block_data = byte_stream[pos:pos + expected_block_size]
            ldpc_blocks.append((block_data, pos))
            print(f"    块 {len(ldpc_blocks)-1}: 位置 {pos}, 长度 {len(block_data)} 字节")
            print(f"      前8字节: {block_data[:8].hex()}")
            print(f"      同步头验证: {block_data[:2].hex() == sync_bytes.hex()}")
        else:
            print(f"    位置 {pos}: 数据不足 ({len(byte_stream) - pos} < {expected_block_size}), 跳过")
    
    print(f"  找到LDPC块: {len(ldpc_blocks)} 个")
    
    # 提取所有LDPC负载数据（移除同步头）
    ldpc_payload_data = []
    for block_data, position in ldpc_blocks:
        ldpc_payload = block_data[2:]  # 移除2字节同步头
        ldpc_payload_data.extend(ldpc_payload)
    
    print(f"  合并LDPC负载数据: {len(ldpc_payload_data)} 字节")
    
    # 步骤5: 螺旋去交织
    print(f"\n[解码步骤5] 螺旋去交织")
    interleaved_ldpc_data = bytes(ldpc_payload_data)
    print(f"  去交织前数据长度: {len(interleaved_ldpc_data)} 字节")
    print(f"  去交织前前16字节: {interleaved_ldpc_data[:16].hex()}")
    
    try:
        # 计算原始LDPC数据的长度
        original_ldpc_length = len(ldpc_data_bytes)
        deinterleaved_data = spiral_decode(interleaved_ldpc_data, original_ldpc_length)
        print(f"  ✓ 螺旋去交织成功")
        print(f"  去交织后数据长度: {len(deinterleaved_data)} 字节")
        print(f"  去交织后前16字节: {deinterleaved_data[:16].hex()}")
        
        # 验证去交织效果
        if deinterleaved_data == ldpc_data_bytes:
            print(f"  ✅ 螺旋交织/去交织完全匹配原始LDPC数据")
        else:
            match_bytes = sum(a == b for a, b in zip(deinterleaved_data, ldpc_data_bytes))
            total_bytes = min(len(deinterleaved_data), len(ldpc_data_bytes))
            print(f"  ⚠ 部分匹配: {match_bytes}/{total_bytes} 字节匹配 ({match_bytes/total_bytes*100:.1f}%)")
    except Exception as e:
        print(f"  ❌ 螺旋去交织失败: {e}")
        deinterleaved_data = interleaved_ldpc_data  # 失败时跳过去交织
    
    # 步骤6: LDPC解码
    print(f"\n[解码步骤6] LDPC解码")
    ldpc_decoded_data = []
    
    # 根据原始RS包数量来分割LDPC数据
    rs_packet_count = len(rs_packets)
    ldpc_packet_size = len(deinterleaved_data) // rs_packet_count if rs_packet_count > 0 else single_ldpc_size
    
    print(f"  预期RS包数量: {rs_packet_count}")
    print(f"  计算LDPC包大小: {ldpc_packet_size} 字节")
    
    for block_idx in range(rs_packet_count):
        try:
            start_pos = block_idx * ldpc_packet_size
            end_pos = min(start_pos + ldpc_packet_size, len(deinterleaved_data))
            ldpc_payload = deinterleaved_data[start_pos:end_pos]
            
            # 使用保存的原始长度（如果可用）
            if block_idx < len(ldpc_lengths):
                original_length = ldpc_lengths[block_idx]
            else:
                # 估算长度
                original_length = rs_n * 8  # RS包的比特长度
            
            decoded = ldpc_decode_pyldpc(ldpc_payload, original_length, ldpc_rate, ldpc_block_size)
            ldpc_decoded_data.extend(decoded)
            
            if block_idx == 0:
                print(f"    块 {block_idx} LDPC解码:")
                print(f"      输入: {len(ldpc_payload)} 字节")
                print(f"      原始长度: {original_length} 比特")
                print(f"      解码输出: {len(decoded)} 字节")
                print(f"      解码数据hex: {decoded.hex()}")
        except Exception as e:
            print(f"    块 {block_idx} LDPC解码失败: {e}")
    
    print(f"  LDPC解码总输出: {len(ldpc_decoded_data)} 字节")
    
    # 步骤7: RS解码
    print(f"\n[解码步骤7] RS解码")
    recovered_chunks = {}
    
    rs_decoder = ReedSolomonDecoder(rs_n, rs_k)
    rs_packets_decoded = extract_rs_packets(bytes(ldpc_decoded_data), rs_n)
    print(f"  提取RS包: {len(rs_packets_decoded)} 个")
    
    for i, packet_bytes in enumerate(rs_packets_decoded[:3]):  # 只处理前3个
        try:
            decoded_data, errors_corrected = rs_decoder.decode(packet_bytes)
            print(f"    包 {i}: RS解码成功，纠错 {errors_corrected} 字节")
            
            # 解析元数据
            metadata = parse_enhanced_metadata(decoded_data)
            if metadata:
                chunk_index = metadata['chunk_index']
                print(f"      分片索引: {chunk_index}")
                print(f"      总分片数: {metadata['total_chunks']}")
                
                # 提取数据部分
                chunk_data = decoded_data[12:]  # 跳过12字节元数据
                recovered_chunks[chunk_index] = chunk_data
                
        except Exception as e:
            print(f"    包 {i}: RS解码失败 - {e}")
    
    # 步骤8: 重组和解压缩
    print(f"\n[解码步骤8] 重组和解压缩")
    if recovered_chunks:
        # 重组数据
        reassembled = b''
        for i in range(max(recovered_chunks.keys()) + 1):
            if i in recovered_chunks:
                chunk_data = recovered_chunks[i]
                # 去除填充的零字节
                useful_length = min(k_data, len(chunk_data))
                reassembled += chunk_data[:useful_length]
        
        print(f"  重组数据长度: {len(reassembled)} 字节")
        
        # 可变重复解码（简化为3倍重复解码）
        original_compressed = variable_repetition_decode(reassembled, len(compressed_data))
        print(f"  重复解码后: {len(original_compressed)} 字节")
        print(f"  与原压缩数据匹配: {'是' if original_compressed == compressed_data else '否'}")
        
        # 解压缩
        try:
            final_decoded = zlib.decompress(original_compressed)
            final_string = final_decoded.decode('utf-8')
            print(f"  解压缩成功: '{final_string}'")
            print(f"  与原始数据匹配: {'是' if final_string == test_data else '否'}")
        except Exception as e:
            print(f"  解压缩失败: {e}")
    
    print(f"\n" + "="*80)
    print("螺旋交织集成诊断完成")
    print("="*80)


def test_rs_encoding():
    """测试Reed-Solomon编码功能"""
    print("\n--- Reed-Solomon编码测试 ---")
    
    test_data = b"Hello, Reed-Solomon!"
    
    rs_configs = [
        (255, 63),
        (255, 20),
        (255, 127),
        (127, 31)
    ]
    
    for rs_n, rs_k in rs_configs:
        try:
            encoder = ReedSolomonEncoder(rs_n, rs_k)
            encoded = encoder.encode(test_data)
            
            print(f"\nRS({rs_n},{rs_k}):")
            print(f"  原始数据长度: {len(test_data)} 字节")
            print(f"  编码后长度: {len(encoded)} 字节")
            print(f"  可纠正错误: {encoder.get_correction_capacity()} 字节")
            print(f"  冗余度: {((rs_n-rs_k)/rs_n)*100:.1f}%")
            
        except Exception as e:
            print(f"RS({rs_n},{rs_k}) 编码失败: {e}")


def test_decode_functionality():
    """测试解码功能"""
    print("\n--- 解码功能测试 ---")
    
    # 先测试一个简单的例子，打开调试
    print("\n=== 调试模式：测试简单消息 ===")
    simple_msg = "Hello"
    required_bits, _ = estimate_required_capacity(simple_msg)
    encoded = encode_binary_to_watermark(simple_msg, max_capacity_bits=required_bits * 2)
    
    print(f"原始消息: {simple_msg}")
    print(f"编码后长度: {len(encoded)} 比特")
    
    # 检查编码后的同步头
    sync_header = 0xB593
    sync_pattern = []
    for bit in range(16):
        sync_pattern.append(bool((sync_header >> (15 - bit)) & 1))
    
    print(f"同步头模式: {sync_pattern}")
    print(f"编码后前16位: {encoded[:16]}")
    
    # 使用调试模式解码
    decoded_str, stats = decode_watermark_to_string(encoded, debug=True)
    print(f"\n解码结果: {decoded_str}")
    print(f"统计信息: {stats}")
    
    print("\n=== 正常测试 ===")
    
    test_messages = [
        "Hello, World!",
        "这是一个中文测试信息。",
        "A longer test message with more content to test the robustness of the system.",
        b'\x00\x01\x02\x03\x04\x05\xFF\xFE\xFD'
    ]
    
    for i, original in enumerate(test_messages):
        print(f"\n测试 {i+1}: {type(original).__name__} 数据")
        
        try:
            required_bits, _ = estimate_required_capacity(original)
            encoded = encode_binary_to_watermark(original, max_capacity_bits=required_bits * 2)
            print(f"原始数据: {original}")
            print(f"编码长度: {len(encoded)} 比特")
            
            # 解码
            if isinstance(original, str):
                decoded_str, stats = decode_watermark_to_string(encoded)
                print(f"解码结果: {decoded_str}")
                print(f"解码成功: {'是' if decoded_str == original else '否'}")
            else:
                decoded_bytes, stats = decode_watermark_to_binary(encoded)
                print(f"解码结果: {decoded_bytes}")
                print(f"解码成功: {'是' if decoded_bytes == original else '否'}")
            
            # 简化统计
            print(f"数据包: {stats['valid_packets']}/{stats['total_packets_found']}")
            print(f"纠错: {stats['total_errors_corrected']} 字节")
            
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()


def validate_ldpc_params(n: int, rate: float) -> tuple:
    """验证并调整 LDPC 参数以满足 pyldpc 库的约束"""
    d_v = 3  # 变量节点度数（通常取较小值）
    d_c = int(d_v / rate)
    
    # 确保 d_c 能整除 n
    original_d_c = d_c
    while n % d_c != 0:
        d_c += 1
    
    # 确保 d_c >= d_v
    if d_c < d_v:
        d_c = d_v
        while n % d_c != 0:
            d_c += 1
    
    # 计算实际参数
    n_equations = d_v * n // d_c
    k_actual = n - n_equations
    actual_rate = k_actual / n
    
    print(f"参数调整: d_c 从 {original_d_c} 调整为 {d_c}")
    print(f"预期码率: {rate:.4f}, 实际码率: {actual_rate:.4f}")
    
    return d_v, d_c, k_actual, actual_rate

# 测试函数
def test_ldpc_encoding_and_decoding():
    """测试 LDPC 编码和解码功能"""
    # 测试数据
    test_data = b"Hello, LDPC encoding and decoding test! This is a longer message to ensure multiple blocks are processed correctly."
    
    # 测试参数
    rate = 0.5  # 增加码率以提高解码成功率
    block_size = 256 # 减小块大小以更好地观察效果
    
    print("="*50)
    print("测试 LDPC 编码和解码")
    print("="*50)
    
    # 验证参数
    d_v, d_c, k_actual, actual_rate = validate_ldpc_params(block_size, rate)
    
    # 编码
    try:
        print("\n--- 编码阶段 ---")
        encoded, data_length = ldpc_encode_pyldpc(test_data, rate, block_size)
        print(f"\n原始数据长度: {len(test_data)} 字节 ({len(test_data)*8} 比特)")
        print(f"编码后数据长度: {len(encoded)} 字节 ({len(encoded)*8} 比特)")
        print(f"扩展比例: {len(encoded)/len(test_data):.2f}")
        print("编码成功！")
        
        # 解码
        print("\n--- 解码阶段 ---")
        decoded = ldpc_decode_pyldpc(encoded,data_length, rate, block_size)
        print(f"解码后数据长度: {len(decoded)} 字节")
        
        # 比较原始数据和解码数据
        if decoded == test_data:
            print("\n解码成功！原始数据与解码数据一致。")
        else:
            print("\n解码失败！原始数据与解码数据不一致。")
            print(f"原始数据: {test_data}")
            print(f"解码数据: {decoded}")
            
            # 找出不一致的地方
            min_len = min(len(test_data), len(decoded))
            diff_idx = -1
            for i in range(min_len):
                if test_data[i] != decoded[i]:
                    diff_idx = i
                    break
            
            if diff_idx != -1:
                print(f"第一个不一致的字节在索引 {diff_idx}:")
                print(f"原始字节: {test_data[diff_idx]:08b} ({test_data[diff_idx]})")
                print(f"解码字节: {decoded[diff_idx]:08b} ({decoded[diff_idx]})")
            else:
                if len(test_data) > len(decoded):
                    print("解码数据比原始数据短。")
                else:
                    print("解码数据比原始数据长。")

    except Exception as e:
        print(f"编码或解码失败: {e}")
        import traceback
        traceback.print_exc()


def test_noise_resilience():
    """测试噪声抗性"""
    print("\n--- 噪声抗性测试 ---")
    
    test_message = "Noise resilience test message"
    
    # 编码
    required_bits, _ = estimate_required_capacity(test_message)
    clean_bitstream = encode_binary_to_watermark(test_message, max_capacity_bits=required_bits * 3)
    
    # 添加不同程度的噪声
    noise_levels = [0.01, 0.05, 0.10, 0.15, 0.20]
    
    print(f"原始消息: {test_message}")
    print(f"比特流长度: {len(clean_bitstream)}")
    
    for noise_level in noise_levels:
        # 复制比特流并添加噪声
        noisy_bitstream = clean_bitstream.copy()
        num_flips = int(len(noisy_bitstream) * noise_level)
        
        # 随机翻转比特
        import random
        flip_positions = random.sample(range(len(noisy_bitstream)), num_flips)
        for pos in flip_positions:
            noisy_bitstream[pos] = not noisy_bitstream[pos]
        
        # 尝试解码
        decoded_str, stats = decode_watermark_to_string(noisy_bitstream)
        
        success = decoded_str == test_message if decoded_str else False
        print(f"\n噪声级别 {noise_level*100:4.1f}%: 翻转 {num_flips:4d} 比特")
        print(f"  解码成功: {'是' if success else '否'}")
        print(f"  有效包: {stats['valid_packets']:3d}, 纠错: {stats['total_errors_corrected']:3d} 字节")
        if decoded_str and not success:
            print(f"  解码结果: {decoded_str[:50]}...")


# 主程序
if __name__ == "__main__":
    print("开始执行鲁棒二进制信息嵌入编码器自动化测试...")
    print("=" * 60)

    # 运行 CRC16 测试
    test_crc16()
    print("-" * 60)

    # 运行 RS 编码测试
    test_rs_encoding()
    print("-" * 60)

    # 运行 LDPC编码测试
    test_ldpc_encoding_and_decoding()

    # 运行诊断测试
    # diagnose_encoding_decoding()
    diagnose_encoding_decoding_with_spiral()
    # print("-" * 60)

    # # 运行完整的编码/解码功能测试
    # test_decode_functionality()
    # print("-" * 60)

    # # 运行噪声抵抗能力测试
    # test_noise_resilience()
    # print("=" * 60)
    print("所有自动化测试已完成。")