## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms
"""
鲁棒二进制信息嵌入编码器(rc1)

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


def crc16_ccitt(data: bytes) -> int:
    """
    计算CRC16-CCITT校验码
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
        self.rs_codec = RSCodec(self.parity_symbols, nsize=n)
        
    def decode(self, encoded_data: bytes) -> Tuple[bytes, int]:
        if len(encoded_data) != self.n:
            if len(encoded_data) < self.n:
                encoded_data = encoded_data + b'\x00' * (self.n - len(encoded_data))
            else:
                encoded_data = encoded_data[:self.n]
        
        try:
            # 使用reedsolo进行解码
            decoded = self.rs_codec.decode(encoded_data)
            # reedsolo可能返回tuple或bytes，需要处理
            if isinstance(decoded, tuple):
                # 如果返回的是元组，第一个元素是解码数据，第二个可能是纠错信息
                decoded_data = decoded[0]

                if len(decoded) > 1:
                    correction_info = decoded[1]
                    if isinstance(correction_info, int):
                        num_corrected = correction_info
                    elif isinstance(correction_info, (list, tuple)):
                        num_corrected = len(correction_info)
                    elif isinstance(correction_info, (bytes, bytearray)):
                        num_corrected = len(correction_info)
                    else:
                        num_corrected = 0
                else:
                    num_corrected = 0
            else:
                # 如果直接返回解码数据
                decoded_data = decoded
                num_corrected = 0
            # 确保decoded_data是bytes类型
            if isinstance(decoded_data, (list, tuple)):
                decoded_data = bytes(decoded_data)
            elif isinstance(decoded_data, bytearray):
                decoded_data = bytes(decoded_data)
            elif not isinstance(decoded_data, bytes):
                decoded_data = bytes(decoded_data)
            # 确保num_corrected是非负整数
            if not isinstance(num_corrected, int) or num_corrected < 0:
                num_corrected = 0
            return decoded_data[:self.k], num_corrected
        except Exception as e:
            raise ValueError(f"RS解码失败: {e}")


def encode_binary_to_watermark(
    data: Union[bytes, str],
    max_capacity_bits: int = None,
    rs_n: int = 255,
    rs_k: int = 63,
    sync_header: int = 0xB593,
    interleave_depth: int = 8
) -> List[bool]:
    """将任意二进制数据编码为水印布尔列表"""
    
    # 1. 数据预处理
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    # 2. 压缩数据
    compressed_data = zlib.compress(data, level=9)
    total_payload_length = len(compressed_data)
    
    # 3. 计算数据分片参数
    metadata_size = 10  # Total_Payload_Length(4) + Chunk_Index(2) + Total_Chunks(2) + Header_CRC(2)
    k_data = rs_k - metadata_size
    
    if k_data <= 0:
        raise ValueError(f"RS码的k值{rs_k}太小，无法容纳元数据（需要至少{metadata_size}字节）")
    
    # 计算需要多少个数据分片
    total_chunks = (total_payload_length + k_data - 1) // k_data
    
    # 4. RS编码器
    rs_encoder = ReedSolomonEncoder(rs_n, rs_k)
    
    # 5. 构建数据包
    packets = []
    
    for chunk_index in range(total_chunks):
        # 提取当前数据分片
        start_idx = chunk_index * k_data
        end_idx = min(start_idx + k_data, total_payload_length)
        data_chunk = compressed_data[start_idx:end_idx]
        
        # 构建元数据
        metadata = struct.pack('>I', total_payload_length)  # Total_Payload_Length (4 bytes, big-endian)
        metadata += struct.pack('>H', chunk_index)          # Chunk_Index (2 bytes)
        metadata += struct.pack('>H', total_chunks)         # Total_Chunks (2 bytes)
        
        # 计算元数据CRC
        header_crc = crc16_ccitt(metadata)
        metadata += struct.pack('>H', header_crc)           # Header_CRC (2 bytes)
        
        # 组合元数据和数据分片
        packet_data = metadata + data_chunk
        
        # 填充到rs_k长度
        if len(packet_data) < rs_k:
            packet_data += b'\x00' * (rs_k - len(packet_data))
        
        # RS编码
        encoded_packet = rs_encoder.encode(packet_data)
        
        # 添加同步头
        sync_header_bytes = struct.pack('>H', sync_header)
        full_packet = sync_header_bytes + encoded_packet
        
        packets.append(full_packet)
    
    # 6. 将所有数据包转换为比特流
    master_bitstream = []
    
    for packet in packets:
        for byte_val in packet:
            for bit in range(8):
                master_bitstream.append(bool((byte_val >> (7 - bit)) & 1))
    
    # 6.5. 添加交织编码
    if interleave_depth > 1:
        master_bitstream = interleave_bitstream(master_bitstream, interleave_depth)
    
    # 7. 重复填充到目标容量
    if max_capacity_bits is None:
        final_bitstream = master_bitstream
    else:
        final_bitstream = []
        bit_index = 0
        
        while len(final_bitstream) < max_capacity_bits:
            final_bitstream.append(master_bitstream[bit_index % len(master_bitstream)])
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
    rs_k: int = 63,
    sync_header: int = 0xB593,
    max_attempts: int = 1000,
    interleave_depth: int = 8,
    debug: bool = False
) -> Tuple[Optional[bytes], Dict]:
    """从水印比特流解码出原始二进制数据"""
    
    # 初始化统计信息
    stats = {
        'total_packets_found': 0,
        'valid_packets': 0,
        'rs_decode_success': 0,
        'rs_decode_failed': 0,
        'metadata_valid': 0,
        'metadata_invalid': 0,
        'total_errors_corrected': 0,
        'chunks_recovered': {},
        'final_payload_length': 0,
        'compression_successful': False,
        'interleaving_applied': interleave_depth > 1,
        'original_bitstream_length': len(bitstream),
        'debug_info': []
    }
    
    if debug:
        print(f"[DEBUG] 原始比特流长度: {len(bitstream)}")
        print(f"[DEBUG] 交织深度: {interleave_depth}")
    
    # 0. 数据去交织处理
    if interleave_depth > 1:
        bitstream = deinterleave_bitstream(bitstream, interleave_depth)
        if debug:
            print(f"[DEBUG] 去交织后比特流长度: {len(bitstream)}")
            print(f"[DEBUG] 去交织前后长度变化: {stats['original_bitstream_length']} -> {len(bitstream)}")
    
    # 1. 提取所有可能的数据包
    raw_packets = extract_packets_from_bitstream(bitstream, rs_n, sync_header)
    stats['total_packets_found'] = len(raw_packets)
    
    if debug:
        print(f"[DEBUG] 找到 {len(raw_packets)} 个数据包")
    
    if not raw_packets:
        return None, stats
    
    # 2. 创建RS解码器
    rs_decoder = ReedSolomonDecoder(rs_n, rs_k)
    
    # 3. 解码数据包并收集有效数据
    valid_chunks = defaultdict(list)
    total_chunks_expected = None
    total_payload_length = None
    
    processed_count = 0
    for packet_idx, (packet_bytes, position) in enumerate(raw_packets):
        if processed_count >= max_attempts:
            break
        processed_count += 1
        
        if debug and packet_idx < 30:  # 只调试前3个包
            print(f"\n[DEBUG] 数据包 {packet_idx} @ 位置 {position}:")
            print(f"  长度: {len(packet_bytes)} 字节")
            print(f"  前16字节: {packet_bytes[:16].hex()}")
        
        try:
            # RS解码
            decoded_data, errors_corrected = rs_decoder.decode(packet_bytes)
            stats['rs_decode_success'] += 1
            stats['total_errors_corrected'] += errors_corrected
            stats['valid_packets'] += 1
            
            if debug and packet_idx < 3:
                print(f"  RS解码成功，纠错: {errors_corrected} 字节")
                print(f"  解码后前16字节: {decoded_data[:16].hex()}")
            
            # 解析元数据
            metadata, chunk_data = parse_packet_metadata(decoded_data)
            
            if metadata is not None:
                stats['metadata_valid'] += 1
                
                if debug and packet_idx < 3:
                    print(f"  元数据有效:")
                    print(f"    总长度: {metadata['total_payload_length']}")
                    print(f"    分片索引: {metadata['chunk_index']}/{metadata['total_chunks']}")
                    print(f"    CRC: 0x{metadata['header_crc']:04X}")
                
                # 设置或验证全局参数
                if total_chunks_expected is None:
                    total_chunks_expected = metadata['total_chunks']
                    total_payload_length = metadata['total_payload_length']
                elif (total_chunks_expected != metadata['total_chunks'] or 
                    total_payload_length != metadata['total_payload_length']):
                    continue
                
                # 存储有效的数据块
                chunk_index = metadata['chunk_index']
                if chunk_index < total_chunks_expected:
                    valid_chunks[chunk_index].append((chunk_data, metadata, errors_corrected))
                
            else:
                stats['metadata_invalid'] += 1
                if debug and packet_idx < 3:
                    print(f"  元数据无效！")
                
        except Exception as e:
            stats['rs_decode_failed'] += 1
            if debug and packet_idx < 3:
                print(f"  RS解码失败: {e}")
            continue
    
    if not valid_chunks or total_chunks_expected is None:
        return None, stats
    
    # 4. 重组数据
    final_chunks = {}
    
    for chunk_index in range(total_chunks_expected):
        if chunk_index in valid_chunks:
            candidates = valid_chunks[chunk_index]
            # 选择纠错数最少的版本
            best_candidate = min(candidates, key=lambda x: x[2])
            final_chunks[chunk_index] = best_candidate[0]
            
            stats['chunks_recovered'][chunk_index] = {
                'candidates': len(candidates),
                'errors_corrected': best_candidate[2]
            }
    
    # 检查是否收集到了所有必需的分片
    missing_chunks = []
    for i in range(total_chunks_expected):
        if i not in final_chunks:
            missing_chunks.append(i)
    
    if missing_chunks:
        if debug:
            print(f"警告：缺失分片 {missing_chunks}，尝试部分恢复...")
    
    # 5. 重组压缩数据
    compressed_data = b''
    metadata_size = 10
    k_data = rs_k - metadata_size
    
    for chunk_index in range(total_chunks_expected):
        if chunk_index in final_chunks:
            chunk_data = final_chunks[chunk_index]
            
            # 计算这个分片应该包含的实际数据长度
            start_pos = chunk_index * k_data
            expected_length = min(k_data, total_payload_length - start_pos)
            
            # 只取需要的数据长度（去除填充）
            actual_data = chunk_data[:expected_length]
            compressed_data += actual_data
        else:
            # 缺失分片，用零填充
            start_pos = chunk_index * k_data
            expected_length = min(k_data, total_payload_length - start_pos)
            compressed_data += b'\x00' * expected_length
    
    stats['final_payload_length'] = len(compressed_data)
    
    # 6. 解压缩数据
    try:
        original_data = zlib.decompress(compressed_data)
        stats['compression_successful'] = True
        
        if debug:
            print(f"[DEBUG] 解压缩成功，最终数据长度: {len(original_data)}")
        
        return original_data, stats
    except Exception as e:
        if debug:
            print(f"解压缩失败: {e}")
        return compressed_data, stats


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


def diagnose_encoding_decoding():
    """诊断编码解码过程"""
    print("\n--- 编码/解码诊断 ---")
    
    # 使用最简单的测试数据
    test_data = b"TEST"
    print(f"测试数据: {test_data}")
    
    # 手动构建一个数据包
    metadata_size = 10
    rs_n, rs_k = 255, 63
    
    # 元数据
    total_payload_length = len(test_data)
    chunk_index = 0
    total_chunks = 1
    
    print(f"\n构建元数据:")
    print(f"  总长度: {total_payload_length}")
    print(f"  分片索引: {chunk_index}")
    print(f"  总分片数: {total_chunks}")
    
    # 构建元数据
    metadata = struct.pack('>I', total_payload_length)
    metadata += struct.pack('>H', chunk_index)
    metadata += struct.pack('>H', total_chunks)
    
    # 计算CRC
    header_crc = crc16_ccitt(metadata)
    print(f"  计算的CRC: 0x{header_crc:04X}")
    
    metadata += struct.pack('>H', header_crc)
    
    # 组合数据包
    packet_data = metadata + test_data
    packet_data += b'\x00' * (rs_k - len(packet_data))
    
    print(f"\n数据包信息:")
    print(f"  元数据长度: {len(metadata)}")
    print(f"  数据长度: {len(test_data)}")
    print(f"  填充后长度: {len(packet_data)}")
    print(f"  前16字节hex: {packet_data[:16].hex()}")
    
    # RS编码
    rs_encoder = ReedSolomonEncoder(rs_n, rs_k)
    encoded_packet = rs_encoder.encode(packet_data)
    print(f"\nRS编码后:")
    print(f"  长度: {len(encoded_packet)}")
    
    # 添加同步头
    sync_header = 0xB593
    sync_header_bytes = struct.pack('>H', sync_header)
    full_packet = sync_header_bytes + encoded_packet
    
    print(f"\n完整数据包:")
    print(f"  同步头: 0x{sync_header:04X}")
    print(f"  总长度: {len(full_packet)}")
    print(f"  前8字节hex: {full_packet[:8].hex()}")
    
    # 转换为比特流
    bitstream = []
    for byte_val in full_packet:
        for bit in range(8):
            bitstream.append(bool((byte_val >> (7 - bit)) & 1))
    
    print(f"\n比特流:")
    print(f"  长度: {len(bitstream)} 比特")
    print(f"  前32比特: {''.join('1' if b else '0' for b in bitstream[:32])}")
    
    # 测试同步头查找
    sync_positions = find_sync_patterns(bitstream, sync_header)
    print(f"\n同步头查找:")
    print(f"  找到位置: {sync_positions}")
    
    # 测试数据包提取
    packets = extract_packets_from_bitstream(bitstream, rs_n, sync_header)
    print(f"\n数据包提取:")
    print(f"  提取到的数据包数: {len(packets)}")
    
    if packets:
        packet_bytes, pos = packets[0]
        print(f"  第一个数据包位置: {pos}")
        print(f"  数据包长度: {len(packet_bytes)}")
        print(f"  前16字节hex: {packet_bytes[:16].hex()}")
        
        # 尝试RS解码
        rs_decoder = ReedSolomonDecoder(rs_n, rs_k)
        try:
            decoded_data, errors = rs_decoder.decode(packet_bytes)
            print(f"\nRS解码:")
            print(f"  成功！纠错: {errors}")
            print(f"  解码后长度: {len(decoded_data)}")
            print(f"  前16字节hex: {decoded_data[:16].hex()}")
            
            # 解析元数据
            parsed_metadata, chunk_data = parse_packet_metadata(decoded_data)
            if parsed_metadata:
                print(f"\n元数据解析:")
                print(f"  成功！")
                print(f"  总长度: {parsed_metadata['total_payload_length']}")
                print(f"  分片索引: {parsed_metadata['chunk_index']}")
                print(f"  总分片数: {parsed_metadata['total_chunks']}")
                print(f"  CRC: 0x{parsed_metadata['header_crc']:04X}")
                print(f"  数据部分: {chunk_data[:len(test_data)]}")
            else:
                print(f"\n元数据解析失败！")
                
        except Exception as e:
            print(f"\nRS解码失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  没有提取到数据包！")


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

    # 运行诊断测试
    diagnose_encoding_decoding()
    print("-" * 60)

    # 运行完整的编码/解码功能测试
    test_decode_functionality()
    print("-" * 60)

    # 运行噪声抵抗能力测试
    test_noise_resilience()
    print("=" * 60)
    print("所有自动化测试已完成。")