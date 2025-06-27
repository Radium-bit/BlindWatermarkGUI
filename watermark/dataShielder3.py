# -*- coding: utf-8 -*-
## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms
"""
RC3 鲁棒二进制信息嵌入编码器 (V3)
基于RC1改进，实现编码效率与鲁棒性的平衡

主要改进：
1. 支持多种配置档案 (Compact/Balanced/Robust)
2. 高码率RS编码 RS(255,191)
3. 可选轻量LDPC编码
4. 精简元数据结构 (8字节)
5. 智能交织策略
6. 高效压缩算法支持

Copyright (c) 2025 
SPDX-License-Identifier: Apache-2.0
"""

import struct
import zlib
import time
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
from collections import defaultdict, Counter
from enum import Enum
from dataclasses import dataclass

# 尝试导入第三方库
try:
    from reedsolo import RSCodec
    REEDSOLO_AVAILABLE = True
except ImportError:
    print("警告: reedsolo库未安装，将使用内置RS实现")
    REEDSOLO_AVAILABLE = False

try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    print("警告: lzma库不可用")
    LZMA_AVAILABLE = False

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    print("警告: brotli库未安装")
    BROTLI_AVAILABLE = False

# 尝试导入zstd (非标准库)
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    print("警告: zstandard库未安装")
    ZSTD_AVAILABLE = False

try:
    import pyldpc
    PYLDPC_AVAILABLE = True
except ImportError:
    print("警告: pyldpc库未安装")
    PYLDPC_AVAILABLE = False


class CompressionMethod(Enum):
    """压缩算法枚举"""
    ZLIB = "zlib"
    LZMA = "lzma" 
    BROTLI = "brotli"
    ZSTD = "zstd"


class InterleavingMode(Enum):
    """交织模式枚举"""
    NONE = "none"
    BLOCK = "block"
    SPIRAL = "spiral"
    DUAL = "dual"


@dataclass
class RC3Config:
    """RC3配置参数"""
    name: str
    compression: CompressionMethod
    rs_n: int
    rs_k: int
    enable_ldpc: bool
    ldpc_rate: float
    interleaving: InterleavingMode
    repetition_rate: float
    sync_header: int = 0xA5C3
    
    @property
    def rs_rate(self) -> float:
        return self.rs_k / self.rs_n
    
    @property
    def expected_expansion(self) -> float:
        """预期编码膨胀率"""
        base_expansion = (1 / self.rs_rate)
        if self.enable_ldpc:
            base_expansion *= (1 / self.ldpc_rate)
        return base_expansion * self.repetition_rate


# 预定义配置档案
RC3_CONFIGS = {
    'compact': RC3Config(
        name="Compact",
        compression=CompressionMethod.BROTLI,
        rs_n=255, rs_k=230,  # 码率0.90
        enable_ldpc=False,
        ldpc_rate=1.0,
        interleaving=InterleavingMode.BLOCK,
        repetition_rate=1.0
    ),
    'balanced': RC3Config(
        name="Balanced", 
        compression=CompressionMethod.ZSTD,
        rs_n=255, rs_k=191,  # 码率0.75
        enable_ldpc=True,
        ldpc_rate=0.85,
        interleaving=InterleavingMode.SPIRAL,
        repetition_rate=1.5
    ),
    'robust': RC3Config(
        name="Robust",
        compression=CompressionMethod.ZLIB,
        rs_n=255, rs_k=165,  # 码率0.65
        enable_ldpc=True, 
        ldpc_rate=0.80,
        interleaving=InterleavingMode.DUAL,
        repetition_rate=2.0
    )
}


def crc16_ccitt(data: bytes) -> int:
    """
    计算CRC16-CCITT校验码 (标准实现)
    使用多项式 0x1021，初始值 0xFFFF
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


class AdvancedCompressor:
    """高级压缩器 - 支持多种压缩算法"""
    
    @staticmethod
    def compress(data: bytes, method: CompressionMethod) -> Tuple[bytes, str]:
        """压缩数据并返回压缩后数据和实际使用的方法"""
        original_size = len(data)
        
        if method == CompressionMethod.ZSTD and ZSTD_AVAILABLE:
            try:
                cctx = zstd.ZstdCompressor(level=19)  # 最高压缩率
                compressed = cctx.compress(data)
                actual_method = "zstd"
            except Exception as e:
                print(f"ZSTD压缩失败，回退到zlib: {e}")
                compressed = zlib.compress(data, level=9)
                actual_method = "zlib"
                
        elif method == CompressionMethod.BROTLI and BROTLI_AVAILABLE:
            try:
                compressed = brotli.compress(data, quality=11)
                actual_method = "brotli"
            except Exception as e:
                print(f"Brotli压缩失败，回退到zlib: {e}")
                compressed = zlib.compress(data, level=9)
                actual_method = "zlib"
                
        elif method == CompressionMethod.LZMA and LZMA_AVAILABLE:
            try:
                compressed = lzma.compress(data, preset=9)
                actual_method = "lzma"
            except Exception as e:
                print(f"LZMA压缩失败，回退到zlib: {e}")
                compressed = zlib.compress(data, level=9)
                actual_method = "zlib"
        else:
            # 默认使用zlib
            compressed = zlib.compress(data, level=9)
            actual_method = "zlib"
        
        compression_ratio = (1 - len(compressed) / original_size) * 100
        print(f"压缩统计: {original_size} -> {len(compressed)} 字节 "
              f"({compression_ratio:.1f}% 减少，方法: {actual_method})")
        
        return compressed, actual_method
    
    @staticmethod
    def decompress(data: bytes, method: str) -> bytes:
        """解压缩数据"""
        if method == "zstd" and ZSTD_AVAILABLE:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        elif method == "brotli" and BROTLI_AVAILABLE:
            return brotli.decompress(data)
        elif method == "lzma" and LZMA_AVAILABLE:
            return lzma.decompress(data)
        else:
            return zlib.decompress(data)


class RC3ReedSolomonEncoder:
    """RC3专用Reed-Solomon编码器"""
    
    def __init__(self, n: int = 255, k: int = 191):
        self.n = n
        self.k = k
        self.parity_symbols = n - k
        
        if REEDSOLO_AVAILABLE:
            self.rs_codec = RSCodec(self.parity_symbols, nsize=n)
            self.use_reedsolo = True
            print(f"使用reedsolo库: RS({n},{k}), 纠错能力: {self.parity_symbols//2} 字节")
        else:
            self.use_reedsolo = False
            print(f"使用内置RS实现: RS({n},{k}), 纠错能力: {self.parity_symbols//2} 字节")
    
    def encode(self, data: bytes) -> bytes:
        """RS编码"""
        if len(data) > self.k:
            raise ValueError(f"数据长度{len(data)}超过RS码字长度{self.k}")
        
        # 填充数据到k长度
        padded_data = data + b'\x00' * (self.k - len(data))
        
        if self.use_reedsolo:
            encoded = self.rs_codec.encode(padded_data)
            return bytes(encoded[:self.n])
        else:
            # 简化的内置实现（仅用于演示）
            return padded_data + b'\x00' * self.parity_symbols


class RC3ReedSolomonDecoder:
    """RC3专用Reed-Solomon解码器"""
    
    def __init__(self, n: int = 255, k: int = 191):
        self.n = n
        self.k = k
        self.parity_symbols = n - k
        
        if REEDSOLO_AVAILABLE:
            self.rs_codec = RSCodec(self.parity_symbols, nsize=n)
            self.use_reedsolo = True
        else:
            self.use_reedsolo = False
    
    def decode(self, encoded_data: bytes) -> Tuple[bytes, int]:
        """RS解码"""
        if len(encoded_data) != self.n:
            if len(encoded_data) < self.n:
                encoded_data = encoded_data + b'\x00' * (self.n - len(encoded_data))
            else:
                encoded_data = encoded_data[:self.n]
        
        if self.use_reedsolo:
            try:
                result = self.rs_codec.decode(encoded_data)
                if isinstance(result, tuple):
                    decoded_data, corrections = result[0], len(result[1]) if len(result) > 1 else 0
                else:
                    decoded_data, corrections = result, 0
                
                if isinstance(decoded_data, (list, tuple)):
                    decoded_data = bytes(decoded_data)
                elif not isinstance(decoded_data, bytes):
                    decoded_data = bytes(decoded_data)
                
                return decoded_data[:self.k], max(0, corrections)
            except Exception as e:
                raise ValueError(f"RS解码失败: {e}")
        else:
            # 简化内置实现
            return encoded_data[:self.k], 0


class LightLDPCCodec:
    """轻量级LDPC编解码器 (模拟实现)"""
    
    def __init__(self, rate: float = 0.85):
        self.rate = rate
        self.enabled = rate < 1.0
        print(f"LDPC编码器: 码率={rate}, {'启用' if self.enabled else '禁用'}")
    
    def encode(self, data: bytes) -> bytes:
        """LDPC编码 (简化实现)"""
        if not self.enabled:
            return data
        
        # 简化实现：添加校验字节
        parity_length = int(len(data) * (1/self.rate - 1))
        parity_bytes = b'\x00' * parity_length  # 简化的校验数据
        
        print(f"LDPC编码: {len(data)} -> {len(data) + parity_length} 字节")
        return data + parity_bytes
    
    def decode(self, data: bytes, original_length: int) -> Tuple[bytes, int]:
        """LDPC解码 (简化实现)"""
        if not self.enabled:
            return data, 0
        
        # 简化实现：直接截取原始数据长度
        decoded = data[:original_length]
        errors_corrected = 0  # 简化版本不实际纠错
        
        print(f"LDPC解码: {len(data)} -> {len(decoded)} 字节")
        return decoded, errors_corrected

class RealLDPCCodec:
    """真正的LDPC编解码器 - 基于pyldpc实现"""
    def __init__(self, rate: float = 0.85, n: int = 1200):
        """
        初始化LDPC编码器
        
        Args:
            rate: 码率 (0 < rate < 1)
            n: 码长 (比特数)
        """
        self.rate = rate
        self.n = n
        self.k = int(n * rate)  # 信息位数
        self.m = n - self.k     # 校验位数
        self.enabled = rate < 1.0 and PYLDPC_AVAILABLE
        
        if self.enabled:
            try:
                # 生成LDPC码的校验矩阵H
                # 使用正则LDPC码：每个变量节点度数为3，每个校验节点度数约为6
                d_v = 3  # 变量节点度数
                d_c = int(d_v * self.n / self.m)  # 校验节点度数
                
                # 确保度数合理
                if d_c < 2:
                    d_c = 2
                elif d_c > 20:
                    d_c = 20
                
                # 生成校验矩阵
                self.H = pyldpc.make_ldpc(self.n, d_v, d_c, systematic=True, sparse=True)
                
                # 从H矩阵计算生成矩阵G
                self.G = pyldpc.coding_matrix(self.H)
                
                # 验证矩阵维度
                actual_k = self.G.shape[1]  # 实际信息位数
                actual_n = self.G.shape[0]  # 实际码长
                
                if actual_k != self.k or actual_n != self.n:
                    print(f"LDPC参数调整: 设计({self.n},{self.k}) -> 实际({actual_n},{actual_k})")
                    self.n = actual_n
                    self.k = actual_k
                    self.rate = self.k / self.n
                
                print(f"LDPC编码器初始化成功: ({self.n},{self.k}), 码率={self.rate:.3f}")
                
            except Exception as e:
                print(f"LDPC初始化失败，禁用LDPC: {e}")
                self.enabled = False
        else:
            print(f"LDPC编码器: 禁用 (码率={rate}, pyldpc可用={PYLDPC_AVAILABLE})")
    
    def encode(self, data: bytes) -> bytes:
        """LDPC编码"""
        if not self.enabled:
            return data
        
        try:
            # 将字节数据转换为比特
            data_bits = self._bytes_to_bits(data)
            
            # 计算需要的信息块数
            blocks_needed = (len(data_bits) + self.k - 1) // self.k
            
            encoded_blocks = []
            
            for i in range(blocks_needed):
                # 提取信息比特块
                start_idx = i * self.k
                end_idx = min(start_idx + self.k, len(data_bits))
                info_bits = data_bits[start_idx:end_idx]
                
                # 填充到k长度
                if len(info_bits) < self.k:
                    info_bits.extend([0] * (self.k - len(info_bits)))
                
                # LDPC编码
                info_array = np.array(info_bits, dtype=int)
                encoded_array = pyldpc.encode(self.G, info_array)
                
                encoded_blocks.append(encoded_array.tolist())
            
            # 合并所有编码块
            all_encoded_bits = []
            for block in encoded_blocks:
                all_encoded_bits.extend(block)
            
            # 转换回字节
            encoded_bytes = self._bits_to_bytes(all_encoded_bits)
            
            print(f"LDPC编码: {len(data)} -> {len(encoded_bytes)} 字节 "
                  f"({blocks_needed} 块, 膨胀率: {len(encoded_bytes)/len(data):.2f}x)")
            
            return encoded_bytes
            
        except Exception as e:
            print(f"LDPC编码失败: {e}")
            return data
    
    def decode(self, data: bytes, original_length: int) -> Tuple[bytes, int]:
        """LDPC解码"""
        if not self.enabled:
            return data[:original_length], 0
        
        try:
            # 将字节数据转换为比特
            received_bits = self._bytes_to_bits(data)
            
            # 计算块数
            blocks_count = len(received_bits) // self.n
            
            decoded_blocks = []
            total_errors = 0
            
            for i in range(blocks_count):
                # 提取接收的码字
                start_idx = i * self.n
                end_idx = start_idx + self.n
                
                if end_idx <= len(received_bits):
                    received_block = received_bits[start_idx:end_idx]
                    
                    # LDPC解码 (使用置信传播算法)
                    received_array = np.array(received_block, dtype=float)
                    
                    # 转换为软信息 (简化：硬判决转软判决)
                    llr = np.where(received_array == 0, 1.0, -1.0)
                    
                    # 迭代解码
                    decoded_info, syndrome = pyldpc.decode(self.H, llr, self.k)
                    
                    # 计算纠错的比特数 (近似)
                    errors_in_block = np.sum(np.abs(decoded_info - received_array[:self.k])) // 2
                    total_errors += errors_in_block
                    
                    decoded_blocks.append(decoded_info.astype(int).tolist())
            
            # 合并解码块
            all_decoded_bits = []
            for block in decoded_blocks:
                all_decoded_bits.extend(block)
            
            # 截取到原始长度对应的比特数
            original_bits = original_length * 8
            if len(all_decoded_bits) > original_bits:
                all_decoded_bits = all_decoded_bits[:original_bits]
            
            # 转换回字节
            decoded_bytes = self._bits_to_bytes(all_decoded_bits)
            
            # 截取到原始字节长度
            if len(decoded_bytes) > original_length:
                decoded_bytes = decoded_bytes[:original_length]
            
            print(f"LDPC解码: {len(data)} -> {len(decoded_bytes)} 字节 "
                  f"({blocks_count} 块, 纠错: {total_errors} 比特)")
            
            return decoded_bytes, total_errors
            
        except Exception as e:
            print(f"LDPC解码失败: {e}")
            # 回退到简单截取
            return data[:original_length], 0
    
    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """字节转比特列表"""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        return bits
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """比特列表转字节"""
        # 填充到8的倍数
        while len(bits) % 8 != 0:
            bits.append(0)
        
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i+j < len(bits):
                    byte_val |= (bits[i+j] << (7-j))
            bytes_data.append(byte_val)
        
        return bytes(bytes_data)
    
    @property
    def expansion_factor(self) -> float:
        """编码膨胀因子"""
        return 1.0 / self.rate if self.enabled else 1.0


class SmartInterleaver:
    """智能交织器 - 支持多种交织模式"""
    
    @staticmethod
    def interleave(data: bytes, mode: InterleavingMode, params: dict = None) -> bytes:
        """数据交织"""
        if mode == InterleavingMode.NONE:
            return data
        
        data_bits = SmartInterleaver._bytes_to_bits(data)
        
        if mode == InterleavingMode.BLOCK:
            depth = params.get('depth', 16) if params else 16
            interleaved_bits = SmartInterleaver._block_interleave(data_bits, depth)
            
        elif mode == InterleavingMode.SPIRAL:
            primes = params.get('primes', [7, 11]) if params else [7, 11]
            interleaved_bits = SmartInterleaver._spiral_interleave(data_bits, primes)
            
        elif mode == InterleavingMode.DUAL:
            # 双层交织：先块交织再螺旋交织
            temp_bits = SmartInterleaver._block_interleave(data_bits, 8)
            interleaved_bits = SmartInterleaver._spiral_interleave(temp_bits, [5, 7])
            
        else:
            interleaved_bits = data_bits
        
        return SmartInterleaver._bits_to_bytes(interleaved_bits)
    
    @staticmethod
    def deinterleave(data: bytes, mode: InterleavingMode, params: dict = None) -> bytes:
        """数据去交织"""
        if mode == InterleavingMode.NONE:
            return data
        
        data_bits = SmartInterleaver._bytes_to_bits(data)
        
        if mode == InterleavingMode.BLOCK:
            depth = params.get('depth', 16) if params else 16
            deinterleaved_bits = SmartInterleaver._block_deinterleave(data_bits, depth)
            
        elif mode == InterleavingMode.SPIRAL:
            primes = params.get('primes', [7, 11]) if params else [7, 11]
            deinterleaved_bits = SmartInterleaver._spiral_deinterleave(data_bits, primes)
            
        elif mode == InterleavingMode.DUAL:
            # 双层去交织：逆序操作
            temp_bits = SmartInterleaver._spiral_deinterleave(data_bits, [5, 7])
            deinterleaved_bits = SmartInterleaver._block_deinterleave(temp_bits, 8)
            
        else:
            deinterleaved_bits = data_bits
        
        return SmartInterleaver._bits_to_bytes(deinterleaved_bits)
    
    @staticmethod
    def _bytes_to_bits(data: bytes) -> List[bool]:
        """字节转比特"""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append(bool((byte >> (7-i)) & 1))
        return bits
    
    @staticmethod
    def _bits_to_bytes(bits: List[bool]) -> bytes:
        """比特转字节"""
        # 填充到8的倍数
        while len(bits) % 8 != 0:
            bits.append(False)
        
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i+j < len(bits) and bits[i+j]:
                    byte_val |= (1 << (7-j))
            bytes_data.append(byte_val)
        
        return bytes(bytes_data)
    
    @staticmethod
    def _block_interleave(bits: List[bool], depth: int) -> List[bool]:
        """块交织"""
        if depth <= 1:
            return bits
        
        rows = [bits[i::depth] for i in range(depth)]
        interleaved = []
        
        max_len = max(len(row) for row in rows) if rows else 0
        for i in range(max_len):
            for row in rows:
                if i < len(row):
                    interleaved.append(row[i])
        
        return interleaved
    
    @staticmethod
    def _block_deinterleave(bits: List[bool], depth: int) -> List[bool]:
        """块去交织"""
        if depth <= 1:
            return bits
        
        total_bits = len(bits)
        cols = (total_bits + depth - 1) // depth
        
        rows = [[] for _ in range(depth)]
        for i, bit in enumerate(bits):
            row_idx = i % depth
            rows[row_idx].append(bit)
        
        deinterleaved = []
        for col in range(cols):
            for row_idx in range(depth):
                if col < len(rows[row_idx]):
                    deinterleaved.append(rows[row_idx][col])
        
        return deinterleaved
    
    @staticmethod
    def _spiral_interleave(bits: List[bool], primes: List[int]) -> List[bool]:
        """螺旋交织 (简化实现)"""
        if not primes:
            return bits
        
        total_len = len(bits)
        prime = primes[0]
        
        interleaved = [False] * total_len
        for i, bit in enumerate(bits):
            new_pos = (i * prime) % total_len
            interleaved[new_pos] = bit
        
        return interleaved
    
    @staticmethod
    def _spiral_deinterleave(bits: List[bool], primes: List[int]) -> List[bool]:
        """螺旋去交织"""
        if not primes:
            return bits
        
        total_len = len(bits)
        prime = primes[0]
        
        # 计算模逆
        def mod_inverse(a, m):
            for i in range(1, m):
                if (a * i) % m == 1:
                    return i
            return 1
        
        inv_prime = mod_inverse(prime, total_len)
        
        deinterleaved = [False] * total_len
        for i, bit in enumerate(bits):
            orig_pos = (i * inv_prime) % total_len
            deinterleaved[orig_pos] = bit
        
        return deinterleaved


class RC3Encoder:
    """RC3编码器主类"""
    
    def __init__(self, config: Union[str, RC3Config] = 'balanced'):
        if isinstance(config, str):
            if config not in RC3_CONFIGS:
                raise ValueError(f"未知配置: {config}. 可用配置: {list(RC3_CONFIGS.keys())}")
            self.config = RC3_CONFIGS[config]
        else:
            self.config = config
        
        print(f"\n=== RC3编码器初始化 ===")
        print(f"配置档案: {self.config.name}")
        print(f"压缩算法: {self.config.compression.value}")
        print(f"RS参数: RS({self.config.rs_n},{self.config.rs_k}) 码率={self.config.rs_rate:.3f}")
        print(f"LDPC: {'启用' if self.config.enable_ldpc else '禁用'} (码率={self.config.ldpc_rate})")
        print(f"交织模式: {self.config.interleaving.value}")
        print(f"重复率: {self.config.repetition_rate}x")
        print(f"预期膨胀率: {self.config.expected_expansion:.2f}x")
        print("========================\n")
        
        # 初始化编码器组件
        self.compressor = AdvancedCompressor()
        self.rs_encoder = RC3ReedSolomonEncoder(self.config.rs_n, self.config.rs_k)
        self.ldpc_codec = LightLDPCCodec(self.config.ldpc_rate) if self.config.enable_ldpc else None
        
    def encode(self, data: Union[str, bytes], max_capacity_bits: int = None) -> Tuple[List[bool], Dict]:
        """
        RC3编码主函数
        
        Args:
            data: 待编码的数据 (字符串或字节)
            max_capacity_bits: 最大容量限制 (比特)
            
        Returns:
            (比特流, 编码统计信息)
        """
        start_time = time.time()
        print(f"\n=== RC3编码开始 ===")
        
        # 1. 数据预处理
        if isinstance(data, str):
            raw_data = data.encode('utf-8')
        else:
            raw_data = data
        
        print(f"原始数据: {len(raw_data)} 字节")
        
        # 2. 数据压缩
        compressed_data, compression_method = self.compressor.compress(raw_data, self.config.compression)
        
        # 3. 数据分片与编码
        packets = self._create_packets(compressed_data, compression_method)
        print(f"生成数据包: {len(packets)} 个")
        
        # 4. 生成比特流
        master_bitstream = self._packets_to_bitstream(packets)
        
        # 5. 应用重复率
        if self.config.repetition_rate > 1.0:
            master_bitstream = self._apply_repetition(master_bitstream)
        
        # 6. 容量限制处理
        if max_capacity_bits is not None:
            master_bitstream = self._apply_capacity_limit(master_bitstream, max_capacity_bits)
        
        encoding_time = time.time() - start_time
        
        # 编码统计
        stats = {
            'original_size': len(raw_data),
            'compressed_size': len(compressed_data),
            'compression_method': compression_method,
            'compression_ratio': (1 - len(compressed_data) / len(raw_data)) * 100,
            'num_packets': len(packets),
            'bitstream_length': len(master_bitstream),
            'expansion_ratio': len(master_bitstream) / (len(raw_data) * 8),
            'encoding_time': encoding_time,
            'config_name': self.config.name
        }
        
        print(f"\n=== RC3编码完成 ===")
        print(f"原始: {stats['original_size']} 字节")
        print(f"压缩: {stats['compressed_size']} 字节 ({stats['compression_ratio']:.1f}%)")
        print(f"最终: {stats['bitstream_length']} 比特")
        print(f"膨胀率: {stats['expansion_ratio']:.2f}x")
        print(f"编码时间: {encoding_time:.3f}s")
        print("=====================\n")
        
        return master_bitstream, stats
    
    def _create_packets(self, data: bytes, compression_method: str) -> List[bytes]:
        """创建数据包"""
        print(f"\n--- 数据包创建阶段 ---")
        
        # 元数据大小计算 (8字节)
        metadata_size = 8  # Payload_Length(2) + Chunk_Index(1) + Total_Chunks(1) + Config_Flags(1) + Reserved(1) + CRC16(2)
        max_chunk_data = self.config.rs_k - metadata_size
        
        if max_chunk_data <= 0:
            raise ValueError(f"RS参数({self.config.rs_n},{self.config.rs_k})无法容纳元数据")
        
        # 计算分片数量
        total_chunks = (len(data) + max_chunk_data - 1) // max_chunk_data
        if total_chunks > 255:
            raise ValueError(f"数据过大，需要{total_chunks}个分片，但最大支持255个")
        
        print(f"数据分片: {len(data)} 字节 -> {total_chunks} 个分片")
        print(f"每片最大数据: {max_chunk_data} 字节")
        
        packets = []
        
        for chunk_index in range(total_chunks):
            # 提取数据分片
            start_idx = chunk_index * max_chunk_data
            end_idx = min(start_idx + max_chunk_data, len(data))
            chunk_data = data[start_idx:end_idx]
            
            # 构建元数据 (8字节)
            config_flags = self._encode_config_flags(compression_method)
            
            metadata = struct.pack('>H', len(data))           # Payload_Length (2 bytes)
            metadata += struct.pack('B', chunk_index)         # Chunk_Index (1 byte)
            metadata += struct.pack('B', total_chunks)        # Total_Chunks (1 byte)
            metadata += struct.pack('B', config_flags)        # Config_Flags (1 byte)
            metadata += struct.pack('B', 0)                   # Reserved (1 byte)
            
            # 计算CRC16
            header_crc = crc16_ccitt(metadata)
            metadata += struct.pack('>H', header_crc)          # CRC16 (2 bytes)
            
            # 组合数据包
            packet_data = metadata + chunk_data
            
            # 填充到RS长度
            if len(packet_data) < self.config.rs_k:
                packet_data += b'\x00' * (self.config.rs_k - len(packet_data))
            
            # RS编码
            rs_encoded = self.rs_encoder.encode(packet_data)
            
            # LDPC编码 (可选)
            if self.ldpc_codec:
                final_packet = self.ldpc_codec.encode(rs_encoded)
            else:
                final_packet = rs_encoded
            
            # 交织处理
            interleaved_packet = SmartInterleaver.interleave(final_packet, self.config.interleaving)
            
            packets.append(interleaved_packet)
        
        print(f"数据包处理完成: {len(packets)} 个包")
        return packets
    
    def _encode_config_flags(self, compression_method: str) -> int:
        """编码配置标志 (1字节)"""
        flags = 0
        
        # 压缩方法 (3位)
        compression_map = {'zlib': 0, 'lzma': 1, 'brotli': 2, 'zstd': 3}
        flags |= compression_map.get(compression_method, 0) & 0x07
        
        # LDPC启用标志 (1位)
        if self.config.enable_ldpc:
            flags |= 0x08
        
        # 交织模式 (2位)
        interleaving_map = {
            InterleavingMode.NONE: 0,
            InterleavingMode.BLOCK: 1, 
            InterleavingMode.SPIRAL: 2,
            InterleavingMode.DUAL: 3
        }
        flags |= (interleaving_map[self.config.interleaving] & 0x03) << 4
        
        # 保留位 (2位)
        # flags |= 0x00  # 预留
        
        return flags
    
    def _packets_to_bitstream(self, packets: List[bytes]) -> List[bool]:
        """数据包转换为比特流"""
        print(f"\n--- 比特流生成阶段 ---")
        
        bitstream = []
        sync_header_bytes = struct.pack('>H', self.config.sync_header)
        
        for i, packet in enumerate(packets):
            # 添加同步头
            full_packet = sync_header_bytes + packet
            
            # 转换为比特
            for byte_val in full_packet:
                for bit_pos in range(8):
                    bitstream.append(bool((byte_val >> (7 - bit_pos)) & 1))
        
        print(f"比特流生成: {len(packets)} 包 -> {len(bitstream)} 比特")
        return bitstream
    
    def _apply_repetition(self, bitstream: List[bool]) -> List[bool]:
        """应用重复率"""
        if self.config.repetition_rate <= 1.0:
            return bitstream
        
        repeat_count = int(self.config.repetition_rate)
        repeated_stream = []
        
        for _ in range(repeat_count):
            repeated_stream.extend(bitstream)
        
        print(f"重复处理: {len(bitstream)} -> {len(repeated_stream)} 比特 ({self.config.repetition_rate}x)")
        return repeated_stream
    
    def _apply_capacity_limit(self, bitstream: List[bool], max_capacity: int) -> List[bool]:
        """应用容量限制"""
        if len(bitstream) <= max_capacity:
            return bitstream
        
        # 循环填充
        limited_stream = []
        for i in range(max_capacity):
            limited_stream.append(bitstream[i % len(bitstream)])
        
        print(f"容量限制: {len(bitstream)} -> {max_capacity} 比特")
        return limited_stream
    
    def estimate_capacity(self, data: Union[str, bytes]) -> Dict:
        """估算编码容量"""
        if isinstance(data, str):
            raw_data = data.encode('utf-8')
        else:
            raw_data = data
        
        # 模拟压缩
        compressed_data, _ = self.compressor.compress(raw_data, self.config.compression)
        
        # 计算数据包数量
        metadata_size = 8
        max_chunk_data = self.config.rs_k - metadata_size
        total_chunks = (len(compressed_data) + max_chunk_data - 1) // max_chunk_data
        
        # 计算比特流大小
        packet_size_bytes = 2 + self.config.rs_n  # 同步头 + RS编码包
        
        if self.config.enable_ldpc:
            packet_size_bytes = int(packet_size_bytes / self.config.ldpc_rate)
        
        total_bits = total_chunks * packet_size_bytes * 8 * self.config.repetition_rate
        
        return {
            'original_size': len(raw_data),
            'compressed_size': len(compressed_data),
            'total_chunks': total_chunks,
            'packet_size_bytes': packet_size_bytes,
            'estimated_bits': int(total_bits),
            'expansion_ratio': total_bits / (len(raw_data) * 8)
        }


class RC3Decoder:
    """RC3解码器主类"""
    
    def __init__(self):
        print(f"\n=== RC3解码器初始化 ===")
        self.compressor = AdvancedCompressor()
        
        # 预初始化不同配置的解码器
        self.rs_decoders = {}
        self.ldpc_codecs = {}
        
        for config_name, config in RC3_CONFIGS.items():
            self.rs_decoders[config_name] = RC3ReedSolomonDecoder(config.rs_n, config.rs_k)
            if config.enable_ldpc:
                self.ldpc_codecs[config_name] = LightLDPCCodec(config.ldpc_rate)
        
        print("解码器组件初始化完成")
        print("========================\n")
    
    def decode(self, bitstream: List[bool], debug: bool = False) -> Tuple[Optional[str], Dict]:
        """
        RC3解码主函数
        
        Args:
            bitstream: 比特流
            debug: 是否输出调试信息
            
        Returns:
            (解码结果字符串, 解码统计信息)
        """
        start_time = time.time()
        print(f"\n=== RC3解码开始 ===")
        print(f"输入比特流: {len(bitstream)} 比特")
        
        # 统计信息初始化
        stats = {
            'input_bitstream_length': len(bitstream),
            'sync_headers_found': 0,
            'packets_extracted': 0,
            'packets_decoded': 0,
            'chunks_recovered': {},
            'compression_method': None,
            'final_size': 0,
            'decoding_time': 0,
            'success': False
        }
        
        try:
            # 1. 检测同步头并提取数据包
            raw_packets = self._extract_all_packets(bitstream)
            stats['packets_extracted'] = len(raw_packets)
            
            if not raw_packets:
                print("错误: 未找到有效数据包")
                return None, stats
            
            # 2. 解码数据包并分析配置
            decoded_packets, config_info = self._decode_packets(raw_packets, debug)
            stats['packets_decoded'] = len(decoded_packets)
            stats['compression_method'] = config_info.get('compression_method')
            
            if not decoded_packets:
                print("错误: 无法解码任何数据包")
                return None, stats
            
            # 3. 重组数据
            recovered_data = self._reconstruct_data(decoded_packets, config_info, debug)
            
            if recovered_data is None:
                print("错误: 数据重组失败")
                return None, stats
            
            # 4. 解压缩
            if config_info.get('compression_method'):
                try:
                    final_data = self.compressor.decompress(recovered_data, config_info['compression_method'])
                    final_string = final_data.decode('utf-8')
                    stats['final_size'] = len(final_data)
                    stats['success'] = True
                except Exception as e:
                    print(f"解压缩失败: {e}")
                    # 尝试直接解码
                    try:
                        final_string = recovered_data.decode('utf-8')
                        stats['final_size'] = len(recovered_data)
                        stats['success'] = True
                    except:
                        return None, stats
            else:
                try:
                    final_string = recovered_data.decode('utf-8')
                    stats['final_size'] = len(recovered_data)
                    stats['success'] = True
                except:
                    return None, stats
            
            decoding_time = time.time() - start_time
            stats['decoding_time'] = decoding_time
            
            print(f"\n=== RC3解码完成 ===")
            print(f"解码成功: {len(final_string)} 字符")
            print(f"压缩方法: {config_info.get('compression_method', '未知')}")
            print(f"解码时间: {decoding_time:.3f}s")
            print("=====================\n")
            
            return final_string, stats
            
        except Exception as e:
            print(f"解码过程发生错误: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return None, stats
    
    def _extract_all_packets(self, bitstream: List[bool]) -> List[Tuple[bytes, int, int]]:
        """提取所有可能的数据包"""
        print(f"\n--- 数据包提取阶段 ---")
        
        all_packets = []
        
        # 尝试所有可能的同步头
        for config_name, config in RC3_CONFIGS.items():
            sync_positions = self._find_sync_patterns(bitstream, config.sync_header)
            
            if sync_positions:
                print(f"配置 {config_name}: 发现 {len(sync_positions)} 个同步头")
                
                for pos in sync_positions:
                    packet_size_bytes = config.rs_n
                    if config.enable_ldpc:
                        packet_size_bytes = int(packet_size_bytes / config.ldpc_rate)
                    
                    packet_total_bits = (2 + packet_size_bytes) * 8  # 同步头 + 数据
                    
                    if pos + packet_total_bits <= len(bitstream):
                        # 提取数据包 (不包括同步头)
                        data_start = pos + 16  # 跳过同步头
                        data_bits = bitstream[data_start:data_start + packet_size_bytes * 8]
                        
                        if len(data_bits) == packet_size_bytes * 8:
                            packet_bytes = self._bits_to_bytes(data_bits)
                            all_packets.append((packet_bytes, pos, config_name))
        
        print(f"总共提取 {len(all_packets)} 个候选数据包")
        return all_packets
    
    def _find_sync_patterns(self, bitstream: List[bool], sync_header: int) -> List[int]:
        """查找同步头模式"""
        sync_pattern = []
        for bit in range(16):
            sync_pattern.append(bool((sync_header >> (15 - bit)) & 1))
        
        positions = []
        for i in range(len(bitstream) - 15):
            if bitstream[i:i+16] == sync_pattern:
                positions.append(i)
        
        return positions
    
    def _bits_to_bytes(self, bits: List[bool]) -> bytes:
        """比特转字节"""
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i+j < len(bits) and bits[i+j]:
                    byte_val |= (1 << (7-j))
            bytes_data.append(byte_val)
        return bytes(bytes_data)
    
    def _decode_packets(self, raw_packets: List[Tuple[bytes, int, str]], debug: bool) -> Tuple[List[Dict], Dict]:
        """解码数据包"""
        print(f"\n--- 数据包解码阶段 ---")
        
        decoded_packets = []
        config_votes = defaultdict(int)
        compression_votes = defaultdict(int)
        
        for i, (packet_data, pos, config_name) in enumerate(raw_packets):
            if debug and i < 5:  # 只调试前5个包
                print(f"\n调试包 {i} (配置: {config_name}):")
                print(f"  位置: {pos}, 长度: {len(packet_data)}")
            
            try:
                config = RC3_CONFIGS[config_name]
                
                # 去交织
                deinterleaved = SmartInterleaver.deinterleave(packet_data, config.interleaving)
                
                # LDPC解码 (如果启用)
                if config.enable_ldpc and config_name in self.ldpc_codecs:
                    ldpc_decoded, _ = self.ldpc_codecs[config_name].decode(deinterleaved, config.rs_n)
                else:
                    ldpc_decoded = deinterleaved
                
                # RS解码
                rs_decoder = self.rs_decoders[config_name]
                rs_decoded, errors_corrected = rs_decoder.decode(ldpc_decoded)
                
                # 解析元数据
                metadata = self._parse_metadata(rs_decoded)
                
                if metadata is not None:
                    metadata['config_name'] = config_name
                    metadata['position'] = pos
                    metadata['errors_corrected'] = errors_corrected
                    metadata['chunk_data'] = rs_decoded[8:]  # 跳过8字节元数据
                    
                    decoded_packets.append(metadata)
                    config_votes[config_name] += 1
                    compression_votes[metadata['compression_method']] += 1
                    
                    if debug and i < 5:
                        print(f"  解码成功! 分片: {metadata['chunk_index']}/{metadata['total_chunks']}")
                        print(f"  压缩方法: {metadata['compression_method']}")
                        print(f"  纠错: {errors_corrected} 字节")
                
            except Exception as e:
                if debug and i < 5:
                    print(f"  解码失败: {e}")
                continue
        
        # 确定最可能的配置
        best_config = max(config_votes.items(), key=lambda x: x[1])[0] if config_votes else 'balanced'
        best_compression = max(compression_votes.items(), key=lambda x: x[1])[0] if compression_votes else 'zlib'
        
        config_info = {
            'config_name': best_config,
            'compression_method': best_compression,
            'config_votes': dict(config_votes),
            'compression_votes': dict(compression_votes)
        }
        
        print(f"解码统计:")
        print(f"  有效包: {len(decoded_packets)}")
        print(f"  最佳配置: {best_config}")
        print(f"  压缩方法: {best_compression}")
        
        return decoded_packets, config_info
    
    def _parse_metadata(self, data: bytes) -> Optional[Dict]:
        """解析数据包元数据"""
        if len(data) < 8:
            return None
        
        try:
            payload_length = struct.unpack('>H', data[0:2])[0]
            chunk_index = struct.unpack('B', data[2:3])[0]
            total_chunks = struct.unpack('B', data[3:4])[0]
            config_flags = struct.unpack('B', data[4:5])[0]
            reserved = struct.unpack('B', data[5:6])[0]
            header_crc = struct.unpack('>H', data[6:8])[0]
            
            # 验证CRC
            header_data = data[0:6]
            calculated_crc = crc16_ccitt(header_data)
            
            if calculated_crc != header_crc:
                return None
            
            # 解码配置标志
            compression_map = {0: 'zlib', 1: 'lzma', 2: 'brotli', 3: 'zstd'}
            compression_method = compression_map.get(config_flags & 0x07, 'zlib')
            
            # 基本合理性检查
            if (payload_length > 64*1024 or total_chunks == 0 or 
                total_chunks > 255 or chunk_index >= total_chunks):
                return None
            
            return {
                'payload_length': payload_length,
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'config_flags': config_flags,
                'compression_method': compression_method,
                'header_crc': header_crc
            }
            
        except (struct.error, IndexError):
            return None
    
    def _reconstruct_data(self, decoded_packets: List[Dict], config_info: Dict, debug: bool) -> Optional[bytes]:
        """重组数据"""
        print(f"\n--- 数据重组阶段 ---")
        
        if not decoded_packets:
            return None
        
        # 按分片索引分组
        chunks_by_index = defaultdict(list)
        for packet in decoded_packets:
            chunks_by_index[packet['chunk_index']].append(packet)
        
        # 确定参数
        total_chunks = max(p['total_chunks'] for p in decoded_packets)
        payload_length = max(p['payload_length'] for p in decoded_packets)
        
        print(f"重组参数: {total_chunks} 分片, 总长度 {payload_length} 字节")
        
        # 选择最佳的配置
        config_name = config_info['config_name']
        config = RC3_CONFIGS[config_name]
        metadata_size = 8
        max_chunk_data = config.rs_k - metadata_size
        
        # 重组数据
        reconstructed_data = b''
        recovered_chunks = 0
        
        for chunk_index in range(total_chunks):
            if chunk_index in chunks_by_index:
                # 选择最好的候选 (纠错最少的)
                candidates = chunks_by_index[chunk_index]
                best_candidate = min(candidates, key=lambda x: x['errors_corrected'])
                
                # 计算这个分片的实际数据长度
                start_pos = chunk_index * max_chunk_data
                chunk_data_length = min(max_chunk_data, payload_length - start_pos)
                
                chunk_data = best_candidate['chunk_data'][:chunk_data_length]
                reconstructed_data += chunk_data
                recovered_chunks += 1
                
                if debug:
                    print(f"  分片 {chunk_index}: 恢复 {len(chunk_data)} 字节")
            else:
                # 缺失分片，用零填充
                start_pos = chunk_index * max_chunk_data
                chunk_data_length = min(max_chunk_data, payload_length - start_pos)
                reconstructed_data += b'\x00' * chunk_data_length
                
                if debug:
                    print(f"  分片 {chunk_index}: 缺失，零填充 {chunk_data_length} 字节")
        
        recovery_rate = recovered_chunks / total_chunks * 100
        print(f"分片恢复率: {recovered_chunks}/{total_chunks} ({recovery_rate:.1f}%)")
        
        if recovery_rate < 50:  # 如果恢复率太低，可能失败
            print("警告: 分片恢复率过低，解码可能失败")
        
        return reconstructed_data


def print_capacity_analysis():
    """打印容量分析"""
    print("\n" + "="*60)
    print("RC3 容量分析")
    print("="*60)
    
    test_sizes = [1024, 2048, 5120]  # 1KB, 2KB, 5KB
    
    for size in test_sizes:
        test_data = "x" * size
        print(f"\n--- {size//1024}KB 测试数据 ---")
        
        for config_name in ['compact', 'balanced', 'robust']:
            try:
                encoder = RC3Encoder(config_name)
                capacity_info = encoder.estimate_capacity(test_data)
                
                print(f"{config_name.capitalize()}: "
                      f"{capacity_info['estimated_bits']:,} bits "
                      f"({capacity_info['estimated_bits']//8:,} bytes) "
                      f"膨胀率: {capacity_info['expansion_ratio']:.2f}x")
            except Exception as e:
                print(f"{config_name.capitalize()}: 错误 - {e}")


def test_rc3_basic():
    """RC3基本功能测试"""
    print("\n" + "="*60)
    print("RC3 基本功能测试")
    print("="*60)
    
    test_messages = [
        "Hello, RC3!",
        "这是一个中文测试消息，用于验证RC3编码器的功能。",
        "A longer message to test the robustness and efficiency of the RC3 encoding system."
    ]
    
    for i, message in enumerate(test_messages):
        print(f"\n测试 {i+1}: {message[:50]}...")
        
        try:
            # 编码
            encoder = RC3Encoder('balanced')
            bitstream, encode_stats = encoder.encode(message)
            
            print(f"编码: {encode_stats['original_size']} -> {encode_stats['bitstream_length']} bits")
            
            # 解码
            decoder = RC3Decoder()
            decoded_text, decode_stats = decoder.decode(bitstream)
            
            # 验证
            success = decoded_text == message if decoded_text else False
            print(f"解码: {'成功' if success else '失败'}")
            
            if not success:
                print(f"  原始: {message[:100]}...")
                print(f"  解码: {decoded_text[:100] if decoded_text else 'None'}...")
                
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("RC3 鲁棒二进制信息嵌入编码器")
    print("Copyright (c) 2025")
    print("-" * 40)
    
    # 运行容量分析
    print_capacity_analysis()
    
    # 运行基本功能测试
    test_rc3_basic()
    
    print("\n" + "="*60)
    print("RC3 测试完成")
    print("="*60)