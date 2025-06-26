## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

import os
import tempfile
import threading
import re
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from blind_watermark import WaterMark
from tkinter import messagebox


class WatermarkExtractor:
    def __init__(self, app):
        self.app = app

    def extract_watermark(self, filepath):
        def worker():
            try:
                self.app.root.after(0, lambda: self.app.show_processing_window("正在提取水印，请稍候..."))
                
                # 创建临时文件
                tmp_in = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                tmp_out = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                
                # 保存图片到临时文件
                img = Image.open(filepath)
                # 如果目标尺寸不为None，则调整图片大小，常用于提取尺寸已被更改的图像
                target_size = self.app.get_target_size()
                if target_size is not None:
                    if img.size != target_size:
                        print("Enter Resize Brench")
                        img = img.resize(target_size, Image.LANCZOS)
                # 保存图像临时文件
                img.save(tmp_in)
                
                # 尝试128x128尺寸提取
                pwd = self.app.get_pwd()
                bwm1 = WaterMark(password_wm=int(pwd), password_img=int(pwd))
                
                # 初始化
                text = None
                img_128 = None
                img_64 = None
                
                try:
                    # 第一次尝试128x128
                    bwm1.extract(filename=tmp_in, wm_shape=(128, 128), out_wm_name=tmp_out)
                    img_128 = Image.open(tmp_out)
                    # 尝试解码二维码
                    qreader = self.app.qreader
                    print(f'128x128临时文件路径: 输入={tmp_in} 输出={tmp_out}')
                    if img_128.mode != 'RGB':
                        img_128 = img_128.convert('RGB')
                    img_array = np.array(img_128)
                    text = qreader.detect_and_decode(image=img_array)[0]
                    
                    # 如果第一次解析失败，尝试增强解析
                    # 先备份原图
                    if self.app.show_orignal_extract_picture.get(): img_128_backup = img_128.copy()
                    if not text:
                        # 1. 尝试调整对比度和亮度
                        enhancer = ImageEnhance.Contrast(img_128)
                        img_128 = enhancer.enhance(2.0)
                        enhancer = ImageEnhance.Brightness(img_128)
                        img_128 = enhancer.enhance(1.5)
                        
                        # 2. 重新尝试解码
                        img_array = np.array(img_128.convert('RGB'))
                        text = qreader.detect_and_decode(image=img_array)[0]
                except Exception as e:
                    print(f"128x128提取失败: {e}")
                    # 清理临时文件
                    for f in [tmp_out]:
                        if os.path.exists(f):
                            os.unlink(f)
                    # import traceback
                    # print("完整错误追踪:")
                    # traceback.print_exc()
                    pass
                
                # 如果128x128失败，尝试64x64
                if not text:
                    print("Entering 64x64 branch")
                    try:
                        qreader = self.app.qreader
                        print(f'64x64临时文件路径: 输入={tmp_in} 输出={tmp_out}')
                        
                        # 在extract调用前添加参数检查
                        if not all([tmp_in, tmp_out]):
                            raise ValueError(f"文件路径参数异常 tmp_in:{tmp_in} tmp_out:{tmp_out}")
                        
                        bwm1.extract(filename=tmp_in, wm_shape=(64, 64), out_wm_name=tmp_out)
                        img_64 = Image.open(tmp_out)
                        
                        if img_64.mode != 'RGB':
                            img_64 = img_64.convert('RGB')
                        img_array = np.array(img_64)
                        text = qreader.detect_and_decode(image=img_array)[0]
                        print(text)
                        print("Has try 64x64")
                        # 如果第一次解析失败，尝试增强解析
                        # 先备份原图
                        if self.app.show_orignal_extract_picture.get(): img_64_backup = img_64.copy()
                        if not text:
                            print("No Text at try1")
                            # 1. 尝试调整对比度和亮度
                            enhancer = ImageEnhance.Contrast(img_64)
                            img_64 = enhancer.enhance(2.0)
                            enhancer = ImageEnhance.Brightness(img_64)
                            img_64 = enhancer.enhance(1.5)
                            
                            # 2. 重新尝试解码
                            img_array = np.array(img_64.convert('RGB'))
                            text = qreader.detect_and_decode(image=img_array)[0]
                    except Exception as e:
                        print(f"64x64提取失败: {e}")
                        # 清理临时文件
                        for f in [tmp_out]:
                            if os.path.exists(f):
                                os.unlink(f)
                        # import traceback
                        # print("完整错误追踪:")
                        # traceback.print_exc()
                        pass
                
                # 如果两次都失败，显示图片和错误信息
                if not text:
                    images = []
                    if img_128:
                        if self.app.show_orignal_extract_picture.get():
                            img_128 = img_128_backup.copy()
                            img_128_backup.close()
                        images.append(("128x128", img_128))
                    if img_64:
                        if self.app.show_orignal_extract_picture.get():
                            img_64 = img_64_backup.copy()
                            img_64_backup.close()
                        images.append(("64x64", img_64))
                    
                    if images:
                        # 将图片对象转换为可序列化的元组格式
                        image_tuples = [(size, np.array(img)) for size, img in images]
                        # 处理多余的tmp_out文件
                        if os.path.exists(tmp_out):
                            os.unlink(tmp_out)
                        self.app.show_qr_code(None, "", None, *image_tuples)
                    else:
                        messagebox.showerror("错误", "水印提取失败")
                    return
                
                # 显示提取的二维码水印
                self.app.root.after(0, lambda: self.app.show_qr_code(tmp_out, text, True))
            except Exception as e:
                self.app.root.after(0, lambda e=e: messagebox.showerror("错误", f"提取水印失败: {str(e)}"))
            finally:
                # 清理临时文件
                for f in [tmp_in]:
                    if os.path.exists(f):
                        os.unlink(f)
                self.app.root.after(0, self.app.hide_processing_window)
                
        threading.Thread(target=worker).start()

    def extract_watermark_v013(self, filepath):
        """旧版本兼容方法 - 从文件名提取ws和size信息"""
        def worker():
            try:
                # 显示处理窗口
                self.app.root.after(0, lambda: self.app.show_processing_window("正在处理图片，请稍候..."))
                
                pwd = self.app.get_pwd()
                name = os.path.basename(filepath)
                ext = os.path.splitext(filepath)[1]

                output_dir = self.app.get_output_dir()
                tmp_in = os.path.join(output_dir, f"input.png")

                # 读取 ws
                wm_len = self.app.get_ws()

                if wm_len is None:
                    m = re.search(r"ws(\d+)", name)
                    if not m:
                        raise ValueError("文件名中未找到 ws（如 ws256）\n可手动输入或改名")
                    wm_len = int(m.group(1))

                # 读取原始尺寸
                target_size = self.app.get_target_size()
                if target_size is None:
                    m = re.search(r"size(\d+)x(\d+)", name)
                    if not m:
                        raise ValueError("文件名中未找到 size（如 size800x600）\n可手动输入或改名")
                    target_size = int(m.group(1)), int(m.group(2))

                # 判断是否需要 resize
                img = Image.open(filepath)
                if img.size == target_size:
                    img.save(tmp_in)
                else:
                    resized = img.resize(target_size, Image.LANCZOS) #使用LANCZOS算法重新匹配图像
                    resized.save(tmp_in, format="PNG")

                bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                wm_extract = bwm1.extract(tmp_in, wm_shape=wm_len, mode='str')
                wm_extract = wm_extract.replace("\\n", "\n")
                # 确保处理窗口关闭
                self.app.root.after(0, self.app.hide_processing_window)
                self.app.root.after(0, lambda: messagebox.showinfo("提取成功", f"水印内容：\n{wm_extract}"))

                if os.path.exists(tmp_in):
                    try:
                        os.remove(tmp_in)
                    except:
                        pass
            except Exception as e:
                # 确保处理窗口关闭
                self.app.root.after(0, lambda: messagebox.showerror("错误", str(e)))
                self.app.root.after(0, self.app.hide_processing_window)
                
        # 启动工作线程
        threading.Thread(target=worker).start()

    def extract_watermark_v3(self, filepath):
        """新版本方法 - 集成鲁棒二进制信息解码器"""
        def worker():
            try:
                # 显示处理窗口
                self.app.root.after(0, lambda: self.app.show_processing_window("正在处理图片，请稍候..."))
                
                pwd = self.app.get_pwd()
                name = os.path.basename(filepath)
                ext = os.path.splitext(filepath)[1]
                output_dir = self.app.get_output_dir()
                os.makedirs(output_dir, exist_ok=True)
                tmp_in = os.path.join(output_dir, f"input.png")
                
                # ===== 从文件名解析编码信息 =====
                # 检测编码类型（RS 或 UTF8）
                encoding_type = "RS" if "RS" in name else ("UTF8" if "UTF8" in name else "unknown")
                
                # 读取 ws（水印长度）
                wm_len = self.app.get_ws()
                if wm_len is None:
                    m = re.search(r"ws(\d+)", name)
                    if not m:
                        raise ValueError("文件名中未找到 ws（如 ws256）\n可手动输入或改名")
                    wm_len = int(m.group(1))
                
                # 读取原始尺寸
                target_size = self.app.get_target_size()
                if target_size is None:
                    m = re.search(r"size(\d+)x(\d+)", name)
                    if not m:
                        raise ValueError("文件名中未找到 size（如 size800x600）\n可手动输入或改名")
                    target_size = int(m.group(1)), int(m.group(2))
                
                print(f"检测到编码类型: {encoding_type}")
                print(f"水印长度: {wm_len} 比特")
                print(f"目标尺寸: {target_size}")
                
                # ===== 图片预处理 =====
                self.app.root.after(0, lambda: self.app.show_processing_window("正在预处理图片..."))
                
                img = Image.open(filepath)
                if img.size == target_size:
                    img.save(tmp_in)
                else:
                    print(f"图片尺寸不匹配，从 {img.size} 调整到 {target_size}")
                    resized = img.resize(target_size, Image.LANCZOS)
                    resized.save(tmp_in, format="PNG")
                
                # ===== 提取水印比特流 =====
                self.app.root.after(0, lambda: self.app.show_processing_window("正在提取水印数据..."))
                
                bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                wm_bits = bwm1.extract(tmp_in, wm_shape=wm_len, mode='bit')
                
                # 处理不同类型的返回值
                if wm_bits is None:
                    raise ValueError("未能从图片中提取到水印数据")
                
                # 如果是numpy数组，转换为Python列表
                if hasattr(wm_bits, 'tolist'):
                    wm_bits = wm_bits.tolist()
                elif hasattr(wm_bits, '__iter__') and not isinstance(wm_bits, (str, bytes)):
                    wm_bits = list(wm_bits)
                
                # 确保是布尔值列表
                if wm_bits and not isinstance(wm_bits[0], bool):
                    wm_bits = [bool(bit) for bit in wm_bits]
                
                if len(wm_bits) == 0:
                    raise ValueError("提取到的水印数据为空")
                
                print(f"提取到 {len(wm_bits)} 比特的水印数据")
                print(f"数据类型: {type(wm_bits)}, 元素类型: {type(wm_bits[0]) if wm_bits else 'N/A'}")
                
                # ===== 根据编码类型解码 =====
                self.app.root.after(0, lambda: self.app.show_processing_window("正在解码水印内容..."))
                
                if encoding_type == "RS":
                    # 使用鲁棒解码器
                    try:
                        from .dataShielder import (
                            decode_watermark_to_string, 
                            decode_watermark_to_binary,
                            analyze_watermark_quality,
                            print_decode_report
                        )
                    
                        print("使用Reed-Solomon解码器...")
                    
                        # 尝试解码为字符串
                        decoded_text, decode_stats = decode_watermark_to_string(wm_bits)
                    
                        if decoded_text is not None:
                            # 解码成功
                            print(f"解码成功: {decoded_text}")
                        
                            # 分析解码质量
                            quality_analysis = analyze_watermark_quality(wm_bits)
                            quality_metrics = quality_analysis['quality_metrics']
                        
                            # 准备详细的解码报告
                            quality_score = quality_metrics['data_integrity_score'] * 100
                            error_correction = decode_stats['total_errors_corrected']
                            valid_packets = decode_stats['valid_packets']
                            total_packets = decode_stats['total_packets_found']
                        
                            success_message = f"""解码成功！

水印内容：
{decoded_text}

解码质量报告：
• 编码方式：Reed-Solomon编码
• 数据完整性：{quality_score:.1f}%
• 有效数据包：{valid_packets}/{total_packets}
• 纠错字节数：{error_correction}
• RS解码成功率：{quality_metrics['rs_success_rate']*100:.1f}%
• 分片恢复率：{quality_metrics['chunk_recovery_rate']*100:.1f}%

{'✓ 数据完整，质量优秀' if quality_score > 90 else '⚠ 数据部分损坏但已恢复' if quality_score > 50 else '❌ 数据严重损坏'}"""
                        
                            # 如果质量较低，显示详细分析
                            if quality_score < 90:
                                print("=== 详细解码分析 ===")
                                print_decode_report(wm_bits)
                        
                            wm_extract = decoded_text
                        
                        else:
                            # 解码失败，尝试降级处理
                            print("RS解码失败，尝试二进制解码...")
                            decoded_binary, decode_stats = decode_watermark_to_binary(wm_bits)
                        
                            if decoded_binary is not None:
                                # 尝试UTF-8解码
                                try:
                                    decoded_text = decoded_binary.decode('utf-8', errors='ignore')
                                    success_message = f"""部分解码成功！

水印内容：
{decoded_text}

警告：
• 使用了降级解码模式
• 数据可能不完整
• 建议检查原始图片质量"""
                                    wm_extract = decoded_text
                                except:
                                    raise ValueError("RS解码失败且无法转换为文本")
                            else:
                                raise ValueError("RS解码完全失败")
                    
                    except ImportError:
                        self.app.root.after(0, lambda: messagebox.showwarning(
                            "解码器未找到", 
                            "鲁棒解码器不可用，将尝试简单UTF-8解码。\n建议安装: pip install reedsolo numpy"
                        ))
                        encoding_type = "UTF8"  # 降级到UTF8模式
                    
                    except Exception as e:
                        print(f"RS解码失败: {e}")
                        # 降级到UTF8解码
                        encoding_type = "UTF8"
                        self.app.root.after(0, lambda e=e: messagebox.showwarning(
                            "RS解码失败", 
                            f"鲁棒解码失败: {str(e)}\n将尝试简单UTF-8解码"
                        ))
                
                if encoding_type == "UTF8" or encoding_type == "unknown":
                    # 使用简单UTF-8解码
                    print("使用简单UTF-8解码器...")
                
                    try:
                        # 确保wm_bits是列表格式
                        if hasattr(wm_bits, 'tolist'):
                            bit_list = wm_bits.tolist()
                        else:
                            bit_list = list(wm_bits)
                        
                        # 将比特流转换为字节
                        if len(bit_list) % 8 != 0:
                            # 填充到8的倍数
                            bit_list.extend([False] * (8 - len(bit_list) % 8))
                        
                        byte_data = []
                        for i in range(0, len(bit_list), 8):
                            byte_bits = bit_list[i:i+8]
                            byte_val = 0
                            for j, bit in enumerate(byte_bits):
                                if bool(bit):  # 确保转换为布尔值
                                    byte_val |= (1 << (7 - j))
                            byte_data.append(byte_val)
                        
                        # 转换为字节串并解码
                        byte_string = bytes(byte_data)
                        
                        # 尝试找到有效的UTF-8结尾
                        decoded_text = None
                        print(f"尝试解码 {len(byte_string)} 字节")
                        for end_pos in range(len(byte_string), max(0, len(byte_string)-1000), -1):  # 限制搜索范围
                            try:
                                test_text = byte_string[:end_pos].decode('utf-8')
                                # 移除空字符和控制字符
                                cleaned_text = ''.join(char for char in test_text if char.isprintable() or char.isspace())
                                if len(cleaned_text.strip()) > 0:
                                    decoded_text = cleaned_text.strip()
                                    break
                            except UnicodeDecodeError:
                                continue
                        
                        if decoded_text:
                            success_message = f"""解码成功！

水印内容：
{decoded_text}

解码信息：
• 编码方式：简单UTF-8编码
• 原始比特长度：{len(bit_list)}
• 解码字节数：{len(decoded_text.encode('utf-8'))}

注意：简单编码抗干扰能力有限"""
                            wm_extract = decoded_text
                        else:
                            raise ValueError("UTF-8解码失败，未找到有效文本")
                
                    except Exception as e:
                        raise ValueError(f"UTF-8解码失败: {str(e)}")
                
                # ===== 处理换行符和格式化 =====
                wm_extract = wm_extract.replace("\\n", "\n")
                
                # 确保处理窗口关闭
                self.app.root.after(0, self.app.hide_processing_window)
                self.app.root.after(0, lambda: messagebox.showinfo("提取成功", success_message))
                
            except Exception as e:
                error_msg = f"提取过程中发生错误: {str(e)}"
                print(error_msg)
                # 确保处理窗口关闭
                self.app.root.after(0, self.app.hide_processing_window)
                self.app.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            
            finally:
                # 清理临时文件
                if os.path.exists(tmp_in):
                    try:
                        os.remove(tmp_in)
                    except:
                        pass
    
        # 启动工作线程
        threading.Thread(target=worker, daemon=True).start()


    # ===== 辅助解码函数 =====

    def detect_encoding_type_from_filename(filename):
        """
        从文件名检测编码类型
        
        参数:
            filename: 文件名字符串
            
        返回:
            编码类型字符串: "RS", "UTF8", 或 "unknown"
        """
        filename_upper = filename.upper()
        
        if "RS" in filename_upper:
            return "RS"
        elif "UTF8" in filename_upper:
            return "UTF8"
        else:
            # 尝试其他模式检测
            if "WATERMARK" in filename_upper:
                # 如果包含watermark但没有明确标识，假设是新版本
                return "RS"
            return "unknown"


    def extract_watermark_parameters(filename):
        """
        从文件名提取水印参数
        
        参数:
            filename: 文件名字符串
            
        返回:
            字典包含: {
                'encoding_type': str,
                'ws_length': int or None,
                'size': tuple or None,
                'rs_params': tuple or None
            }
        """
        import re
        
        params = {
            'encoding_type': detect_encoding_type_from_filename(filename),
            'ws_length': None,
            'size': None,
            'rs_params': None
        }
        
        # 提取ws长度
        ws_match = re.search(r"ws(\d+)", filename)
        if ws_match:
            params['ws_length'] = int(ws_match.group(1))
        
        # 提取尺寸
        size_match = re.search(r"size(\d+)x(\d+)", filename)
        if size_match:
            params['size'] = (int(size_match.group(1)), int(size_match.group(2)))
        
        # 提取RS参数（如果有）
        rs_match = re.search(r"RS(\d+)-(\d+)", filename)
        if rs_match:
            params['rs_params'] = (int(rs_match.group(1)), int(rs_match.group(2)))
        
        return params


    def validate_extracted_text(text, min_length=1, max_length=10000):
        """
        验证提取的文本
        
        参数:
            text: 提取的文本
            min_length: 最小长度
            max_length: 最大长度
            
        返回:
            (is_valid, cleaned_text, message)
        """
        if not text:
            return False, "", "提取的文本为空"
        
        # 清理文本
        cleaned = text.strip()
        
        # 移除过多的空白字符
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 移除不可打印字符（保留换行符和制表符）
        cleaned = ''.join(char for char in cleaned 
                        if char.isprintable() or char in '\n\t')
        
        if len(cleaned) < min_length:
            return False, cleaned, f"文本太短，少于{min_length}字符"
        
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
            return True, cleaned, f"文本过长，已截断到{max_length}字符"
        
        return True, cleaned, "文本验证通过"


    def analyze_extraction_quality(decode_stats, quality_metrics):
        """
        分析提取质量并给出建议
        
        参数:
            decode_stats: 解码统计信息
            quality_metrics: 质量指标
            
        返回:
            分析报告字典
        """
        integrity_score = quality_metrics.get('data_integrity_score', 0) * 100
        
        # 质量等级
        if integrity_score >= 95:
            quality_level = "优秀"
            reliability = "非常可靠"
            suggestions = ["数据完整，无需额外处理"]
        elif integrity_score >= 80:
            quality_level = "良好"
            reliability = "比较可靠"
            suggestions = [
                "数据基本完整",
                "如有疑问建议多次提取对比"
            ]
        elif integrity_score >= 60:
            quality_level = "一般"
            reliability = "部分可靠"
            suggestions = [
                "数据部分损坏但已恢复",
                "建议检查原始图片质量",
                "可尝试不同的密码参数"
            ]
        else:
            quality_level = "较差"
            reliability = "可靠性低"
            suggestions = [
                "数据严重损坏",
                "建议使用原始无损图片",
                "检查图片是否经过压缩或编辑",
                "确认密码和参数是否正确"
            ]
        
        return {
            'quality_level': quality_level,
            'reliability': reliability,
            'integrity_score': integrity_score,
            'suggestions': suggestions,
            'error_correction': decode_stats.get('total_errors_corrected', 0),
            'valid_packets_ratio': decode_stats.get('valid_packets', 0) / max(decode_stats.get('total_packets_found', 1), 1)
        }