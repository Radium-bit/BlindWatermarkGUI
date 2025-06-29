## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

import os
import tempfile
import threading
import re
import time
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
                basename = os.path.basename(filepath)
                # 创建临时文件
                tmp_in = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                tmp_out = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                
                # 保存图片到临时文件
                img = Image.open(filepath)
                # 如果目标尺寸不为None，则调整图片大小，常用于提取尺寸已被更改的图像
                target_size = self.app.get_target_size()
                if target_size is None:
                    m = re.search(r"size(\d+)x(\d+)", basename)
                    if not m: 
                        result = messagebox.askyesno("注意","文件名中未找到 size(如 size800x600)\n可取消后手动输入或改名\n\n或者继续？（使用当前图像的长宽）")
                        if not result:
                            if os.path.exists(tmp_in): os.remove(tmp_out)
                            raise ValueError("用户取消操作，请手动输入原图长宽")
                    target_size = (int(m.group(1)), int(m.group(2))) if m else None
                self.app.root.after(0, lambda: self.app.show_processing_window("正在提取水印，请稍候..."))
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

    def extract_watermark_bit_advanced(self, filepath, threshold=0.5, auto_threshold=False):
        """高级版本：支持自动阈值调整的二进制水印提取，保存到.bin文件"""
        def worker():
            try:
                # 显示处理窗口
                self.app.root.after(0, lambda: self.app.show_processing_window("正在提取二进制水印，请稍候..."))
                
                pwd = self.app.get_pwd()
                name = os.path.basename(filepath)
                ext = os.path.splitext(filepath)[1]

                output_dir = self.app.get_output_dir()
                tmp_in = os.path.join(output_dir, f"input.png")

                # 读取 ws (水印长度)
                wm_len = self.app.get_ws()
                if wm_len is None:
                    m = re.search(r"ws(\d+)", name)
                    if not m:
                        raise ValueError("文件名中未找到 ws（如 ws6）\n可手动输入或改名")
                    wm_len = int(m.group(1))

                # 读取原始尺寸
                target_size = self.app.get_target_size()
                if target_size is None:
                    m = re.search(r"size(\d+)x(\d+)", name)
                    if not m:
                        raise ValueError("文件名中未找到 size（如 size800x600）\n可手动输入或改名")
                    target_size = int(m.group(1)), int(m.group(2))

                # 处理图片
                img = Image.open(filepath)
                if img.size == target_size:
                    img.save(tmp_in)
                else:
                    resized = img.resize(target_size, Image.LANCZOS)
                    resized.save(tmp_in, format="PNG")

                # 提取水印
                bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                wm_extract = bwm1.extract(tmp_in, wm_shape=wm_len, mode='bit')
                
                # 自动调整阈值（可选）
                if auto_threshold:
                    # 使用中位数作为阈值
                    threshold = np.median(wm_extract)
                    print(f"自动计算阈值: {threshold}")
                
                # 转换为布尔值
                bit_result = [value > threshold for value in wm_extract]
                
                # 计算提取质量指标
                confidence_scores = [abs(value - threshold) for value in wm_extract]
                avg_confidence = np.mean(confidence_scores)
                
                # 生成输出文件路径：与原文件同名，但扩展名为.bin，保存在源目录
                source_dir = os.path.dirname(filepath)
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                bin_filepath = os.path.join(source_dir, f"{base_name}.bin")
                
                # 保存二进制数据到文件
                with open(bin_filepath, 'wb') as f:
                    # 布尔值表转字节流
                    binary_data = bytes([int(bit) for bit in bit_result])
                    f.write(binary_data)
                
                # 格式化结果消息
                result_str = "二进制水印提取完成!\n"
                result_str += f"水印长度: {wm_len} 位\n"
                result_str += f"使用阈值: {threshold:.3f}\n"
                result_str += f"平均置信度: {avg_confidence:.3f}\n"
                result_str += f"结果已保存到:\n{bin_filepath}\n"
                
                # 显示置信度统计
                high_conf = sum(1 for score in confidence_scores if score > 0.2)
                medium_conf = sum(1 for score in confidence_scores if 0.1 < score <= 0.2)
                low_conf = sum(1 for score in confidence_scores if score <= 0.1)
                
                result_str += f"\n置信度分布:\n"
                result_str += f"高置信度 (>0.2): {high_conf} 位\n"
                result_str += f"中置信度 (0.1-0.2): {medium_conf} 位\n"
                result_str += f"低置信度 (≤0.1): {low_conf} 位"
                
                # 显示结果
                self.app.root.after(0, self.app.hide_processing_window)
                self.app.root.after(0, lambda: messagebox.showinfo("提取成功", result_str))

                # 清理临时文件
                if os.path.exists(tmp_in):
                    try:
                        os.remove(tmp_in)
                    except:
                        pass
                        
            except Exception as e:
                self.app.root.after(0, lambda: messagebox.showerror("错误", str(e)))
                self.app.root.after(0, self.app.hide_processing_window)
                
        threading.Thread(target=worker).start()

    def extract_watermark_bit_advanced_with_rc1(self, filepath, use_rc1_flag=True):
            def worker():
                tmp_in = None
                tmp_out = None
                basename = os.path.basename(filepath)
                temp_img_main = None
                use_rc1=use_rc1_flag
                try:
                    if not os.path.exists(filepath):
                        self.app.root.after(0, lambda: messagebox.showerror("错误", f"图片文件不存在：{filepath}\n请检查文件路径是否正确？"))
                        self.app.root.after(0, self.app.hide_processing_window)
                        return
                    
                    # Show processing window
                    self.app.root.after(0, lambda: self.app.show_processing_window("正在读取图片，请稍候..."))
                    
                    # Read the watermarked image
                    image = Image.open(filepath)
                    # 如果目标尺寸不为None，则调整图片大小，常用于提取尺寸已被更改的图像
                    target_size = self.app.get_target_size()
                    if target_size is None:
                        m = re.search(r"size(\d+)x(\d+)", basename)
                        if not m: 
                            result = messagebox.askyesno("注意","文件名中未找到 size(如 size800x600)\n可取消后手动输入或改名\n\n或者继续？（使用当前图像的长宽）")
                            if not result:
                                raise ValueError("用户取消操作，请手动输入原图长宽")
                        target_size = (int(m.group(1)), int(m.group(2))) if m else None
                    self.app.root.after(0, lambda: self.app.show_processing_window("正在读取图片，请稍候..."))
                    if target_size is not None:
                        if image.size != target_size:
                            print("Enter Resize Brench")
                            image = image.resize(target_size, Image.LANCZOS)
                            
                    width, height = image.size
                    name, ext = os.path.splitext(os.path.basename(filepath))
                    
                    # Create temporary file for processing
                    tmp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                    
                    
                    # Save the input image for blind-watermark library
                    with open(tmp_in, 'wb') as f:
                        image.save(f)
                    
                    # Get password
                    pwd = self.app.get_pwd()
                    
                    # Extract watermark bits
                    self.app.root.after(0, lambda: self.app.show_processing_window("正在提取水印，请稍候..."))
                    
                    try:
                        # Create WaterMark instance for extraction
                        bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                        
                        # Extract watermark length from filename if available
                        wm_len = None
                        if '-ws' in name:
                            try:
                                ws_part = name.split('-ws')[1].split('-')[0]
                                wm_len = int(ws_part)
                                print(f"从文件名检测到水印长度: {wm_len} 比特")
                            except:
                                print("无法从文件名获取水印长度，将尝试自动检测")
                        
                        # Extract watermark
                        wm_bits = bwm1.extract(tmp_in, wm_shape=wm_len, mode='bit')
                        
                        # Convert watermark bits to boolean list
                        bit_array = []
                        for bit in wm_bits:
                            bit_array.append(bool(bit))
                        
                        print(f"提取的比特流长度: {len(bit_array)} 比特")
                        
                    except Exception as e:
                        self.app.root.after(0, lambda error_msg=e: messagebox.showerror("错误", f"水印提取失败: {str(error_msg)}\n请确保图片包含有效水印"))
                        return
                    
                    # Process extracted bits based on encoding type
                    if use_rc1:
                        self.app.root.after(0, lambda: self.app.show_processing_window("正在使用RC1纠错解码处理水印数据，请稍候..."))
                        
                        try:
                            from .dataShielder import decode_watermark_to_binary, decode_watermark_to_string
                            
                            # Decode using RC1 decoder
                            binary_data, stats = decode_watermark_to_binary(bit_array)
                            
                            if binary_data is None:
                                # Try with different parameters or without interleaving
                                print("标准RC1解码失败，尝试不同的参数...")
                                binary_data, stats = decode_watermark_to_binary(bit_array, interleave_depth=1)
                            
                            if binary_data is None:
                                self.app.root.after(0, lambda: messagebox.showerror("错误", 
                                    f"RC1解码失败\n"
                                    f"找到的数据包: {stats.get('total_packets_found', 0)}\n"
                                    f"有效数据包: {stats.get('valid_packets', 0)}\n"
                                    f"请确保水印是使用RC1编码嵌入的"))
                                return
                            
                            decoding_info = "RC1解码"
                            
                            # Print decoding statistics
                            print(f"RC1解码统计:")
                            print(f"  找到的数据包: {stats['total_packets_found']}")
                            print(f"  有效数据包: {stats['valid_packets']}")
                            print(f"  纠正的错误: {stats['total_errors_corrected']} 字节")
                            print(f"  恢复的分片: {len(stats['chunks_recovered'])}")
                            
                        except ImportError:
                            print("警告：RC1解码器不可用，回退到原始方法")
                            use_rc1 = False
                    
                    if not use_rc1:
                        # Original method: directly convert bit array to binary data
                        self.app.root.after(0, lambda: self.app.show_processing_window("正在将比特流转换为二进制数据，请稍候..."))
                        
                        # Convert bit array to bytes
                        binary_data = bytearray()
                        for i in range(0, len(bit_array), 8):
                            if i + 8 <= len(bit_array):
                                byte_val = 0
                                for j in range(8):
                                    if bit_array[i + j]:
                                        byte_val |= (1 << (7 - j))
                                binary_data.append(byte_val)
                        
                        binary_data = bytes(binary_data)
                        decoding_info = "原始解码"
                        stats = {'valid_packets': 0, 'total_packets_found': 0}
                    
                    # Determine output filename and extension
                    output_dir = self.app.get_output_dir()
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Try to detect file type from binary data
                    file_ext = self.detect_file_type(binary_data)
                    if file_ext is None:
                        # Ask user for file extension
                        self.app.root.after(0, self.app.hide_processing_window)
                        
                        # Create a simple dialog to get file extension
                        from tkinter import simpledialog
                        file_ext = self.app.root.after(0, lambda: simpledialog.askstring(
                            "文件类型", 
                            "无法自动检测文件类型。\n请输入文件扩展名（例如: txt, pdf, jpg, exe）：",
                            parent=self.app.root
                        ))
                        
                        if not file_ext:
                            file_ext = "bin"
                        elif not file_ext.startswith('.'):
                            file_ext = '.' + file_ext
                        
                        self.app.root.after(0, lambda: self.app.show_processing_window("正在保存文件，请稍候..."))
                    
                    # Generate output filename
                    encoding_suffix = "-RC1" if use_rc1 else "-RAW"
                    output_filename = f"{name}-Extracted{encoding_suffix}{file_ext}"
                    output_path = os.path.join(os.path.dirname(filepath), output_filename)
                    
                    # Save the extracted binary data
                    with open(output_path, 'wb') as f:
                        f.write(binary_data)
                    
                    # Ensure processing window is closed
                    self.app.root.after(0, self.app.hide_processing_window)
                    
                    # Show success message with detailed information
                    file_size = len(binary_data)
                    success_message = f"""提取成功！输出文件：{output_path}
                    
    解码信息：
    • 解码方式：{decoding_info}
    • 提取的数据大小：{file_size} 字节
    • 文件类型：{file_ext}
    • 图像尺寸：{width}x{height}
    • 比特流长度：{len(bit_array)} 比特"""
                    
                    if use_rc1 and stats:
                        success_message += f"""
    • 有效数据包：{stats['valid_packets']}/{stats['total_packets_found']}
    • 纠正的错误：{stats.get('total_errors_corrected', 0)} 字节"""
                    
                    self.app.root.after(0, lambda: messagebox.showinfo("提取成功", success_message))
                    
                except Exception as e:
                    # Catch other unforeseen errors and display them
                    self.app.root.after(0, lambda e_val=e: messagebox.showerror("错误", str(e_val)))
                    import traceback
                    traceback.print_exc()
                finally:
                    # Regardless of success or failure, hide the processing window at the end
                    self.app.root.after(0, self.app.hide_processing_window)
                    # Clean up temporary files
                    for f in [tmp_in, tmp_out, temp_img_main]:
                        if f and os.path.exists(f):
                            try:
                                os.remove(f)
                                print(f"已清理临时文件: {f}")
                            except Exception as cleanup_e:
                                print(f"清理临时文件失败 {f}: {cleanup_e}")
            
            # Start worker thread
            threading.Thread(target=worker, daemon=True).start()
        
    # 尝试识别文件头来命名提取的文件
    def detect_file_type(self, data):
        """Detect file type from binary data using magic bytes"""
        if len(data) < 4:
            return None
        # Common file signatures (magic bytes)
        signatures = {
            b'\xFF\xD8\xFF': '.jpg',
            b'\x89PNG': '.png',
            b'GIF87a': '.gif',
            b'GIF89a': '.gif',
            b'%PDF': '.pdf',
            b'PK\x03\x04': '.zip',
            b'PK\x05\x06': '.zip',
            b'PK\x07\x08': '.zip',
            b'\x50\x4B\x03\x04': '.docx',  # Also for xlsx, pptx
            b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1': '.doc',  # Also for xls, ppt
            b'MZ': '.exe',
            b'\x7FELF': '.elf',
            b'\xCA\xFE\xBA\xBE': '.class',
            b'\xFE\xED\xFA': '.mach',
            b'#!': '.sh',
            b'\x1F\x8B': '.gz',
            b'BZh': '.bz2',
            b'\x37\x7A\xBC\xAF\x27\x1C': '.7z',
            b'Rar!': '.rar',
            b'\x00\x00\x00\x14ftypMP4': '.mp4',
            b'\x00\x00\x00\x14ftyp': '.mp4',
            b'\x00\x00\x00\x18ftypmp4': '.mp4',
            b'\x1A\x45\xDF\xA3': '.mkv',
            b'OggS': '.ogg',
            b'RIFF': '.wav',  # Also for avi
            b'ID3': '.mp3',
            b'\xFF\xFB': '.mp3',
            b'\xFF\xF3': '.mp3',
            b'\xFF\xF2': '.mp3',
            b'fLaC': '.flac',
        }
        
        # Check each signature
        for sig, ext in signatures.items():
            if data.startswith(sig):
                return ext
        
        # Special checks for text files
        try:
            # Try to decode as UTF-8
            data[:1000].decode('utf-8')
            # Check if it looks like text (printable characters)
            if sum(1 for b in data[:100] if 32 <= b <= 126 or b in (9, 10, 13)) > 90:
                return '.txt'
        except:
            pass
        
        # Default to .bin if cannot detect
        return '.bin'

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


#===================extract version 3 part==========================

    def extract_watermark_qr_bit_direct(self, filepath):
        """
        提取直接bit模式嵌入的QR码水印
        流程：载图 → 提取bit数组 → 重构QR码图像 → QR解码
        
        Args:
            filepath (str): 含水印图片的文件路径
        """
        def worker():
            tmp_in = None
            qr_images_found = []
            
            try:
                # 显示处理窗口
                self.app.root.after(0, lambda: self.app.show_processing_window("正在提取QR码水印，请稍候..."))
                
                # 获取密码
                pwd = self.app.get_pwd()
                
                # 创建临时文件
                tmp_in = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                
                # 读取并处理图片
                self.app.root.after(0, lambda: self.app.show_processing_window("正在读取图片，请稍候..."))
                img = Image.open(filepath)
                
                # 检查是否需要调整尺寸（从文件名或用户输入获取原始尺寸）
                basename = os.path.basename(filepath)
                
                # 如果目标尺寸不为None，则调整图片大小，常用于提取尺寸已被更改的图像
                target_size = self.app.get_target_size()
                if target_size is None:
                    # 优先查找标准的尺寸格式
                    m = re.search(r"(\d+)x(\d+)", basename)
                    if not m:
                        # 如果没找到，尝试查找size格式
                        m = re.search(r"size(\d+)x(\d+)", basename)
                        if not m:
                            self.app.root.after(0, self.app.hide_processing_window)
                            result = messagebox.askyesno("注意","文件名中未找到 size(如 size800x600)\n可取消后手动输入或改名\n\n或者继续？（使用当前图像的长宽）")
                            if not result:
                                raise ValueError("用户取消操作，请手动输入原图长宽")
                            self.app.root.after(0, lambda: self.app.show_processing_window("正在提取水印，请稍候..."))
                        else:
                            target_size = (int(m.group(1)), int(m.group(2)))
                            print(f"从文件名识别原始尺寸(size格式): {target_size}")
                    else:
                        target_size = (int(m.group(1)), int(m.group(2)))
                        print(f"从文件名识别原始尺寸: {target_size}")
                
                # 调整图片尺寸（如果需要）
                if target_size is not None and img.size != target_size:
                    print(f"Enter Resize Branch: {img.size} → {target_size}")
                    img = img.resize(target_size, Image.LANCZOS)
                
                # 保存图像临时文件
                img.save(tmp_in)
                
                # 常见的QR码尺寸列表（从小到大尝试）
                # 根据embed代码，主要使用的是直接计算的尺寸
                qr_sizes_to_try = []
                
                # 1. 尝试从文件名中提取QR码信息
                version_match = re.search(r"QRV(\d+)\+([LMQH])", basename)
                size_match = re.search(r"ws(\d+)", basename)
                
                if version_match and size_match:
                    version = int(version_match.group(1))
                    error_correction = version_match.group(2)
                    bit_count = int(size_match.group(1))  # 修复：应该是group(1)而不是group(2)
                    
                    # 计算对应的QR码尺寸
                    estimated_size = int(bit_count ** 0.5)
                    qr_sizes_to_try.append(estimated_size)
                    print(f"从文件名推断QR码参数: V{version}+{error_correction}, bit数: {bit_count}, 预估尺寸: {estimated_size}×{estimated_size}")
                
                # 2. 如果没有从文件名获取到ws信息，尝试手动输入
                if not size_match:
                    # 读取 ws (水印长度)
                    wm_len = self.app.get_ws()
                    if wm_len is None:
                        name = os.path.splitext(basename)[0]  # 去掉扩展名
                        m = re.search(r"ws(\d+)", name)
                        if not m:
                            self.app.root.after(0, self.app.hide_processing_window)
                            error_msg = "文件名中未找到 ws（如 ws6）\n可手动输入或改名"
                            self.app.root.after(0, lambda: messagebox.showerror("参数缺失", error_msg))
                            raise ValueError(error_msg)
                        wm_len = int(m.group(1))
                    
                    if wm_len:
                        estimated_size = int(wm_len ** 0.5)
                        qr_sizes_to_try.append(estimated_size)
                        print(f"从ws参数推断QR码尺寸: ws{wm_len} → {estimated_size}×{estimated_size}")
                
                # 3. 如果从ws参数计算出了预期尺寸，就只尝试这个尺寸
                if size_match or (wm_len and len(qr_sizes_to_try) > 0):
                    # 从ws参数得到了预期尺寸，只尝试这个尺寸
                    expected_qr_size = qr_sizes_to_try[0] if qr_sizes_to_try else None
                    if expected_qr_size:
                        print(f"将仅尝试预期的QR码尺寸: {expected_qr_size}×{expected_qr_size}")
                        qr_sizes_to_try = [expected_qr_size]  # 只保留预期尺寸
                else:
                    # 没有ws信息，添加常见尺寸进行尝试
                    common_sizes = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
                    for size in common_sizes:
                        if size not in qr_sizes_to_try:
                            qr_sizes_to_try.append(size)
                    
                    # 4. 基于图片容量计算可能的尺寸范围
                    img_width, img_height = img.size
                    max_capacity = int(img_width * img_height * 0.25)  # 估算最大容量
                    max_qr_size = int(max_capacity ** 0.5)
                    
                    # 添加容量范围内的尺寸
                    for size in range(50, min(max_qr_size, 400), 16):  # 步长16
                        if size not in qr_sizes_to_try:
                            qr_sizes_to_try.append(size)
                
                print(f"将尝试的QR码尺寸: {qr_sizes_to_try[:10]}..." if len(qr_sizes_to_try) > 10 else f"将尝试的QR码尺寸: {qr_sizes_to_try}")
                
                # 初始化WaterMark对象
                bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                
                # 逐个尝试不同的QR码尺寸
                for i, qr_size in enumerate(qr_sizes_to_try):
                    try:
                        self.app.root.after(0, lambda s=qr_size, idx=i+1, total=len(qr_sizes_to_try): 
                                        self.app.show_processing_window(f"正在尝试提取 {s}×{s} QR码 ({idx}/{total})..."))
                        
                        print(f"尝试提取 {qr_size}×{qr_size} QR码...")
                        
                        # 提取bit数组水印
                        wm_shape = qr_size * qr_size  # bit数组的长度
                        wm_extract = bwm1.extract(tmp_in, wm_shape=wm_shape, mode='bit')
                        
                        if wm_extract is None or len(wm_extract) == 0:
                            print(f"  提取失败: 空数据")
                            continue
                        
                        print(f"  提取到 {len(wm_extract)} 个浮点数")
                        print(f"  数值范围: {min(wm_extract):.3f} ~ {max(wm_extract):.3f}")
                        
                        # 检查数据长度
                        expected_length = qr_size * qr_size
                        if len(wm_extract) != expected_length:
                            print(f"  数据长度不匹配: 期望 {expected_length}, 实际 {len(wm_extract)}")
                            continue
                        
                        # 将bit数组重构为2D图像
                        # 直接使用浮点数值作为灰度强度，不进行二值化
                        img_data = []
                        for row in range(qr_size):
                            row_data = []
                            for col in range(qr_size):
                                idx = row * qr_size + col
                                # 直接将0-1的浮点数转换为0-255的灰度值
                                gray_val = int(wm_extract[idx] * 255)
                                gray_val = max(0, min(255, gray_val))  # 确保在有效范围内
                                row_data.append(gray_val)
                            img_data.append(row_data)
                        
                        # 转换为numpy数组并创建灰度图像
                        img_array = np.array(img_data, dtype=np.uint8)
                        qr_img = Image.fromarray(img_array, mode='L')
                        
                        # 放大图像以便更好的识别（如果QR码太小）
                        if qr_size < 100:
                            scale_factor = max(2, 200 // qr_size)
                            new_size = qr_size * scale_factor
                            qr_img = qr_img.resize((new_size, new_size), Image.NEAREST)
                            print(f"  QR码放大: {qr_size}×{qr_size} → {new_size}×{new_size}")
                        
                        # 转换为RGB模式以便qreader识别
                        qr_img_rgb = qr_img.convert('RGB')
                        
                        # 尝试解码QR码
                        qreader = self.app.qreader
                        img_array_rgb = np.array(qr_img_rgb)
                        
                        try:
                            text = qreader.detect_and_decode(image=img_array_rgb)[0]
                            
                            if text and len(text.strip()) > 0:
                                print(f"  ✓ QR码解码成功: '{text}'")
                                
                                # 保存成功的结果
                                qr_images_found.append({
                                    'size': qr_size,
                                    'text': text.strip(),
                                    'image': qr_img_rgb,
                                    'bit_count': len(wm_extract),
                                    'gray_range': f"{min(wm_extract):.3f}-{max(wm_extract):.3f}",
                                    'avg_intensity': np.mean(img_array)
                                })
                                
                                # 找到有效结果后立即停止搜索
                                print(f"  找到有效QR码，停止搜索其他尺寸")
                                break
                            else:
                                print(f"  解码失败: 未识别到文本")
                                
                                # 尝试图像增强
                                try:
                                    # 增强对比度
                                    enhancer = ImageEnhance.Contrast(qr_img_rgb)
                                    enhanced_img = enhancer.enhance(2.0)
                                    
                                    # 增强亮度
                                    enhancer = ImageEnhance.Brightness(enhanced_img)
                                    enhanced_img = enhancer.enhance(1.2)
                                    
                                    # 再次尝试解码
                                    enhanced_array = np.array(enhanced_img)
                                    text = qreader.detect_and_decode(image=enhanced_array)[0]
                                    
                                    if text and len(text.strip()) > 0:
                                        print(f"  ✓ 增强后解码成功: '{text}'")
                                        qr_images_found.append({
                                            'size': qr_size,
                                            'text': text.strip(),
                                            'image': enhanced_img,
                                            'bit_count': len(wm_extract),
                                            'gray_range': f"{min(wm_extract):.3f}-{max(wm_extract):.3f}",
                                            'avg_intensity': np.mean(img_array),
                                            'enhanced': True
                                        })
                                        
                                        # 找到有效结果后立即停止搜索
                                        print(f"  找到有效QR码(增强)，停止搜索其他尺寸")
                                        break
                                    else:
                                        print(f"  增强后仍解码失败")
                                except Exception as enhance_e:
                                    print(f"  图像增强失败: {enhance_e}")
                        
                        except Exception as decode_e:
                            print(f"  QR码解码异常: {decode_e}")
                            continue
                    
                    except Exception as e:
                        print(f"  提取 {qr_size}×{qr_size} 失败: {e}")
                        continue
                    
                    # 如果已经找到有效QR码，退出主循环
                    if qr_images_found:
                        print(f"已找到有效QR码，停止尝试其他尺寸")
                        break
                
                # 处理结果
                if qr_images_found:
                    # 按尺寸排序，选择最佳结果
                    qr_images_found.sort(key=lambda x: x['size'])
                    
                    best_result = qr_images_found[0]  # 或者可以根据其他标准选择最佳结果
                    
                    print(f"\n=== QR码水印提取成功 ===")
                    print(f"QR码尺寸: {best_result['size']}×{best_result['size']}")
                    print(f"解码文本: '{best_result['text']}'")
                    print(f"数据统计:")
                    print(f"  总浮点数: {best_result['bit_count']}")
                    print(f"  浮点数范围: {best_result['gray_range']}")
                    print(f"  平均灰度: {best_result['avg_intensity']:.1f}")
                    
                    if len(qr_images_found) > 1:
                        print(f"\n找到 {len(qr_images_found)} 个有效结果:")
                        for i, result in enumerate(qr_images_found):
                            enhanced_flag = " (增强)" if result.get('enhanced') else ""
                            print(f"  {i+1}. {result['size']}×{result['size']}: '{result['text']}'{enhanced_flag}")
                    
                    # 隐藏处理窗口
                    self.app.root.after(0, self.app.hide_processing_window)
                    
                    # 显示结果（使用现有的show_qr_code方法）
                    # 创建临时文件保存QR码图像
                    tmp_qr = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                    best_result['image'].save(tmp_qr)
                    
                    # 显示QR码和文本
                    # show_qr_code方法签名: show_qr_code(qr_path, text=None, status=None, *images)
                    # 对于成功情况: qr_path=临时文件路径, text=解码文本, status=True
                    self.app.root.after(0, lambda: self.app.show_qr_code(
                        tmp_qr, 
                        best_result['text'], 
                        True
                    ))
                    
                else:
                    # 没有找到有效的QR码，参考旧版extract_watermark的处理方式
                    self.app.root.after(0, self.app.hide_processing_window)
                    
                    # 如果是从ws参数预期的尺寸失败，显示提取的图像
                    if len(qr_sizes_to_try) == 1 and (size_match or wm_len):
                        # 尝试提取并显示失败的QR码图像
                        failed_images = []
                        expected_size = qr_sizes_to_try[0]
                        
                        try:
                            print(f"尝试显示失败的QR码图像: {expected_size}×{expected_size}")
                            wm_shape = expected_size * expected_size
                            wm_extract = bwm1.extract(tmp_in, wm_shape=wm_shape, mode='bit')
                            
                            if wm_extract is not None and len(wm_extract) == wm_shape:
                                # 重构图像
                                img_data = []
                                for i in range(expected_size):
                                    row = []
                                    for j in range(expected_size):
                                        idx = i * expected_size + j
                                        gray_val = int(wm_extract[idx] * 255)
                                        gray_val = max(0, min(255, gray_val))
                                        row.append(gray_val)
                                    img_data.append(row)
                                
                                img_array = np.array(img_data, dtype=np.uint8)
                                failed_qr_img = Image.fromarray(img_array, mode='L').convert('RGB')
                                
                                # 添加到失败图像列表
                                failed_images.append((failed_qr_img.size, np.array(failed_qr_img)))
                                
                        except Exception as extract_e:
                            print(f"提取失败图像时出错: {extract_e}")
                        
                        if failed_images:
                            # 显示失败的图像，参考旧版的处理方式
                            # show_qr_code方法签名: show_qr_code(qr_path, text=None, status=None, *images)
                            # 对于失败情况: qr_path=None, text="", status=None, *images=失败图像列表
                            self.app.root.after(0, lambda: self.app.show_qr_code(None, "", None, *failed_images))
                        else:
                            error_message = f"""QR码水印提取失败

    预期QR码尺寸: {expected_size}×{expected_size}
    可能的原因:
    1. 密码不正确
    2. 图片尺寸已被修改
    3. 水印损坏或质量过低
    4. ws参数不匹配实际水印尺寸

    建议:
    • 确认密码是否正确
    • 检查图片是否为原始带水印图片
    • 验证ws参数是否正确"""
                            
                            self.app.root.after(0, lambda: messagebox.showerror("提取失败", error_message))
                    else:
                        # 通用的提取失败信息
                        error_message = f"""未能提取到有效的QR码水印

    可能的原因:
    1. 图片未包含QR码水印
    2. 密码不正确
    3. 图片尺寸已被修改
    4. QR码损坏或质量过低

    建议:
    • 确认密码是否正确
    • 检查图片是否为原始带水印图片
    • 尝试手动输入原始图片尺寸"""
                        
                        self.app.root.after(0, lambda: messagebox.showerror("提取失败", error_message))
            
            except Exception as e:
                self.app.root.after(0, lambda e_val=e: messagebox.showerror("错误", f"提取过程中发生错误: {str(e_val)}"))
                print(f"提取过程异常: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # 清理临时文件
                self.app.root.after(0, self.app.hide_processing_window)
                if tmp_in and os.path.exists(tmp_in):
                    try:
                        os.remove(tmp_in)
                        print(f"已清理临时文件: {tmp_in}")
                    except Exception as cleanup_e:
                        print(f"清理临时文件失败: {cleanup_e}")
        
        # 启动工作线程
        threading.Thread(target=worker, daemon=True).start()