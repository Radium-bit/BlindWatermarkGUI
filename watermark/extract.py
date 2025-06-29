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
                        print("Enter Resize Branch")
                        img = img.resize(target_size, Image.LANCZOS)
                # 保存图像临时文件
                img.save(tmp_in)
                
                # 定义要尝试的尺寸列表
                sizes_to_try = [256, 128, 96, 64]
                pwd = self.app.get_pwd()
                bwm1 = WaterMark(password_wm=int(pwd), password_img=int(pwd))
                
                # 初始化变量
                text = None
                extracted_images = {}  # 存储每个尺寸的提取结果 {size: (img, img_backup)}
                success_size = None
                
                # 按顺序尝试不同尺寸
                for size in sizes_to_try:
                    try:
                        self.app.root.after(0, lambda s=size: self.app.show_processing_window(f"正在尝试提取{s}x{s}尺寸水印，请稍候..."))
                        print(f"尝试提取{size}x{size}尺寸水印")
                        
                        # 提取水印
                        bwm1.extract(filename=tmp_in, wm_shape=(size, size), out_wm_name=tmp_out)
                        extracted_img = Image.open(tmp_out)
                        
                        # 转换为RGB模式
                        if extracted_img.mode != 'RGB':
                            extracted_img = extracted_img.convert('RGB')
                        
                        # 备份原图（如果需要显示原始提取图片）
                        img_backup = None
                        if self.app.show_orignal_extract_picture.get():
                            img_backup = extracted_img.copy()
                        
                        # 尝试解码二维码
                        qreader = self.app.qreader
                        print(f'{size}x{size}临时文件路径: 输入={tmp_in} 输出={tmp_out}')
                        
                        img_array = np.array(extracted_img)
                        text = qreader.detect_and_decode(image=img_array)[0]
                        
                        # 如果第一次解析失败，尝试增强解析
                        if not text:
                            print(f"{size}x{size}第一次解析失败，尝试增强解析")
                            # 1. 尝试调整对比度和亮度
                            enhancer = ImageEnhance.Contrast(extracted_img)
                            extracted_img = enhancer.enhance(2.0)
                            enhancer = ImageEnhance.Brightness(extracted_img)
                            extracted_img = enhancer.enhance(1.5)
                            
                            # 2. 重新尝试解码
                            img_array = np.array(extracted_img.convert('RGB'))
                            text = qreader.detect_and_decode(image=img_array)[0]
                        
                        # 存储提取结果
                        extracted_images[size] = (extracted_img, img_backup)
                        
                        # 如果成功解码，记录成功的尺寸并跳出循环
                        if text:
                            success_size = size
                            print(f"使用{size}x{size}尺寸成功提取水印: {text}")
                            break
                        else:
                            print(f"{size}x{size}尺寸解码失败，尝试下一个尺寸")
                            
                    except Exception as e:
                        print(f"{size}x{size}提取失败: {e}")
                        # 清理临时文件
                        if os.path.exists(tmp_out):
                            try:
                                os.unlink(tmp_out)
                            except:
                                pass
                        continue
                
                # 如果所有尺寸都失败，显示提取的图片和错误信息
                if not text:
                    print("所有尺寸的水印提取都失败")
                    if extracted_images:
                        # 准备显示的图片列表
                        images_to_show = []
                        for size in sizes_to_try:
                            if size in extracted_images:
                                img, img_backup = extracted_images[size]
                                # 如果需要显示原始图片且有备份，使用备份
                                display_img = img_backup if (img_backup is not None and self.app.show_orignal_extract_picture.get()) else img
                                images_to_show.append((f"{size}x{size}", display_img))
                        
                        if images_to_show:
                            # 将图片对象转换为可序列化的元组格式
                            image_tuples = [(size_str, np.array(img)) for size_str, img in images_to_show]
                            # 处理多余的tmp_out文件
                            if os.path.exists(tmp_out):
                                os.unlink(tmp_out)
                            self.app.show_qr_code(None, "", None, *image_tuples)
                            
                            # 清理备份图片
                            for size, (img, img_backup) in extracted_images.items():
                                if img_backup:
                                    img_backup.close()
                        else:
                            messagebox.showerror("错误", "水印提取失败")
                    else:
                        messagebox.showerror("错误", "水印提取失败")
                    return
                
                # 成功提取水印，显示结果
                self.app.root.after(0, lambda: self.app.show_qr_code(tmp_out, text, True))
                
                # 清理其他尺寸的备份图片
                for size, (img, img_backup) in extracted_images.items():
                    if size != success_size and img_backup:
                        img_backup.close()
                        
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