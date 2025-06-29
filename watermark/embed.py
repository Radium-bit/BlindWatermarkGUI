## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

import os
import tempfile
import time
import threading
import qrcode
from PIL import Image, ImageCms
from blind_watermark import WaterMark
import numpy as np
from noise import pnoise2
import shutil
from tkinter import messagebox


class WatermarkEmbedder:
    def __init__(self, app):
        self.app = app
    
    def generate_qr_watermark(self, size=128):
        """生成二维码水印文件
        Args:size: 二维码尺寸 (默认128)
        """
        try:
            wm_text = self.app.text_wm.get("1.0", "end").strip()
            if not wm_text:
                messagebox.showerror("错误", "请输入水印文本")
                return None
                
            # 创建临时文件
            tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
            
            # 生成二维码
            qr = qrcode.QRCode(
                version=4,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=10,
                border=1,
            )
            qr.add_data(wm_text)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")            
            # 调整二维码尺寸
            img = img.resize((size, size), Image.LANCZOS)
            # 保存为JPG（缩减体积）
            img.save(tmp_file, "JPEG", quality=100)
            
            return tmp_file
        except Exception as e:
            messagebox.showerror("错误", f"生成二维码失败: {str(e)}")
            return None

    def process_image_pre_watermark(self, image_path, target_size):
        """
        处理水印图像，确保其为 RGB 格式，并将其尺寸调整为目标尺寸。
        如果图像是 JPG，则会进行色彩空间检查。
        Args:
            image_path (str): 水印图像的路径。
            target_size (int): 目标尺寸（边长）。
        Returns:
            PIL.Image.Image or None: 处理后的图像数据，如果处理失败则返回 None。
        """
        try:
            img = Image.open(image_path)
            # 转换为 RGB 模式，以确保兼容性
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 调整图像尺寸
            # Image.LANCZOS 是高质量的下采样滤镜
            img = img.resize((target_size, target_size), Image.LANCZOS)

            return img
        except Exception as e:
            # 使用 Tkinter 的 messagebox 显示错误
            self.app.root.after(0, lambda error_msg=e: messagebox.showerror("错误", f"处理水印图片失败: {str(error_msg)}"))
            return None


    def embed_watermark_custom_image(self, filepath, w_filepath):
        """
        将自定义图像水印嵌入到主图片中。

        Args:
            filepath (str): 主图片的文件路径。
            w_filepath (str): 用作水印的图像文件路径。
        """
        def worker():
            tmp_in = None
            tmp_out = None
            temp_img_main = None # 用于主图片临时转换的变量
            processed_w_filepath = None # 用于处理后的水印图片的临时文件路径

            try:
                if not os.path.exists(w_filepath):
                    self.app.root.after(0, lambda: messagebox.showerror("错误", f"水印文件不存在：{w_filepath}\n请检查水印路径是否正确？"))
                    self.app.root.after(0, self.app.hide_processing_window)
                    return # 提前退出，不再执行后续操作
                
                # 显示处理窗口
                self.app.root.after(0, lambda: self.app.show_processing_window("正在处理图片，请稍候..."))
                output_dir = self.app.get_output_dir()
                os.makedirs(output_dir, exist_ok=True)

                name, ext = os.path.splitext(os.path.basename(filepath))
                # 读取主图片
                image = Image.open(filepath)
                width, height = image.size

                # 检查并转换主图片的 JPG 色彩空间(仅当输出格式为JPG时)
                if self.app.output_format.get() == "JPG":
                    if filepath.lower().endswith(('.jpg', '.jpeg')):
                        # 检查主图片是否需要色彩空间转换
                        if image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0':
                            # 启动长时间操作提示
                            self.app.root.after(0, lambda: self.app.show_processing_window("正在转换主图片色彩空间，请稍候..."))
                            start_time = time.time()
                            image = image.convert('RGB')
                            # 如果转换时间超过1秒，保持提示窗口
                            if time.time() - start_time > 1:
                                self.app.root.after(0, lambda: messagebox.showinfo("色彩空间转换", "为确保兼容性，已将主图片色彩空间转换为sRGB 4:2:0"))
                            else:
                                self.app.root.after(0, self.app.hide_processing_window)

                # --- 处理 w_filepath (水印图片) ---
                # 先尝试 128x128 尺寸
                current_w_target_size = 128
                processed_w_image = self.process_image_pre_watermark(w_filepath, current_w_target_size)
                if not processed_w_image:
                    return # 如果处理失败，直接返回

                # 保存处理后的水印图片到临时文件
                processed_w_filepath = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
                processed_w_image.save(processed_w_filepath)

                # 定义所有临时文件变量
                tmp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                tmp_out = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name

                # 临时保存转换后的主图片 (仅当需要转换时)
                if self.app.output_format.get() == "JPG" and filepath.lower().endswith(('.jpg', '.jpeg')) and \
                    (image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0'):
                        temp_img_main = os.path.join(os.path.dirname(filepath), "temp_converted_main_image.jpg")
                        image.save(temp_img_main, "JPEG", subsampling="4:2:0", quality=100)
                        image = Image.open(temp_img_main)
                        width, height = image.size # 更新尺寸

                # 保存输入主图片并确保文件关闭，用于 Blind-Watermark 库读取
                with open(tmp_in, 'wb') as f:
                    image.save(f)

                # 获取密码
                pwd = self.app.get_pwd()

                if self.app.enhanced_mode.get():
                    try:
                        # 读取临时文件
                        img = Image.open(tmp_in).convert("RGB") # 读图，转RGB模型
                        arr = np.array(img).astype(np.float32)  # 避免uint8溢出
                        print("Useing Enhanced Mode...")
                        # 生成2D柏林噪声
                        noise = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.float32)
                        for i in range(arr.shape[0]):
                            for j in range(arr.shape[1]):
                                noise[i][j] = pnoise2(i / 50.0, j / 50.0, octaves=2)

                        # 扩展为3D通道，应用到每个颜色通道
                        noise_3d = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
                        arr += noise_3d * 12.8  # 控制噪声强度
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                        # 回写文件
                        Image.fromarray(arr).save(tmp_in)
                    except Exception as e:
                        print(f"噪声处理失败: {e}")

                # 尝试水印嵌入
                try:
                    bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                    bwm1.read_img(tmp_in) # 读取主图片
                    bwm1.read_wm(processed_w_filepath) # 读取处理后的水印图片
                    bwm1.embed(tmp_out) # 嵌入水印
                except Exception as e:
                    # 如果首次嵌入失败 (128x128 尺寸)，尝试 64x64 尺寸
                    if processed_w_filepath and os.path.exists(processed_w_filepath):
                        os.remove(processed_w_filepath) # 删除之前的临时水印文件

                    current_w_target_size = 64 # 尝试 64x64 尺寸
                    processed_w_image = self.process_image_pre_watermark(w_filepath, current_w_target_size)
                    if not processed_w_image:
                        # 如果 64x64 处理也失败，则报错并返回
                        self.app.root.after(0, lambda error_msg=e: messagebox.showerror("错误", f"水印图像处理失败: {str(error_msg)}\n请尝试使用更小的水印尺寸"))
                        return

                    # 保存新的处理后的水印图片到临时文件
                    processed_w_filepath = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                    processed_w_image.save(processed_w_filepath)

                    try:
                        bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                        bwm1.read_img(tmp_in)
                        bwm1.read_wm(processed_w_filepath) # 使用新的处理后的水印图片
                        bwm1.embed(tmp_out)
                    except Exception as e2:
                        # 如果第二次嵌入也失败，则报错并返回
                        self.app.root.after(0, lambda: messagebox.showerror("错误", f"水印嵌入失败: {str(e2)}\n请尝试使用更小的水印尺寸"))
                        return

                wm_len = len(bwm1.wm_bit) # 获取嵌入水印的长度
                output_ext = ".jpg" if self.app.output_format.get() == "JPG" else ext # 确定输出文件扩展名
                dst_img = os.path.join(
                    os.path.dirname(filepath),
                    f"{name}-Watermark-ws{wm_len}-size{width}x{height}{output_ext}"
                )

                if self.app.output_format.get() == "JPG":
                    # 创建 sRGB ICC 配置文件
                    srgb_profile = ImageCms.createProfile("sRGB")
                    # 保存带 ICC 配置的 JPG
                    img_to_save = Image.open(tmp_out).convert('RGB')
                    img_to_save.save(dst_img, "JPEG", quality=100, subsampling="4:2:0",
                                    icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())
                else:
                    # 对于非 JPG 格式，直接复制临时输出文件到目标路径
                    shutil.copy2(tmp_out, dst_img)

                # 确保处理窗口关闭
                self.app.root.after(0, self.app.hide_processing_window)
                # 显示成功信息
                self.app.root.after(0, lambda: messagebox.showinfo("嵌入成功", f"输出文件：\n{dst_img}\n\n水印长度：{wm_len} 尺寸：{width}x{height}"))

            except Exception as e:
                # 捕获其他未预料的错误并显示
                self.app.root.after(0, lambda e_val=e: messagebox.showerror("错误", str(e_val)))
            finally:
                # 无论成功或失败，最后都隐藏处理窗口
                self.app.root.after(0, self.app.hide_processing_window)
                # 清理临时文件
                for f in [tmp_in, tmp_out, temp_img_main, processed_w_filepath]:
                    if f and os.path.exists(f):
                        try:
                            os.remove(f)
                            print(f"已清理临时文件: {f}")
                        except Exception as cleanup_e:
                            print(f"清理临时文件失败 {f}: {cleanup_e}")

        # 启动工作线程
        threading.Thread(target=worker, daemon=True).start()

        self.app.hide_processing_window()


    def embed_watermark_custom_binary(self, filepath, w_filepath):
        """
        将任意二进制文件作为水印嵌入到主图片中。

        Args:
            filepath (str): 主图片的文件路径。
            w_filepath (str): 用作水印的二进制文件路径。
        """
        def worker():
            tmp_in = None
            tmp_out = None
            temp_img_main = None  # 用于主图片临时转换的变量
            tmp_watermark_file = None  # 用于复制水印文件的临时路径

            try:
                if not os.path.exists(w_filepath):
                    self.app.root.after(0, lambda: messagebox.showerror("错误", f"水印文件不存在：{w_filepath}\n请检查水印路径是否正确？"))
                    self.app.root.after(0, self.app.hide_processing_window)
                    return  # Exit early, do not proceed with further operations
                
                # Show processing window
                self.app.root.after(0, lambda: self.app.show_processing_window("正在处理图片，请稍候..."))
                output_dir = self.app.get_output_dir()
                os.makedirs(output_dir, exist_ok=True)

                name, ext = os.path.splitext(os.path.basename(filepath))
                # Read the main image
                image = Image.open(filepath)
                width, height = image.size

                # Check and convert JPG color space of the main image (only if output format is JPG)
                if self.app.output_format.get() == "JPG":
                    if filepath.lower().endswith(('.jpg', '.jpeg')):
                        # Check if the main image needs color space conversion
                        if image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0':
                            # Start long-running operation prompt
                            self.app.root.after(0, lambda: self.app.show_processing_window("正在转换主图片色彩空间，请稍候..."))
                            start_time = time.time()
                            image = image.convert('RGB')
                            # If conversion time exceeds 1 second, keep the prompt window
                            if time.time() - start_time > 1:
                                self.app.root.after(0, lambda: messagebox.showinfo("色彩空间转换", "为确保兼容性，已将主图片色彩空间转换为sRGB 4:2:0"))
                            else:
                                self.app.root.after(0, self.app.hide_processing_window)

                # --- Copy watermark file to temporary directory (to avoid Chinese path issues) ---
                self.app.root.after(0, lambda: self.app.show_processing_window("正在处理水印文件，请稍候..."))
                _, w_ext = os.path.splitext(w_filepath)
                tmp_watermark_file = tempfile.NamedTemporaryFile(suffix=w_ext, delete=False).name
                shutil.copy2(w_filepath, tmp_watermark_file)

                # Define all temporary file variables
                tmp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                tmp_out = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name

                # Temporarily save the converted main image (only if conversion is needed)
                if self.app.output_format.get() == "JPG" and filepath.lower().endswith(('.jpg', '.jpeg')) and \
                    (image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0'):
                        temp_img_main = os.path.join(os.path.dirname(filepath), "temp_converted_main_image.jpg")
                        image.save(temp_img_main, "JPEG", subsampling="4:2:0", quality=100)
                        image = Image.open(temp_img_main)
                        width, height = image.size  # Update dimensions

                # Save the input main image and ensure the file is closed, for Blind-Watermark library to read
                with open(tmp_in, 'wb') as f:
                    image.save(f)

                # Get password
                pwd = self.app.get_pwd()

                if self.app.enhanced_mode.get():
                    try:
                        # Read temporary file
                        img = Image.open(tmp_in).convert("RGB")  # Read image, convert to RGB model
                        arr = np.array(img).astype(np.float32)  # Avoid uint8 overflow
                        print("Using Enhanced Mode...")
                        # Generate 2D Perlin noise
                        noise = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.float32)
                        for i in range(arr.shape[0]):
                            for j in range(arr.shape[1]):
                                noise[i][j] = pnoise2(i / 50.0, j / 50.0, octaves=2)

                        # Extend to 3D channels, apply to each color channel
                        noise_3d = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
                        arr += noise_3d * 12.8  # Control noise intensity
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                        # Write back to file
                        Image.fromarray(arr).save(tmp_in)
                    except Exception as e:
                        print(f"Noise processing failed: {e}")

                # Read binary watermark file and convert to bit array
                self.app.root.after(0, lambda: self.app.show_processing_window("正在读取二进制水印文件，请稍候..."))
                with open(tmp_watermark_file, 'rb') as f:
                    binary_data = f.read()
                
                # Convert binary data to bit array
                bit_array = []
                for byte in binary_data:
                    for i in range(8):
                        bit_array.append(bool(byte & (1 << (7 - i))))

                # Watermark embedding
                self.app.root.after(0, lambda: self.app.show_processing_window("正在嵌入水印，请稍候..."))
                try:
                    bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                    bwm1.read_img(tmp_in)  # Read main image
                    bwm1.read_wm(bit_array, mode='bit')  # Read binary bit array
                    bwm1.embed(tmp_out)  # Embed watermark
                except Exception as e:
                    # If embedding fails, report error and return
                    self.app.root.after(0, lambda error_msg=e: messagebox.showerror("错误", f"水印嵌入失败: {str(error_msg)}\n请尝试使用更小的二进制文件或更大的主图片"))
                    return

                wm_len = len(bit_array)  # Get the length of the embedded watermark
                output_ext = ".jpg" if self.app.output_format.get() == "JPG" else ext  # Determine output file extension
                dst_img = os.path.join(
                    os.path.dirname(filepath),
                    f"{name}-Watermark-ws{wm_len}-size{width}x{height}{output_ext}"
                )

                if self.app.output_format.get() == "JPG":
                    # Create sRGB ICC profile
                    srgb_profile = ImageCms.createProfile("sRGB")
                    # Save JPG with ICC profile
                    img_to_save = Image.open(tmp_out).convert('RGB')
                    img_to_save.save(dst_img, "JPEG", quality=100, subsampling="4:2:0",
                                    icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())
                else:
                    # For non-JPG formats, directly copy the temporary output file to the target path
                    shutil.copy2(tmp_out, dst_img)

                # Ensure processing window is closed
                self.app.root.after(0, self.app.hide_processing_window)
                # Show success message
                self.app.root.after(0, lambda: messagebox.showinfo("嵌入成功", f"输出文件：\n{dst_img}\n\n水印长度：{wm_len} bits 尺寸：{width}x{height}\n原始文件大小：{len(binary_data)} bytes"))

            except Exception as e:
                # Catch other unforeseen errors and display them
                self.app.root.after(0, lambda e_val=e: messagebox.showerror("错误", str(e_val)))
            finally:
                # Regardless of success or failure, hide the processing window at the end
                self.app.root.after(0, self.app.hide_processing_window)
                # Clean up temporary files
                for f in [tmp_in, tmp_out, temp_img_main, tmp_watermark_file]:
                    if f and os.path.exists(f):
                        try:
                            os.remove(f)
                            print(f"已清理临时文件: {f}")
                        except Exception as cleanup_e:
                            print(f"清理临时文件失败 {f}: {cleanup_e}")  # Only print cleanup failure info

        # Start worker thread
        threading.Thread(target=worker, daemon=True).start()

        self.app.hide_processing_window()

    def confirm_watermark_embedding(self, binary_data, available_capacity, safety_margin=0.90):
        """
        在嵌入水印前检查容量并请求用户确认
    
        Args:
            binary_data: 要嵌入的二进制数据
            available_capacity: 可用容量（比特数）
            safety_margin: 安全边际（默认0.90，即90%）
    
        Returns:
            tuple: (是否继续, 适配后的数据)
        """
        from .dataShielder import adapt_to_watermark_capacity_binary, estimate_required_capacity
        
        # 计算数据大小（字节）
        binary_data_size = len(binary_data)
        
        try:
            # 估算实际需要的容量（比特）
            required_bits, _ = estimate_required_capacity(binary_data)
            required_bytes = (required_bits + 7) // 8  # 转换为字节显示
            
            # 可用容量转换为字节显示
            available_bytes = available_capacity // 8
            
            # 如果需要的容量超过可用容量，显示确认对话框
            if required_bits > available_capacity:
                # 转换为KB显示
                data_size_kb = binary_data_size / 1024
                required_kb = required_bytes / 1024
                available_kb = available_bytes / 1024
                
                # 计算截断后的容量
                safe_capacity_bits = int(available_capacity * safety_margin)
                safe_capacity_bytes = safe_capacity_bits // 8
                safe_capacity_kb = safe_capacity_bytes / 1024
            
                # 创建确认消息
                message = (
                    f"水印数据容量不足！\n\n"
                    f"原始数据大小: {data_size_kb:.2f} KB\n"
                    f"编码后需要容量: {required_kb:.2f} KB\n"
                    f"图像可用容量: {available_kb:.2f} KB\n"
                    f"安全容量限制: {safe_capacity_kb:.2f} KB\n\n"
                    f"继续嵌入将自动截断数据到安全容量范围内，\n"
                    f"可能会丢失部分数据内容。\n\n"
                    f"是否继续？"
                )
            
                # 显示确认对话框
                result = messagebox.askyesno("容量不足确认", message)
            
                if not result:
                    return False, None
                    
            else:
                print(f"容量检查通过：需要 {required_bits} 比特，可用 {available_capacity} 比特")
        
            # 如果用户确认或容量足够，调用适配函数
            bit_array = adapt_to_watermark_capacity_binary(
                binary_data,
                available_capacity,
                safety_margin=safety_margin  # 确保参数名正确
            )
        
            return True, bit_array
            
        except Exception as e:
            print(f"容量确认过程出错: {e}")
            # 显示错误信息
            error_message = f"容量检查失败: {str(e)}\n\n是否尝试强制嵌入？"
            result = messagebox.askyesno("容量检查错误", error_message)
            
            if not result:
                return False, None
                
            # 尝试强制嵌入
            try:
                bit_array = adapt_to_watermark_capacity_binary(
                    binary_data,
                    available_capacity,
                    safety_margin=safety_margin
                )
                return True, bit_array
            except Exception as force_error:
                messagebox.showerror("嵌入失败", f"强制嵌入也失败了: {str(force_error)}")
                return False, None

    def embed_watermark_custom_binary_with_rc1(self, filepath, w_filepath, use_rc1=True):
        def worker():
            tmp_in = None
            tmp_out = None
            temp_img_main = None  # 用于主图片临时转换的变量
            tmp_watermark_file = None  # 用于复制水印文件的临时路径
            use_rc1_encoding = use_rc1
            try:
                if not os.path.exists(w_filepath):
                    self.app.root.after(0, lambda: messagebox.showerror("错误", f"水印文件不存在：{w_filepath}\n请检查水印路径是否正确？"))
                    self.app.root.after(0, self.app.hide_processing_window)
                    return  # Exit early, do not proceed with further operations

                # Show processing window
                self.app.root.after(0, lambda: self.app.show_processing_window("正在处理图片，请稍候..."))
                output_dir = self.app.get_output_dir()
                os.makedirs(output_dir, exist_ok=True)
                name, ext = os.path.splitext(os.path.basename(filepath))
                # Read the main image
                image = Image.open(filepath)
                width, height = image.size
                # Check and convert JPG color space of the main image (only if output format is JPG)
                if self.app.output_format.get() == "JPG":
                    if filepath.lower().endswith(('.jpg', '.jpeg')):
                        # Check if the main image needs color space conversion
                        if image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0':
                            # Start long-running operation prompt
                            self.app.root.after(0, lambda: self.app.show_processing_window("正在转换主图片色彩空间，请稍候..."))
                            start_time = time.time()
                            image = image.convert('RGB')
                            # If conversion time exceeds 1 second, keep the prompt window
                            if time.time() - start_time > 1:
                                self.app.root.after(0, lambda: messagebox.showinfo("色彩空间转换", "为确保兼容性，已将主图片色彩空间转换为sRGB 4:2:0"))
                            else:
                                self.app.root.after(0, self.app.hide_processing_window)
                # --- Copy watermark file to temporary directory (to avoid Chinese path issues) ---
                self.app.root.after(0, lambda: self.app.show_processing_window("正在处理水印文件，请稍候..."))
                _, w_ext = os.path.splitext(w_filepath)
                tmp_watermark_file = tempfile.NamedTemporaryFile(suffix=w_ext, delete=False).name
                shutil.copy2(w_filepath, tmp_watermark_file)
                # Define all temporary file variables
                tmp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                tmp_out = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                # Temporarily save the converted main image (only if conversion is needed)
                if self.app.output_format.get() == "JPG" and filepath.lower().endswith(('.jpg', '.jpeg')) and \
                        (image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0'):
                    temp_img_main = os.path.join(os.path.dirname(filepath), "temp_converted_main_image.jpg")
                    image.save(temp_img_main, "JPEG", subsampling="4:2:0", quality=100)
                    image = Image.open(temp_img_main)
                    width, height = image.size  # Update dimensions
                # Save the input main image and ensure the file is closed, for Blind-Watermark library to read
                with open(tmp_in, 'wb') as f:
                    image.save(f)
                # Get password
                pwd = self.app.get_pwd()
                if self.app.enhanced_mode.get():
                    try:
                        # Read temporary file
                        img = Image.open(tmp_in).convert("RGB")  # Read image, convert to RGB model
                        arr = np.array(img).astype(np.float32)  # Avoid uint8 overflow
                        print("Using Enhanced Mode...")
                        # 生成2D柏林噪声
                        noise = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.float32)
                        for i in range(arr.shape[0]):
                            for j in range(arr.shape[1]):
                                noise[i][j] = pnoise2(i / 50.0, j / 50.0, octaves=2)
                        # 扩展3D通道
                        noise_3d = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
                        arr += noise_3d * 12.8  # Control noise intensity
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                        # Write back to file
                        Image.fromarray(arr).save(tmp_in)
                    except Exception as e:
                        print(f"Noise processing failed: {e}")
                
                # 读取二进制水印文件
                with open(tmp_watermark_file, 'rb') as f:
                    binary_data = f.read()

                # ========== RC1纠错编码处理开始 ==========
                if use_rc1_encoding:
                    self.app.root.after(0, lambda: self.app.show_processing_window("正在使用RC1纠错编码处理水印文件，请稍候..."))

                    # 创建WaterMark实例来获取精确的可用容量
                    self.app.root.after(0, lambda: self.app.show_processing_window("正在计算图像可用容量，请稍候..."))
                    try:
                        # 创建WaterMark实例来获取容量信息
                        bwm_temp = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                        bwm_temp.read_img(tmp_in)

                        # WaterMarkCore有一个方法可以获取可用容量，如果没有则使用估算
                        # 这里需要根据实际的WaterMark类实现来调整
                        try:
                            # 尝试获取实际可用容量
                            bwm_temp.bwm_core.init_block_index()
                            available_capacity = bwm_temp.bwm_core.block_num
                            capacity_method = "精确计算"
                        except AttributeError:
                            # 如果方法不存在，回退到估算
                            available_capacity = int(width * height * 0.25)
                            capacity_method = "估算"

                        print(f"原始水印文件大小: {len(binary_data)} 字节")
                        print(f"图像可用容量({capacity_method}): {available_capacity} 比特")

                    except Exception as e:
                        print(f"获取容量信息失败: {e}")
                        # 回退到基本估算
                        available_capacity = int(width * height * 0.25)
                        capacity_method = "基本估算"
                        print(f"使用基本估算容量: {available_capacity} 比特")

                    # 获取RC1编码统计信息并进行容量确认
                    try:
                        from .dataShielder import get_encoding_stats, adapt_to_watermark_capacity, print_encoding_report

                        # 打印编码前的报告
                        print_encoding_report(binary_data)

                        # 先进行容量确认，然后使用RC1编码器处理数据
                        continue_embedding, bit_array = self.confirm_watermark_embedding(
                            binary_data,
                            available_capacity,
                            safety_margin=0.90  # 使用90%的安全边际
                        )

                        if continue_embedding:
                            # 用户确认继续，使用返回的适配数据
                            wm_len = len(bit_array)
                            compression_info = f"RC1编码"
                            print(f"RC1编码后比特长度: {wm_len}")
                            if 'available_capacity' in locals():
                                print(f"容量利用率: {(wm_len/available_capacity)*100:.1f}%")
                        else:
                            # 用户取消操作
                            print("用户取消了水印嵌入操作")
                            self.app.root.after(0, lambda: messagebox.showinfo("操作取消", "水印嵌入操作已取消"))
                            self.app.root.after(0, self.app.hide_processing_window)
                            return

                    except ImportError as import_error:
                        # 如果RC1编码器不可用，直接报错退出
                        error_msg = f"RC1编码器不可用: {import_error}\n\n由于原始编码方法不可靠，无法继续嵌入。\n请确保RC1编码模块正确安装。"
                        print(f"错误：{error_msg}")
                        self.app.root.after(0, lambda: messagebox.showerror("编码器不可用", error_msg))
                        self.app.root.after(0, self.app.hide_processing_window)
                        return
                    except Exception as rc1_error:
                        # RC1编码过程中的其他错误，直接报错退出
                        error_msg = f"RC1编码失败: {str(rc1_error)}\n\n由于原始编码方法不可靠，无法继续嵌入。\n请检查水印文件或尝试使用更大的图片。"
                        print(f"RC1编码过程出错: {rc1_error}")
                        self.app.root.after(0, lambda: messagebox.showerror("编码失败", error_msg))
                        self.app.root.after(0, self.app.hide_processing_window)
                        return

                else:
                    # 如果不使用RC1编码，直接报错
                    error_msg = "原始编码方法不可靠，请启用RC1编码模式进行水印嵌入。"
                    print(f"错误：{error_msg}")
                    self.app.root.after(0, lambda: messagebox.showerror("编码模式错误", error_msg))
                    self.app.root.after(0, self.app.hide_processing_window)
                    return
                # ========== RC1纠错编码处理结束 ==========
                
                # Watermark embedding
                self.app.root.after(0, lambda: self.app.show_processing_window("正在嵌入水印，请稍候..."))
                try:
                    bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                    bwm1.read_img(tmp_in)  # Read main image
                    bwm1.read_wm(bit_array, mode='bit')  # Read binary bit array
                    bwm1.embed(tmp_out)  # Embed watermark
                except Exception as e:
                    # If embedding fails, report error and return
                    self.app.root.after(0, lambda error_msg=e: messagebox.showerror("错误", f"水印嵌入失败: {str(error_msg)}\n请尝试使用更小的二进制文件或更大的主图片"))
                    return
                
                output_ext = ".jpg" if self.app.output_format.get() == "JPG" else ext  # Determine output file extension

                # 修改输出文件名，包含编码信息
                encoding_suffix = "-RC1" if use_rc1_encoding else "-RAW"
                dst_img = os.path.join(
                    os.path.dirname(filepath),
                    f"{name}-Watermark{encoding_suffix}-ws{wm_len}-size{width}x{height}{output_ext}"
                )
                if self.app.output_format.get() == "JPG":
                    # Create sRGB ICC profile
                    srgb_profile = ImageCms.createProfile("sRGB")
                    # Save JPG with ICC profile
                    img_to_save = Image.open(tmp_out).convert('RGB')
                    img_to_save.save(dst_img, "JPEG", quality=100, subsampling="4:2:0",
                                    icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())
                else:
                    # For non-JPG formats, directly copy the temporary output file to the target path
                    shutil.copy2(tmp_out, dst_img)
                # Ensure processing window is closed
                self.app.root.after(0, self.app.hide_processing_window)

                # 显示详细的成功信息
                original_size = len(binary_data) if 'binary_data' in locals() else 0
                # 计算容量利用率
                capacity_utilization = (wm_len / available_capacity) * 100 if 'available_capacity' in locals() and available_capacity > 0 else 0

                success_message = f"""嵌入成功！
    输出文件：{dst_img}

    编码信息：
    • 编码方式：{compression_info}
    • 原始数据：{original_size} 字节
    • 水印长度：{wm_len} 比特
    • 图像尺寸：{width}x{height}
    • 容量利用率：{capacity_utilization:.1f}%

    提示：{'使用RC1纠错编码可提供更好的抗噪性能' if use_rc1_encoding else '建议使用RC1编码以获得更好的鲁棒性'}"""
                
                self.app.root.after(0, lambda: messagebox.showinfo("嵌入成功", success_message))
                
            except Exception as e:
                # Catch other unforeseen errors and display them
                self.app.root.after(0, lambda e_val=e: messagebox.showerror("错误", str(e_val)))
            finally:
                # Regardless of success or failure, hide the processing window at the end
                self.app.root.after(0, self.app.hide_processing_window)
                # Clean up temporary files
                for f in [tmp_in, tmp_out, temp_img_main, tmp_watermark_file]:
                    if f and os.path.exists(f):
                        try:
                            os.remove(f)
                            print(f"已清理临时文件: {f}")
                        except Exception as cleanup_e:
                            print(f"清理临时文件失败 {f}: {cleanup_e}")  # Only print cleanup failure info
        # Start worker thread
        threading.Thread(target=worker, daemon=True).start()
        self.app.hide_processing_window()

    def embed_watermark(self, filepath):
        def worker():
            try:
                # 显示处理窗口
                self.app.root.after(0, lambda: self.app.show_processing_window("正在处理图片，请稍候..."))
                output_dir = self.app.get_output_dir()
                os.makedirs(output_dir, exist_ok=True)
                # 生成二维码水印
                # 先尝试128x128尺寸
                qr_path = self.generate_qr_watermark(128)
                if not qr_path:
                    return
                # 确保二维码文件存在
                if not os.path.exists(qr_path):
                    self.app.root.after(0, lambda: messagebox.showerror("错误", "二维码水印生成失败"))
                    return
                
                name, ext = os.path.splitext(os.path.basename(filepath))
                # 读取图片
                image = Image.open(filepath)
                width, height = image.size
                
                # 检查并转换JPG色彩空间(仅当输出格式为JPG时)
                if self.app.output_format.get() == "JPG":
                    if filepath.lower().endswith(('.jpg', '.jpeg')):
                        if image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0':
                            # 启动长时间操作提示
                            self.app.root.after(0, lambda: self.app.show_processing_window("正在转换色彩空间，请稍候..."))
                            
                            start_time = time.time()
                            image = image.convert('RGB')
                            
                            # 如果转换时间超过1秒，保持提示窗口
                            if time.time() - start_time > 1:
                                self.app.root.after(0, lambda: messagebox.showinfo("色彩空间转换", "为确保兼容性，已将图片色彩空间转换为sRGB 4:2:0"))
                            else:
                                self.app.root.after(0, self.app.hide_processing_window)
                            
                # 定义所有临时文件变量
                tmp_in = tempfile.NamedTemporaryFile(suffix=ext,delete=False).name
                tmp_out = tempfile.NamedTemporaryFile(suffix=ext,delete=False).name
                temp_img = None
                
                # 临时保存转换后的图片(仅当需要转换时)
                if self.app.output_format.get() == "JPG" and filepath.lower().endswith(('.jpg', '.jpeg')) and \
                (image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0'):
                    temp_img = os.path.join(os.path.dirname(filepath), "temp_converted.jpg")
                    image.save(temp_img, "JPEG", subsampling="4:2:0", quality=100)
                    image = Image.open(temp_img)
                    width, height = image.size
                
                # 保存输入图片并确保文件关闭
                with open(tmp_in, 'wb') as f:
                    image.save(f)
                
                # 确保二维码文件已关闭
                if os.path.exists(qr_path):
                    with open(qr_path, 'rb') as f:
                        pass  # 确保文件已关闭
                
                wm_text = self.app.get_wm_text()
                pwd = self.app.get_pwd()

                # 应用增强模式
                if self.app.enhanced_mode.get():
                    try:
                        # 读取临时文件
                        img = Image.open(tmp_in).convert("RGB") # 读图，转RGB模型
                        arr = np.array(img).astype(np.float32)  # 避免uint8溢出
                        print("Useing Enhanced Mode...")
                        # 生成2D柏林噪声
                        noise = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.float32)
                        for i in range(arr.shape[0]):
                            for j in range(arr.shape[1]):
                                noise[i][j] = pnoise2(i / 50.0, j / 50.0, octaves=2)

                        # 扩展为3D通道，应用到每个颜色通道
                        noise_3d = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
                        arr += noise_3d * 12.8  # 控制噪声强度
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                        # 回写文件
                        Image.fromarray(arr).save(tmp_in)
                    except Exception as e:
                        print(f"噪声处理失败: {e}")

                # 先尝试128x128二维码嵌入
                try:
                    bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                    bwm1.read_img(tmp_in)
                    bwm1.read_wm(qr_path)
                    bwm1.embed(tmp_out)
                except Exception as e:
                    # 如果失败，尝试64x64尺寸
                    if os.path.exists(qr_path):
                        os.remove(qr_path)
                    qr_path = self.generate_qr_watermark(64)
                    if not qr_path:
                        messagebox.showerror("错误", f"二维码生成失败: {str(e)}\n请尝试使用更小的水印尺寸")
                        return
                    
                    try:
                        bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                        bwm1.read_img(tmp_in)
                        bwm1.read_wm(qr_path)
                        bwm1.embed(tmp_out)
                    except Exception as e2:
                        messagebox.showerror("错误", f"水印嵌入失败: {str(e2)}\n请尝试使用更小的水印尺寸")
                        if os.path.exists(qr_path):
                            try:
                                os.unlink(qr_path)
                            except Exception as e:
                                print(f"删除临时文件失败: {e}")
                        return
                
                # 延迟清理临时文件
                if temp_img and os.path.exists(temp_img):
                    os.remove(temp_img)
                if qr_path and os.path.exists(qr_path):
                    os.remove(qr_path)

                wm_len = len(bwm1.wm_bit)
                output_ext = ".jpg" if self.app.output_format.get() == "JPG" else ext
                dst_img = os.path.join(
                    os.path.dirname(filepath),
                    f"{name}-Watermark-ws{wm_len}-size{width}x{height}{output_ext}"
                )
                
                if self.app.output_format.get() == "JPG":
                    # 创建sRGB ICC配置文件
                    srgb_profile = ImageCms.createProfile("sRGB")
                    # 保存带ICC配置的JPG
                    img = Image.open(tmp_out).convert('RGB')
                    img.save(dst_img, "JPEG", quality=100, subsampling="4:2:0", 
                            icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())
                else:
                    shutil.copy2(tmp_out, dst_img)
                # 确保处理窗口关闭
                self.app.root.after(0, self.app.hide_processing_window)
                self.app.root.after(0, lambda: messagebox.showinfo("嵌入成功", f"输出文件：\n{dst_img}\n\n水印长度：{wm_len} 尺寸：{width}x{height}"))
            except Exception as e:
                self.app.root.after(0, lambda e=e: messagebox.showerror("错误", str(e)))
            finally:
                self.app.root.after(0, self.app.hide_processing_window)
                for f in [tmp_in, tmp_out] if 'tmp_in' in locals() and 'tmp_out' in locals() else []:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except:
                            pass

        # 启动工作线程
        threading.Thread(target=worker, daemon=True).start()
        
        # 隐藏处理窗口
        self.app.hide_processing_window()

    def embed_watermark_v013(self, filepath):
        """旧版本兼容方法 - 使用文本水印而非二维码"""
        def worker():
            try:
                # 显示处理窗口
                self.app.root.after(0, lambda: self.app.show_processing_window("正在处理图片，请稍候..."))
                output_dir = self.app.get_output_dir()
                os.makedirs(output_dir, exist_ok=True)

                name, ext = os.path.splitext(os.path.basename(filepath))
                # 读取图片
                image = Image.open(filepath)
                width, height = image.size
                
                # 检查并转换JPG色彩空间(仅当输出格式为JPG时)
                if self.app.output_format.get() == "JPG":
                    if filepath.lower().endswith(('.jpg', '.jpeg')):
                        if image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0':
                            # 启动长时间操作提示
                            self.app.root.after(0, lambda: self.app.show_processing_window("正在转换色彩空间，请稍候..."))
                            
                            start_time = time.time()
                            image = image.convert('RGB')
                            
                            # 如果转换时间超过1秒，保持提示窗口
                            if time.time() - start_time > 1:
                                self.app.root.after(0, lambda: messagebox.showinfo("色彩空间转换", "为确保兼容性，已将图片色彩空间转换为sRGB 4:2:0"))
                            else:
                                self.app.root.after(0, self.app.hide_processing_window)
                            
                # 定义所有临时文件变量
                tmp_in = tempfile.NamedTemporaryFile(suffix=ext,delete=False).name
                tmp_out = tempfile.NamedTemporaryFile(suffix=ext,delete=False).name
                temp_img = None
                
                # 临时保存转换后的图片(仅当需要转换时)
                if self.app.output_format.get() == "JPG" and filepath.lower().endswith(('.jpg', '.jpeg')) and \
                (image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0'):
                    temp_img = os.path.join(os.path.dirname(filepath), "temp_converted.jpg")
                    image.save(temp_img, "JPEG", subsampling="4:2:0", quality=100)
                    image = Image.open(temp_img)
                    width, height = image.size
                
                # 保存输入图片
                image.save(tmp_in)
                
                # 清理临时文件
                if temp_img and os.path.exists(temp_img):
                    os.remove(temp_img)                
                wm_text = self.app.get_wm_text()
                pwd = self.app.get_pwd()

                bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                bwm1.read_img(tmp_in)
                bwm1.read_wm(wm_text, mode='str')
                bwm1.embed(tmp_out)

                wm_len = len(bwm1.wm_bit)
                output_ext = ".jpg" if self.app.output_format.get() == "JPG" else ext
                dst_img = os.path.join(
                    os.path.dirname(filepath),
                    f"{name}-Watermark-ws{wm_len}-size{width}x{height}{output_ext}"
                )
                
                if self.app.output_format.get() == "JPG":
                    # 创建sRGB ICC配置文件
                    srgb_profile = ImageCms.createProfile("sRGB")
                    # 保存带ICC配置的JPG
                    img = Image.open(tmp_out).convert('RGB')
                    img.save(dst_img, "JPEG", quality=100, subsampling="4:2:0", 
                            icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())
                else:
                    shutil.copy2(tmp_out, dst_img)
                # 确保处理窗口关闭
                self.app.root.after(0, self.app.hide_processing_window)
                self.app.root.after(0, lambda: messagebox.showinfo("嵌入成功", f"输出文件：\n{dst_img}\n\n【旧版水印！请完善保存以下内容！】\n水印长度：{wm_len} 尺寸：{width}x{height}"))
            except Exception as e:
                self.app.root.after(0, lambda e=e: messagebox.showerror("错误", str(e)))
            finally:
                self.app.root.after(0, self.app.hide_processing_window)
                for f in [tmp_in, tmp_out]:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except:
                            pass

        # 启动工作线程
        threading.Thread(target=worker, daemon=True).start()
        
        # 删除临时转换的图片
        if 'temp_img' in locals() and os.path.exists(temp_img):
            os.remove(temp_img)
        
        # 隐藏处理窗口
        self.app.hide_processing_window()

#====================3.x QR Embed Funcs========================

    def check_qr_capacity(self, text, version, error_correction):
        """检查QR码容量是否足够"""
        
        # QR码容量表 (字节)
        capacity_table = {
            (1, 'L'): 17, (1, 'M'): 14, (1, 'Q'): 11, (1, 'H'): 7,
            (2, 'L'): 32, (2, 'M'): 26, (2, 'Q'): 20, (2, 'H'): 14,
            (3, 'L'): 53, (3, 'M'): 42, (3, 'Q'): 32, (3, 'H'): 24,
            (4, 'L'): 78, (4, 'M'): 62, (4, 'Q'): 46, (4, 'H'): 34,
            (5, 'L'): 106, (5, 'M'): 84, (5, 'Q'): 60, (5, 'H'): 44,
            (6, 'L'): 134, (6, 'M'): 106, (6, 'Q'): 74, (6, 'H'): 58,
        }
        
        max_capacity = capacity_table.get((version, error_correction), 0)
        text_bytes = len(text.encode('utf-8'))
        
        return text_bytes <= max_capacity


    def embed_watermark_qr_direct(self, filepath):
        """
        使用QR码的直接水印嵌入方法
        流程：原图 → 取得水印文字 → 自适应QRCode生成 → 容量检查 → 直接bit模式嵌入
        
        Args:
            filepath (str): 主图片的文件路径
        """
        def worker():
            tmp_in = None
            tmp_out = None
            temp_img_main = None
            
            try:
                # 显示处理窗口
                self.app.root.after(0, lambda: self.app.show_processing_window("正在初始化，请稍候..."))
                
                # 获取水印文字
                watermark_text = self.app.get_wm_text()
                
                # 准备输出目录和文件名
                output_dir = self.app.get_output_dir()
                os.makedirs(output_dir, exist_ok=True)
                name, ext = os.path.splitext(os.path.basename(filepath))
                
                # 读取主图片
                self.app.root.after(0, lambda: self.app.show_processing_window("正在读取主图片，请稍候..."))
                image = Image.open(filepath)
                width, height = image.size
                print(f"主图片尺寸: {width}x{height}")
                
                # JPG色彩空间处理
                if self.app.output_format.get() == "JPG":
                    if filepath.lower().endswith(('.jpg', '.jpeg')):
                        if image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0':
                            self.app.root.after(0, lambda: self.app.show_processing_window("正在转换主图片色彩空间，请稍候..."))
                            start_time = time.time()
                            image = image.convert('RGB')
                            if time.time() - start_time > 1:
                                self.app.root.after(0, lambda: messagebox.showinfo("色彩空间转换", "为确保兼容性，已将主图片色彩空间转换为sRGB 4:2:0"))
                
                # 创建临时文件
                tmp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                tmp_out = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                
                # 处理主图片转换（如需要）
                if self.app.output_format.get() == "JPG" and filepath.lower().endswith(('.jpg', '.jpeg')) and \
                        (image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0'):
                    temp_img_main = os.path.join(os.path.dirname(filepath), "temp_converted_main_image.jpg")
                    image.save(temp_img_main, "JPEG", subsampling="4:2:0", quality=100)
                    image = Image.open(temp_img_main)
                    width, height = image.size
                
                # 保存临时输入文件
                with open(tmp_in, 'wb') as f:
                    image.save(f)
                
                # 获取密码
                pwd = self.app.get_pwd()
                
                # 增强模式处理
                if self.app.enhanced_mode.get():
                    try:
                        self.app.root.after(0, lambda: self.app.show_processing_window("正在应用增强模式，请稍候..."))
                        img = Image.open(tmp_in).convert("RGB")
                        arr = np.array(img).astype(np.float32)
                        print("Using Enhanced Mode...")
                        
                        # 生成2D柏林噪声
                        noise = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.float32)
                        for i in range(arr.shape[0]):
                            for j in range(arr.shape[1]):
                                noise[i][j] = pnoise2(i / 50.0, j / 50.0, octaves=2)
                        
                        # 扩展到3D通道
                        noise_3d = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
                        arr += noise_3d * 12.8
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                        Image.fromarray(arr).save(tmp_in)
                    except Exception as e:
                        print(f"增强模式处理失败: {e}")
                
                # ========== 获取图像可用容量开始 ==========
                self.app.root.after(0, lambda: self.app.show_processing_window("正在计算图像容量，请稍候..."))
                
                try:
                    bwm_temp = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                    bwm_temp.read_img(tmp_in)
                    
                    try:
                        bwm_temp.bwm_core.init_block_index()
                        available_capacity = bwm_temp.bwm_core.block_num
                        capacity_method = "精确计算"
                    except AttributeError:
                        available_capacity = int(width * height * 0.25)
                        capacity_method = "估算"
                    
                    print(f"图像可用容量({capacity_method}): {available_capacity} 比特")
                    
                except Exception as e:
                    print(f"获取容量信息失败: {e}")
                    available_capacity = int(width * height * 0.25)
                    capacity_method = "基本估算"
                    print(f"使用基本估算容量: {available_capacity} 比特")
                # ========== 获取图像可用容量结束 ==========
                
                # ========== 自适应QR码生成开始 ==========
                self.app.root.after(0, lambda: self.app.show_processing_window("正在生成自适应QR码，请稍候..."))
                
                # 自适应选择QR码参数，根据可用容量
                qr_versions = [1, 2, 3, 4, 5, 6]  # 支持的QR码版本
                error_corrections = ['L', 'M', 'Q', 'H']  # 纠错等级，从低到高
                
                bit_array = None
                qr_config = None
                selected_params = None
                
                # 计算合适的QR码尺寸，确保不超过容量
                max_qr_size = int(available_capacity ** 0.5)  # 最大可能的正方形尺寸
                print(f"根据容量计算的最大QR码尺寸: {max_qr_size}×{max_qr_size}")
                
                # 尝试不同的QR码参数组合，找到合适的就停止
                best_bit_array = None
                best_qr_config = None
                best_selected_params = None
                best_utilization = 0
                target_utilization = 90.0  # 目标利用率90%，达到就停止
                
                for version in qr_versions:
                    if best_utilization >= target_utilization:
                        print(f"已达到目标利用率 {target_utilization}%，停止搜索")
                        break
                        
                    for error_correction in error_corrections:
                        if self.check_qr_capacity(watermark_text, version, error_correction):
                            # 计算这个版本的理论尺寸
                            qr_modules = (version - 1) * 4 + 21
                            
                            print(f"尝试 V{version}+{error_correction}: QR模块={qr_modules}×{qr_modules}")
                            
                            # 计算合适的目标尺寸：在容量范围内尽可能大
                            target_bits = int(available_capacity * target_utilization)
                            target_size = int(target_bits ** 0.5)
                            
                            # 确保不小于QR码的最小尺寸
                            min_size = qr_modules + 2  # 最小尺寸
                            target_size = max(target_size, min_size)
                            target_size = min(target_size, max_qr_size)  # 不超过最大尺寸
                            
                            print(f"目标尺寸: {target_size}×{target_size} (目标比特数: {target_bits})")
                            
                            # 生成QR码
                            bit_array, qr_config = self.generate_watermark_qr_bit_direct(
                                watermark_text, size=target_size, version=version, error_correction=error_correction
                            )
                            
                            if bit_array is not None:
                                qr_bit_count = len(bit_array)
                                utilization = (qr_bit_count / available_capacity) * 100
                                
                                print(f"生成结果: V{version}+{error_correction}, 实际尺寸: {qr_config['size']}×{qr_config['size']}")
                                print(f"比特数: {qr_bit_count}, 容量利用率: {utilization:.1f}%")
                                
                                if qr_bit_count <= available_capacity:
                                    # 如果容量利用率更高，则选择这个
                                    if utilization > best_utilization:
                                        best_bit_array = bit_array
                                        best_qr_config = qr_config
                                        best_selected_params = f"V{version}+{error_correction}"
                                        best_utilization = utilization
                                        print(f"更新最佳选择: {best_selected_params}, 利用率: {best_utilization:.1f}%")
                                        
                                        # 如果达到目标利用率，提前停止
                                        if best_utilization >= target_utilization:
                                            print(f"达到目标利用率 {target_utilization}%，提前结束搜索")
                                            break
                                else:
                                    print(f"超出容量，跳过")
                        
                        # 如果内层循环提前结束，外层也要结束
                        if best_utilization >= target_utilization:
                            break
                
                # 使用最佳选择
                if best_bit_array is not None:
                    bit_array = best_bit_array
                    qr_config = best_qr_config
                    selected_params = best_selected_params
                    print(f"最终选择: {selected_params}, 尺寸: {qr_config['size']}×{qr_config['size']}, 利用率: {best_utilization:.1f}%")
                else:
                    bit_array = None
                
                if bit_array is None:
                    error_msg = f"无法为文字 '{watermark_text}' 生成适合容量的QR码\n\n可用容量: {available_capacity} 比特\n\n请尝试：\n1. 使用更短的水印文字\n2. 使用更大的图片"
                    self.app.root.after(0, lambda: messagebox.showerror("QR码生成失败", error_msg))
                    self.app.root.after(0, self.app.hide_processing_window)
                    return
                
                wm_len = len(bit_array)
                print(f"最终QR码生成成功: {wm_len} 比特")
                # ========== 自适应QR码生成结束 ==========
                
                # ========== 水印嵌入开始 ==========
                self.app.root.after(0, lambda: self.app.show_processing_window("正在嵌入QR码水印，请稍候..."))
                
                try:
                    bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                    bwm1.read_img(tmp_in)
                    bwm1.read_wm(bit_array, mode='bit')
                    bwm1.embed(tmp_out)
                except Exception as e:
                    self.app.root.after(0, lambda error_msg=e: messagebox.showerror("错误", f"水印嵌入失败: {str(error_msg)}\n请尝试使用更短的水印文字或更大的主图片"))
                    return
                # ========== 水印嵌入结束 ==========
                
                # 保存输出文件
                output_ext = ".jpg" if self.app.output_format.get() == "JPG" else ext
                
                qr_info = f"QR{selected_params}" if selected_params else "QR"
                dst_img = os.path.join(
                    os.path.dirname(filepath),
                    f"{name}-Watermark3-{qr_info}-ws{wm_len}-{width}x{height}{output_ext}"
                )
                
                if self.app.output_format.get() == "JPG":
                    srgb_profile = ImageCms.createProfile("sRGB")
                    img_to_save = Image.open(tmp_out).convert('RGB')
                    img_to_save.save(dst_img, "JPEG", quality=100, subsampling="4:2:0",
                                    icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())
                else:
                    shutil.copy2(tmp_out, dst_img)
                
                # 隐藏处理窗口
                self.app.root.after(0, self.app.hide_processing_window)
                
                # 显示成功信息
                capacity_utilization = (wm_len / available_capacity) * 100 if available_capacity > 0 else 0
                
                success_message = f"""QR码水印嵌入成功！

    输出文件：{dst_img}

    水印信息：
    • QR码参数：{selected_params} ({qr_config['size']}×{qr_config['size']})
    • 编码方式：Bit嵌入

    数据统计：
    • QR码比特数：{wm_len} 比特
    • 图像尺寸：{width}×{height}
    • 可用容量：{available_capacity} 比特
    • 容量利用率：{capacity_utilization:.1f}%

    优势：
    • 无编码开销，数据量最小
    • 1-bit黑白图像，质量最佳
    • 快速嵌入和提取

    提示：使用相同密码可以从图片中提取QR码"""
                
                self.app.root.after(0, lambda: messagebox.showinfo("嵌入成功", success_message))
                
            except Exception as e:
                self.app.root.after(0, lambda e_val=e: messagebox.showerror("错误", f"处理过程中发生错误：{str(e_val)}"))
            finally:
                # 清理临时文件
                self.app.root.after(0, self.app.hide_processing_window)
                for f in [tmp_in, tmp_out, temp_img_main]:
                    if f and os.path.exists(f):
                        try:
                            os.remove(f)
                            print(f"已清理临时文件: {f}")
                        except Exception as cleanup_e:
                            print(f"清理临时文件失败 {f}: {cleanup_e}")
        
        # 启动工作线程
        threading.Thread(target=worker, daemon=True).start()
        self.app.hide_processing_window()


    def generate_watermark_qr_bit_direct(self, text, size=128, version=4, error_correction='H'):
        """生成1-bit黑白QR码并转换为bit数组，用于直接bit模式嵌入
    
        Args:
            text: 要编码的文本
            size: QR码目标尺寸
            version: QR码版本
            error_correction: 纠错等级
    
        Returns:
            tuple: (bit_array_list, qr_config) 或 (None, None)
        """
    
        error_correction_map = {
            'L': qrcode.constants.ERROR_CORRECT_L,
            'M': qrcode.constants.ERROR_CORRECT_M,
            'Q': qrcode.constants.ERROR_CORRECT_Q,
            'H': qrcode.constants.ERROR_CORRECT_H
        }
    
        try:
            print(f"生成直接嵌入QR码: {size}×{size}, V{version}+{error_correction}")
        
            # 检查QR码容量
            if not self.check_qr_capacity(text, version, error_correction):
                print(f"文本超出V{version}+{error_correction}容量限制")
                return None, None
            
            # 计算QR码的实际模块尺寸
            qr_modules = (version - 1) * 4 + 21
            
            # 计算最佳的box_size和border组合
            best_box_size = 1
            best_border = 1
            best_actual_size = 0
            
            # 尝试不同的border值
            for border in [1, 2, 3, 4]:
                total_modules = qr_modules + 2 * border
                box_size = size // total_modules
                
                if box_size >= 1:
                    actual_size = total_modules * box_size
                    if abs(actual_size - size) < abs(best_actual_size - size):
                        best_box_size = box_size
                        best_border = border
                        best_actual_size = actual_size
            
            print(f"QR参数: 模块={qr_modules}, box_size={best_box_size}, border={best_border}, 实际尺寸={best_actual_size}")
        
            # 生成QR码
            qr = qrcode.QRCode(
                version=version,
                error_correction=error_correction_map[error_correction],
                box_size=best_box_size,
                border=best_border,
            )
        
            qr.add_data(text)
            qr.make(fit=True)
        
            # 生成1-bit黑白图像
            img = qr.make_image(fill_color="black", back_color="white")
            
            print(f"QR码生成: 尺寸={img.size}, 模式={img.mode}")
            
            # 确保是1-bit模式
            if img.mode != '1':
                print(f"转换图像模式从 {img.mode} 到 1")
                img = img.convert('1')  # 转换为1-bit黑白图像
            
            actual_width, actual_height = img.size
            
            # 转换为numpy数组
            img_array = np.array(img, dtype=bool)
            print(f"numpy数组: 形状={img_array.shape}, 数据类型={img_array.dtype}")
        
            # 对于1-bit图像，True=白色，False=黑色
            # 我们需要黑色=True，所以要反转
            bit_array = ~img_array  # 反转：黑色=True，白色=False
        
            # 验证转换结果
            print(f"True像素(黑色): {np.sum(bit_array)}")
            print(f"False像素(白色): {np.sum(~bit_array)}")
        
            # 展平为一维数组并转为Python列表
            bit_array_1d = bit_array.flatten().tolist()
        
            # 配置信息
            qr_config = {
                'size': actual_width,  # 使用实际尺寸
                'version': version,
                'error_correction': error_correction,
                'bit_count': len(bit_array_1d),
                'box_size': best_box_size,
                'border': best_border,
                'qr_modules': qr_modules,
                'mode': '1-bit'
            }
        
            print(f"QR码bit数组生成成功:")
            print(f"  实际尺寸: {actual_width}×{actual_height}")
            print(f"  位数: {len(bit_array_1d):,}位")
            print(f"  黑色模块: {sum(bit_array_1d):,}个 ({sum(bit_array_1d)/len(bit_array_1d)*100:.1f}%)")
            
            # 保存调试图像（目前注释掉不保存）
            # debug_path = f"DEBUG_QR_Direct_{actual_width}x{actual_height}_V{version}+{error_correction}.png"
            # img.save(debug_path)
            # print(f"调试：QR码已保存到 {debug_path}")
        
            return bit_array_1d, qr_config
        
        except Exception as e:
            print(f"QR码bit数组生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None