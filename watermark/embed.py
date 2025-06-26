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
from .dataShielder import ReedSolomonEncoder


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
                version=1,
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
                            print(f"清理临时文件失败 {f}: {cleanup_e}") # 仅打印清理失败信息

        # 启动工作线程
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
                self.app.root.after(0, lambda: messagebox.showerror("错误", str(e)))
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



    def embed_watermark_v3(self, filepath):
        """新版本方法 - 集成鲁棒二进制信息嵌入编码器"""
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
                    
                # 获取预嵌入文本
                wm_text = self.app.get_wm_text()
                # 取得密码
                pwd = self.app.get_pwd()
                
                # ===== 集成鲁棒二进制信息嵌入编码器 =====
                self.app.root.after(0, lambda: self.app.show_processing_window("正在编码水印数据..."))
                
                try:
                    # 创建WaterMark实例来获取容量信息
                    bwm_temp = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                    bwm_temp.read_img(tmp_in)
                    
                    # 假设WaterMark有一个方法可以获取可用容量，如果没有则使用估算
                    # 这里需要根据实际的WaterMark类实现来调整
                    try:
                        # 尝试获取实际可用容量
                        available_capacity = getattr(bwm_temp, 'block_num', None)
                        if available_capacity is None:
                            # 如果没有block_num属性，进行估算
                            # 通常水印容量与图片大小相关
                            estimated_capacity = (width * height) // 64  # 粗略估算
                            available_capacity = estimated_capacity
                            print(f"估算可用容量: {available_capacity} 比特")
                        else:
                            print(f"实际可用容量: {available_capacity} 比特")
                    except Exception as e:
                        # 如果获取容量失败，使用保守估算
                        estimated_capacity = min(10000, (width * height) // 64)
                        available_capacity = estimated_capacity
                        print(f"使用保守估算容量: {available_capacity} 比特 (原因: {e})")
                    
                    # 使用鲁棒编码器编码文本
                    from .dataShielder import (
                        adapt_to_watermark_capacity, 
                        get_encoding_stats,
                        print_encoding_report
                    )
                    
                    # 获取编码统计信息
                    stats = get_encoding_stats(wm_text)
                    print(f"原始文本: '{wm_text}' ({stats['original_size']} 字节)")
                    print(f"压缩后: {stats['compressed_size']} 字节")
                    print(f"需要最小容量: {stats['min_capacity_bits']} 比特")
                    
                    # 根据可用容量自动适配编码
                    encoded_watermark_bits = adapt_to_watermark_capacity(
                        wm_text, 
                        available_capacity,
                        safety_margin=0.98  # 使用98%的可用容量，塞满它逝世
                    )
                    
                    print(f"编码完成，生成 {len(encoded_watermark_bits)} 比特的水印数据")
                    
                    # 显示编码报告（可选，用于调试）
                    if len(wm_text) > 10:  # 只对较长文本显示详细报告
                        print_encoding_report(wm_text)
                    
                except ImportError:
                    # 如果无法导入编码器，回退到简单UTF-8编码
                    self.app.root.after(0, lambda: messagebox.showwarning(
                        "编码器未找到", 
                        "鲁棒编码器不可用，将使用简单UTF-8编码。\n建议安装: pip install reedsolo numpy crc"
                    ))
                    
                    # 简单的UTF-8转比特编码作为回退
                    utf8_bytes = wm_text.encode('utf-8')
                    encoded_watermark_bits = []
                    for byte in utf8_bytes:
                        for bit in range(8):
                            encoded_watermark_bits.append(bool((byte >> (7 - bit)) & 1))
                    
                    print(f"使用简单UTF-8编码，生成 {len(encoded_watermark_bits)} 比特")
                    
                except Exception as e:
                    # 编码失败时的处理
                    error_msg = f"编码失败: {str(e)}"
                    print(error_msg)
                    self.app.root.after(0, lambda: messagebox.showerror("编码错误", error_msg))
                    return
                
                # 检查编码后的数据是否为空
                if not encoded_watermark_bits:
                    self.app.root.after(0, lambda: messagebox.showerror("编码错误", "编码后的水印数据为空"))
                    return
                
                # 限制最大长度以防止嵌入时间过长
                max_bits = min(len(encoded_watermark_bits), available_capacity if available_capacity else 50000)
                if len(encoded_watermark_bits) > max_bits:
                    encoded_watermark_bits = encoded_watermark_bits[:max_bits]
                    print(f"水印数据被截断到 {max_bits} 比特以优化性能")
                
                # ===== 嵌入编码后的水印 =====
                self.app.root.after(0, lambda: self.app.show_processing_window("正在嵌入水印..."))
                
                bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
                bwm1.read_img(tmp_in)
                bwm1.read_wm(encoded_watermark_bits, mode='bit')  # 使用编码后的比特列表
                bwm1.embed(tmp_out)

                wm_len = len(bwm1.wm_bit)
                output_ext = ".jpg" if self.app.output_format.get() == "JPG" else ext
                
                # 生成包含编码信息的文件名
                rs_info = "RS" if 'adapt_to_watermark_capacity' in locals() else "UTF8"
                dst_img = os.path.join(
                    os.path.dirname(filepath),
                    f"{name}-Watermark-{rs_info}-ws{wm_len}-size{width}x{height}{output_ext}"
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
                    
                # 准备成功消息
                encoding_type = "Reed-Solomon编码" if 'adapt_to_watermark_capacity' in locals() else "简单UTF-8编码"
                
                # 计算冗余倍数
                original_utf8_bits = len(wm_text.encode('utf-8')) * 8
                redundancy_ratio = wm_len / original_utf8_bits if original_utf8_bits > 0 else 1
                
                success_message = f"""嵌入成功！

输出文件：
{dst_img}

编码信息：
• 编码方式：{encoding_type}
• 原始文本：{len(wm_text)} 字符
• 水印长度：{wm_len} 比特
• 冗余倍数：{redundancy_ratio:.1f}x
• 图片尺寸：{width}x{height}

{'✓ 具备强抗干扰能力' if 'adapt_to_watermark_capacity' in locals() else '⚠ 建议安装鲁棒编码器以提高抗干扰能力'}"""
                
                # 确保处理窗口关闭
                self.app.root.after(0, self.app.hide_processing_window)
                self.app.root.after(0, lambda: messagebox.showinfo("嵌入成功", success_message))
                
            except Exception as e:
                error_msg = f"处理过程中发生错误: {str(e)}"
                print(error_msg)
                self.app.root.after(0, lambda: messagebox.showerror("错误", error_msg))
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


    # ===== 辅助函数 =====

    def estimate_watermark_capacity(width, height, compression_factor=64):
        """
        估算图片的水印容量
        
        参数:
            width: 图片宽度
            height: 图片高度  
            compression_factor: 压缩因子，值越大容量越小
            
        返回:
            估算的比特容量
        """
        # 基于图片像素数量的粗略估算
        # 实际容量取决于具体的水印算法
        pixel_count = width * height
        estimated_bits = pixel_count // compression_factor
        
        # 设置合理的上下限
        min_capacity = 1000     # 至少1000比特
        max_capacity = 100000   # 最多100000比特
        
        return max(min_capacity, min(estimated_bits, max_capacity))


    def get_watermark_instance_capacity(watermark_instance):
        """
        尝试从WaterMark实例获取实际容量
        
        参数:
            watermark_instance: WaterMark类的实例
            
        返回:
            容量（比特数），如果无法获取则返回None
        """
        # 尝试不同的可能属性名
        capacity_attrs = ['block_num', 'capacity', 'max_bits', 'available_bits']
        
        for attr in capacity_attrs:
            if hasattr(watermark_instance, attr):
                try:
                    capacity = getattr(watermark_instance, attr)
                    if isinstance(capacity, (int, float)) and capacity > 0:
                        return int(capacity)
                except:
                    continue
        
        return None


    def validate_encoded_watermark(encoded_bits, min_length=8, max_length=200000):
        """
        验证编码后的水印数据
        
        参数:
            encoded_bits: 编码后的比特列表
            min_length: 最小长度
            max_length: 最大长度
            
        返回:
            (is_valid, message)
        """
        if not encoded_bits:
            return False, "编码后的水印数据为空"
        
        if not isinstance(encoded_bits, list):
            return False, "水印数据格式错误，应为列表"
        
        if len(encoded_bits) < min_length:
            return False, f"水印数据太短，至少需要{min_length}比特"
        
        if len(encoded_bits) > max_length:
            return False, f"水印数据太长，最多支持{max_length}比特"
        
        # 检查数据类型
        if not all(isinstance(bit, bool) for bit in encoded_bits[:100]):  # 只检查前100个
            return False, "水印数据应为布尔值列表"
        
        return True, "水印数据验证通过"
