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
                tmp_in = os.path.join(output_dir, f"input{ext}")
                tmp_out = os.path.join(output_dir, f"output{ext}")
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
                tmp_in = os.path.join(output_dir, f"input{ext}")
                tmp_out = os.path.join(output_dir, f"output{ext}")
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