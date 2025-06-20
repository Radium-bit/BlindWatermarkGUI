## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

# 程序版本号
VERSION = r"0.2.0"
# 开发模型路径
DEV_MODEL_PATH = r""
# SitePackagePath
SITE_PACKAGE_PATH = r""

import os
import sys
from dotenv import load_dotenv
import shutil
import tkinter as tk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import tempfile
import re
import json
from pathlib import Path
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
from blind_watermark import WaterMark
import webbrowser
import time
import threading
import qrcode
from io import BytesIO
from pyzbar.pyzbar import decode



class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        # 设置模型路径环境变量（必须在导入qreader前执行）
        self.set_model_path()
        # 初始化QReader
        from debug.module_tracker import start_tracking ##Module import Tracker(Debug USE)
        ## [DEBUG] 用于测试记录QReader所使用的隐藏导入模块
        # start_tracking()
        from qreader import QReader
        self.qreader = QReader()
        self.processing_window = None
        self.processing_label = None
        self.processing_animation = None
        self.processing_active = False
        self.config_path = os.path.join(os.environ["USERPROFILE"], "radiumbit.blindwatermark.config.json")
        self._saved_before_close = False
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title(f"BlindWatermarkGUI v{VERSION}")
        self.geometry("580x560")
        self.configure(bg="white")
        self.qr_window = None

        self.mode = tk.StringVar(value="embed")

        # 模式选择
        frm_mode = tk.Frame(self, bg="white")
        frm_mode.pack(pady=5)
        tk.Label(frm_mode, text="选择模式：", bg="white").pack(side="left")
        tk.Radiobutton(frm_mode, text="嵌入水印", variable=self.mode, value="embed", bg="white").pack(side="left")
        tk.Radiobutton(frm_mode, text="提取水印", variable=self.mode, value="extract", bg="white").pack(side="left")

        # 输出格式选择
        self.output_format = tk.StringVar(value="PNG")
        format_frame = tk.Frame(self, bg="white")
        format_frame.pack(pady=5, fill="x", padx=20)
        tk.Label(format_frame, text="输出格式:", bg="white").pack(side="left")
        tk.Radiobutton(format_frame, text="PNG", variable=self.output_format, value="PNG", bg="white").pack(side="left", padx=5)
        tk.Radiobutton(format_frame, text="JPG", variable=self.output_format, value="JPG", bg="white").pack(side="left", padx=5)
        
        # 密码输入
        frm_pwd = tk.Frame(self, bg="white")
        frm_pwd.pack(pady=5, fill="x", padx=20)
        tk.Label(frm_pwd, text="密码（可空，默认为1234）：", bg="white").pack(side="left")
        self.entry_pwd = tk.Entry(frm_pwd, show="*")
        self.entry_pwd.insert(0, "")
        self.entry_pwd.pack(side="left", fill="x", expand=True)

        # 水印文本输入
        frm_wm = tk.Frame(self, bg="white")
        frm_wm.pack(pady=5, fill="both", padx=20)
        tk.Label(frm_wm, text="水印文本（仅嵌入时有效）：", bg="white").pack(anchor="w")
        self.text_wm = tk.Text(frm_wm, height=4)
        self.text_wm.pack(fill="both", expand=True)
        self.load_config()

        # 水印长度输入
        frm_len = tk.Frame(self, bg="white")
        frm_len.pack(pady=5, fill="x", padx=20)
        tk.Label(frm_len, text="提取水印长度（可空）：", bg="white").pack(side="left")
        self.entry_ws = tk.Entry(frm_len)
        self.entry_ws.insert(0, "")
        self.entry_ws.pack(side="left", fill="x", expand=True)

        # 原图尺寸输入
        frm_size = tk.Frame(self, bg="white")
        frm_size.pack(pady=5, fill="x", padx=20)
        tk.Label(frm_size, text="原图尺寸（如1920x1080，可空）：", bg="white").pack(side="left")
        self.entry_size = tk.Entry(frm_size)
        self.entry_size.insert(0, "")
        self.entry_size.pack(side="left", fill="x", expand=True)

        # 输出目录
        frm_out = tk.Frame(self, bg="white")
        frm_out.pack(pady=5, fill="x", padx=20)
        tk.Label(frm_out, text="输出目录（默认系统临时目录）：", bg="white").pack(side="left")
        self.entry_out = tk.Entry(frm_out)
        self.entry_out.insert(0, tempfile.gettempdir())
        self.entry_out.pack(side="left", fill="x", expand=True)
        
        # 重置配置按钮
        frm_reset = tk.Frame(self, bg="white")
        frm_reset.pack(pady=5, fill="x", padx=20)
        tk.Button(frm_reset, text="重置配置", command=self.reset_config).pack(side="right")

        # 拖拽区域
        lbl = tk.Label(self, text="请将图片拖入此区域", bg="#f0f0f0", fg="black", relief="ridge", borderwidth=2, height=10)
        lbl.pack(expand=True, fill="both", padx=20, pady=20)
        
        # 二维码水印设置
        self.qr_frame = tk.Frame(self)
        self.qr_frame.pack(pady=10)
        
        tk.Label(self.qr_frame, text="水印文本:").pack(side=tk.LEFT)
        self.wm_text = tk.Entry(self.qr_frame, width=30)
        self.wm_text.pack(side=tk.LEFT, padx=5)
        lbl.drop_target_register(DND_FILES)
        lbl.dnd_bind('<<Drop>>', self.on_drop)
        
        # 项目地址链接
        self.project_address_label = tk.Label(self, text="项目地址:", font=("Arial", 10))
        self.project_address_label.pack(pady=(10,0))
        self.project_link = tk.Label(self, text="Radium-bit/BlindWatermarkGUI", fg="blue", cursor="hand2", font=("Arial", 10))
        self.project_link.pack(pady=(0,5))
        self.project_link.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/Radium-bit/BlindWatermarkGUI"))

    def on_drop(self, ev):
        for f in self.tk.splitlist(ev.data):
            f = f.strip('{}')
            try:
                if not os.path.exists(f):
                    messagebox.showerror("错误", "文件不存在或路径无效")
                    return
                    
                if self.mode.get() == "embed":
                    self.embed_watermark(f)
                else:
                    self.extract_watermark(f)
            except Exception as e:
                messagebox.showerror("错误", str(e))
                # 确保处理窗口关闭
                if hasattr(self, 'hide_processing_window'):
                    self.hide_processing_window()

    def get_pwd(self):
        pwd = self.entry_pwd.get().strip()
        return pwd if pwd else "1234"

    def get_wm_text(self):
        text = self.text_wm.get("1.0", "end").rstrip()
        self.save_config(text)
        return text
        
    def set_model_path(self):
        """设置qreader模型路径环境变量"""
        if getattr(sys, 'frozen', False):
            # 打包环境：使用临时解压目录
            base_path = sys._MEIPASS
            model_path = os.path.join(base_path, "qrdet", ".model", "qrdet-s.pt")
        else:
            # 开发环境：使用默认安装路径，从DOT.ENV加载
            if not os.path.exists('DEV.ENV'):
                print("警告：未找到DEV.ENV文件，请创建并配置路径")
            else:
                load_dotenv('DEV.ENV')
                envpath = os.getenv('SITE_PACKAGE_PATH')
                model_path = os.path.join(envpath, "qrdet", ".model", "qrdet-s.pt")
        
        # 关键！设置环境变量阻止下载
        os.environ["QRDET_MODEL_PATH"] = model_path
        print(f"模型路径已设置为: {model_path}")  # 调试用

    def load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.entry_pwd.delete(0, tk.END)
                    self.entry_pwd.insert(0, config.get("pwd", ""))
                    self.text_wm.delete("1.0", tk.END)
                    self.text_wm.insert("1.0", config.get("last_wm_text", "Copyright@\nAuthor@"))
            else:
                self.text_wm.delete("1.0", tk.END)
                self.text_wm.insert("1.0", "Copyright@\nAuthor@")
        except Exception as e:
            print(f"加载配置失败: {e}")
            self.text_wm.delete("1.0", tk.END)
            self.text_wm.insert("1.0", "Copyright@\nAuthor@")
            
    def save_config(self, wm_text):
        try:
            config = {
                "pwd": self.entry_pwd.get().strip(),
                "last_wm_text": wm_text
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"保存配置失败: {e}")
            
    def on_close(self):
        from debug.module_tracker import stop_and_save_tracking
        ## [DEBUG] 用于测试记录QReader所使用的隐藏导入模块
        # stop_and_save_tracking()
        if not self._saved_before_close:
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "pwd": self.entry_pwd.get().strip(),
                        "last_wm_text": self.text_wm.get("1.0", "end").rstrip()
                    }, f, ensure_ascii=False, indent=4)
                app._saved_before_close = True
            except Exception as e:
                print(f"保存配置时出错: {e}")
        self.destroy()
            
    def show_processing_window(self, message):
        """显示处理中窗口"""
        if not self.processing_window:
            self.processing_window = tk.Toplevel(self)
            self.processing_window.title("处理中")
            self.processing_window.geometry("300x100")
            self.processing_window.resizable(False, False)
            self.processing_window.protocol("WM_DELETE_WINDOW", lambda: None)  # 禁用关闭按钮
            
            self.processing_label = tk.Label(self.processing_window, text=message)
            self.processing_label.pack(pady=10)
            
            # 简单的动画效果
            self.processing_animation = tk.Label(self.processing_window, text="◐")
            self.processing_animation.pack()
            self.animate_processing()
            
        self.processing_active = True
    
    def hide_processing_window(self):
        """隐藏处理中窗口"""
        if self.processing_window:
            self.processing_active = False
            self.processing_window.destroy()
            self.processing_window = None
            
    def generate_qr_watermark(self, size=128):
        """生成二维码水印文件
        Args:size: 二维码尺寸 (默认128)
        """
        try:
            wm_text = self.text_wm.get("1.0", "end").strip()
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
            
    def show_qr_code(self, qr_path, text=None, status=None, *images):
        ## 显示二维码窗口
        if self.qr_window:
            self.qr_window.destroy()
            
        self.qr_window = tk.Toplevel(self)
        self.qr_window.title("水印提取结果")
        self.qr_window.geometry("600x700")
        
        # 清理文件回调
        def cleanup_callback():
            return True

        # 添加窗口关闭事件处理
        def on_close():
            if qr_path and os.path.exists(qr_path):
                try:
                    os.remove(qr_path)
                    return cleanup_callback
                except:
                    return cleanup_callback
                    pass
            self.qr_window.destroy()
            
        self.qr_window.protocol("WM_DELETE_WINDOW", on_close)
        
        # 显示状态标题
        if status==True:
            tk.Label(self.qr_window, text="水印提取成功", font=("Arial", 12, "bold")).pack()
        else:
            tk.Label(self.qr_window, text="水印提取失败，请检查下方是否存在二维码图像", font=("Arial", 12, "bold")).pack()
        
        # 处理传入的图片
        if images:
            for img_data in images:
                if img_data is not None:
                    size, img_array = img_data
                    # 将numpy数组转换回PIL Image
                    img = Image.fromarray(img_array)
                    # 对小于256的图片进行放大
                    if max(img.size) < 256:
                        scale = 256 / max(img.size)
                        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img_tk = ImageTk.PhotoImage(img)
                    label = tk.Label(self.qr_window, image=img_tk)
                    label.image = img_tk  # 保持引用
                    label.pack(pady=5)
                    # 显示尺寸标签
                    tk.Label(self.qr_window, text=f"尺寸: {size}").pack()
        elif qr_path:
            try:
                img = Image.open(qr_path)
                img_tk = ImageTk.PhotoImage(img)
                label = tk.Label(self.qr_window, image=img_tk)
                label.image = img_tk  # 保持引用
                label.pack(pady=5)
            except Exception as e:
                messagebox.showerror("错误", f"无法显示图片: {str(e)}")
        # 显示解码文本（如果有）
        if text:
            tk.Label(self.qr_window, text="解码文本:", font=("Arial", 10, "bold")).pack()
            text_label = tk.Label(self.qr_window, text=text, wraplength=380, justify="left")
            text_label.pack(pady=5)
        
        # 关闭按钮
        close_btn = tk.Button(self.qr_window, text="关闭", 
                            command=lambda: [os.unlink(qr_path) if qr_path and os.path.exists(qr_path) else None,
                            self.qr_window.destroy()])
        close_btn.pack(pady=10)
    
    def animate_processing(self):
        """处理动画效果"""
        if not self.processing_active:
            return
            
        frames = ["◐", "◓", "◑", "◒"]
        current_frame = self.processing_animation.cget("text")
        next_frame = frames[(frames.index(current_frame) + 1) % len(frames)]
        self.processing_animation.config(text=next_frame)
        self.after(200, self.animate_processing)
    
    def reset_config(self):
        try:
            if os.path.exists(self.config_path):
                os.remove(self.config_path)
                self.entry_pwd.delete(0, tk.END)
                self.text_wm.delete("1.0", tk.END)
                self.text_wm.insert("1.0", "Copyright@\nAuthor@")
                messagebox.showinfo("重置成功", "配置文件已删除，已恢复默认设置")
            else:
                messagebox.showinfo("提示", "配置文件不存在，无需重置")
        except Exception as e:
            messagebox.showerror("重置失败", f"重置配置时出错: {e}")

    def get_output_dir(self):
        path = self.entry_out.get().strip()
        if not path:
            path = tempfile.gettempdir()
        if not os.path.isdir(path):
            raise ValueError(f"输出目录不存在：{path}")
        return path

    def get_ws(self):
        val = self.entry_ws.get().strip()
        return int(val) if val.isdigit() else None

    def get_target_size(self):
        val = self.entry_size.get().strip()
        if re.match(r'^\d+x\d+$', val):
            w, h = map(int, val.lower().split('x'))
            return w, h
        return None

    def embed_watermark(self, filepath):
        def worker():
            try:
                # 显示处理窗口
                self.after(0, lambda: self.show_processing_window("正在处理图片，请稍候..."))
                output_dir = self.get_output_dir()
                os.makedirs(output_dir, exist_ok=True)
                # 生成二维码水印
                # 先尝试128x128尺寸
                qr_path = self.generate_qr_watermark(128)
                if not qr_path:
                    return
                # 确保二维码文件存在
                if not os.path.exists(qr_path):
                    self.after(0, lambda: messagebox.showerror("错误", "二维码水印生成失败"))
                    return
                
                name, ext = os.path.splitext(os.path.basename(filepath))
                # 读取图片
                image = Image.open(filepath)
                width, height = image.size
                
                # 检查并转换JPG色彩空间(仅当输出格式为JPG时)
                if self.output_format.get() == "JPG":
                    if filepath.lower().endswith(('.jpg', '.jpeg')):
                        if image.mode != 'RGB' or image.info.get('subsampling') != '4:2:0':
                            # 启动长时间操作提示
                            self.after(0, lambda: self.show_processing_window("正在转换色彩空间，请稍候..."))
                            
                            start_time = time.time()
                            image = image.convert('RGB')
                            
                            # 如果转换时间超过1秒，保持提示窗口
                            if time.time() - start_time > 1:
                                self.after(0, lambda: messagebox.showinfo("色彩空间转换", "为确保兼容性，已将图片色彩空间转换为sRGB 4:2:0"))
                            else:
                                self.after(0, self.hide_processing_window)
                            
                # 定义所有临时文件变量
                tmp_in = os.path.join(output_dir, f"input{ext}")
                tmp_out = os.path.join(output_dir, f"output{ext}")
                temp_img = None
                
                # 临时保存转换后的图片(仅当需要转换时)
                if self.output_format.get() == "JPG" and filepath.lower().endswith(('.jpg', '.jpeg')) and \
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
                
                wm_text = self.get_wm_text()
                pwd = self.get_pwd()

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
                output_ext = ".jpg" if self.output_format.get() == "JPG" else ext
                dst_img = os.path.join(
                    os.path.dirname(filepath),
                    f"{name}-Watermark-ws{wm_len}-size{width}x{height}{output_ext}"
                )
                
                if self.output_format.get() == "JPG":
                    from PIL import ImageCms
                    # 创建sRGB ICC配置文件
                    srgb_profile = ImageCms.createProfile("sRGB")
                    # 保存带ICC配置的JPG
                    img = Image.open(tmp_out).convert('RGB')
                    img.save(dst_img, "JPEG", quality=100, subsampling="4:2:0", 
                            icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())
                else:
                    shutil.copy2(tmp_out, dst_img)
                # 确保处理窗口关闭
                self.after(0, self.hide_processing_window)
                self.after(0, lambda: messagebox.showinfo("嵌入成功", f"输出文件：\n{dst_img}\n\n【请完善保存以下内容！】\n水印长度：{wm_len} 尺寸：{width}x{height}"))
            except Exception as e:
                self.after(0, lambda e=e: messagebox.showerror("错误", str(e)))
            finally:
                self.after(0, self.hide_processing_window)
                for f in [tmp_in, tmp_out] if 'tmp_in' in locals() and 'tmp_out' in locals() else []:
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
        self.hide_processing_window()



    def extract_watermark(self, filepath):
        def worker():
            try:
                self.after(0, lambda: self.show_processing_window("正在提取水印，请稍候..."))
                
                # 创建临时文件
                tmp_in = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                tmp_out = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                
                # 保存图片到临时文件
                img = Image.open(filepath)
                img.save(tmp_in)
                
                # 尝试128x128尺寸提取
                pwd = self.get_pwd()
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
                    qreader = self.qreader
                    print(f'128x128临时文件路径: 输入={tmp_in} 输出={tmp_out}')
                    if img_128.mode != 'RGB':
                        img_128 = img_128.convert('RGB')
                    img_array = np.array(img_128)
                    text = qreader.detect_and_decode(image=img_array)[0]
                    
                    # 如果第一次解析失败，尝试增强解析
                    if not text:
                        # 1. 尝试调整对比度和亮度
                        enhancer = ImageEnhance.Contrast(img_128)
                        img_128 = enhancer.enhance(2.0)
                        enhancer = ImageEnhance.Brightness(img_128)
                        img_128 = enhancer.enhance(1.5)
                        
                        # 2. 应用中值滤波去噪
                        img_128 = img_128.filter(ImageFilter.MedianFilter(size=3))
                        
                        # 3. 转换为灰度并应用自适应阈值
                        img_128 = img_128.convert('L')
                        img_128 = img_128.point(lambda x: 0 if x < 128 else 255, '1')
                        
                        # 4. 重新尝试解码
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
                        qreader = self.qreader
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
                        if not text:
                            print("No Text at try1")
                            # 1. 尝试调整对比度和亮度
                            enhancer = ImageEnhance.Contrast(img_64)
                            img_64 = enhancer.enhance(2.0)
                            enhancer = ImageEnhance.Brightness(img_64)
                            img_64 = enhancer.enhance(1.5)
                            # 2. 应用中值滤波去噪
                            img_64 = img_64.filter(ImageFilter.MedianFilter(size=3))
                            
                            # 3. 转换为灰度并应用自适应阈值
                            img_64 = img_64.convert('L')
                            img_64 = img_64.point(lambda x: 0 if x < 128 else 255, '1')
                            
                            # 4. 重新尝试解码
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
                        images.append(("128x128", img_128))
                    if img_64:
                        images.append(("64x64", img_64))
                    
                    if images:
                        # 将图片对象转换为可序列化的元组格式
                        image_tuples = [(size, np.array(img)) for size, img in images]
                        # 处理多余的tmp_out文件
                        if os.path.exists(tmp_out):
                            os.unlink(tmp_out)
                        self.show_qr_code(None, "",None, *image_tuples)
                    else:
                        messagebox.showerror("错误", "水印提取失败")
                    return
                
                # 显示提取的二维码水印
                self.after(0, lambda: self.show_qr_code(tmp_out, text, True))
            except Exception as e:
                self.after(0, lambda e=e: messagebox.showerror("错误", f"提取水印失败: {str(e)}"))
            finally:
                # 清理临时文件
                for f in [tmp_in]:
                    if os.path.exists(f):
                        os.unlink(f)
                self.after(0, self.hide_processing_window)
                
        threading.Thread(target=worker).start()

# 抑制QReader的特定编码解析警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Double decoding failed')

if __name__ == "__main__":
    try:
        from hooks.torch_fixes import apply_torch_numpy_fix, disable_problematic_features
        disable_problematic_features()
        apply_torch_numpy_fix()
    except ImportError:
        print("Torch修复模块未找到，继续运行...")
    app = App()
    try:
        app.mainloop()
    except Exception as e:
        print(f"程序运行时出错: {e}")
    finally:
        try:
            # 在窗口关闭前保存配置
            if hasattr(app, '_saved_before_close') and not app._saved_before_close:
                wm_text = app.text_wm.get("1.0", "end").rstrip()
                app.save_config(wm_text)
                app._saved_before_close = True
        except Exception as e:
            print(f"保存配置时出错: {e}")
