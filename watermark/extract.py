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