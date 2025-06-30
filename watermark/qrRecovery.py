## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

import numpy as np
from PIL import Image
import cv2
from qreader import QReader

class IntegratedQRRecovery:
    def __init__(self, qreader=None):
        # 初始化QReader
        if qreader is not None:
            self.qreader = qreader
        else:
            print("QReader Not Found")
            self.qreader = QReader(model_size='l', min_confidence=0.3, reencode_to='cp65001')
    
    def decode_qr(self, image):
        """尝试解码二维码，返回文本或None"""
        try:
            # 确保输入是PIL Image或numpy array
            if isinstance(image, Image.Image):
                # PIL Image转numpy array
                img_array = np.array(image)
            else:
                img_array = image
            
            # 如果是灰度图，转换为BGR
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # RGB转BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 使用QReader解码
            decoded = self.qreader.detect_and_decode(image=img_array)
            
            if decoded and len(decoded) > 0 and decoded[0] is not None:
                # 偏好UTF-8解码
                result = decoded[0]
                if isinstance(result, bytes):
                    try:
                        result = result.decode('utf-8')
                    except:
                        result = result.decode('utf-8', errors='ignore')
                return result
            return None
            
        except Exception as e:
            print(f"解码失败: {e}")
            return None

# 自定义：根据希望的半径（radius_px）计算合适的 kernel size（必须为奇数）
def radius_to_kernel(radius_px):
    k = int(radius_px * 2) + 1
    if k % 2 == 0:
        k += 1
    return k

def enhanced_qr_recovery(extracted_img, size, qreader=None):
    """
    针对严重损坏的二维码进行恢复
    使用边缘引导 + 高斯模糊的组合策略
    返回 (recovered_image, decoded_text, method_name) 或 (None, None, None)
    """
    method_name = "边缘引导恢复"
    try:
        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(np.array(extracted_img), cv2.COLOR_RGB2GRAY)
        
        # 统一缩放到1024x1024，无论输入尺寸如何
        upsampled = cv2.resize(img_cv, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        
        # 轻微高斯模糊去噪
        blurred = cv2.GaussianBlur(upsampled, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 30, 100)
        
        # 膨胀边缘以连接断裂的部分
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 结合原图和边缘信息
        edge_guided = cv2.addWeighted(blurred, 0.7, edges, 0.3, 0)
        
        # Otsu自适应阈值二值化
        _, binary = cv2.threshold(edge_guided, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel_size = radius_to_kernel(radius_px=32)
        # 关键步骤：固定使用32px半径的高斯模糊
        final_result = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)  # 32px半径对应卷积核
        
        # 转换回PIL格式
        recovered_image = Image.fromarray(final_result)
        
        # 尝试解码恢复后的图像
        try:
            # 创建临时解码器实例，传递qreader参数
            temp_decoder = IntegratedQRRecovery(qreader)
            decoded_text = temp_decoder.decode_qr(recovered_image)
            return recovered_image, decoded_text, method_name
        except:
            # 如果解码失败，仍然返回恢复的图像，但文本为None
            return recovered_image, None, method_name
        
    except Exception as e:
        print(f"边缘引导恢复失败: {e}")
        return None, None, None

class ComprehensiveQRRecovery:
    def __init__(self, qreader=None):
        self.qr_decoder = IntegratedQRRecovery(qreader)
    
    def enhance_contrast(self, image, method='clahe'):
        """对比度增强"""
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(image)
        elif method == 'histogram':
            return cv2.equalizeHist(image)
        elif method == 'gamma':
            gamma = 1.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        return image
    
    def denoise(self, image, method='gaussian'):
        """降噪处理"""
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'nlm':
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return image
    
    def binarize(self, image, method='adaptive'):
        """二值化处理"""
        if method == 'adaptive':
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        elif method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == 'fixed':
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary
        return image
    
    def morphological_operations(self, image):
        """形态学操作"""
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return closing
    
    def process_with_method(self, image, enhance_method, denoise_method, binarize_method):
        """使用指定方法处理图像，返回(processed_image, decoded_text, method_name)"""
        method_name = f"{enhance_method}+{denoise_method}+{binarize_method}"
        try:
            # 转换为灰度图
            if isinstance(image, Image.Image):
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            else:
                gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. 对比度增强
            enhanced = self.enhance_contrast(gray, enhance_method)
            
            # 2. 降噪
            denoised = self.denoise(enhanced, denoise_method)
            
            # 3. 二值化
            binary = self.binarize(denoised, binarize_method)
            
            # 4. 形态学操作
            processed = self.morphological_operations(binary)
            
            # 5. 尝试不同旋转角度
            for angle in [0, 90, 180, 270]:
                current_method = method_name
                if angle != 0:
                    current_method = f"{method_name}+旋转{angle}度"
                    
                if angle == 0:
                    test_image = processed
                else:
                    rows, cols = processed.shape
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                    test_image = cv2.warpAffine(processed, M, (cols, rows))
                
                # 尝试解码
                result = self.qr_decoder.decode_qr(test_image)
                if result:
                    # 返回处理后的图像、解码文本和方法名
                    processed_pil = Image.fromarray(test_image)
                    return processed_pil, result, current_method
            
            return None, None, method_name
            
        except Exception as e:
            print(f"处理方法失败: {e}")
            return None, None, method_name
    
    def comprehensive_recovery(self, image, size=None):
        """
        综合恢复方法：结合原有方法和多种新方法
        返回 (recovered_image, decoded_text, method_name) 或 (None, None, None)
        """
        print("开始综合二维码恢复...")
        
        # 方法1：原有的边缘引导方法
        print("尝试方法1: 边缘引导恢复")
        try:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
                
            enhanced_img, result, method_name = enhanced_qr_recovery(pil_image, size, self.qr_decoder.qreader)
            if result:
                print(f"✓ 边缘引导方法成功! 内容: {result}")
                return enhanced_img, result, method_name
            elif enhanced_img:
                # 如果有恢复图像但没有解码成功，尝试用类内解码器再试一次
                result = self.qr_decoder.decode_qr(enhanced_img)
                if result:
                    print(f"✓ 边缘引导方法二次解码成功! 内容: {result}")
                    return enhanced_img, result, f"{method_name}+二次解码"
        except Exception as e:
            print(f"✗ 边缘引导方法失败: {e}")
        
        # 方法2-8：多种图像处理方法组合
        methods = [
            ('clahe', 'gaussian', 'adaptive'),
            ('clahe', 'bilateral', 'otsu'),
            ('histogram', 'median', 'adaptive'),
            ('gamma', 'nlm', 'otsu'),
            ('clahe', 'gaussian', 'otsu'),
            ('histogram', 'gaussian', 'adaptive'),
            ('gamma', 'bilateral', 'adaptive'),
        ]
        
        for i, (enhance, denoise, binarize) in enumerate(methods, 2):
            method_desc = f"方法{i}: {enhance}+{denoise}+{binarize}"
            print(f"尝试{method_desc}")
            processed_img, result, method_name = self.process_with_method(image, enhance, denoise, binarize)
            if result:
                print(f"✓ {method_desc}成功! 内容: {result}")
                return processed_img, result, method_name
            else:
                print(f"✗ {method_desc}失败")
        
        # 方法9：直接尝试原始图像
        print("尝试方法9: 直接解码原始图像")
        result = self.qr_decoder.decode_qr(image)
        if result:
            print(f"✓ 原始图像直接解码成功! 内容: {result}")
            # 返回原始图像和解码结果
            if isinstance(image, np.ndarray):
                original_pil = Image.fromarray(image)
            else:
                original_pil = image
            return original_pil, result, "原始图像直接解码"
        
        print("✗ 所有方法都无法解码二维码")
        return None, None, None

# 便捷函数
def recover_qr_code(image, size=None, qreader=None):
    """
    便捷的二维码恢复函数
    
    Args:
        image: PIL Image 或 numpy array
        size: 可选的尺寸参数
        qreader: 可选的QReader实例，如果不提供会创建新的
    
    Returns:
        tuple: (recovered_image, decoded_text, method_name) 或 (None, None, None)
    """
    recovery = ComprehensiveQRRecovery(qreader)
    return recovery.comprehensive_recovery(image, size)

# 使用示例
if __name__ == "__main__":
    # 示例用法
    try:
        # 从文件加载图像
        image_path = "test.png"  # 你的二维码图像路径
        image = Image.open(image_path)
        
        # 恢复二维码
        recovered_img, result, method_used = recover_qr_code(image)
        
        if result:
            print(f"最终解码结果: {result}")
            print(f"使用的方法: {method_used}")
            # [DEBUG] 保存图像
            # if recovered_img:
            #     recovered_img.save("recovered_output.png")
            #     print("恢复后的图像已保存为 recovered_output.png")
        else:
            print("无法恢复二维码内容")
            
    except FileNotFoundError:
        print("请确保test.png文件存在")
    except Exception as e:
        print(f"运行出错: {e}")