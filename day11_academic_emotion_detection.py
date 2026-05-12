import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2 
import numpy as np

# a. 加载训练模型
def load_trained_model(model_path):
    #创建和训练时一模一样的 mobilenet_v3_small 模型
    model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
    # 修改最后一层为8分类
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 8)

    # 加载你训练好的权重
    # map_location='cpu' 确保即使你用GPU训练的，在CPU上也能加载
    model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
    # 设置为评估模式
    model.eval()

    return model

# b. 阅读状态映射
def emotion_to_reading_state(emotion_label):
    mapping = {
        'anger': 'frustrated',
        'contempt': 'boredom',
        'disgust': 'frustrated',
        'fear': 'confused',
        'happy': 'engagement',
        'sad': 'frustrated',
        'surprise': 'confused',
        'neutral': 'engagement',
        }
    #我先试着从字典里找这个表情对应的状态。如果找到了，就返回它；如果没找到（不管什么原因），我就默认返回中性，绝对不让程序崩溃。
    return mapping.get(emotion_label, 'neutral')

# c. 单张人脸预测函数
def predict_emotion(model, face_image):
    # 把OpenCV的BGR格式转成RGB格式
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    # 预处理图片
    image_tensor = transform(face_rgb).unsqueeze(0) # 增加batch维度

    # 预测
    # torch.no_grad()：关闭梯度计算，推理时必须加，否则会非常慢
    with torch.no_grad():
        outputs = model(image_tensor)
        # torch.max返回两个值：最大值本身和最大值的索引
        _, predicted = torch.max(outputs.data, 1)
        # 计算置信度
        #softmax 函数会把这 7 个原始分转换成 0-1 之间的概率，并且 7 个概率加起来等于 1 
        #probabilities = [0.03, 0.02, 0.02, 0.923, ..., ...]
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    label_idx = predicted.item()  # 把张量3变成普通数字3
    confidence = probabilities[0][label_idx].item() # 取出第0张图片，第3个类别的概率0.923
    label = emotion_labels[label_idx] # 根据索引从表情列表里取出对应的表情字符串

    # ✅ 新增：返回所有8个表情的置信度数组
    all_probs = probabilities[0].cpu().numpy()

    return emotion_labels[label_idx], confidence, label_idx, all_probs

# d.时序平滑：滑动窗口类，避免卡顿
class SlidingWindow:
    def __init__(self, window_size = 15, num_classes = 8):
        """
        初始化滑动窗口
        参数：
            window_size：窗口大小，保存最近多少帧的结果（默认15帧，约0.5秒）
            num_classes：表情类别数（我们是8类）
        """
        self.window_size = window_size # 窗口最多存15帧
        self.num_classes = num_classes # 8种表情
        # 创建一个队列，保存最近window_size帧的所有类别的置信度
        # 每一个元素是一个长度为8的列表，对应8个表情的置信度\
        self.confidence_history = []

    def add(self, probabilities):
         """
        把新一帧的置信度结果添加到窗口里
        参数：
            probabilities：模型输出的8个表情的置信度数组（numpy数组或列表）
        """
         # 把tensor转换成numpy数组，方便计算
         #hasattr = 检查这个东西有没有某个功能 
         #如果是 GPU 张量（tensor） → 有 .cpu() 功能
         if hasattr(probabilities, 'cpu'):
             probabilities = probabilities.cpu().numpy()

         # 添加到队列末尾
         self.confidence_history.append(probabilities)

         # 如果窗口满了，就把最老的一帧（队列第一个）扔掉
         if len(self.confidence_history) > self.window_size:
             self.confidence_history.pop(0)
    
    def get_average(self):
        # 如果窗口是空的，返回空
        if len(self.confidence_history) == 0:
            return None, -1, 0.0

        # 把所有帧的置信度转换成numpy数组，然后求平均值
        history_array = np.array(self.confidence_history)
        #axis=0 = 竖着算平均（按列求平均）
        avg_probs = np.mean(history_array, axis = 0)

        # 找到平均置信度最高的表情
        best_idx = np.argmax(avg_probs)
        best_conf = avg_probs[best_idx]
        return avg_probs, best_idx, best_conf

# f.状态锁定逻
class StateLock:
    def __init__(self, threshold = 0.6, duration_seconds = 2.0, fps =30):
         """
        初始化状态锁定
        参数：
            threshold：置信度阈值（默认0.6）
            duration_seconds：需要持续的时间（默认2秒）
            fps：摄像头的帧率（默认30fps）
        """
         self.threshold = threshold
         # 计算需要持续多少帧：2秒 × 30fps = 60帧 这里定义了需要连续60帧负面表情 才会被锁定
         self.required_frames = int(duration_seconds * fps)
         # 计数器：记录负面状态持续了多少帧
         self.negative_counter = 0
         self.positive_counter = 0
         self.is_locked = False
         self.locked_state = None

    def update(self, avg_probs):
          """
        更新状态锁定
        参数：
            avg_probs：滑动窗口输出的平均置信度数组
        返回：
            is_locked：是否处于锁定状态
            locked_state：锁定的状态
        """
          # 获取confused和frustrated的平均置信度
          # 注意：这里的索引要和你的emotion_labels对应
          # confused对应surprise(6)，frustrated对应anger(0), disgust(2), fear(3), sad(5)
          confused_conf = avg_probs[6] #surprise → confused
          frustrated_conf = avg_probs[0] + avg_probs[2] + avg_probs[3] + avg_probs[5]
          happy_conf = avg_probs[4]
          neutral_conf = avg_probs[7]

          # 计算总的负面置信度
          total_negative_conf = max(confused_conf, frustrated_conf)
          total_positive_conf = happy_conf + neutral_conf

          # 如果负面置信度超过阈值，计数器加1
          if total_negative_conf > self.threshold and total_negative_conf > total_positive_conf:
              #负面状态：负面计数器+1，正面计数器清零
              self.negative_counter += 1
              self.positive_counter = 0

              # 检查是否达到负面锁定条件
              if self.negative_counter >= self.required_frames:
                  self.is_locked = True
                  # 锁定置信度更高的那个状态
                  if confused_conf > frustrated_conf:
                      self.locked_state = 'confused'
                  else:
                      self.locked_state = 'frustrated'
          elif total_positive_conf > self.threshold and total_positive_conf > total_negative_conf:
                  # 正面状态：正面计数器+1，负面计数器清零
                  self.negative_counter = 0
                  self.positive_counter += 1

                  #检查是否达到正面锁定条件
                  if self.positive_counter >= self.required_frames:
                       self.is_locked = True 
                       self.locked_state = 'engagement'
          else:
              # 都不满足：两个计数器都清零，解除锁定
              self.negative_counter = 0
              self.positive_counter = 0
              self.is_locked = False
              self.locked_state = None

          return self.is_locked, self.locked_state

# 1. 加载人脸检测器（OpenCV自带的Haar级联分类器）
# 这个文件是OpenCV自带的，不需要额外下载
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2.数据预处理
transform = transforms.Compose([
    # 1. 把OpenCV的numpy数组转换成PIL Image格式（模型只接受PIL Image或Tensor）
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    # 3. 把图片转换成Tensor格式，并且把像素值从0-255归一化到0-1之间
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    ])

# 3.表情标签对应关系
emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_colors = [
    (0, 0, 255),      # anger - 红色
    (128, 0, 128),    # contempt - 紫色
    (255, 0, 0),       # disgust - 蓝色
    (0, 255, 255),     # fear - 青色
    (0, 255, 0),       # happy - 绿色
    (255, 255, 0),     # sad - 黄色
    (255, 165, 0),     # surprise - 橙色
    (128, 128, 128)    # neutral - 灰色
]

# 4. 主程序：打开摄像头，实时识别
def main():
    model_path = 'best_mobilenet_affectnet.pth'

    print("正在加载模型...")
    model = load_trained_model(model_path)
    print("✅ 模型加载完成！")

    # ✅ 修复1：初始化滑动窗口和状态锁定
    window = SlidingWindow(window_size = 15)
    state_lock = StateLock(threshold = 0.6, duration_seconds = 2.0)

    print("正在打开摄像头...")
    # 打开摄像头（0是默认摄像头）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return
    print("✅ 摄像头已打开！按 'q' 键退出。")

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
             print("❌ 无法获取视频帧！")
             break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,# 图像缩放比例，值越小检测越仔细，但速度越慢
            minNeighbors = 5,# 每个候选框需要多少个邻居才被认为是人脸
            minSize = (100, 100)  # 最小人脸尺寸，避免误检
            )
        # 对每一张检测到的人脸进行识别
        for (x, y, w, h) in faces:
            # 裁剪出人脸区域
            face_img = frame[y:y+h, x:x+w]
            # 预测表情
            try:
                # 用try-except捕获可能的错误，防止单张人脸识别失败导致整个程序崩溃
                #（逐帧独立）
                #label, confidence, label_idx = predict_emotion(model, face_img)
                #滑动窗口平均
                # 1. 先得到单帧的所有8个表情的置信度
                label, confidence, label_idx, all_probs = predict_emotion(model, face_img)
                # 2. 把这一帧的置信度加到滑动窗口里
                window.add(all_probs)
                # 3. 计算最近15帧的平均置信度
                avg_probs, smooth_idx, smooth_conf = window.get_average()
                # 4. 用平均后的结果来显示
                smooth_label = emotion_labels[smooth_idx]

                #新增：把原始表情映射成阅读状态
                reading_state = emotion_to_reading_state(smooth_label)
                #调用状态锁定更新方法，接收返回值
                is_locked, locked_state = state_lock.update(avg_probs)

                # 选择颜色
                color = emotion_colors[smooth_idx]
                # 画人脸框
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                # 准备显示的文字
                text = f"{smooth_label}({smooth_conf:.1%})"
                # ✅ 新增：阅读状态的文字
                state_text = f"reading_state: {reading_state}"
                # 显示表情和置信度（在框上方）
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                # 把文字放在人脸框上方
                #显示阅读状态
                state_text = f"state: {reading_state}"
                cv2.putText(
                    frame,
                    state_text,
                    (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                    )
                
                #升级后的锁定状态显示（区分正面和负面）
                if is_locked:
                    if locked_state == 'engagement':
                        # 正面状态：绿色文字
                        lock_text = f"✅ detecting engagement！"
                        lock_color = (0, 255, 0)  # 绿色
                    else:
                        # 负面状态：红色文字
                        lock_text = f"⚠️ detecting {locked_state}！"
                        lock_color = (0, 0, 255)  # 红色
                    
                    cv2.putText(
                        frame, lock_text, (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, lock_color, 2
                    )

            except Exception as e:
                # ✅ 修复：打印错误信息，而不是直接 pass
                print(f"❌ 识别出错: {e}")
                #pass
        # 显示结果
        cv2.imshow('real-time FER', frame)
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出。")

if __name__ == "__main__":
    main()


#这个模型目前准确度不高 需要更准确的训练模型 