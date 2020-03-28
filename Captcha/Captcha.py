import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
#首先，导入图像分割函数要用到的label、regionprops函数
from skimage.measure import label, regionprops
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import words
from sklearn.metrics import confusion_matrix
from nltk.metrics import edit_distance
from operator import itemgetter
from matplotlib import pyplot as plt

#生成验证码的基础函数
#这个函数接收一个单词和错切值（通常在0到0.5之间）
#返回用numpy数组格式表示的图像。
def create_captcha(text, shear=0, size=(100, 24),scale=1):
        
    # 我们使用字母L来生成一张黑白图像，为`ImageDraw`类初始化一个实例。这样，我们就可以用`PIL`绘图
    im = Image.new("L", size, "black")   
    draw = ImageDraw.Draw(im)
        
    # 指定验证码文字所使用的字体。这里要用到字体文件，下面代码中的文件名（Coval.otf）应该指向文件存放位置（我把它放到当前笔记本所在目录）。
    font = ImageFont.truetype(r"Coval-Black.ttf", 22)
    draw.text((2, 2), text, fill=1, font=font)
        
    #把PIL图像转换为`numpy`数组，以便用`scikit-image`库为图像添加错切变化效果。`scikit-image`大部分计算都使用`numpy`数组格式。
    image = np.array(im)
        
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
        
    #最后一行代码对图像特征进行归一化处理，确保特征值落在0到1之间。归一化处理可在数据预处理、分类或其他阶段进行
    return image / image.max()


#生成验证码图像并显示它
image = create_captcha("GENE", shear=0.2)
plt.imshow(image, cmap='Greys')

#图像分割函数接收图像，返回小图像列表，每张小图像为单词的一个字母，函数声明如下：
def segment_image(image):

    
    # 我们要做的第一件事就是检测每个字母的位置，这就要用到`scikit-image`的`label`函数，它能找出图像中像素值相同且又连接在一起的像素块。这有点像第7章中的连通分支。`label`函数的参数为图像数组，返回跟输入同型的数组。在返回的数组中，图像**连接在一起的区域**用不同的值来表示，在这些区域以外的像素用0来表示。
    labeled_image = label(image > 0)
    
    #抽取每一张小图像，将它们保存到一个列表中。
    subimages = []
    
    #`scikit-image`库还提供抽取连续区域的函数：`regionprops`。遍历这些区域，分别对它们进行处理。
    for region in regionprops(labeled_image):
        start_x, start_y, end_x, end_y = region.bbox
        
        
        #用这两组坐标作为索引就能抽取到小图像（`image`对象为`numpy`数组，可以直接用索引值），然后，把它保存到`subimages`列表中。
        subimages.append(image[start_x:end_x,start_y:end_y])
        
    #最后（循环外面），返回找到的小图像，每张（希望如此）小图像包含单词的一个字母区域。没有找到小图像的情况，直接把原图像作为子图返回。
    if len(subimages) == 0:
	        return [image,]
    return subimages

subimages = segment_image(image)

f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
for i in range(len(subimages)):
    axes[i].imshow(subimages[i], cmap="gray")

#指定随机状态值，创建字母列表，指定错切值
random_state = check_random_state(14)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)

#生成一条训练数据
def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    
    #返回字母图像及表示图像中字母属于哪个类别的数值。字母A为类别0，B为类别1，C为类别2，以此类推
    return create_captcha(letter, shear=shear, size=(30, 30)),letters.index(letter)

image, target = generate_sample(random_state)
plt.imshow(image, cmap="Greys")
print("The target for this image is: {0}".format(letters[target]))

#调用几千次该函数，就能生成足够的训练数据
dataset, targets = zip(*(generate_sample(random_state) for i in range(1000)))
dataset = np.array([tf.resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
dataset = np.array(dataset, dtype='float') 
targets = np.array(targets)

#对类别使用一位有效码编码方法
onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0],1)) 
y = y.todense()

#现在，就可以对每条数据运行`segment_image`函数，将得到的小图像调整为20像素见方
dataset = np.array([resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
    
#最后，创建我们的数据集。`dataset`数组为三维的，因为它里面存储的是二维图像信息。由于分类器接收的是二维数组，因此，需要将最后两维扁平化
X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
    
#使用`scikit-learn`中的`train_test_split`函数，把数据集切分为训练集和测试集 
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=0.9)

clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=14)

clf.fit(X_train, y_train)

print (len(clf.coefs_))
print (clf.coefs_[0].shape)
print (clf.coefs_[1].shape)

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(20, 20), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

y_pred = clf.predict(X_test)
f1_score(y_pred=y_pred, y_true=y_test, average='macro')

print(classification_report(y_pred=y_pred, y_true=y_test))

#接收验证码，用神经网络进行训练，返回单词预测结果
def predict_captcha(captcha_image, neural_network):
    
    #使用前面定义的图像切割函数， segment_iamge 抽取小图像
    subimages = segment_image(captcha_image)
    
    #整理数据集
    dataset = np.array([tf.resize(subimage, (20, 20)) for subimage in subimages])
    X_test = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
    
    #进行预测
    y_pred = neural_network.predict_proba(X_test)
    predictions = np.argmax(y_pred, axis=1)
    
    #检查输入输出集是否相等
    assert len(y_pred) == len(X_test)
    
    #将预测的字母拼接起来
    predicted_word = str.join("", [letters[prediction] for prediction in predictions])
    return predicted_word

word = "GENE"
captcha = create_captcha(word, shear=0.2)
print(predict_captcha(captcha, clf))

def test_prediction(word, net, shear=0.2):
    captcha = create_captcha(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    prediction = prediction[:4]
    return word == prediction, word, prediction


#借助NLTK模块创建单词数据集
nltk.download('words')
valid_words = [word.upper() for word in words.words() if len(word) ==4]

num_correct = 0
num_incorrect = 0
for word in valid_words:
    correct, word, prediction = test_prediction(word, clf,shear=0.2)
    if correct:
        num_correct += 1
    else:
        num_incorrect += 1
print("Number correct is {0}".format(num_correct))
print("Number incorrect is {0}".format(num_incorrect))

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

plt.figure(figsize=(10, 10))
plt.imshow(cm)
tick_marks = np.arange(len(letters))
plt.xticks(tick_marks, letters)
plt.yticks(tick_marks, letters)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

steps = edit_distance("STEP", "STOP")
print("The number of steps needed is: {0}".format(steps))

#距离度量函数
def compute_distance(prediction, word):
    len_word = min(len(prediction), len(word))
    return len_word - sum([prediction[i] == word[i] for i in range(len_word)])

def improved_prediction(word, net, dictionary, shear=0.2):
    captcha = create_captcha(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    prediction = prediction[:4]

    # 到目前为止，代码跟之前的预测函数一致，还是取到预测结果的前四个字母。然而，这次还要检测单词是否在词典里。如果在，把单词作为预测结果返回；如果不在，找到最相似的单词返回。
    if prediction not in dictionary:
        distances = sorted([(word, compute_distance(prediction, word)) for word in dictionary],key=itemgetter(1))
        
    # 找到最为匹配的单词——也就是距离最小的单词——随后把这个单词作为预测结果返回。
        best_word = distances[0]
        prediction = best_word[0]
        
    # 函数返回结果跟之前相同：预测结果是否正确，验证码中的单词和预测结果。
    return word == prediction, word, prediction

num_incorrect = 0
for word in valid_words:
    correct, word, prediction = improved_prediction(word, clf, valid_words, shear=0.2)
    if correct:
        num_correct += 1
    else:
        num_incorrect += 1
print("Number correct is {0}".format(num_correct))
print("Number incorrect is {0}".format(num_incorrect))