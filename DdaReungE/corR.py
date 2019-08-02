import matplotlib.pyplot as plt
import platform


# 한글폰트 처리하기
from matplotlib import font_manager, rc

plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)

plt.figure()
# 크기:figsize=(10,10)넓이, 높이 inch 기준
# data_result['소계'].plot(kind='barh', grid=True, figsize=(10,10))
data_result['소계'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))