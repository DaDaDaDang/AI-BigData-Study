import pandas as pd
import numpy as np

#데이터 불러오기 (csv)
DRE=pd.read_csv('c:/pdata/DdaReungE/r_s_count.csv', encoding='euc-kr')
#불러온 데이터에 index 설정
DRE.set_index('구명', inplace=True)
#전처리한 데이터 파일 (.csv) 만들기
DRE.to_csv("dre.csv", index=True)

# print(DRE.sort_index)

#DRE.rename(columns={DRE.columns[0] : '지역'}, inplace=True)
# print(DRE.head())


#데이터 불러오기 (csv)
rsInfo = pd.read_csv('c:/pdata/DdaReungE/Rental_Shop_Info.csv', encoding='utf-8')
rdMonth1_6 = pd.read_csv('c:/pdata/DdaReungE/Rent_Data_Month.csv', encoding='euc-kr') #utf-8로 인코딩이 안되서 euc-kr로 바꿈
rdMonth7_11 = pd.read_csv('c:/pdata/DdaReungE/Rent_Data_Month2.csv', encoding='utf-8')

# data_result = pd.merge(rsInfo, rdMonth1_6, on='대여소번호')
#
# del data_result['대여소ID']
# del data_result['대여소 주소']
# del data_result['거치대수']
# del data_result['위도']
# del data_result['경도']
# del data_result['대여소']

#대여소 번호를 key값으로 하여 머지시키기 - 함수화
def data_processing(rs , rdMonth) :
    data_result = pd.merge(rs, rdMonth, on='대여소번호')

    del data_result['대여소ID']
    del data_result['대여소 주소']
    del data_result['거치대수']
    del data_result['위도']
    del data_result['경도']
    del data_result['대여소']

    return data_result


#데이터 무결성을 위한 전처리 과정
def indexing(data_result) :
    data_result.set_index('구명', inplace=True)
    real_result = data_result.groupby(['구명', '대여일자'], as_index=True).sum()
    del real_result['대여소번호']
    print(real_result)
    return real_result

###############main으로 들어갈 부분######################
data_result1 = data_processing(rsInfo , rdMonth1_6)
real_result1=indexing(data_result1)

data_result2 = data_processing(rsInfo, rdMonth7_11)
##########################################################
# data_result2 = pd.merge(rsInfo, rdMonth7_11, on='대여소번호')
#
# del data_result2['대여소ID']
# del data_result2['대여소 주소']
# del data_result2['거치대수']
# del data_result2['위도']
# del data_result2['경도']
# del data_result2['대여소명_y']

data_result2.rename(columns={data_result2.columns[2] : '대여소명'}, inplace=True)
real_result2=indexing(data_result2)


data_result1.to_csv('result1.csv')
data_result2.to_csv('result2.csv')

real_result1.to_csv('real_result.csv')
real_result2.to_csv('real_result2.csv')

combined_csv = pd.concat([real_result1, real_result2])

combined_csv.groupby(['구명', '대여일자'], as_index=True).sum()

combined_csv.to_csv("combined_csv.csv", index=True)

rComb=combined_csv.groupby(['구명', '대여일자'], as_index=True).sum()
# del rComb['대여소번호']

print("combined_csv")
print(rComb.head(11))

rComb.to_csv("rComb.csv", index=True)

final = pd.merge(rComb, DRE, on='구명')
final.to_csv("final.csv", index=False)

#월별 따릉이 이용현환 데이터 불러오기 - 함수화
def month_us() :
    a=pd.read_csv('c:/pdata/DdaReungE/1.csv', encoding='euc-kr')
    b=pd.read_csv('c:/pdata/DdaReungE/2.csv', encoding='euc-kr')
    c=pd.read_csv('c:/pdata/DdaReungE/3.csv', encoding='euc-kr')
    d=pd.read_csv('c:/pdata/DdaReungE/4.csv', encoding='euc-kr')
    e=pd.read_csv('c:/pdata/DdaReungE/5.csv', encoding='euc-kr')
    f=pd.read_csv('c:/pdata/DdaReungE/6.csv', encoding='euc-kr')
    g=pd.read_csv('c:/pdata/DdaReungE/7.csv', encoding='euc-kr')
    h=pd.read_csv('c:/pdata/DdaReungE/8.csv', encoding='euc-kr')
    i=pd.read_csv('c:/pdata/DdaReungE/9.csv', encoding='euc-kr')
    j=pd.read_csv('c:/pdata/DdaReungE/10.csv', encoding='euc-kr')
    k=pd.read_csv('c:/pdata/DdaReungE/11.csv', encoding='euc-kr')

    #구명으로 월별 이용현황과 따릉이 대여소를 머지한다.
    m1 = pd.merge(a, DRE, on='구명')
    m2 = pd.merge(b, DRE, on='구명')
    m3 = pd.merge(c, DRE, on='구명')
    m4 = pd.merge(d, DRE, on='구명')
    m5 = pd.merge(e, DRE, on='구명')
    m6 = pd.merge(f, DRE, on='구명')
    m7 = pd.merge(g, DRE, on='구명')
    m8 = pd.merge(h, DRE, on='구명')
    m9 = pd.merge(i, DRE, on='구명')
    m10 = pd.merge(j, DRE, on='구명')
    m11 = pd.merge(k, DRE, on='구명')

#############################
month_us()
#############################

#시각화

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

#그래프 지역별 대여소 수
plt.figure()
# 크기:figsize=(10,10)넓이, 높이 inch 기준
# data_result['소계'].plot(kind='barh', grid=True, figsize=(10,10))
DRE['대여소 수'].sort_values().plot(kind='barh', grid=True, figsize=(10,10), label='대여소 수')
plt.show()

#분포도(산포도) 1월 지역별 대여건수와 지역별 대여소 수
plt.figure(figsize=(6,6))
plt.xlim(0,100)
plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?

#라인따기
fp1 = np.polyfit(m1['대여소 수'], m1['대여건수'], 1)
#y축값 구하기
f1 = np.poly1d(fp1)
#x축값 구하기
fx=np.linspace(10,100,100)

plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')

m1['오차']= np.abs(m1['대여건수']-f1(m1['대여소 수']))
df_sort = m1.sort_values(by='오차', ascending=False)
#오차정보 출력하며, 문자열 함께 출력
for n in range(5):
    plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)


plt.scatter(m1['대여소 수'], m1['대여건수'], s=50)


plt.xlabel('대여소 수')
plt.ylabel('1월 대여 합계')
plt.grid()
plt.show()

# m1['대여소 당 대여건수'] = m1['대여건수'] / m1['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m1['대여소 수'], m1['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m1['오차']= np.abs(m1['대여소 당 대여건수']-f1(m1['대여소 수']))
# #오차기준정렬
# df_sort = m1.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m1['대여소 수'], m1['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('1월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m2['대여소 수'], m2['대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# m2['오차']= np.abs(m2['대여건수']-f1(m2['대여소 수']))
# df_sort = m2.sort_values(by='오차', ascending=False)
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
#
#
# plt.scatter(m2['대여소 수'], m2['대여건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('2월 대여 합계')
# plt.grid()
# plt.show()
#
# m2['대여소 당 대여건수'] = m2['대여건수'] / m2['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m2['대여소 수'], m2['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m2['오차']= np.abs(m2['대여소 당 대여건수']-f1(m2['대여소 수']))
# #오차기준정렬
# df_sort = m2.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m2['대여소 수'], m2['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('2월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m3['대여소 수'], m3['대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# m3['오차']= np.abs(m3['대여건수']-f1(m3['대여소 수']))
# df_sort = m3.sort_values(by='오차', ascending=False)
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
#
#
# plt.scatter(m3['대여소 수'], m3['대여건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('3월 대여 합계')
# plt.grid()
# plt.show()
#
# m3['대여소 당 대여건수'] = m3['대여건수'] / m3['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m3['대여소 수'], m3['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m3['오차']= np.abs(m3['대여소 당 대여건수']-f1(m3['대여소 수']))
# #오차기준정렬
# df_sort = m3.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m3['대여소 수'], m3['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('3월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#

plt.figure(figsize=(6,6))
plt.xlim(0,100)
plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#라인따기
fp1 = np.polyfit(m4['대여소 수'], m4['대여건수'], 1)
#y축값 구하기
f1 = np.poly1d(fp1)
#x축값 구하기
fx=np.linspace(10,100,100)

plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
m4['오차']= np.abs(m4['대여건수']-f1(m4['대여소 수']))
df_sort = m4.sort_values(by='오차', ascending=False)
#오차정보 출력하며, 문자열 함께 출력
for n in range(5):
    plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)


plt.scatter(m4['대여소 수'], m4['대여건수'], s=50)
plt.xlabel('대여소 수')
plt.ylabel('4월 대여 합계')
plt.grid()
plt.show()
#
# m4['대여소 당 대여건수'] = m4['대여건수'] / m4['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m4['대여소 수'], m4['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m4['오차']= np.abs(m4['대여소 당 대여건수']-f1(m4['대여소 수']))
# #오차기준정렬
# df_sort = m4.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m4['대여소 수'], m4['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('4월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m5['대여소 수'], m5['대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m5['대여소 수'], m5['대여건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('5월 대여 합계')
# plt.grid()
# plt.show()
#
# m5['대여소 당 대여건수'] = m5['대여건수'] / m5['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m5['대여소 수'], m5['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m5['오차']= np.abs(m5['대여소 당 대여건수']-f1(m5['대여소 수']))
# #오차기준정렬
# df_sort = m5.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m5['대여소 수'], m5['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('5월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#

plt.figure(figsize=(6,6))
plt.xlim(0,100)
plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#라인따기
fp1 = np.polyfit(m6['대여소 수'], m6['대여건수'], 1)
#y축값 구하기
f1 = np.poly1d(fp1)
#x축값 구하기
fx=np.linspace(10,100,100)

plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
m6['오차']= np.abs(m6['대여건수']-f1(m6['대여소 수']))
df_sort = m6.sort_values(by='오차', ascending=False)
#오차정보 출력하며, 문자열 함께 출력
for n in range(5):
    plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)


plt.scatter(m6['대여소 수'], m6['대여건수'], s=50)
plt.xlabel('대여소 수')
plt.ylabel('6월 대여 합계')
plt.grid()
plt.show()
#
# m6['대여소 당 대여건수'] = m6['대여건수'] / m6['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m6['대여소 수'], m6['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m6['오차']= np.abs(m6['대여소 당 대여건수']-f1(m6['대여소 수']))
# #오차기준정렬
# df_sort = m6.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m6['대여소 수'], m6['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('6월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m7['대여소 수'], m7['대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m7['대여소 수'], m7['대여건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('7월 대여 합계')
# plt.grid()
# plt.show()
#
# m7['대여소 당 대여건수'] = m7['대여건수'] / m7['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m7['대여소 수'], m7['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m7['오차']= np.abs(m7['대여소 당 대여건수']-f1(m7['대여소 수']))
# #오차기준정렬
# df_sort = m7.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m7['대여소 수'], m7['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('7월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
# #라인따기
# fp1 = np.polyfit(m8['대여소 수'], m8['대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m8['대여소 수'], m8['대여건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('8월 대여 합계')
# plt.grid()
# plt.show()
#
# m8['대여소 당 대여건수'] = m8['대여건수'] / m8['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m8['대여소 수'], m8['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m8['오차']= np.abs(m8['대여소 당 대여건수']-f1(m8['대여소 수']))
# #오차기준정렬
# df_sort = m8.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m8['대여소 수'], m8['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('8월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#

plt.figure(figsize=(6,6))
plt.xlim(0,100)
plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#라인따기
fp1 = np.polyfit(m9['대여소 수'], m9['대여건수'], 1)
#y축값 구하기
f1 = np.poly1d(fp1)
#x축값 구하기
fx=np.linspace(10,100,100)

plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
m9['오차']= np.abs(m9['대여건수']-f1(m9['대여소 수']))
df_sort = m9.sort_values(by='오차', ascending=False)
#오차정보 출력하며, 문자열 함께 출력
for n in range(5):
    plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)


plt.scatter(m9['대여소 수'], m9['대여건수'], s=50)
plt.xlabel('대여소 수')
plt.ylabel('9월 대여 합계')
plt.grid()
plt.show()
#
# m9['대여소 당 대여건수'] = m9['대여건수'] / m9['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m9['대여소 수'], m9['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m9['오차']= np.abs(m9['대여소 당 대여건수']-f1(m9['대여소 수']))
# #오차기준정렬
# df_sort = m1.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m9['대여소 수'], m9['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('9월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
# #라인따기
# fp1 = np.polyfit(m10['대여소 수'], m10['대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# m10['오차']= np.abs(m10['대여건수']-f1(m10['대여소 수']))
# df_sort = m10.sort_values(by='오차', ascending=False)
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
#
#
# plt.scatter(m1['대여소 수'], m1['대여건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('10월 대여 합계')
# plt.grid()
# plt.show()
#
# m10['대여소 당 대여건수'] = m10['대여건수'] / m10['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m10['대여소 수'], m10['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m10['오차']= np.abs(m10['대여소 당 대여건수']-f1(m10['대여소 수']))
# #오차기준정렬
# df_sort = m10.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m10['대여소 수'], m10['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('10월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m11['대여소 수'], m11['대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m11['대여소 수'], m11['대여건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('11월 대여 합계')
# plt.grid()
# plt.show()
#
# m11['대여소 당 대여건수'] = m11['대여건수'] / m11['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m11['대여소 수'], m11['대여소 당 대여건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m11['오차']= np.abs(m11['대여소 당 대여건수']-f1(m11['대여소 수']))
# #오차기준정렬
# df_sort = m11.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m11['대여소 수'], m11['대여소 당 대여건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 대여건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('11월 대여건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#

plt.figure(figsize=(6,6))
plt.xlim(0,100)
plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#라인따기
fp1 = np.polyfit(m1['대여소 수'], m1['반납건수'], 1)
#y축값 구하기
f1 = np.poly1d(fp1)
#x축값 구하기
fx=np.linspace(10,100,100)

plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
m1['오차']= np.abs(m1['반납건수']-f1(m1['대여소 수']))
df_sort = m1.sort_values(by='오차', ascending=False)
#오차정보 출력하며, 문자열 함께 출력
for n in range(5):
    plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)


plt.scatter(m1['대여소 수'], m1['반납건수'], s=50)
plt.xlabel('대여소 수')
plt.ylabel('1월 반납 합계')
plt.grid()
plt.show()
#
# m1['대여소 당 반납건수'] = m1['반납건수'] / m1['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m1['대여소 수'], m1['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m1['오차']= np.abs(m1['대여소 당 반납건수']-f1(m1['대여소 수']))
# #오차기준정렬
# df_sort = m1.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m1['대여소 수'], m1['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('1월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m2['대여소 수'], m2['반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m2['대여소 수'], m2['반납건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('2월 반납 합계')
# plt.grid()
# plt.show()
#
# m2['대여소 당 반납건수'] = m2['반납건수'] / m2['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m2['대여소 수'], m2['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m2['오차']= np.abs(m2['대여소 당 반납건수']-f1(m2['대여소 수']))
# #오차기준정렬
# df_sort = m2.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m2['대여소 수'], m2['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('2월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m3['대여소 수'], m3['반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m3['대여소 수'], m3['반납건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('3월 반납 합계')
# plt.grid()
# plt.show()
#
# m3['대여소 당 반납건수'] = m3['반납건수'] / m3['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m3['대여소 수'], m3['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m3['오차']= np.abs(m3['대여소 당 반납건수']-f1(m3['대여소 수']))
# #오차기준정렬
# df_sort = m3.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m3['대여소 수'], m3['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('3월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#

plt.figure(figsize=(6,6))
plt.xlim(0,100)
plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#라인따기
fp1 = np.polyfit(m4['대여소 수'], m4['반납건수'], 1)
#y축값 구하기
f1 = np.poly1d(fp1)
#x축값 구하기
fx=np.linspace(10,100,100)

plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')

m4['오차']= np.abs(m4['반납건수']-f1(m4['대여소 수']))
df_sort = m4.sort_values(by='오차', ascending=False)
#오차정보 출력하며, 문자열 함께 출력
for n in range(5):
    plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)


plt.scatter(m4['대여소 수'], m4['반납건수'], s=50)
plt.xlabel('대여소 수')
plt.ylabel('4월 반납 합계')
plt.grid()
plt.show()
#
# m4['대여소 당 반납건수'] = m4['반납건수'] / m4['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m4['대여소 수'], m4['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m4['오차']= np.abs(m4['대여소 당 반납건수']-f1(m4['대여소 수']))
# #오차기준정렬
# df_sort = m4.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m4['대여소 수'], m4['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('4월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m5['대여소 수'], m5['반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m5['대여소 수'], m5['반납건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('5월 반납 합계')
# plt.grid()
# plt.show()
#
# m5['대여소 당 반납건수'] = m5['대여건수'] / m5['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m5['대여소 수'], m5['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m5['오차']= np.abs(m5['대여소 당 반납건수']-f1(m5['대여소 수']))
# #오차기준정렬
# df_sort = m5.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m5['대여소 수'], m5['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('5월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#

plt.figure(figsize=(6,6))
plt.xlim(0,100)
plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#라인따기
fp1 = np.polyfit(m6['대여소 수'], m6['반납건수'], 1)
#y축값 구하기
f1 = np.poly1d(fp1)
#x축값 구하기
fx=np.linspace(10,100,100)

plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')

m6['오차']= np.abs(m6['반납건수']-f1(m6['대여소 수']))
df_sort = m6.sort_values(by='오차', ascending=False)
#오차정보 출력하며, 문자열 함께 출력
for n in range(5):
    plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)



plt.scatter(m6['대여소 수'], m6['반납건수'], s=50)
plt.xlabel('대여소 수')
plt.ylabel('6월 반납 합계')
plt.grid()
plt.show()
#
# m6['대여소 당 반납건수'] = m6['반납건수'] / m6['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m6['대여소 수'], m6['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m6['오차']= np.abs(m6['대여소 당 반납건수']-f1(m6['대여소 수']))
# #오차기준정렬
# df_sort = m6.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m6['대여소 수'], m6['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('6월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m7['대여소 수'], m7['반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m7['대여소 수'], m7['반납건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('7월 반납 합계')
# plt.grid()
# plt.show()
#
# m7['대여소 당 반납건수'] = m7['반납건수'] / m7['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m7['대여소 수'], m7['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m7['오차']= np.abs(m7['대여소 당 반납건수']-f1(m7['대여소 수']))
# #오차기준정렬
# df_sort = m7.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m7['대여소 수'], m7['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('7월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
# #라인따기
# fp1 = np.polyfit(m8['대여소 수'], m8['반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m8['대여소 수'], m8['반납건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('8월 반납 합계')
# plt.grid()
# plt.show()
#
# m8['대여소 당 반납건수'] = m8['반납건수'] / m8['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m8['대여소 수'], m8['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m8['오차']= np.abs(m8['대여소 당 반납건수']-f1(m8['대여소 수']))
# #오차기준정렬
# df_sort = m8.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m8['대여소 수'], m8['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('8월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#

plt.figure(figsize=(6,6))
plt.xlim(0,100)
plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?

#라인따기
fp1 = np.polyfit(m9['대여소 수'], m9['반납건수'], 1)
#y축값 구하기
f1 = np.poly1d(fp1)
#x축값 구하기
fx=np.linspace(10,100,100)

plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')

m9['오차']= np.abs(m9['반납건수']-f1(m9['대여소 수']))
df_sort = m9.sort_values(by='오차', ascending=False)
#오차정보 출력하며, 문자열 함께 출력
for n in range(5):
    plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)


plt.scatter(m9['대여소 수'], m9['반납건수'], s=50)
plt.xlabel('대여소 수')
plt.ylabel('9월 반납 합계')
plt.grid()
plt.show()
#
# m9['대여소 당 반납건수'] = m9['반납건수'] / m9['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m9['대여소 수'], m9['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m9['오차']= np.abs(m9['대여소 당 반납건수']-f1(m9['대여소 수']))
# #오차기준정렬
# df_sort = m1.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m9['대여소 수'], m9['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('9월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#

# plt.figure(figsize=(6,6))
# plt.xlim(0,100)
# plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
# #라인따기
# fp1 = np.polyfit(m10['대여소 수'], m10['반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# m10['오차']= np.abs(m10['반납건수']-f1(m10['대여소 수']))
# df_sort = m10.sort_values(by='오차', ascending=False)
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
#
#
# plt.scatter(m1['대여소 수'], m1['반납건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('10월 반납 합계')
# plt.grid()
# plt.show()
#
# m10['대여소 당 반납건수'] = m10['반납건수'] / m10['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m10['대여소 수'], m10['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m10['오차']= np.abs(m10['대여소 당 반납건수']-f1(m10['대여소 수']))
# #오차기준정렬
# df_sort = m10.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m10['대여소 수'], m10['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('10월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
#
#
# #라인따기
# fp1 = np.polyfit(m11['대여소 수'], m11['반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(10,100,100)
#
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
#
# plt.scatter(m11['대여소 수'], m11['반납건수'], s=50)
# plt.xlabel('대여소 수')
# plt.ylabel('11월 대여 합계')
# plt.grid()
# plt.show()
#
# m11['대여소 당 반납건수'] = m11['반납건수'] / m11['대여소 수']
#
# #라인따기
# fp1 = np.polyfit(m11['대여소 수'], m11['대여소 당 반납건수'], 1)
# #y축값 구하기
# f1 = np.poly1d(fp1)
# #x축값 구하기
# fx=np.linspace(0,100,100)
#
# #수평직선과이 오차값을 구해서 컬럼 추가
# m11['오차']= np.abs(m11['대여소 당 반납건수']-f1(m11['대여소 수']))
# #오차기준정렬
# df_sort = m11.sort_values(by='오차', ascending=False)
# plt.figure(figsize=(14, 6))
# plt.xlim(0,100)
# #plt.ylim(0,225000) #모두다 같은 범위로 봐야하나?
#
# print(df_sort['구명'])
#
# plt.scatter(m11['대여소 수'], m11['대여소 당 반납건수'], s=50)
# # 라인값 설정
# plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
# #오차정보 출력하며, 문자열 함께 출력
# for n in range(5):
#     plt.text(df_sort['대여소 수'][df_sort.index[n]]*1.02, df_sort['대여소 당 반납건수'][df_sort.index[n]]*0.98, df_sort['구명'][df_sort.index[n]], fontsize=12)
# # x축, y축 라벨 설정
# plt.xlabel('대여소 수')
# plt.ylabel('11월 반납건수')
# plt.colorbar()
# plt.grid()
# plt.show()
