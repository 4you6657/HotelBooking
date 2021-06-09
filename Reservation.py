##导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import  Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

from mpl_toolkits.mplot3d import Axes3D

##数据提取
dataCustomer = pd.read_csv('D:\data\customer.csv')
dataHolidayMST = pd.read_csv('D:\data\holiday_mst.csv')
dataHotel = pd.read_csv('D:\data\hotel.csv')
dataMonthMST = pd.read_csv('D:\data\month_mst.csv')
dataMonthlyIndex = pd.read_csv('D:\data\monthly_index.csv')
dataProduction = pd.read_csv('D:\data\production.csv')
dataProductionMissingCategory = pd.read_csv('D:\data\production_missing_category.csv')
dataProductionMissingNum = pd.read_csv('D:\data\production_missing_num.csv')
dataProductionMissingNum4RedShift = pd.read_csv('D:\data\production_missing_num_4_redshift.csv')
dataReserve = pd.read_csv('D:/data/reserve.csv')
#检查数据是否导入完成
#print(dataCustomer)
#print(dataHolidayMST)
#print(dataHotel)
#print(dataMonthMST)
#print(dataMonthlyIndex)
#print(dataProduction)
#print(dataProductionMissingCategory)
#print(dataProductionMissingNum)
#print(dataProductionMissingNum4RedShift)
#print(dataReserve)

#查看数据的前两行
#print("查看数据的前两行:")
#print(dataCustomer.head(2))
#查看维数(raw and column)
#print("dataReserve的维数为:")
#print(dataReserve.shape)
#describe获取任何数值型列的描述性统计量
#print("dataHolidayMST的描述性统计量:")
#print(dataHolidayMST.describe())
#print("dataCustomer的描述性统计量:")
#print(dataCustomer.describe())


#分界线
print("--------------------------------------------------")
#将TRUE和FALSE转化为数字
a=dataHotel['is_business']
b=dataHotel['hotel_id']
c=dataHotel['big_area_name']
d=dataHotel['small_area_name']
a=pd.DataFrame(a)
for u in a.columns:
    if a[u].dtype==bool:
        a[u]=a[u].astype('int')
#dataHotel.drop(['is_business','hotel_id','big_area_name','small_area_name'],axis=1,inplace=True)
#dataHotel.insert(dataHotel.shape[-1],'is_business',a)
#导出数据
#dataHotel.to_csv('D:\data\customer.csv')
#print(dataHotel)

#数字化处理
#1)customer
#将sex 转化为 0、1
dataCustomer = dataCustomer.replace("man",1)
dataCustomer = dataCustomer.replace("woman",0)
#print(dataCustomer)
#dataCustomer.drop(['customer_id'],axis=1,inplace=True)
#dataCustomer.to_csv('D:\data\customer.csv')
#print(dataCustomer)

#dataHolidayMST
#f=dataHolidayMST['holiday_flg']
#g=dataHolidayMST['nextday_is_holiday_flg']
#f=pd.DataFrame(f)
#g=pd.DataFrame(g)
#for u in f.columns:
    #if f[u].dtype==bool:
     #   f[u]=f[u].astype('int')
#for u in g.columns:
   # if g[u].dtype==bool:
      #  g[u]=g[u].astype('int')
#dataHolidayMST.drop([holidayday_flg','nextday_is_holiday_flg'],axis=1,inplace=True)
#dataHolidayMST.insert(dataHolidayMST.shape[-1],'holidayday_flg',f)
#dataHolidayMST.insert(dataHolidayMST.shape[-1],'nextday_is_holiday_flg',g)
#rint(dataHolidayMST)

#聚合函数整理数据
#dataCustomer_id_sex = dataCustomer.groupby(['customer_id','sex']).sum()
#print(dataCustomer_id_sex)
#针对reserve表格中的顾客ID和酒店ID进行整理关联
#针对reserve表格中的到达并登记的日期和退订的日期进行整理关联
#dataReserve_customerID_hotelID = dataReserve.groupby(['customer_id','hotel_id']).sum()
#print(dataReserve_customerID_hotelID)
#dataReserve_checkin_checkout = dataReserve.groupby(['checkin_date','checkin_time','checkout_date']).sum()
#print(dataReserve_checkin_checkout)
#针对hotel表格中酒店所处的经度和纬度进行整理关联
#dataHotel_latitude_longitude = dataHotel.groupby(['hotel_latitude','hotel_longitude']).sum()
#print(dataHotel_latitude_longitude)
#dataHotel_hotelid_isbusiness = dataHotel.groupby(['hotel_id','is_business']).sum()
#print(dataHotel_hotelid_isbusiness)

#查看描述性数据统计量
#print(dataProductionMissingCategoryColumn1.describe())
#print("查看描述性数据统计量")
#print("dataProductionMissingCategory")
#print(dataProductionMissingCategory.describe())
#print("----------")
#print("dataCustomer")
#print(dataCustomer.describe())
#print("----------")
#print("dataReserve")
#print(dataReserve.describe())
#print("----------")
#print("dataProduction")
#print(dataProduction.describe())
#print("----------")
#print("dataHolidayMST")
#print(dataHolidayMST.describe())
#print("----------")
#print("dataHotel")
#print(dataHotel.describe())
#print("----------")
#print("dataMonthlyIndex")
#print(dataMonthlyIndex.describe())
#print("----------")
#print("dataMonthMST")
#print(dataMonthMST.describe())
#print("----------")
#print("dataProductionMissingNum")
#print(dataProductionMissingNum.describe())
#print("----------")
#print("dataProductionMissingNum4RedShift")
#print(dataProductionMissingNum4RedShift.describe())
#print("----------")

#对指定数据列的提取操作
#eg.
#dataCustomerColumn0_2 = dataCustomer.iloc[:,0:2]
#print(dataCustomerColumn0_2)

#dataHolidayMSTColumn0_3 = dataHolidayMST.iloc[:,0:3]
#print(dataHolidayMSTColumn0_3)

#单独提取出某一列数据
#dataCustomerColumn3 = dataCustomer.iloc[:,[2]]
#print(dataCustomerColumn3)
#选择第一行
#print("显示dataCustomer的第一行:")
#print(dataCustomer.iloc[0])
#选择第一列
#print("显示dataCustomer的第一列:")
#print(dataCustomer.iloc[:,0])
#使用':'来定义想要选择哪些行,比如选择第2、3、4行:
#选择三行
#print("显示dataCustomer的第2、3、4行")
#print(dataCustomer.iloc[1:4])
#选择三列
#print("获取dataCustomer的第2、3、4列")
#print(dataCustomer.iloc[:,1:4])
#获取到第4行为止的所有行:
#print("获取到第4行为止的所有行:")
#print(dataCustomer.iloc[:4])
#获取到第4列为止的所有列:
#print("获取到第4列为止的所有列:")
#print(dataCustomer.iloc[:,:4])

#设置索引
#dataReserveCustomerID = dataReserve.set_index(dataReserve['customer_id'])
#dataReserveTotalPrice = dataReserve.set_index(dataReserve['total_price'])
#print(dataReserveTotalPrice)
#print(dataReserveCustomerID)
#查看行
#print("通过索引查看行:")
#print(dataReserveCustomerID.loc['c_1'])
#print(dataReserveCustomerID.loc['c_4'])


######数据替换#####
#用pandas的replace方法将sex中的woman/man替换成0/1
#sexModifiedInt = dataCustomer['sex'].replace(["woman","man"],["0","1"])
sexModifiedInt = dataCustomer.replace(["woman","man"],["0","1"])
print("替换后的性别表示为:")
print(sexModifiedInt)

#用pandas的replace方法将True\False替换成1/0
#boolModifiedInt = dataHotel['is_business'].astype('int')#replace(["TRUE","FALSE"],["true","false"])
#boolModifiedInt = dataHotel['is_business'].replace(["TRUE","FALSE"],["true","false"])
#print("替换后的布尔值表示为:")
#print(boolModifiedInt)
#production_fault_flg = dataProduction['fault_flg'].replace(["TRUE","FALSE"],["1","0"])
#print(production_fault_flg)
#分界线
print("===================================================")
#rename 方法重命名列,只查看两行数据:
#print(dataProduction.rename(columns={'type':'renametype'}).head(2))
#print(dataProduction.rename(columns={'length':'renamelength'}).head(2))
#修改多个列名参数
#print(dataProduction.rename(columns={'type':'renametype','length':'renamelength'}).head(2))

#分界线
print("===================================================")
#观察数据概况(均值、方差、标准差等等)
#print("求方差、标准差、均值等等衡量数据的因素")
#求方差(以dataCustomer为例)
#print("求各列数据的标准差:")
#print(dataCustomer['age'].std())
#print(dataCustomer['home_latitude'].std())
#print(dataCustomer['home_longitude'].std())
#print("求下列各数据的方差:")
#print(dataHotel['base_price'].var())
#print(dataHotel['hotel_latitude'].var())
#print(dataHotel['hotel_longitude'].var())
#print("取出下列数据的众数、中位数:")
#print(dataHotel['base_price'].median())
#print(dataHotel['hotel_latitude'].median())


#分界线
print("==================================================")
#计算描述统计量:
#print('Maximum:',dataReserve['total_price'].max())
#print('Minimum:',dataReserve['total_price'].min())
#print('Mean:',dataReserve['total_price'].mean())
#print('Sum:',dataReserve['total_price'].sum())
#print('Count:',dataReserve['total_price'].count())
#分界线
print("==================================================")
#查找唯一值
#筛选出唯一值
#print(dataReserve['total_price'].unique())
#查看计数
#print(dataReserve['total_price'].value_counts()) #显示所有的唯一值以及他们出现的次数
#print(dataCustomer['sex'].value_counts()) #筛选出唯一值,往往用于分类.
#print(dataCustomer['sex'].unique())
#分界线
print("==================================================")

#######处理日期和时间数据:######
#print(dataReserve['reserve_datetime'])
#dataReserve['checkin_date']
#dataReserve['checkin_time']
#dataReserve['checkout_date']
#选择日期和时间
#dataReserve[(dataReserve['reserve_datetime']>'2002/1/1 01:00:00')&(dataReserve['reserve_datatime']<='2002/1/1 04:00:00')]


#分界线
print("==================================================")


#缺失值的检查
#dataCustomerColumn1 = dataCustomer.iloc[:,[1],]
#print(dataCustomerColumn1)
#以 dataProductionMissingCategory 为例
#dataProductionMissingCategory.isnull()
#isnull 和 nonull 都能够返回布尔型的值来表示一个值是否缺失:
#print("判断缺失值是否存在:")
#print(dataProductionMissingNum['thickness'].isnull())
#print("dataProductionMissingCategory")
#print(dataProductionMissingCategory.isnull())
print("--------------------------------------------------")
#print("dataProductionMissingNum4RedShift")
#print(dataProductionMissingNum4RedShift.isnull())
print("--------------------------------------------------")
#print("dataProductionMissingNum")
#print(dataProductionMissingNum.isnull())
print("--------------------------------------------------")
#print("dataHotel")
#print(dataHotel.isnull())
print("--------------------------------------------------")
#print("dataMonthMST")
#print(dataMonthMST.isnull())
print("--------------------------------------------------")
#print("dataProduction")
#print(dataProduction.isnull())
print("--------------------------------------------------")
#print("dataMonthlyIndex")
#print(dataMonthlyIndex.isnull())
print("--------------------------------------------------")
#print("dataReserve")
#print(dataReserve.isnull())
print("--------------------------------------------------")
#print("dataCustomer")
#print(dataCustomer.isnull())
print("--------------------------------------------------")
#判断哪些列存在缺失值及其缺失值的数量
#print("判断哪些列存在缺失值及其缺失值的数量:")
#print("dataProductionMissingCategory")
#print(dataProductionMissingCategory.isnull().sum())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataProductionMissingCategory.isnull().any())
print("--------------------------------------------------")
#print("dataProductionMissingNum4RedShift")
#print(dataProductionMissingNum4RedShift.isnull().sum())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataProductionMissingNum4RedShift.isnull().any())
print("--------------------------------------------------")
#print("dataProductionMissingNum")
#print(dataProductionMissingNum.isnull().sum())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataProductionMissingNum.isnull().any())
print("--------------------------------------------------")
#在对dataProductionMissingNum表进行缺失值判断时发现该表格中"thickness"对应的列有"None"值
#但是程序执行的结果显示此列并没有NAN值的反馈,这是由于表格中NAN值是用字符串类型"None"表示的,因此
#要先找到"None"值的位置,再将其替换为"NAN",最后执行程序.
#dataProductionMissingNum['thicknessModified']=dataProductionMissingNum['thickness']
#dataProductionMissingNum.loc[dataProductionMissingNum['thickness']=='None','thicknessModified']=None
#print("修改后的thickness为:")
#print(dataProductionMissingNum.isnull().sum())

#或者:
#dataProductionMissingNum['thickness'] =dataProductionMissingNum['thickness'].replace('None',np.nan)
#print("用nan替换掉None后的表格:")
#print(dataProductionMissingNum['thickness'])
#print(dataProductionMissingNum['thickness'].isnull().sum()) #统计缺失值的数量
#print(dataProductionMissingNum['thickness'].isnull().any()) #判断是否有缺失值存在
#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#分界线
#print("==================================================")
#dataProductionMissingNum['thickness'] = dataProductionMissingNum['thickness'].replace(np.nan,'None')
#print("用None替换掉nan后的表格:")
#print(dataProductionMissingNum['thickness'])
#print(dataProductionMissingNum['thickness'].isnull().sum())
#print(dataProductionMissingNum['thickness'].isnull().any())
#分界线
print("==================================================")
#删除某一列,使用drop方法,并传入参数axis=1(即坐标轴列):
#dataCustomer.drop('age',axis=1)
#print(dataCustomer.drop('age',axis=1))#删除一列属性值
#print(dataCustomer.drop(['age','sex'],axis=1))#删除多列属性值
#如果某一列没有名字,可以使用dataframe.columns通过指定列下标的方式删除该列:
#print("#如果某一列没有名字,可以使用dataframe.columns通过指定列下标的方式删除该列:")
#print(dataCustomer.drop(dataCustomer.columns[1],axis=1).head(2))
#分界线
print("==================================================")
#根据值对行分组:groupby()方法
#print("groupby()方法")
#print(dataCustomer.groupby('sex').mean()) #对sex列的值进行对行分组,并计算每一组的平均值
#print(dataCustomer.groupby('sex').sum())

#按照时间段对行进行分组:
#print(dataReserve['checkin_date'])
#print("=======================================")
#print(dataReserve['checkin_date'].head(3))
#print(dataReserve['checkout_date'].resample('M').sum())
#print(dataReserve['reserve_datetime']) #每一条记录的日期和时间都是数据帧的索引,这是因为resample要求索引的类型必须是类datetime的值
#print(dataReserve['reserve_datetime'].resample('M').mean())
print("=======================================")

#遍历一个列的数据,并且对其进行某种操作.
#for name in dataCustomer['customer_id'][0:2]:
    #print(name.upper())
    #print(name.lower())

#对一列的所有元素应用某个函数
#创建一个函数
#def uppercase(x):
   # return x.upper()
#应用函数,查看两行.
#print(dataCustomer['customer_id'].apply(uppercase)[0:2])





#print(dataProductionMissingNum.isnull().any())
#此时能够找到表格中的确有108个被标记为“None”的属性值,这些值就是NAN值
#找到空值后,使用该列的均值填充空缺值



#dataProductionMissingNum_thicknessfill = dataProductionMissingNum_thickness.fillna(thickness_mean,inplace=True)
#print(dataProductionMissingNum_thicknessfill)

print("--------------------------------------------------")
#print("dataCustomer")
#print(dataCustomer.isnull().sum())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataCustomer.isnull().any())
print("--------------------------------------------------")
#print("dataReserve")
#print(dataReserve.isnull().sum())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataReserve.isnull().any())
print("--------------------------------------------------")
#print("dataProduction")
#print(dataProduction.isnull().sum())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataProduction.isnull().any())
print("--------------------------------------------------")
#print("dataHotel")
#print(dataHotel.isnull().sum())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataHotel.isnull().any())
print("--------------------------------------------------")
#print("dataMonthlyIndex")
#print(dataMonthlyIndex.isnull().sum())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataMonthlyIndex.isnull().any())
print("--------------------------------------------------")
#print("dataMonthMST")
#print(dataMonthMST.isnull().sum())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataMonthMST.isnull().any())
print("--------------------------------------------------")


#分界线
print("==================================================")


#缺失值的填充(以dataProductionMissingCategory为例)
#print("填充dataProductionMissingCategory中的缺失值")
dataProductionMissingCategory['typeModified']=dataProductionMissingNum['type']
dataProductionMissingCategory.loc[dataProductionMissingNum['type']==None,'typeModified']="B"
#print(dataProductionMissingCategory.isnull())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataProductionMissingCategory.isnull().any())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(dataProductionMissingCategory.isnull().sum())
print("--------------------------------------------------")
#填充空缺值的方法:pandas中的fillna()方法
#print("用fillna()方法填充缺失值")
dataProductionMissingCategory_type = dataProductionMissingCategory.iloc[:,[0]]
dataProductionMissingCategory_type_mean = np.mean(dataProductionMissingCategory_type)
#print(dataProductionMissingCategory_type)
filler = dataProductionMissingCategory_type.fillna('B')
#print(filler)
#数据表合并
#df3 = pd.merge(df1,df2,how='inner'\'left'\'right'\'outer')
#append
#result = df1.append(df2)

#异常值的识别
#一、酒店数据
#1）酒店数据(dataHotel)
datahotel_baseprice = dataHotel.iloc[:,[1]]
datahotel_hotel_latitude = dataHotel.iloc[:,[4]]
datahotel_hotel_longitude = dataHotel.iloc[:,[5]]
#print(datahotel_baseprice.describe())
print("---------------------------------")
#print(datahotel_hotel_latitude.describe())
print("---------------------------------")
#print(datahotel_hotel_longitude.describe())
#2) 顾客数据(dataCustomer)
dataCustomer_age = dataCustomer.iloc[:,[1]]
dataCustomer_home_latitude = dataCustomer.iloc[:,[3]]
dataCustomer_home_longitude = dataCustomer.iloc[:,[4]]
#print(dataCustomer_age.describe())
print("---------------------------------")
#print(dataCustomer_home_latitude.describe())
print("---------------------------------")
#print(dataCustomer_home_longitude.describe())
#3)预订数据(dataReserve)
dataReserve_datetime = dataReserve.iloc[:,[3]]
dataReserve_checkindate = dataReserve.iloc[:,[4]]
dataReserve_checkintime = dataReserve.iloc[:,[5]]
dataReserve_checkoutdate = dataReserve.iloc[:,[6]]
dataReserve_peoplenum = dataReserve.iloc[:,[7]]
dataReserve_totalprice = dataReserve.iloc[:,[8]]
#print(dataReserve_datetime.describe())
#print("---------------------------------")
#print(dataReserve_checkindate.describe())
#print("---------------------------------")
#print(dataReserve_checkintime.describe())
#print("---------------------------------")
#print(dataReserve_checkoutdate.describe())
#print("---------------------------------")
#print(dataReserve_peoplenum.describe())
#print("---------------------------------")
#print(dataReserve_totalprice.describe())
#1)酒店totalprice\baseprice数据的处理
#outliersTotalPrice = dataReserve[abs(dataReserve['total_price']-np.mean(dataReserve['total_price']))/np.std(dataReserve['total_price'])>=3].reset_index()
#print(outliersTotalPrice)

#清除异常点后的数据集
#cleanedCustomerData = dataCustomer[abs(dataCustomer-np.mean(dataCustomer))/np.std(dataCustomer)<3].reset_index()
#cleanedReserveData = dataReserve[abs(dataReserve-np.mean(dataReserve))/np.std(dataReserve)<3].reset_index()
cleanedReserveData = dataReserve[abs(dataReserve['total_price']-np.mean(dataReserve['total_price']))/np.std(dataReserve['total_price'])<3].reset_index()
cleanedHotelData = dataHotel[abs(dataHotel['base_price']-np.mean(dataHotel['base_price']))/np.std(dataHotel['base_price'])<3].reset_index()
cleanedMonthlyIndexData = dataMonthlyIndex[abs(dataMonthlyIndex-np.mean(dataMonthlyIndex))/np.std(dataMonthlyIndex)<3].reset_index()
#print("清除异常点后的数据集为:")
#print(cleanedReserveData)
print("清洗后的cleanedReserveData输出结果为:")
print(cleanedReserveData)
print("清洗后的clleanMonthlyIndexData输出结果为:")
print(cleanedMonthlyIndexData)
#print(cleanedHotelData)
#print(cleanedCustomerData)
#print(cleanedHotelData['base_price'].head(50))

#outliersBasePrice = dataHotel[abs(dataHotel['base_price']-np.mean(dataHotel['base_price']))/np.std(dataHotel['base_price'])>=3].reset_index()
#print(outliersBasePrice)
#2)产品length\thickness数据的处理
#outliersLength = dataProduction[abs(dataProduction['length']-np.mean(dataProduction['length']))/np.std(dataProduction['length'])>=3].reset_index()
#print(outliersLength)
#outliersThickness = dataProduction[abs(dataProduction['thickness']-np.mean(dataProduction['thickness']))/np.std(dataProduction['thickness'])>=3].reset_index()
#print(outliersThickness)
#分界线
print("======================================================")

#特征值标准化\归一化:对某个特征进行转换,使其平均值为0,标准差为1.
#1)最大最小标准化:
#获取各个指标的最大值和最小值
#hotel_baseprice
#basePriceMax = np.max(datahotel_baseprice)
#basePriceMin = np.min(datahotel_baseprice)
#print(basePriceMax)
#print(basePriceMin)
#datahotel_baseprice = (datahotel_baseprice-basePriceMin)/(basePriceMax-basePriceMin)
#print(datahotel_baseprice)
#reserve_totalprice
#totalPriceMax = np.max(dataReserve_totalprice)
#totalPriceMin = np.min(dataReserve_totalprice)
#print(totalPriceMax)
#print(totalPriceMin)
#dataReserve_totalprice = (dataReserve_totalprice-totalPriceMin)/(totalPriceMax-totalPriceMin)
#print(dataReserve_totalprice)

#2)零均值标准化
def Z_ScoreNormalize(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    data = (data - data_mean)/data_std
    return data
#对清理异常值后的数据进行零标准化
#normalize_totalprice = Z_ScoreNormalize(cleanedReserveData['total_price'])
temp3 = Z_ScoreNormalize(cleanedMonthlyIndexData)
print("零均质化后的MonthlyIndexData:")
print(temp3)
newMonthlyIndex = temp3
newMonthlyIndex['year_month'] = dataMonthlyIndex['year_month']
print("清洗、整合标准化后的MonthlyIndex数据:")
print(newMonthlyIndex)
#temp2=Z_ScoreNormalize(cleanedReserveData)
#print("零均值化后的ReserveData:")
#print(temp2)
#newReserve = temp2
#newReserve['reserve_id'] = cleanedReserveData['reserve_id']
#newReserve['hotel_id'] = cleanedReserveData['hotel_id']
#newReserve['customer_id'] = cleanedReserveData['customer_id']
#newReserve['reserve_datetime'] = cleanedReserveData['reserve_datetime']
#newReserve['checkin_date'] = cleanedReserveData['checkin_date']
#newReserve['checkin_time'] = cleanedReserveData['checkin_time']
#newReserve['checkout_date'] = cleanedReserveData['checkout_date']
#print("清洗标准化后的Reserve数据:")
#print(newReserve)
#print(normalize_totalprice)
#normalize_baseprice = Z_ScoreNormalize(cleanedHotelData['base_price'])
#normalize_latitude = Z_ScoreNormalize(cleanedHotelData['hotel_latitude'])
#normalize_longitude = Z_ScoreNormalize(cleanedHotelData['hotel_latitude'])
temp=Z_ScoreNormalize(cleanedHotelData)
#print(normalizeHotelData)
newHotel = temp
newHotel['big_area_name'] = cleanedHotelData['big_area_name']
newHotel['hotel_id'] = cleanedHotelData['hotel_id']
newHotel['small_area_name'] = cleanedHotelData['small_area_name']
print("清洗标准化后的Hotel数据:")
print(newHotel)

#print(normalize_latitude)
#print(normalize_baseprice)
#normalize_baseprice = Z_ScoreNormalize(datahotel_baseprice)
#(normalize_baseprice)
#分界线
print("======================================================")
#属性字段的规整化(将一些属性列中的非数值型数据转化为数值型数据)

#分界线
print("======================================================")

#将特征离散化
#创建特征
#age = np.array(dataCustomer['age'])
#print(age)
#创建二值化器
#binarizer = Binarizer(18)
#转换特征
#binarizer.fit_transform(age)
#print(binarizer.fit_transform(age))
#2)根据多个阈值将数值型特征离散化:
#print(np.digitize(age,bins=[20,30,64])) #bins参数中的每个数字表示的是每个区间的左边界(左闭右开)
#分界线
print("======================================================")

#两个字段之间的相关性分析
#print("相关性分析")
#relevance = dataReserve['people_num'].corr(dataReserve['total_price']) #相关系数在-1到1之间,接近1为正相关,接近-1为负相关,0为不相关
#print(relevance)
#通过相关性分析得出:顾客的数量与酒店的总收益之间具有一定的相关性
#将顾客数量和九点总收益进行数据可视化,分析其关联
#用Matplotlib画散点图,展示peoplenum 与 totalprice之间的关系
#peoplenum = dataReserve_peoplenum
#totalprice = dataReserve_totalprice
#plt.scatter(peoplenum,totalprice,marker='.')
#plt.show()

#使用主成分进行特征降维
#思想:对于给定的一组特征,在保留信息量的同时减少特征的数量.
#解决方案:使用scikit-learn库中的主成分分析工具(PCA):

##加载数据##
#print(dataMonthlyIndex['year_month'].describe())
#print(dataMonthlyIndex['sales_amount'].describe())
#print(dataMonthlyIndex['customer_number'].describe())

#digits = datasets.load_digits()
#标准化特征矩阵
#features = StandardScaler().fit_transform(digits.data)
#创建可以保留80%信息量(用方差表示)的PCA
#pca = PCA(n_components=0.80,whiten=True)
#执行PCA
#features_pca = pca.fit_transform(features)
#显示结果
#print(features_pca)
#print("Original number of features:",features.shape[1])
#print("Reduced number of features:",features_pca.shape[1])

#scaler = preprocessing.StandardScaler()
#data = cleanedReserveData.loc[:,['people_num','total_price']]

data2 = newHotel.loc[:,['base_price','hotel_latitude','hotel_longitude']]
print(data2)
pca = PCA(n_components=0.85,whiten=True)
data3 = newMonthlyIndex.loc[:,['sales_amount','customer_number']]
print("data3:")
print(data3)
data3_pca = pca.fit_transform(data3)
print("data3_pca:")
print(data3_pca)
plt.plot(data3_pca,"r*")
plt.show()


#print(cleanedHotelData.loc[:,['base_price','hotel_latitude','hotel_longitude']])
#print(data)
pca = PCA(n_components=0.85,whiten=True)
#data_pca = pca.fit_transform(data)
data2_pca = pca.fit_transform(data2)
#print(data_pca)
#print(data.shape[1])
#(data_pca.shape[1])

#print(data2_pca)
#print(data2.shape[1])
#print(data2_pca.shape[1])

#可视化
#plt.plot(data2,"k*")
#plt.show()
#plt.plot(data2_pca,'k*')
#plt.show()


#自定义数据集
#Review = pd.read_csv('')















































