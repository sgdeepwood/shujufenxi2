import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("./train.csv")
test  = pd.read_csv("./test.csv")
print ('训练数据train:',train.shape,'测试数据test:',test.shape)#查看两个数据集的形状

full = train.append( test , ignore_index = True )#合并数据集
print ('合并后的数据集:',full.shape)


print('处理前:')#age，fare
print(full.isnull().sum())
full['Age']=full['Age'].fillna(full['Age'].mean())
full['Fare']=full['Fare'].fillna(full['Age'].mean())
print('处理后:')
print(full.isnull().sum())

print(full['Embarked'].value_counts())#embarked
full['Embarked']=full['Embarked'].fillna('S')
print(full)

print(full['Cabin'].value_counts())#cabin
full['Cabin']=full['Cabin'].fillna('U')#缺失数据多，暂标未知U

#特征提取
sex_mapdict={'male':1,'female':0}#sex
full['Sex']=full['Sex'].map(sex_mapdict)

embarkedDf=pd.DataFrame()#embarked的提取
embarkedDf=pd.get_dummies(full['Embarked'],prefix='Embarked')
print("提取后的embarked:",embarkedDf)
pclassDf=pd.DataFrame()#pclassdf的提取
pclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
print("提取后的Pclass:",pclassDf)



def gettitle(name):#处理姓名
      str1=name.split(',')[1]
      str2=str1.split('.')[0]
      str3=str2.strip()
      return str3

titleDf=pd.DataFrame()
titleDf['Title']=full['Name'].map(gettitle)



title_mapdict={
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
titleDf['Title']=titleDf['Title'].map(title_mapdict)
titleDf=pd.get_dummies(titleDf['Title'])


cabinDf=pd.DataFrame()#cabin
full['Cabin']=full['Cabin'].map(lambda c : c[0])
cabinDf=pd.get_dummies(full['Cabin'],prefix='Cabin')

familyDf=pd.DataFrame()#parch，sibsp+1
familyDf['FamilySize']=full['Parch']+full['SibSp']+1
familyDf['Family_Single']=familyDf['FamilySize'].map(lambda s:1 if s == 1 else 0)
familyDf['Family_Small']=familyDf['FamilySize'].map(lambda s:1 if 2 <= s <=4 else 0)
familyDf['Family_Large']=familyDf['FamilySize'].map(lambda  s:1 if 5 <= s else 0)


full=pd.concat([full,embarkedDf],axis=1)#将提取后的特征后并进数据集中
full.drop('Embarked',axis=1,inplace=True)
full=pd.concat([full,pclassDf],axis=1)
full.drop('Pclass',axis=1,inplace=True)
full=pd.concat([full,titleDf],axis=1)
full.drop('Name',axis=1,inplace=True)
full=pd.concat([full,cabinDf],axis=1)
full.drop('Cabin',axis=1,inplace=True)
full=pd.concat([full,familyDf],axis=1)
full.drop('FamilySize',axis=1,inplace=True)

'''特征选择'''
corrDf=full.corr()#计算各个特征的相关系数
print(corrDf)
corrDf['Survived'].sort_values(ascending=False)
full_X = pd.concat( [titleDf,pclassDf, familyDf, full['Fare'], full['Sex'],cabinDf, embarkedDf, ] , axis=1 )#提取与survived相关系数较高的列
print("提取的属性:",full_X)

#建立模型用的训练数据集和测试数据集
sourceRaw=891
source_x=full_X.loc[0:sourceRaw-1,:]
source_y=full.loc[0:sourceRaw-1,'Survived']
pred_X = full_X.loc[sourceRaw:,:]

size=np.arange(0.6,1,0.1)#（0.6~0.9）
scorelist=[[],[],[],[],[],[],[]]
from sklearn.model_selection import train_test_split
for i in range(0,4):
    train_X, test_X, train_y, test_y = train_test_split(source_x ,
                                                        source_y,
                                                      train_size=size[i],
                                                        random_state=5)
#，train_x、train_y为训练集的特征变量和目标变量数据，test_x、test_y则为测试集的特征变量和目标变量数据

    #逻辑回归
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit( train_X , train_y )
    scorelist[0].append(model.score(test_X , test_y ))

    #随机森林
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit( train_X , train_y )
    scorelist[1].append(model.score(test_X , test_y ))

    #支持向量机
    from sklearn.svm import SVC
    model = SVC()
    model.fit( train_X , train_y )
    scorelist[2].append(model.score(test_X , test_y ))

    #决策树分类
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit( train_X , train_y )
    scorelist[3].append(model.score(test_X , test_y ))

    #KNN最邻近算法
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit( train_X , train_y )
    scorelist[4].append(model.score(test_X , test_y ))

    #朴素贝叶斯分类
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit( train_X , train_y )
    scorelist[5].append(model.score(test_X , test_y ))

    #神经网络
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier()
    model.fit( train_X , train_y )
    scorelist[6].append(model.score(test_X , test_y ))

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
color_list = ('red', 'blue', 'lightgreen', 'cornflowerblue', 'turquoise', 'magenta','black')
for i in range(0,7):
    plt.plot(size,scorelist[i],color=color_list[i])
plt.legend(['逻辑回归', '随机森林','支持向量机','决策树分类', 'KNN最邻近算法','朴素贝叶斯','神经网络'])

plt.xlabel('训练集占比')
plt.ylabel('准确率')
plt.title('不同的模型随着训练集占比变化曲线')
plt.show()

'''方案实施'''
pred_Y=model.predict(pred_X)
passenger_id=full.loc[sourceRaw:,'PassengerId']
predDf=pd.DataFrame({'PassengerId':passenger_id,'Survived':pred_Y})
predDf.shape
predDf.head()
predDf.to_csv('result.csv',index=False)
