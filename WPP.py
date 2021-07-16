from tkinter import *
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import tree
    
#Import Data
df = pd.read_csv("Data.csv")
features = ["Award_Work","Care-taker_WhileWorking","SP_Supp","Using_DCC","ComfyLeave_ChildSick","MatPat_Leave","Flex_Whours","Eval_WorkProduced"]
labels = ["Prod"]

#Ordinal Encoding
ord_enc = OrdinalEncoder()
headers = ["Award_Work","Care-taker_WhileWorking","SP_Supp","Using_DCC","ComfyLeave_ChildSick","MatPat_Leave","Flex_Whours","Eval_WorkProduced", "Prod"]
df["Award_Work"] = ord_enc.fit_transform(df[["Award_Work"]])
df["Care-taker_WhileWorking"] = ord_enc.fit_transform(df[["Care-taker_WhileWorking"]])
df["SP_Supp"] = ord_enc.fit_transform(df[["SP_Supp"]])
df["Using_DCC"] = ord_enc.fit_transform(df[["Using_DCC"]])
df["ComfyLeave_ChildSick"] = ord_enc.fit_transform(df[["ComfyLeave_ChildSick"]])
df["MatPat_Leave"] = ord_enc.fit_transform(df[["MatPat_Leave"]])
df["Flex_Whours"] = ord_enc.fit_transform(df[["Flex_Whours"]])
df["Eval_WorkProduced"] = ord_enc.fit_transform(df[["Eval_WorkProduced"]])
df["Prod"] = ord_enc.fit_transform(df[["Prod"]])

print(df.info())

#Test-Train Split
X = df[features]
y = df[labels]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.30)

#RFE
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=4, step=1)
selector = selector.fit(X, y.values.ravel())
#print(selector.support_)
#print(selector.ranking_)

l1=['Yes','No']
prod=['Increased','Decreased']
l2=[]
for i in range(0,len(features)):
    l2.append(0)
    
def DecisionTree():
    #Model
    dtc = tree.DecisionTreeClassifier(max_depth=5) 
    dtc = dtc.fit(X,y)
    
    #Feature Importance
    dtc.feature_importances_
    df = pd.DataFrame({ 'Feature_names':X.columns, 'Importances':dtc.feature_importances_})
    print(df.sort_values(by='Importances',ascending=False))
    
    #Visualize the Decision Tree
    fig, axes = plt.subplots(figsize = (8,5), dpi=300)
    fig.suptitle('DT of Working Productivity Prediction', fontsize=16)
    tree.plot_tree(dtc)
    plt.show()
    fig.savefig('dtc.png')

    #Predict
    y_pred=dtc.predict(X_test)
    print('Accuracy: ',accuracy_score(y_test, y_pred))
    
    #Read User Input
    pfeatures = [Feature1.get(),Feature2.get(),Feature3.get(),Feature4.get(),Feature5.get(),Feature6.get(),Feature7.get(),Feature8.get()]

    for z in range(0,len(pfeatures)):
        if(pfeatures[z]=='Yes'):
            l2[z]=1
        elif(pfeatures[z]=='Choose Option'):   
            l2[z]=2
        else:
            l2[z]=0
            
    inputtest = [l2]
    exist = 2 in l2
    if (exist == True):
        t1.delete("1.0", END)
        t1.insert(END, "Answer All Questions!")
    else:
        predict = dtc.predict(inputtest)
        predicted=predict[0] 
    
        if(predicted==1.0):
            t1.delete("1.0", END)
            t1.insert(END, "Productivity Will Increase")
        else:
            t1.delete("1.0", END)
            t1.insert(END, "Productivity Will Decrease")  
    
    #Print Text representation of Decision Tree 
    #text_representation = tree.export_text(dtc)
    #print(text_representation)        

def logisticregression():
    #Model
    lrc = LogisticRegression()
    lrc.fit(X,y.values.ravel())
    
    #Predict
    y_pred=lrc.predict(X_test)
    print('Accuracy: ',accuracy_score(y_test, y_pred))
    
    #Read User Input
    pfeatures = [Feature1.get(),Feature2.get(),Feature3.get(),Feature4.get(),Feature5.get(),Feature6.get(),Feature7.get(),Feature8.get()]

    for z in range(0,len(pfeatures)):
        if(pfeatures[z]=='Yes'):
            l2[z]=1
        elif(pfeatures[z]=='Choose Option'):   
            l2[z]=2
        else:
            l2[z]=0
            
    inputtest = [l2]
    exist = 2 in l2
    if (exist == True):
        t2.delete("1.0", END)
        t2.insert(END, "Answer All Questions!")
    else:
        predict = lrc.predict(inputtest)
        predicted=predict[0] 
    
        if(predicted==1.0):
            t2.delete("1.0", END)
            t2.insert(END, "Productivity Will Increase")
        else:
            t2.delete("1.0", END)
            t2.insert(END, "Productivity Will Decrease")  

# GUI     
root = Tk()
root.wm_title("WPP | Lubaba Islam")
root.configure(background='gray17')
Feature1 = StringVar()
Feature1.set("Choose Option")
Feature2 = StringVar()
Feature2.set("Choose Option")
Feature3 = StringVar()
Feature3.set("Choose Option")
Feature4 = StringVar()
Feature4.set("Choose Option")
Feature5 = StringVar()
Feature5.set("Choose Option")
Feature6 = StringVar()
Feature6.set("Choose Option")
Feature7 = StringVar()
Feature7.set("Choose Option")
Feature8 = StringVar()
Feature8.set("Choose Option")
#Name = StringVar()

w2 = Label(root, text="", fg="dark orange", bg="gray17")
w2.config(font=("Roboto",1,"bold"))
w2.grid(row=1, column=0, columnspan=4, pady=10)

w2 = Label(root, text="", fg="dark orange", bg="gray17")
w2.config(font=("Roboto",1,"bold"))
w2.grid(row=22, column=0, columnspan=4, pady=7)

w2 = Label(root, text="WORKING PRODUCTIVITY PREDICTOR", fg="White", bg="gray17")
w2.config(font=("Roboto",14,"bold"))
w2.grid(row=2, column=0, columnspan=4)

w2 = Label(root, text="", fg="dark orange", bg="gray17")
w2.config(font=("Roboto",2,"bold"))
w2.grid(row=17, column=0, columnspan=4)

w3 = Label(root, text="________________________________________________________________________________________________", fg="dark orange", bg="gray17")
w3.config(font=("Roboto",12,"bold"))
w3.grid(row=3, column=0, columnspan=4)

w3 = Label(root, text="________________________________________________________________________________________________", fg="dark orange", bg="gray17")
w3.config(font=("Roboto",12,"bold"))
w3.grid(row=15, column=0, columnspan=4)

w3 = Label(root, text="________________________________________________________________________________________________", fg="dark orange", bg="gray17")
w3.config(font=("Roboto",12,"bold"))
w3.grid(row=19, column=0, columnspan=4)

S1Lb = Label(root, text="Does your performance get recognized with Award?", fg="White", bg="gray17")
S1Lb.config(font=("Roboto",12,"bold"))
S1Lb.grid(row=6, column=0, pady=10, padx=20, sticky=W)
S2Lb = Label(root, text="Do you have care taker for your children while working?", fg="White", bg="gray17")
S2Lb.config(font=("Roboto",12,"bold"))
S2Lb.grid(row=7, column=0, pady=10, padx=20, sticky=W)
S3Lb = Label(root, text="Does your family (spouse) support you?", fg="White", bg="gray17")
S3Lb.config(font=("Roboto",12,"bold"))
S3Lb.grid(row=8, column=0, pady=10, padx=20, sticky=W)
S4Lb = Label(root, text="Are you using day care center for your child?", fg="White", bg="gray17")
S4Lb.config(font=("Roboto",12,"bold"))
S4Lb.grid(row=9, column=0, pady=10, padx=20, sticky=W)
S5Lb = Label(root, text="Are you comfortable taking leave while your child is sick?", fg="White", bg="gray17")
S5Lb.config(font=("Roboto",12,"bold"))
S5Lb.grid(row=10, column=0, pady=10, padx=20, sticky=W)
S6Lb = Label(root, text="Does your company provide maternity/paternity leave?", fg="White", bg="gray17")
S6Lb.config(font=("Roboto",12,"bold"))
S6Lb.grid(row=11, column=0, pady=10, padx=20, sticky=W)
S7Lb = Label(root, text="Do you have flexible working hours?", fg="White", bg="gray17")
S7Lb.config(font=("Roboto",12,"bold"))
S7Lb.grid(row=12, column=0, pady=10, padx=20, sticky=W)
S8Lb = Label(root, text="Is your performance evaluation based on the work you produced?", fg="White", bg="gray17")
S8Lb.config(font=("Roboto",12,"bold"))
S8Lb.grid(row=13, column=0, pady=10, padx=20, sticky=W)

dst = Label(root, text="Productivity Prediction (Decision Tree):", fg="White", bg="gray17")
dst.config(font=("Roboto",12,"bold"))
dst.grid(row=16, column=0, pady=10, padx=20, sticky=W)

lrLb = Label(root, text="Productivity Prediction (Logistic Regression):", fg="White", bg="gray17")
lrLb.config(font=("Roboto",12,"bold"))
lrLb.grid(row=20, column=0, pady=10, padx=20, sticky=W)

OPTIONS = sorted(l1)

S1 = OptionMenu(root, Feature1,*OPTIONS)
S1.grid(row=6, column=1)
S1.config(bg = "dark olivegreen1", fg="gray17")
S2 = OptionMenu(root, Feature2,*OPTIONS)
S2.grid(row=7, column=1)
S2.config(bg = "dark olivegreen1", fg="gray17")
S3 = OptionMenu(root, Feature3,*OPTIONS)
S3.grid(row=8, column=1)
S3.config(bg = "dark olivegreen1", fg="gray17")
S4 = OptionMenu(root, Feature4,*OPTIONS)
S4.grid(row=9, column=1)
S4.config(bg = "dark olivegreen1", fg="gray17")
S5 = OptionMenu(root, Feature5,*OPTIONS)
S5.grid(row=10, column=1)
S5.config(bg = "dark olivegreen1", fg="gray17")
S6 = OptionMenu(root, Feature6,*OPTIONS)
S6.grid(row=11, column=1)
S6.config(bg = "dark olivegreen1", fg="gray17")
S7 = OptionMenu(root, Feature7,*OPTIONS)
S7.grid(row=12, column=1)
S7.config(bg = "dark olivegreen1", fg="gray17")
S8 = OptionMenu(root, Feature8,*OPTIONS)
S8.grid(row=13, column=1)
S8.config(bg = "dark olivegreen1", fg="gray17")

dst = Button(root, text="PREDICT", command=DecisionTree, fg="gray17", bg="dark olivegreen3",width="10")
dst.config(font=("Roboto",10,"bold"))
dst.grid(row=17, column=0, columnspan=3)

rfc = Button(root, text="PREDICT", command=logisticregression, fg="gray17", bg="dark olivegreen3",width="10")
rfc.config(font=("Roboto",10,"bold"))
rfc.grid(row=21, column=0, columnspan=3)

t1 = Text(root, height=1.3, width=22, bg="white",fg="black")
t1.config(font=("Roboto",10,"bold"))
t1.grid(row=16, column=1, padx=20)

t2 = Text(root, height=1.3, width=22, bg="white",fg="black")
t2.config(font=("Roboto",10,"bold"))
t2.grid(row=20, column=1, padx=20)

root.mainloop()