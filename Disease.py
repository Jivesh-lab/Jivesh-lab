from tkinter import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
# from gui_stuff import *

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

def clustering():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform clustering using KMeans
    n_clusters = len(disease)  # Number of clusters should be equal to the number of disease classes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Group data points by cluster label
    cluster_data = {}
    for cluster, disease_label in zip(cluster_labels, np.ravel(y)):
        if cluster not in cluster_data:
            cluster_data[cluster] = []
        cluster_data[cluster].append(disease_label)

    # Find the majority disease label for each cluster
    cluster_majority_label = {}
    for cluster, diseases in cluster_data.items():
        cluster_majority_label[cluster] = max(set(diseases), key=diseases.count)

    print("Cluster Labels:", cluster_labels)
    print("Cluster Data:", cluster_data)
    print("Cluster Majority Label:", cluster_majority_label)

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    l2 = [0] * len(l1)

    for k in range(len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    input_test = [l2]
    input_test_scaled = scaler.transform(input_test)

    # Predict the cluster label for the input test data
    test_cluster_label = kmeans.predict(input_test_scaled)

    print("Test Cluster Label:", test_cluster_label)

    # Get the majority disease label for the predicted cluster
    predicted = cluster_majority_label[test_cluster_label[0]]

    if 0 <= predicted < len(disease):
        t6.delete("1.0", END)
        t6.insert(END, disease[predicted])
    else:
        t6.delete("1.0", END)
        t6.insert(END, "Not Found")

def knn():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=5)  # You can choose the number of neighbors (k) based on your dataset

    # Perform k-fold cross-validation
    scores = cross_val_score(clf, X_scaled, np.ravel(y), cv=5)  # 5-fold cross-validation
    avg_accuracy = np.mean(scores)

    clf.fit(X_scaled, np.ravel(y))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    l2 = [0] * len(l1)

    for k in range(len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    input_test = [l2]
    input_test_scaled = scaler.transform(input_test)

    predict = clf.predict(input_test_scaled)
    predicted = int(predict[0])

    if 0 <= predicted < len(disease):
        t5.delete("1.0", END)
        t5.insert(END, disease[predicted] + "\nAverage Accuracy: {:.2f}".format(avg_accuracy))
    else:
        t5.delete("1.0", END)
        t5.insert(END, "Not Found")

def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

# gui_stuff------------------------------------------------------------------------------------

root = Tk()
root.configure(background='lightgrey')

# entry variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# Set the background color for the entire window
root.configure(bg="lightblue")

# Heading
w2 = Label(root, justify=LEFT, text="Disease Predictor", fg="Black",bg="lightblue")
w2.config(font=("Agency FB", 44, "bold","italic"))
w2.grid(row=1, column=0, columnspan=4, padx=400, pady=20)

# Labels
NameLb = Label(root, text="Name of the Patient", fg="black", bg="orange")
NameLb.grid(row=3, column=0, pady=15, padx=20, sticky='w')

S1Lb = Label(root, text="Symptom 1", fg="black", bg="orange")
S1Lb.grid(row=6, column=0, pady=10, padx=20, sticky='w')

S2Lb = Label(root, text="Symptom 2", fg="black", bg="orange")
S2Lb.grid(row=7, column=0, pady=10, padx=20, sticky='w')

S3Lb = Label(root, text="Symptom 3", fg="black", bg="orange")
S3Lb.grid(row=8, column=0, pady=10, padx=20, sticky='w')

S4Lb = Label(root, text="Symptom 4", fg="black", bg="orange")
S4Lb.grid(row=10, column=0, pady=10, padx=20, sticky='w')

S5Lb = Label(root, text="Symptom 5", fg="black", bg="orange")
S5Lb.grid(row=11, column=0, pady=10, padx=20, sticky='w')

lrLb = Label(root, text="DecisionTree", fg="white", bg="green")
lrLb.grid(row=13, column=0, pady=15, padx=30, sticky='w')

destreeLb = Label(root, text="RandomForest", fg="white", bg="green")
destreeLb.grid(row=15, column=0, pady=15, padx=30, sticky='w')

ranfLb = Label(root, text="NaiveBayes", fg="white", bg="green")
ranfLb.grid(row=17, column=0, pady=15, padx=30, sticky='w')

kn= Label(root, text="K-nearest Neighbour", fg="white", bg="green")
kn.grid(row=20, column=0, pady=15, padx=30, sticky='w')

cluster = Label(root, text="K-means", fg="white", bg="green")
cluster.grid(row=23, column=0, pady=15, padx=30, sticky='w')
# Entries
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name, font=("Arial", 16))
NameEn.grid(row=3, column=1, padx=20)

S1En = OptionMenu(root, Symptom1, *OPTIONS)
S1En.grid(row=6, column=1, padx=20)

S2En = OptionMenu(root, Symptom2, *OPTIONS)
S2En.grid(row=7, column=1, padx=20)

S3En = OptionMenu(root, Symptom3, *OPTIONS)
S3En.grid(row=8, column=1, padx=20)

S4En = OptionMenu(root, Symptom4, *OPTIONS)
S4En.grid(row=10, column=1, padx=20)

S5En = OptionMenu(root, Symptom5, *OPTIONS)
S5En.grid(row=11, column=1, padx=20)

dst = Button(root, text="DecisionTree", command=DecisionTree, bg="darkgreen", fg="white", font=("Arial", 16))
dst.grid(row=5, column=3, padx=20, pady=10)

rnf = Button(root, text="RandomForest", command=randomforest, bg="darkgreen", fg="white", font=("Arial", 16))
rnf.grid(row=6, column=3, padx=20, pady=10)

lr = Button(root, text="NaiveBayes", command=NaiveBayes, bg="darkgreen", fg="white", font=("Arial", 16))
lr.grid(row=7, column=3, padx=20, pady=10)

knn_btn = Button(root, text="k-Nearest Neighbors", command=knn, bg="darkgreen", fg="white", font=("Arial", 16))
knn_btn.grid(row=8, column=3, padx=20, pady=10)

clustering_btn = Button(root, text="Clustering", command=clustering, bg="darkgreen", fg="white", font=("Arial", 16))
clustering_btn.grid(row=11, column=3, padx=20, pady=10)

# Text fields
t1 = Text(root, height=1, width=40, bg="#36454F", fg="white", font=("Arial", 14))
t1.grid(row=13, column=1, padx=20, pady=10)

t2 = Text(root, height=1, width=40, bg="#36454F", fg="white", font=("Arial", 14))
t2.grid(row=15, column=1, padx=20, pady=10)

t3 = Text(root, height=1, width=40, bg="#36454F", fg="white", font=("Arial", 14))
t3.grid(row=17, column=1, padx=20, pady=10)

t5 = Text(root, height=1, width=40, bg="#36454F", fg="white", font=("Arial", 14))
t5.grid(row=20, column=1, padx=20, pady=10)

t6 = Text(root, height=1, width=40, bg="#36454F", fg="white", font=("Arial", 14))
t6.grid(row=23, column=1, padx=20, pady=10)

root.geometry("900x500")
root.mainloop()
