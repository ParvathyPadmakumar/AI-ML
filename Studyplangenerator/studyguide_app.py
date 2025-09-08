import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
relation=pd.read_csv("prerequisite_annotations.csv")
id_map=pd.read_csv("topics_to_resources.csv")


id_to_name=dict(zip(id_map['200_topic_id'],id_map['200_topic_name']))
print(id_to_name)
relation.drop(['annotator_id'],axis=1,inplace=True)

relation=relation[relation['prereq_relation']==1]


import networkx as nx

G=nx.DiGraph()
for _, row in relation.iterrows():
    src=id_to_name.get(row["source_topic_id"])
    dst=id_to_name.get(row["target_topic_id"])
    if src and dst:
        G.add_edge(src, dst)#src->dst



def build_pre_tree(G, topic, visited=None):
    if visited is None:
        visited = set()
    if topic in visited:
        return None,0
    visited.add(topic)
    tree_of_prerelations = {}
    count_pre=0
    for parent in sorted(G.predecessors(topic)):
        if parent not in visited:
            subtree,subcount= build_pre_tree(G, parent, visited)
            if subtree is not None:
                tree_of_prerelations[parent]=subtree
                count_pre+=1+subcount
            else:
                tree_of_prerelations[parent]={}
    return tree_of_prerelations,count_pre

def build_de_tree(G, topic, visited=None):
    if visited is None:
        visited = set()
    if topic in visited:
        return {},0
    visited.add(topic)
    tree_of_ancrelations={}
    count_de=0

    for child in sorted(G.successors(topic)):
        subtree,subcount2 = build_de_tree(G, child, visited)
        if subtree is not None:
            tree_of_ancrelations[child]=subtree
            count_de+=1+subcount2
    return tree_of_ancrelations,count_de


def print_tree(tree_of_relations,visited=None,indent=0):
    if visited is None:
        visited = set()
    lines=[]
    for topic, subtopics in tree_of_relations.items():
        if topic in visited:
            continue
        visited.add(topic)
        lines.append(f"--> {topic}")
        if subtopics: 
            lines.extend(print_tree(subtopics, visited))
    return lines

def extract_count(G):
    data=[]
    for node in G.nodes():
        prereq=build_pre_tree(G,node)[1]
        deps=build_de_tree(G,node)[1]
        total_degree=prereq+deps
        data.append({"topic": node,
            "prerequisites_count": prereq,
            "dependents_count": deps,
            "total_degree": total_degree})
    return pd.DataFrame(data)
features_Add=extract_count(G)

features_Add["study_hours"] = (
    features_Add["prerequisites_count"] * 0.45 +
    features_Add["dependents_count"] * 1 +
    3
)

#Model Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
model=LinearRegression()

x = features_Add[['prerequisites_count', 'dependents_count', 'total_degree']]
y = features_Add['study_hours']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#Student stress analyzer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
df=pd.read_csv("Student_Lifestyle_Dataset.csv")
encoder=OrdinalEncoder(categories=[["Low", "Moderate", "High"]])
df['Stress_Level']=encoder.fit_transform(df[['Stress_Level']])
X=df.drop(columns=['Student_ID','Stress_Level','GPA'])
Y=df['Stress_Level']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42,stratify=Y)
svm_model=Pipeline([('scaler',StandardScaler()),('svm',SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])
svm_model.fit(X_train,Y_train)
Y_pred=svm_model.predict(X_test)
accuracy=svm_model.score(X_test,Y_test)

#FRONTEND
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Study Plan Generator")
mode = st.selectbox("Select Mode",[ "Prediction", "Training"])

if mode == "Training":
    st.subheader("Training Mode")
    new_topic = st.text_input("Enter topic name")
    prereq = st.number_input("Prerequisites count", min_value=0, step=1)
    depend = st.number_input("Dependents count", min_value=0, step=1)
    total = prereq + depend
    newrow= pd.DataFrame({
        'topic': [new_topic],
        'prerequisites_count': [prereq],
        'dependents_count': [depend],
        'total_degree': [total],
        'study_hours': [0]
    })
    if st.button("Add and Retrain Model"):
        df = pd.concat([features_Add, newrow], ignore_index=True)
        X = df[["prerequisites_count", "dependents_count", "total_degree"]]
        y = df["study_hours"]
        model.fit(X, y)
        st.success(f"Model retrained with new topic: {new_topic}")


elif mode == "Prediction":
    st.subheader("Prediction Mode")
    topic = st.selectbox("Choose a topic:", features_Add["topic"])
    row = features_Add[features_Add["topic"] == topic].iloc[0]
    st.write("### Topic Details")
    st.write(f"- **Prerequisites count:** {row['prerequisites_count']}")
    st.write(f"- **Dependents count:** {row['dependents_count']}")
    st.write(f"- **Total degree:** {row['total_degree']}")
    if st.button("View Suggested Study Plan"):
        st.write("### Here's Your Suggested Study Tree!")
        st.write("**Prerequisites:**")
        pre_tree = build_pre_tree(G, topic)
        pre_tree=pre_tree[0]
        st.write(print_tree(pre_tree))
        st.write("**Dependencies:**")
        de_tree = build_de_tree(G,topic)[0]
        if de_tree:
            lines = print_tree(de_tree)
            st.text("\n".join(lines))
        else:
            st.write("No dependencies found.")

    if st.button("Predict Study Hours"):
            features = [[row["prerequisites_count"], row["dependents_count"], row["total_degree"]]]
            pred = model.predict(features)[0]
            st.write(f"Predicted Study Hours: {pred}")

st.title("Student Stress Level Analyzer")
st.write(f"Stress Level Prediction Model Accuracy: {accuracy:.2f}")

st.subheader("Provide Your Lifestyle Details")
study = st.slider("Study Hours per Day", 0, 8, 4)
extra = st.slider("Extracurricular Hours per Day", 0, 4, 1)
sleep = st.slider("Sleep Hours per Day", 0, 10, 7)
social = st.slider("Social Hours per Day", 0, 6, 2)
physical = st.slider("Physical Activity Hours per Day", 0, 13, 1)

user_input = pd.DataFrame({
    "Study_Hours_Per_Day": [study],
    "Extracurricular_Hours_Per_Day": [extra],
    "Sleep_Hours_Per_Day": [sleep],
    "Social_Hours_Per_Day": [social],
    "Physical_Activity_Hours_Per_Day": [physical]
})

st.write("Here's a quick summary of your input:")
st.table(user_input)
prediction = int(svm_model.predict(user_input)[0])
probabilities = svm_model.predict_proba(user_input)[0]

stress_mapping={0:"LOW",1: "MODERATE",2: "HIGH"}
predicted_label=stress_mapping[prediction]

if prediction == 0:
    st.success(f"You are likely to have a efficient study-life balance WITH {predicted_label} stress levels! Confidence: {probabilities[0]:.2f}")
elif prediction == 1:
    st.warning(f"You may need to improve your study-life balance. Predicted {predicted_label} stress levels. Confidence: {probabilities[1]:.2f}")
elif prediction == 2:
    st.error(f"You are at risk of burnout. Predicted {predicted_label} stress levels. Confidence: {probabilities[2]:.2f}")
