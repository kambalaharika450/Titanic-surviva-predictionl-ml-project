import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
import os

# Set Python executable for Spark workers
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

# Initialize Spark session
@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("TitanicLogisticRegression").master("local[*]").getOrCreate()

spark = get_spark()

# Load and clean Titanic data
@st.cache_resource
def load_data():
    df = spark.read.csv("train.csv", header=True, inferSchema=True)
    df = df.dropna(subset=["Survived"])
    avg_age = df.select(avg("Age")).first()[0]
    df = df.fillna({"Age": avg_age, "Embarked": "S"})
    return df

data = load_data()

# Fit feature transformers
@st.cache_resource
def fit_transformers(_df):
    gender_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_Index").fit(_df)
    embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_Index").fit(_df)
    indexed_data = embarked_indexer.transform(_df)
    embarked_encoder = OneHotEncoder(inputCols=["Embarked_Index"], outputCols=["Embarked_OneHot"]).fit(indexed_data)

    assembler = VectorAssembler(
        inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_Index", "Embarked_OneHot"],
        outputCol="features"
    )

    return gender_indexer, embarked_indexer, embarked_encoder, assembler

gender_indexer_model, embarked_indexer_model, embarked_encoder_model, assembler = fit_transformers(data)

# Train logistic regression model
@st.cache_resource
def train_model(_df):
    _df = gender_indexer_model.transform(_df)
    _df = embarked_indexer_model.transform(_df)
    _df = embarked_encoder_model.transform(_df)
    _df = assembler.transform(_df).select("Survived", "features")

    train_data, _ = _df.randomSplit([0.8, 0.2], seed=42)
    model = LogisticRegression(labelCol="Survived", featuresCol="features").fit(train_data)
    return model

lr_model = train_model(data)

# ==== Streamlit UI ====
st.title("🚢 Titanic Survival Predictor (Logistic Regression)")

with st.form("input_form"):
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0.42, 80.0, 30.0)
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    fare = st.slider("Fare Paid", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = spark.createDataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    input_df = gender_indexer_model.transform(input_df)
    input_df = embarked_indexer_model.transform(input_df)
    input_df = embarked_encoder_model.transform(input_df)
    input_df = assembler.transform(input_df)

    prediction = lr_model.transform(input_df)
    result = prediction.select("prediction", "probability").collect()[0]

    survived = int(result["prediction"])
    confidence = result["probability"][survived] * 100

    st.markdown(f"""
        ### 🧾 Prediction Result:
        - *Outcome:* {'🟢 Survived' if survived == 1 else '🔴 Did not survive'}
        - *Confidence:* {confidence:.2f}%
    """)
