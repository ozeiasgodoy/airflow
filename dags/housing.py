from datetime import datetime, timedelta
from textwrap import dedent

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.decorators import dag, task
from airflow.models.baseoperator import chain


#Importação das bibliotecas para processamento do dataframe
import pandas as pd
import numpy as np
import zipfile
import requests
from io import BytesIO
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from joblib import dump, load

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(schedule_interval=None, start_date=datetime(2021, 1, 1), catchup=False, tags=['example'])
def Housing():
    """
    Flow para apredizagem de maquina sobre o dataset housing
    """
    @task()
    def start():
        print("iniciando...")
    

    @task()
    def criar_diretorio():
       os.makedirs('data/housing', exist_ok=True)

    @task()
    def download_housing_file():
       #download do zip com o csv
        url = "https://github.com/ozeiasgodoy/notebooks/blob/main/dados/housing.zip?raw=true"

        filebytes_housing = BytesIO(
            requests.get(url).content
        )
        with open("data/housing/housing.zip", "wb") as outfile:
            outfile.write(filebytes_housing.getbuffer())

    @task()
    def extract_housing_file():
        myzip = zipfile.ZipFile("data/housing/housing.zip")
        myzip.extractall('data/housing')

    @task()
    def criar_novos_campos():
        #Carregando o arquivo extraido para um dataframe
        housing = pd.read_csv('data/housing/housing.csv')

        #numero de comodos por domicilio
        housing['rooms_per_household'] = housing['total_rooms']/housing['households']

        #numero de comodos
        housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']

        #população por domicilio
        housing['population_per_household'] = housing['population']/housing['households']

        housing.to_csv("data/housing/housing_campos_novos.csv")

    @task()
    def tratar_campos_nulos():
        #Carregando o arquivo extraido para um dataframe
        housing = pd.read_csv('data/housing/housing_campos_novos.csv')
        housing['total_bedrooms'] = housing['total_bedrooms'].fillna(housing['total_bedrooms'].mean())
        housing['bedrooms_per_room'] =housing['bedrooms_per_room'].fillna(housing['bedrooms_per_room'].mean())
        housing.to_csv("data/housing/housing_sem_campos_nulos.csv")

    @task()
    def aplicar_one_hot_encoding():
        #Carregando o arquivo extraido para um dataframe
        housing = pd.read_csv('data/housing/housing_sem_campos_nulos.csv')
        housing  = pd.get_dummies(housing, columns=['ocean_proximity'])

        

        housing.to_csv("data/housing/housing_hot_encoding.csv")

    @task()
    def normalizar_dados():
        #Carregando o arquivo extraido para um dataframe
        housing = pd.read_csv('data/housing/housing_hot_encoding.csv')
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(housing)

        housing[['longitude', 'latitude', 'housing_median_age', 'total_rooms',	'total_bedrooms', 'population',
         'households', 	'median_income', 'rooms_per_household','bedrooms_per_room',
         'population_per_household']] = min_max_scaler.fit_transform(
             housing[['longitude', 'latitude', 'housing_median_age', 'total_rooms',	'total_bedrooms', 'population',
            'households', 	'median_income', 'rooms_per_household','bedrooms_per_room','population_per_household']]
         )

        housing.to_csv("data/housing/housing_normalizado.csv")

    @task()
    def dividir_dados_treino_teste():
        housing = pd.read_csv('data/housing/housing_normalizado.csv')

        housing_train, housing_test = train_test_split(housing, test_size=0.3, random_state=42)

        housing_train.to_csv("data/housing/housing_train.csv")
        housing_test.to_csv("data/housing/housing_test.csv")



    @task()
    def treinar_LinearRegression():
        housing = pd.read_csv('data/housing/housing_train.csv')
        X_train = housing.drop(["median_house_value"], axis=1)
        Y_train = housing["median_house_value"]
 
        #Modelos de classificação
        lr = LinearRegression()
        lr.fit(X_train, Y_train)
        dump(lr, "data/housing/LinearRegression_housing.joblib")

    @task()
    def treinar_DecisionTreeRegressor():
        housing = pd.read_csv('data/housing/housing_train.csv')
        X_train = housing.drop(["median_house_value"], axis=1)
        Y_train = housing["median_house_value"]
 
        #Modelos de classificação
        lr = DecisionTreeRegressor()
        lr.fit(X_train, Y_train)
        dump(lr, "data/housing/DecisionTreeRegressor(housing.joblib")

    @task()
    def treinar_RandomForestRegressor():
        housing = pd.read_csv('data/housing/housing_train.csv')
        X_train = housing.drop(["median_house_value"], axis=1)
        Y_train = housing["median_house_value"]
 
        #Modelos de classificação
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, Y_train)
        dump(rf, "data/housing/RandomForestRegressor_housing.joblib")

    @task()
    def treinar_SVC():
        housing = pd.read_csv('data/housing/housing_train.csv')
        X_train = housing.drop(["median_house_value"], axis=1)
        Y_train = housing["median_house_value"]
 
        #Modelos de classificação
        svc = SVC(kernel='linear', gamma='scale', random_state=42)
        svc.fit(X_train, Y_train)
        dump(svc, "data/housing/SVC_housing.joblib")
    
    @task()
    def treinar_KNeighborsRegressor():
        housing = pd.read_csv('data/housing/housing_train.csv')
        X_train = housing.drop(["median_house_value"], axis=1)
        Y_train = housing["median_house_value"]
        
        #Modelos de classificação
        knn = KNeighborsRegressor()
        knn.fit(X_train, Y_train)
        dump(knn, "data/housing/KNeighborsRegressor_housing.joblib")


    #start() >> criar_diretorio() >> [[download_housing_file() >> extract_housing_file()], [download_maps() >> extract_maps()]]
 
    chain(start() , criar_diretorio(), download_housing_file(),  extract_housing_file(), 
        criar_novos_campos(), tratar_campos_nulos(), aplicar_one_hot_encoding(), normalizar_dados(), dividir_dados_treino_teste(),
        [treinar_LinearRegression(), treinar_DecisionTreeRegressor(),  treinar_KNeighborsRegressor(), treinar_RandomForestRegressor()]
    )
dag = Housing()
