import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="Telco Customer Churn - EDA",
    page_icon="📊",
    layout="wide"
)

# =========================
# Sidebar (menú principal)
# =========================
st.sidebar.title("📌 Navegación")

modulo = st.sidebar.radio(
    "Selecciona un módulo:",
    (
        "Home",
        "Módulo 2: Carga del Dataset y EDA",
        "Conclusiones Finales"
    )
)

st.sidebar.markdown("---")
st.sidebar.caption("Primer Proyecto de Portafolio Profesional")

# =========================
# MÓDULO 1: HOME
# =========================
if modulo == "Home":
    st.title("📊 Análisis Exploratorio de Datos - Telco Customer Churn")
    st.subheader("Proyecto aplicado de Análisis Exploratorio de Datos (EDA) en Python con Streamlit")

    st.markdown(
        """
        Este proyecto tiene como objetivo realizar un **Análisis Exploratorio de Datos (EDA)** 
        sobre el dataset `TelcoCustomerChurn`, con el fin de identificar **patrones y factores 
        asociados a la fuga de clientes (churn)** en una empresa de telecomunicaciones.

        👉 **Importante:**  
        El foco del proyecto **no** es construir modelos predictivos, sino:
        - Limpiar, transformar y explorar los datos.  
        - Generar visualizaciones claras e interpretables.  
        - Extraer insights que puedan apoyar la toma de decisiones.
        """
    )

    st.markdown("---")
    st.header("👤 Datos del autor")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            - **Nombre completo:** Yoseline Carolina Sanchez Quino  
            - **Curso / Especialización:** Especialización en Python For Analytics – Python DMC  
            - **Año:** 2026  
            """
        )

    with col2:
        st.markdown(
            """
            - **Ciudad / País:** Lima, Perú  
            - **Rol:** Estudiante / Analista en formación  
            - **Proyecto:** Primer Proyecto de Portafolio Profesional  
            """
        )

    st.markdown("---")
    st.header("📁 Descripción del dataset: TelcoCustomerChurn")

    st.markdown(
        """
        El dataset TelcoCustomerChurn reúne información detallada sobre los clientes de una empresa de telecomunicaciones, 
        incluyendo sus características personales, los servicios que tienen contratados, su comportamiento de uso, 
        su historial de facturación y si han abandonado o no la empresa.

        En resumen, este dataset permite analizar el comportamiento de los clientes y comprender qué factores están asociados 
        a la fuga de clientes (churn) lo cual es un indicador clave para la rentabilidad del negocio.
        """
    )

    st.markdown("---")
    st.header("🛠️ Tecnologías y librerías utilizadas")

    st.markdown(
        """
        - **Python 3**  
        - **Pandas**  
        - **NumPy**  
        - **Matplotlib**  
        - **Seaborn**  
        - **Streamlit**  
        """
    )

    st.markdown("---")
    st.info("Usa el menú lateral para navegar por los distintos módulos del proyecto.")

# =========================
# CLASE POO PARA EL EDA
# =========================
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def variable_types(self):
        numeric = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical = self.df.select_dtypes(exclude=np.number).columns.tolist()
        return numeric, categorical

    def missing_values(self):
        return self.df.isnull().sum()

    def descriptive_stats(self):
        return self.df.describe()

    def plot_histogram(self, column):
        fig, ax = plt.subplots()
        sns.histplot(self.df[column], kde=True, ax=ax)
        ax.set_title(f"Distribución de {column}")
        return fig

    def plot_bar(self, column):
        fig, ax = plt.subplots()
        self.df[column].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Conteo de {column}")
        return fig

    def plot_numeric_vs_categorical(self, num_col, cat_col):
        fig, ax = plt.subplots()
        sns.boxplot(x=self.df[cat_col], y=self.df[num_col], ax=ax)
        ax.set_title(f"{num_col} vs {cat_col}")
        return fig

    def plot_cat_vs_cat(self, col1, col2):
        fig, ax = plt.subplots()
        sns.countplot(x=self.df[col1], hue=self.df[col2], ax=ax)
        ax.set_title(f"{col1} vs {col2}")
        return fig

# =========================
# MÓDULO 2: CARGA DEL DATASET Y EDA
# =========================
if modulo == "Módulo 2: Carga del Dataset y EDA":

    st.title("📊 Módulo 2: Carga del Dataset y Análisis Exploratorio de Datos")

    file = st.file_uploader("Sube el archivo CSV del dataset TelcoCustomerChurn", type=["csv"])

    if file is None:
        st.warning("⚠️ Por favor carga un archivo CSV para continuar.")
        st.stop()

    df = pd.read_csv(file)
    analyzer = DataAnalyzer(df)

    st.success("Archivo cargado correctamente ✔️")

    st.subheader("👀 Vista previa del dataset")
    st.dataframe(df.head())

    st.write(f"**Filas:** {df.shape[0]} | **Columnas:** {df.shape[1]}")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "1. Info General",
        "2. Clasificación de Variables",
        "3. Estadísticas Descriptivas",
        "4. Valores Faltantes",
        "5. Distribución Numéricas",
        "6. Variables Categóricas",
        "7. Numérico vs Categórico",
        "8. Categórico vs Categórico",
        "9. Análisis Dinámico",
        "10. Hallazgos Clave"
    ])

    # ÍTEM 1
    with tab1:
        st.header("Información general del dataset")

        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()

        st.text(info_str)

        st.subheader("Tipos de datos")
        st.write(df.dtypes)

        st.subheader("Valores nulos")
        st.write(analyzer.missing_values())



    # ÍTEM 2
    with tab2:
        st.header("Clasificación de variables")
        numeric, categorical = analyzer.variable_types()

        st.write("### Variables numéricas:", numeric)
        st.write("### Variables categóricas:", categorical)

        st.metric("Cantidad numéricas", len(numeric))
        st.metric("Cantidad categóricas", len(categorical))

    # ÍTEM 3
    with tab3:
        st.header("Estadísticas descriptivas")
        st.write(analyzer.descriptive_stats())

    # ÍTEM 4
    with tab4:
        st.header("Valores faltantes")
        st.write(analyzer.missing_values())

        st.subheader("Visualización")
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, ax=ax)
        st.pyplot(fig)

    # ÍTEM 5
    with tab5:
        st.header("Distribución de variables numéricas")
        numeric, _ = analyzer.variable_types()

        col = st.selectbox("Selecciona una variable numérica", numeric)
        fig = analyzer.plot_histogram(col)
        st.pyplot(fig)

    # ÍTEM 6
    with tab6:
        st.header("Análisis de variables categóricas")
        _, categorical = analyzer.variable_types()

        col = st.selectbox("Selecciona una variable categórica", categorical)
        fig = analyzer.plot_bar(col)
        st.pyplot(fig)

    # ÍTEM 7
    with tab7:
        st.header("Análisis numérico vs categórico")
        numeric, categorical = analyzer.variable_types()

        num_col = st.selectbox("Variable numérica", numeric)
        cat_col = st.selectbox("Variable categórica", categorical)

        fig = analyzer.plot_numeric_vs_categorical(num_col, cat_col)
        st.pyplot(fig)

    # ÍTEM 8
    with tab8:
        st.header("Análisis categórico vs categórico")
        _, categorical = analyzer.variable_types()

        col1 = st.selectbox("Variable categórica 1", categorical)
        col2 = st.selectbox("Variable categórica 2", categorical)

        fig = analyzer.plot_cat_vs_cat(col1, col2)
        st.pyplot(fig)

    # ÍTEM 9
    with tab9:
        st.header("Análisis dinámico")
        cols = st.multiselect("Selecciona columnas para analizar", df.columns)

        if cols:
            st.dataframe(df[cols].describe(include="all"))

    # ÍTEM 10
    with tab10:
        st.header("Hallazgos clave")
        st.markdown("""

        - Los clientes con menor **tenure** presentan mayor churn.
        - El contrato **Month-to-month** es el más asociado al abandono.
        - El servicio **Fiber optic** muestra mayor churn que DSL.
        - El método de pago **Electronic check** tiene la mayor tasa de fuga.
        - Cargos mensuales altos incrementan la probabilidad de churn.
        - Servicios adicionales como **OnlineSecurity** y **TechSupport** reducen el churn.
        - No hay diferencias significativas por género.
        - Clientes con **dependientes o pareja** tienden a permanecer más tiempo.

        ### Conclusión
        El churn está influenciado principalmente por el tipo de contrato, el costo mensual, el método de pago y la permanencia del cliente.
        """)


# =========================
# MÓDULO 3: CONCLUSIONES
# =========================

if modulo == "Conclusiones Finales":
    st.title("📝 Conclusiones Finales del Proyecto")

    st.markdown("""
    ## 🧠 Hallazgos Clave del Análisis Exploratorio

    Tras realizar el Análisis Exploratorio de Datos (EDA) del dataset **Telco Customer Churn**, 
    se identificaron patrones importantes que ayudan a comprender los factores asociados a la fuga de clientes:

    ### 🔹 1. La permanencia (tenure) es un factor determinante
    Los clientes con menor tiempo en la empresa presentan una mayor probabilidad de abandonar el servicio.
    Esto indica que los primeros meses son críticos para la retención.

    ### 🔹 2. Los contratos mensuales presentan mayor churn
    El tipo de contrato **Month-to-month** concentra la mayor tasa de abandono, 
    mientras que los contratos anuales o bianuales muestran mayor estabilidad.

    ### 🔹 3. El servicio de internet influye en la fuga
    Los clientes con **Fiber optic** presentan mayor churn que los que usan DSL, 
    lo que podría estar relacionado con costos o calidad percibida.

    ### 🔹 4. El método de pago es un indicador relevante
    El método **Electronic check** está asociado a una mayor probabilidad de abandono, 
    posiblemente por ser utilizado por clientes más sensibles al precio.

    ### 🔹 5. Los cargos mensuales altos aumentan el riesgo de churn
    Los clientes con **MonthlyCharges elevados** tienden a abandonar más, 
    lo que sugiere que el costo es un factor clave en la decisión.

    ### 🔹 6. Los servicios adicionales reducen la fuga
    Servicios como **OnlineSecurity**, **TechSupport** y **DeviceProtection** 
    están asociados a una mayor permanencia, ya que incrementan el valor percibido.

    ### 🔹 7. El género no es un factor relevante
    No se observan diferencias significativas en el churn entre hombres y mujeres.

    ### 🔹 8. Los clientes con pareja o dependientes abandonan menos
    Estos clientes suelen mostrar mayor estabilidad y menor probabilidad de fuga.

    ---

    ## 🎯 Conclusión General

    El churn está influenciado principalmente por:
    - La permanencia del cliente  
    - El tipo de contrato  
    - El método de pago  
    - El costo mensual  
    - El tipo de servicio de internet  

    Estos hallazgos permiten orientar estrategias de retención enfocadas en:
    - Incentivar contratos de mayor duración  
    - Ofrecer paquetes de valor (seguridad, soporte, protección)  
    - Revisar precios de servicios premium  
    - Identificar y contactar tempranamente a clientes nuevos o con cargos altos  

    El análisis exploratorio proporciona una base sólida para comprender el comportamiento de los clientes 
    y diseñar acciones que reduzcan la fuga en un contexto donde retener es mucho más rentable que adquirir nuevos clientes.
    """)
