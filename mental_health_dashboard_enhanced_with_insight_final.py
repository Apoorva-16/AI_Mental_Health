
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Enhanced Mental Health Dashboard", layout="wide")
st.title("🧠 Enhanced Global Mental Health Dashboard")

# Load and clean data
df1 = pd.read_csv("mental-illnesses-prevalence.csv")
df1 = df1.dropna()

# Sidebar selections
st.sidebar.header("🔎 Country Selection")
countries = sorted(df1["Entity"].unique())
selected_countries = st.sidebar.multiselect("Select Country or Countries", countries, default=["United States", "India"])

# Prediction toggle
predict_toggle = st.sidebar.radio("Would you like to enable future prediction & insight?", ["No", "Yes"])

# Feature columns
features = ['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized',
            'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized',
            'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized',
            'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized']
target_col = "Eating disorders (share of population) - Sex: Both - Age: Age-standardized"

# Historical trend chart
st.header("📈 Historical Trends Over Time")
trend_df = df1[df1["Entity"].isin(selected_countries)][["Entity", "Year", target_col]]
fig = px.line(trend_df, x="Year", y=target_col, color="Entity", markers=True,
              labels={target_col: "Eating Disorder %"}, title="Eating Disorder Trends Over Time")
st.plotly_chart(fig, use_container_width=True)

# Latest values
st.header("📊 Latest Reported Eating Disorder Rates")
latest_year = df1["Year"].max()
latest_data = df1[(df1["Entity"].isin(selected_countries)) & (df1["Year"] == latest_year)]
if not latest_data.empty:
    display_data = latest_data[["Entity", target_col]].rename(columns={target_col: "Eating Disorder %"})
    st.dataframe(display_data.set_index("Entity").style.format({"Eating Disorder %": "{:.4f}"}))

# Prediction and insights
if predict_toggle == "Yes":
    st.header("🔮 Predict Eating Disorder Rate for a Future Year")
    future_year = st.sidebar.slider("Select Year to Predict", 2024, 2035, 2026)

    # Train models
    X = df1[features]
    y = df1[target_col]
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
    y_log = transformer.fit_transform(y.values.reshape(-1, 1)).ravel()
    X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train_log, epochs=200, batch_size=32, validation_split=0.1, verbose=0)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train_log)

    # Predictions
    predictions = []
    insights = []
    for country in selected_countries:
        recent_data = df1[df1["Entity"] == country].sort_values("Year", ascending=False)
        if not recent_data.empty:
            recent_year = recent_data["Year"].max()
            recent_rate = recent_data[recent_data["Year"] == recent_year][target_col].values[0]
            avg_features = recent_data.head(3)[features].mean().values.reshape(1, -1)
            scaled_input = scaler.transform(avg_features)
            pred_nn = model.predict(scaled_input).flatten()
            pred_rf = rf.predict(scaled_input).flatten()
            predicted_rate = (transformer.inverse_transform(pred_nn.reshape(-1, 1)).ravel() +
                              transformer.inverse_transform(pred_rf.reshape(-1, 1)).ravel()) / 2
            predictions.append((country, predicted_rate[0]))

            if predicted_rate[0] > recent_rate:
                insights.append(f"""
**🟢 Observation:**  
The predicted rate for **{country}** in **{future_year}** is **{predicted_rate[0]:.4f}%**, compared to **{recent_rate:.4f}%** in **{recent_year}**.

**📈 Trend:** The rate has **increased**.

**📌 Recommendation:**
- Expand mental health education and awareness campaigns.
- Improve access to early diagnosis and counseling.
- Encourage healthy discussions around self-image and eating habits.
- Increase funding for school-based and community mental health services.
""")
            else:
                insights.append(f"""
**🟢 Observation:**  
The predicted rate for **{country}** in **{future_year}** is **{predicted_rate[0]:.4f}%**, compared to **{recent_rate:.4f}%** in **{recent_year}**.

**📉 Trend:** The rate has **decreased**.

**📌 Recommendation:**
- Maintain existing awareness and support programs.
- Continue promoting body positivity and mental well-being.
- Incentivize schools and healthcare providers to stay proactive.
""")

    # Display insights
    for insight in insights:
        st.markdown(insight)

    # Prediction results
    if predictions:
        pred_df = pd.DataFrame(predictions, columns=["Country", f"Predicted % in {future_year}"])
        st.subheader("📌 Summary of Predictions")
        st.dataframe(pred_df.set_index("Country").style.format({f"Predicted % in {future_year}": "{:.4f}"}))

        # Plot prediction bar chart
        fig_pred = px.bar(pred_df, x="Country", y=f"Predicted % in {future_year}", text_auto=".4f",
                          title=f"Predicted Eating Disorder % in {future_year}", labels={"value": "%"})
        st.plotly_chart(fig_pred, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit, Keras, Plotly, and Scikit-learn.")
