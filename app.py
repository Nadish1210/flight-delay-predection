import gradio as gr
import pandas as pd
import numpy as np
import joblib

# ────────────────────────────────────────────────
# Load model & encoders
# ────────────────────────────────────────────────
MODEL_FILE = "flight_delay_model.pkl"
ENCODERS_FILE = "label_encoders.pkl"

try:
    model = joblib.load(MODEL_FILE)
    label_encoders = joblib.load(ENCODERS_FILE)
    print("Model & encoders loaded successfully")
except Exception as e:
    print(f"Loading failed: {e}")
    model = None
    label_encoders = {}

# Model ke expected columns (exact order important!)
EXPECTED_FEATURES = [
    'DayOfWeek', 'DepTime', 'ArrTime', 'CRSArrTime', 'UniqueCarrier',
    'Airline', 'FlightNum', 'ActualElapsedTime', 'CRSElapsedTime',
    'AirTime', 'DepDelay', 'Origin', 'Org_Airport', 'Dest',
    'Dest_Airport', 'Distance', 'TaxiIn', 'TaxiOut',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay',
    'LateAircraftDelay'
]

def predict_delay(
    day_of_week,        # 1 = Monday ... 7 = Sunday
    dep_time,
    arr_time,
    crs_arr_time,
    airline,            # e.g. "AA"
    flight_num,
    actual_elapsed_time,
    crs_elapsed_time,
    air_time,
    dep_delay,          # departure delay minutes
    origin,             # IATA code e.g. "JFK"
    dest,               # IATA code e.g. "LAX"
    distance,
    taxi_in,
    taxi_out,
    carrier_delay,
    weather_delay,
    nas_delay,
    security_delay,
    late_aircraft_delay
):
    if model is None:
        return "❌ Model load nahi hua. Files check karen."

    try:
        # Safe conversion
        def safe_float(v, default=np.nan):
            if v in (None, "", " "): return default
            try: return float(v)
            except: return default

        def safe_int(v, default=0):
            if v in (None, "", " "): return default
            try: return int(float(v))
            except: return default

        row = {
            'DayOfWeek': safe_int(day_of_week, 1),
            'DepTime': safe_float(dep_time),
            'ArrTime': safe_float(arr_time),
            'CRSArrTime': safe_float(crs_arr_time),
            'UniqueCarrier': str(airline).strip().upper(),
            'Airline': str(airline).strip().upper(),
            'FlightNum': safe_int(flight_num),
            'ActualElapsedTime': safe_float(actual_elapsed_time),
            'CRSElapsedTime': safe_float(crs_elapsed_time),
            'AirTime': safe_float(air_time),
            'DepDelay': safe_float(dep_delay, 0),
            'Origin': str(origin).strip().upper(),
            'Org_Airport': str(origin).strip().upper(),       # same as Origin
            'Dest': str(dest).strip().upper(),
            'Dest_Airport': str(dest).strip().upper(),        # same as Dest
            'Distance': safe_float(distance, 0),
            'TaxiIn': safe_float(taxi_in, 0),
            'TaxiOut': safe_float(taxi_out, 0),
            'CarrierDelay': safe_float(carrier_delay, 0),
            'WeatherDelay': safe_float(weather_delay, 0),
            'NASDelay': safe_float(nas_delay, 0),
            'SecurityDelay': safe_float(security_delay, 0),
            'LateAircraftDelay': safe_float(late_aircraft_delay, 0),
        }

        df = pd.DataFrame([row])

        # Exact column order jo model ko chahiye
        df = df[EXPECTED_FEATURES]

        # Label encoding (unknown → -1)
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Prediction
        pred = model.predict(df)[0]
        return f"**Predicted arrival delay: {float(pred):.1f} minutes**"

    except Exception as e:
        return f"Error: {str(e)}"


# ────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✈️ Flight Delay Predictor (minutes) By Nadish")
    gr.Markdown("This project built at zaitoon ishraf IT camp in AI AND DATA SCIENCE  completion cermony 'HACHATHON'")

    with gr.Row():
        day_of_week = gr.Dropdown([1,2,3,4,5,6,7], label="Day of Week (1=Mon ... 7=Sun)", value=3)
        dep_time = gr.Textbox(label="DepTime (hhmm)", placeholder="930 or 1430")
        crs_arr_time = gr.Textbox(label="Scheduled ArrTime (hhmm)", placeholder="1200")

    with gr.Row():
        airline = gr.Textbox(label="Airline code", placeholder="AA / DL / UA")
        origin = gr.Textbox(label="Origin (IATA)", placeholder="JFK")
        dest = gr.Textbox(label="Destination (IATA)", placeholder="LAX")

    with gr.Row():
        distance = gr.Number(label="Distance (miles)", value=2000)
        dep_delay = gr.Number(label="Dep Delay (min) – estimate", value=0)

    with gr.Accordion("Aur details (optional / post-flight)", open=False):
        arr_time = gr.Number(label="Actual ArrTime (hhmm)", value=0)
        flight_num = gr.Textbox(label="Flight Number", value="123")
        actual_elapsed_time = gr.Number(label="Actual Elapsed Time (min)", value=0)
        crs_elapsed_time = gr.Number(label="Scheduled Elapsed Time (min)", value=0)
        air_time = gr.Number(label="Air Time (min)", value=0)
        taxi_in = gr.Number(label="Taxi In (min)", value=0)
        taxi_out = gr.Number(label="Taxi Out (min)", value=0)
        carrier_delay = gr.Number(label="Carrier Delay (min)", value=0)
        weather_delay = gr.Number(label="Weather Delay (min)", value=0)
        nas_delay = gr.Number(label="NAS Delay (min)", value=0)
        security_delay = gr.Number(label="Security Delay (min)", value=0)
        late_aircraft_delay = gr.Number(label="Late Aircraft Delay (min)", value=0)

    output = gr.Markdown()

    btn = gr.Button("Predict Delay", variant="primary")

    inputs = [
        day_of_week, dep_time, arr_time, crs_arr_time, airline,
        flight_num, actual_elapsed_time, crs_elapsed_time, air_time,
        dep_delay, origin, dest, distance, taxi_in, taxi_out,
        carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay
    ]

    btn.click(predict_delay, inputs=inputs, outputs=output)

if __name__ == "__main__":
    demo.launch()