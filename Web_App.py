import streamlit as st
import numpy as np
import joblib
import paho.mqtt.client as mqtt
import threading
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# =========================
# 🔧 CONFIGURATION
# =========================
MQTT_BROKER = "185.194.217.124" 
MQTT_PORT = 1883

TOPIC_ROOT_SOIL = "i4Technology/i4TCSF0XTUWVXYZ1/I4TCSFPSM0XTUWVXYZ1J/Sensors/Readings/EightReadingsinOneSensor"
TOPIC_ROOT_LEAF = "i4Technology/i4TCSF0XTUWVXYZ1/I4TCSFPSM0XTUWVXYZ1J/Sensors/Readings/LeafSensor"

COMPANY_LOGO_URL = "https://ryplabs.com/wp-content/uploads/2025/09/Picture1.png"
MODEL_PATH = "plant-disease-model-complete.pth"

# Disease Classes
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

st.set_page_config(page_title="Smart Farm AI", page_icon="🌿", layout="centered")

# =========================
# 🎨 STYLING
# =========================
page_style = """
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1628730749750-430bb849209a?q=80&w=720&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.7); 
        z-index: -1;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(0, 20, 0, 0.9);
        color: #ffffff;
    }
    h1, h2, h3 { color: #ffffff !important; text-shadow: 2px 2px 4px #000000; }
    p, li, .stMarkdown, .stMetricLabel, .stChatInput { color: #e0e0e0 !important; font-weight: 500 !important; }
    .stMetricValue { color: #4CAF50 !important; font-weight: bold !important; text-shadow: 1px 1px 2px black; }
    .stChatMessage { background-color: rgba(0, 50, 0, 0.6); border: 1px solid #4CAF50; border-radius: 10px; }
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)
# =========================
# 🧠 MODEL CLASSES
# =========================
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        return loss
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)           
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x["val_accuracy"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {"val_loss": epoch_loss.item(), "val_accuracy": epoch_acc.item()}
    def epoch_end(self, epoch, result): pass

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True) 
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases))
        
    def forward(self, xb): 
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# =========================
# 🔄 GLOBAL SHARED DATA & MQTT 
# =========================
# This class must be defined globally
class IoTDataStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.connection_status = "🔴 Disconnected"
        self.last_msg_topic = "Waiting..."
        self.last_msg_payload = "Waiting..."
        self.data = {
            "Soil_Moisture": 0.0, "Ambient_Temperature": 0.0, "Soil_Temperature": 0.0,
            "Humidity": 0.0, "Light_Intensity": 0.0, "Soil_pH": 0.0,
            "Nitrogen_Level": 0.0, "Phosphorus_Level": 0.0, "Potassium_Level": 0.0,
            "Chlorophyll_Content": 0.0, "Electrochemical_Signal": 0.0
        }
    def update_data(self, key, value):
        with self.lock:
            self.data[key] = value
    def set_status(self, status):
        with self.lock:
            self.connection_status = status
    def log_raw(self, topic, payload):
        with self.lock:
            self.last_msg_topic = topic
            self.last_msg_payload = payload
    def get_snapshot(self):
        with self.lock:
            return self.data.copy(), self.last_msg_topic, self.last_msg_payload, self.connection_status

@st.cache_resource
def get_shared_store():
    return IoTDataStore()

shared_store = get_shared_store()

# =========================
# 🏗️ HEADER
# =========================
col_header, col_logo = st.columns([5, 1.25])
with col_header:
    st.markdown('<h1 style="white-space: nowrap; margin-top: 0;">🌿 Smart Farm AI Ecosystem</h1>', unsafe_allow_html=True)
    st.subheader("Your company name is here")
with col_logo:
    st.image(COMPANY_LOGO_URL, width=400)
# =========================
# 🔐 GLOBAL SIDEBAR & MQTT INIT
# =========================
st.sidebar.header("🔐 Login Credentials")

# Define the callback function to clear data BEFORE the page reloads
def logout_callback():
    st.session_state["mqtt_user"] = ""
    st.session_state["mqtt_pass"] = ""
    st.cache_resource.clear()

# 1. Widgets with keys
mqtt_user_input = st.sidebar.text_input("Username", type="default", key="mqtt_user")
mqtt_pass_input = st.sidebar.text_input("Password", type="password", key="mqtt_pass")

col_login, col_logout = st.sidebar.columns(2)

with col_login:
    if st.button("Login", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

with col_logout:
    # 2. Assign the callback function to the button
    # Note: We do NOT use 'if st.button():' here because the work is done in the callback
    st.button("Logout", on_click=logout_callback, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.header("📡 IoT Status")

# --- MQTT CLIENT STARTUP ---
@st.cache_resource
def start_mqtt_client(user_input, pass_input):
    def on_connect(client, userdata, flags, rc):
        store = userdata
        if rc == 0:
            store.set_status("🟢 Connected")
            client.subscribe(TOPIC_ROOT_SOIL + "/#")
            client.subscribe(TOPIC_ROOT_LEAF + "/#")
        else:
            store.set_status(f"🔴 Failed (Code: {rc})")

    def on_message(client, userdata, msg):
        try:
            topic = msg.topic
            payload_str = msg.payload.decode()
            store = userdata
            store.log_raw(topic, payload_str)
            try:
                val = float(payload_str.strip())
            except:
                return 
            
            sensor_name = topic.split("/")[-1]
            if sensor_name == "SOIL_Temperature": store.update_data("Soil_Temperature", val)
            elif sensor_name == "SOIL_Moisture": store.update_data("Soil_Moisture", val)
            elif sensor_name == "SOIL_EC" or sensor_name == "SOIL EC": store.update_data("Electrochemical_Signal", val/100)
            elif sensor_name == "SOIL_PH" or sensor_name == "SOIL_Ph": store.update_data("Soil_pH", val)
            elif sensor_name == "SOIL_Nitrogen": store.update_data("Nitrogen_Level", val)
            elif sensor_name == "SOIL_Phosphorus": store.update_data("Phosphorus_Level", val)
            elif sensor_name == "SOIL_Potassium": store.update_data("Potassium_Level", val)
            elif sensor_name == "LF_Temperature": store.update_data("Ambient_Temperature", val)
            elif sensor_name == "LF_Moisture": store.update_data("Humidity", val)
        except Exception as e:
            print(f"Error: {e}")

    client = mqtt.Client()
    client.user_data_set(shared_store)
    client.on_connect = on_connect
    client.on_message = on_message
    
    if user_input and pass_input:
        client.username_pw_set(user_input, pass_input)
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start() 
    except Exception as e:
        st.error(f"MQTT Error: {e}")
        
    return client

# Start Client Globally (Outside Tabs)
start_mqtt_client(mqtt_user_input, mqtt_pass_input)

# Get Data Snapshot for Sidebar Status
_, _, _, status = shared_store.get_snapshot()
st.sidebar.write(f"**Connection:** {status}")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()



# =========================
# 📑 TABS
# =========================
tab1, tab2 = st.tabs(["📊 IoT Monitoring", "📸 Disease Detection"])

# ==========================================
# TAB 1: IOT SENSOR MONITORING
# ==========================================
with tab1:
    st.subheader("📡 Live Sensor Analysis (MQTT)")
    
    # Get Data Snapshot
    current_data, last_topic, last_val, status = shared_store.get_snapshot()

    # DISPLAY METRICS
    st.markdown("### 🌡️ Temperature & Humidity")
    col1, col2, col3 = st.columns(3)
    col1.metric("Soil Temperature", f"{current_data['Soil_Temperature']} °C") 
    col2.metric("Ambient Temperature", f"{current_data['Ambient_Temperature']} °C")
    col3.metric("Ambient Moisture", f"{current_data['Humidity']} %")

    st.markdown("### 💧 Soil Status")
    col4, col5, col6 = st.columns(3)
    col4.metric("Soil Moisture", f"{current_data['Soil_Moisture']} %")
    col5.metric("pH Level", f"{current_data['Soil_pH']}")
    col6.metric("Soil EC", f"{current_data['Electrochemical_Signal']}")
    
    st.markdown("### 🧪 Nutrients")
    col7, col8, col9 = st.columns(3)
    col7.metric("Nitrogen (N)", f"{current_data['Nitrogen_Level']} mg/kg")
    col8.metric("Phosphorus (P)", f"{current_data['Phosphorus_Level']} mg/kg")
    col9.metric("Potassium (K)", f"{current_data['Potassium_Level']} mg/kg")

    # PREDICTION
    st.markdown("---")
    st.subheader("🤖 AI Health Diagnosis")
    
    try:
        tabular_model = joblib.load("plant_health_model.pkl")
        
        input_values = [
            current_data['Soil_Moisture'], current_data['Ambient_Temperature'], current_data['Soil_Temperature'],
            current_data['Humidity'], current_data['Light_Intensity'], current_data['Soil_pH'],
            current_data['Nitrogen_Level'], current_data['Phosphorus_Level'], current_data['Potassium_Level'],
            current_data['Chlorophyll_Content'], current_data['Electrochemical_Signal']
        ]
        
        input_arr = np.array([input_values])
        pred = tabular_model.predict(input_arr)[0]
        prediction_proba = tabular_model.predict_proba(input_arr)[0]
        
        labels = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}
        colors_status = {0: "#4CAF50", 1: "#FFC107", 2: "#FF5252"}

        st.session_state['iot_status'] = labels[pred]
        
        st.markdown(f"""
            <div style="background-color: {colors_status[pred]}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
                <h2 style="color: white; margin:0;">Status: {labels[pred]}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 Confidence Levels")
        col_prob1, col_prob2 = st.columns([1, 1])

        with col_prob1:
            st.write(f"🟢 **Healthy:** {prediction_proba[0]:.2%}")
            st.write(f"🟠 **Moderate Stress:** {prediction_proba[1]:.2%}")
            st.write(f"🔴 **High Stress:** {prediction_proba[2]:.2%}")

        with col_prob2:
            fig, ax = plt.subplots(figsize=(5, 5), facecolor='none') 
            ax.set_facecolor('none')
            pie_labels = ["Healthy", "Moderate", "High"]
            pie_colors = ["#4CAF50", "#FFC107", "#FF5252"]
            wedges, texts, autotexts = ax.pie(prediction_proba, labels=None, autopct="%1.1f%%",
                    startangle=90, colors=pie_colors, radius=0.75, pctdistance=1.1,
                    textprops={"fontsize": 14, "color":"white", "weight":"bold"})
            ax.axis("equal")
            legend = ax.legend(wedges, pie_labels, title="Status", loc="center", bbox_to_anchor=(0.5, -0.15), fontsize=12)
            plt.setp(legend.get_title(), color='black', fontsize=14)
            for text in legend.get_texts(): text.set_color("black")
            st.pyplot(fig, transparent=True)
            
        # Feature Importance
        st.markdown("---")
        st.subheader("🔍 Model Insights")
        st.info("The chart below shows which environmental factors had the biggest impact on this prediction.")
        if hasattr(tabular_model, "feature_importances_"):
            feature_names = ['Soil Moisture', 'Ambient Temperature', 'Soil Temperature', 'Humidity', 'Light Intensity', 'Soil pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Chlorophyll', 'EC Signal']
            importances = tabular_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            sorted_names = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]
            
            fig_feat, ax_feat = plt.subplots(figsize=(12, 8), facecolor='none')
            ax_feat.set_facecolor('none')
            colors_bar = plt.cm.YlGnBu_r(np.linspace(0.2, 0.8, len(importances)))
            bars = ax_feat.barh(range(len(indices)), sorted_importances, align='center', color=colors_bar)
            ax_feat.set_yticks(range(len(indices)))
            ax_feat.set_yticklabels(sorted_names, color='white', fontsize=20)
            ax_feat.invert_yaxis()
            ax_feat.set_xlabel('Importance Score', color='white', fontsize=20)
            ax_feat.tick_params(colors='white')
            for spine in ax_feat.spines.values(): spine.set_edgecolor('white')
            st.pyplot(fig_feat, transparent=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ==========================================
# TAB 2: DISEASE DETECTION
# ==========================================
with tab2:
    st.subheader("📸 Upload Plant Leaf Image")
    @st.cache_resource
    def load_image_model():
        try:
            model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
            model.eval()
            return model
        except: return None

    disease_model = load_image_model()
    # Add key to maintain state
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="leaf_uploader")

    if uploaded_file and disease_model:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Leaf', use_container_width=True)
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = disease_model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            clean_label = DISEASE_CLASSES[idx.item()].replace("___", " - ").replace("_", " ")
            confidence_score = conf.item() * 100
            
            st.session_state['disease_result'] = {"label": clean_label, "confidence": confidence_score}

        if confidence_score > 50:
            st.success(f"**Diagnosis:** {clean_label} ({confidence_score:.2f}%)")
        else:
            st.warning(f"⚠️ Prediction: {clean_label} ({confidence_score:.2f}%)")
