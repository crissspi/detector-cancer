import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

st.set_page_config(
    page_title="Detector de Cáncer",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

def validar_imagen(image):
    img_hsv = image.convert('HSV')
    s_canal = np.array(img_hsv)[:, :, 1] 
    saturacion_media = np.mean(s_canal)
    
    if saturacion_media < 20: 
        return False, "La imagen no tiene suficiente color (posible escala de grises)."
    
    return True, "OK"

@st.cache_resource 
def cargar_modelo():
    model = timm.create_model('resnet50', pretrained=False, num_classes=2)
    try:
        model.load_state_dict(torch.load('modelo_cancer.pth', map_location=torch.device('cpu')))
    except:
        try:
            model.load_state_dict(torch.load('modelo_cancer_tihare.pth', map_location=torch.device('cpu')))
        except:
            st.error("Error: No se encuentra el archivo del modelo.")
            st.stop()     
    model.eval()
    return model

col_info, col_app = st.columns([1, 1.2], gap="large")

with col_info:
    st.title("Detector de Cáncer de Mama")
    st.markdown("##### Herramienta de apoyo al diagnóstico histopatológico.")
    
    st.markdown("---")
    
    st.subheader("Sobre el Cáncer de Mama")
    st.write("""
    El cáncer de mama es una de las principales causas de patología en mujeres a nivel mundial. 
    La detección temprana es fundamental para el pronóstico.
    
    Este sistema utiliza Inteligencia Artificial para identificar patrones microscópicos en biopsias y alertar sobre posibles malignidades.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True) 
    
    st.subheader("Enfoque Social")
    st.info("""
    **Proyecto para Comunidades Aymara**
    
    Esta herramienta nace con el objetivo de acercar tecnología de triaje a zonas rurales del norte de Chile, 
    facilitando el acceso a un pre-diagnóstico donde los especialistas pueden ser escasos.
    """)

with col_app:
    st.subheader("Análisis de Muestra")
    
    try:
        model = cargar_modelo()
        st.success("Sistema y Modelo IA operativos.")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    archivo = st.file_uploader("Cargar imagen de biopsia (JPG, PNG, TIF)", type=["jpg", "png", "jpeg", "tif"])
    
    if archivo is not None:
        image = Image.open(archivo).convert('RGB')
        
        st.image(image, caption='Vista previa de la muestra', use_container_width=True)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if st.button('Ejecutar Análisis', type="primary"):
            
            es_valida, mensaje = validar_imagen(image)
            
            if not es_valida:
                st.error(f"IMAGEN RECHAZADA: {mensaje}")
                st.warning("El sistema solo acepta imágenes histopatológicas (Tinción H&E).")
            
            else:
                with st.spinner('Procesando...'):
                    img_tensor = transform(image).unsqueeze(0)
                    with torch.no_grad():
                        output = model(img_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        prob_maligno = probs[0][1].item()
                    
                    st.markdown("---")
                    
                    if prob_maligno > 0.5:
                        st.error("RESULTADO: MALIGNO")
                        st.progress(prob_maligno, text=f"Probabilidad: {prob_maligno:.1%}")
                        st.write("**Recomendación:** Derivación urgente a patología.")
                    else:
                        st.success("RESULTADO: BENIGNO")
                        confianza = 1 - prob_maligno
                        st.progress(confianza, text=f"Probabilidad: {confianza:.1%}")
                        st.write("**Recomendación:** Mantener control habitual.")

    else:
        st.info("Esperando carga de imagen para iniciar análisis.")