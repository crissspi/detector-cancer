import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="Detector IA", page_icon="🩺")

st.title("Detector de Cáncer de Mama con IA")
st.write("Sube una imagen de histopatología (microscopio) y la Inteligencia Artificial te dará un pre-diagnóstico.")

@st.cache_resource 
def cargar_modelo():
    model = timm.create_model('resnet50', pretrained=False, num_classes=2)
    
    model.load_state_dict(torch.load('modelo_cancer.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = cargar_modelo()
    st.success("Sistema IA cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

archivo = st.file_uploader("Sube la imagen aquí", type=["jpg", "png", "jpeg", "tif"])

if archivo is not None:
    image = Image.open(archivo).convert('RGB')
    
    st.image(image, caption='Imagen subida', use_container_width=True)
    
    if st.button('Analizar Imagen'):
        with st.spinner('La IA está analizando...'):
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                prob_maligno = probs[0][1].item()
                
            st.divider()
            if prob_maligno > 0.5:
                st.error(f"RESULTADO: MALIGNO")
                st.progress(prob_maligno, text=f"Confianza de la IA: {prob_maligno:.1%}")
                st.warning("Se recomienda revisión urgente por un patólogo.")
            else:
                st.success(f"RESULTADO: BENIGNO")
                confianza = 1 - prob_maligno
                st.progress(confianza, text=f"Confianza de la IA: {confianza:.1%}")
                st.info("ℹNo se detectan anomalías graves, pero se sugiere control habitual.")