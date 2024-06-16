
from PIL import Image

import pandas as pd
import streamlit as st
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src.models.models import ResNet18
from src.models.preprocess import classes


nutrition_df = pd.read_csv("/foodimgclassifier/data/food_data.csv")


def find_nutrition(name: str) -> pd.Series:
    for i in range(10):
        if name == nutrition_df.iloc[i]['영문']:
            return nutrition_df.iloc[i]


def print_nutrition(df: pd.Series) -> str:
    return f'''
        식품명: {df[1]}({df[0]})\n
        {df.index[2]}: {df[2]}kcal\n
        {df.index[3]}: {df[3]}g\n
        {df.index[4]}: {df[4]}g\n
        {df.index[5]}: {df[5]}g\n
        {df.index[6]}: {df[7]}mg\n
        {df.index[8]}: {df[8]}mg\n
        {df.index[9]}: {df[9]}mg\n
    '''


st.title('음식 영양성분 출력 페이지')
st.markdown(
    '''
    * ### 페이지 기능\n
      * 업로드된 음식 사진을 추론합니다.\n
      * 추론된 음식의 영양성분을 출력합니다.\n\n
    * ### 추론 가능한 음식\n
      * 비빔밥\n
      * 불고기\n
      * 자장면\n
      * 삼겹살\n
      * 라면\n
      * 유부초밥\n
      * 김밥\n
      * 만두\n
      * 미역국\n
      * 꿀떡\n\n
    '''
    )

st.markdown("* ### 기능 사용하기\n")
file = st.file_uploader('음식 사진을 올려주세요.', type=['jpg', 'png'])

if file is None:
    st.text('이미지를 업로드 하면 추론이 시작됩니다.')
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    image = plt.imread(file)
    resize = transforms.Resize((224, 224))
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image = normalize(resize(torch.tensor(image, dtype=torch.float32).T))
    image = np.array([image])
    image = torch.tensor(image)

    net = ResNet18()
    PATH = '/foodimgclassifier/foodimgclassifier/outputs/ResNet18.pth'
    net.load_state_dict(torch.load(PATH))

    prediction = net(image)
    _, prediction = torch.max(prediction.data, 1)
    food_name = classes[int(prediction)]

    nutrition_info = find_nutrition(food_name)
    st.success(print_nutrition(nutrition_info))
