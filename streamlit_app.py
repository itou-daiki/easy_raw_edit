import streamlit as st
import rawpy
import imageio
import numpy as np
from io import BytesIO
from scipy import ndimage

# ページ設定
st.set_page_config(page_title="RAW画像 カスタマイズ可能プリセット変換アプリ", layout="wide")

# アプリのタイトルと制作者
st.title("RAW画像 カスタマイズ可能プリセット変換アプリ")
st.caption("Created by Dit-Lab.(Daiki Ito)")

# 概要
st.markdown("""
## **概要**
このウェブアプリケーションでは、RAW画像にカスタマイズ可能なプリセットを適用し、JPG形式でダウンロードすることができます。
iPadなどのデバイスでも対応しています。
""")

# ファイルアップローダー
uploaded_file = st.file_uploader("RAW画像をアップロードしてください", type=["raw", "arw", "cr2", "nef"])

if uploaded_file is not None:
    # プリセットの選択
    preset = st.selectbox(
        "プリセットを選択してください",
        ("サマー", "カスタム", "ビビッド", "モノクロ")
    )
    
    # パラメータ設定
    col1, col2 = st.columns(2)
    with col1:
        brightness = st.slider("明るさ", 0.5, 2.0, 1.3, 0.05)
        contrast = st.slider("コントラスト", 0.5, 2.0, 0.8, 0.05)
        saturation = st.slider("彩度", 0.0, 2.0, 0.9, 0.05)
    with col2:
        warmth = st.slider("色温度", 0.5, 1.5, 0.85, 0.025)
        highlight = st.slider("ハイライト", 0.0, 0.5, 0.1, 0.05)
        blur = st.slider("ぼかし", 0.0, 2.0, 1.0, 0.25)
    
    # プリセットに基づいてパラメータを設定
    if preset == "サマー":
        brightness, contrast, saturation = 1.3, 0.8, 0.9
        warmth, highlight, blur = 0.85, 0.1, 1.0
    elif preset == "ビビッド":
        brightness, contrast, saturation = 1.1, 1.3, 1.5
        warmth, highlight, blur = 1.1, 0.3, 0.0
    elif preset == "モノクロ":
        saturation = 0.0
        contrast = 1.3
    
    # スピナーを使用して処理中であることを表示
    with st.spinner('画像を処理中です。しばらくお待ちください...'):
        # RAW画像の処理
        with rawpy.imread(uploaded_file) as raw:
            rgb = raw.postprocess()
        
        # 0-1の範囲に正規化
        image = rgb.astype(np.float32) / 255.0
        
        # 明るさ調整
        image = image * brightness
        
        # コントラスト調整
        image = (image - 0.5) * contrast + 0.5
        
        # 彩度調整
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        image[..., :3] = np.clip(image[..., :3] + (image[..., :3] - gray[..., None]) * (saturation - 1), 0, 1)
        
        # 色温度調整（warmth）
        image[..., 2] = np.clip(image[..., 2] * (2 - warmth), 0, 1)  # 青チャンネルを調整
        image[..., 0] = np.clip(image[..., 0] * warmth, 0, 1)  # 赤チャンネルを調整
        
        # ハイライトの強調
        highlight_threshold = 0.7
        highlights = np.where(image > highlight_threshold, image, 0)
        image = np.clip(image + highlights * highlight, 0, 1)
        
        # ぼかし効果
        if blur > 0:
            image = ndimage.gaussian_filter(image, sigma=blur)
        
        # 青みの強調
        image[..., 2] = np.clip(image[..., 2] * 1.1, 0, 1)
        
        # 最終的なクリッピング
        processed = np.clip(image, 0, 1)
        
        # 0-255の範囲に戻す
        processed = (processed * 255).astype(np.uint8)
    
    # 処理後の画像を表示
    st.image(processed, caption='処理後の画像', use_column_width=True)
    
    # JPG形式でダウンロード
    buf = BytesIO()
    imageio.imwrite(buf, processed, format='jpg')
    btn = st.download_button(
        label="JPGとしてダウンロード",
        data=buf.getvalue(),
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )

st.write("注意: このアプリはRAW画像の処理に時間がかかる場合があります。")

# Copyright
st.write("")
st.subheader('© 2022-2024 Dit-Lab.(Daiki Ito). All Rights Reserved.')