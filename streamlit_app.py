import streamlit as st
import rawpy
import imageio
import numpy as np
from io import BytesIO
from scipy import ndimage

# ページ設定
st.set_page_config(page_title="RAW画像 高度カスタマイズ可能プリセット変換アプリ", layout="wide")

# アプリのタイトルと制作者
st.title("RAW画像 高度カスタマイズ可能プリセット変換アプリ")
st.caption("Created by Dit-Lab.(Daiki Ito)")

# 概要
st.markdown("""
## **概要**
このウェブアプリケーションでは、RAW画像に高度にカスタマイズ可能なプリセットを適用し、JPG形式でダウンロードすることができます。
RAW画像の特性を考慮した調整が可能です。
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
    col1, col2, col3 = st.columns(3)
    with col1:
        exposure = st.slider("露出補正", -2.0, 2.0, 0.0, 0.1)
        brightness = st.slider("明るさ", 0.5, 2.0, 1.3, 0.05)
        contrast = st.slider("コントラスト", 0.5, 2.0, 0.8, 0.05)
        highlights = st.slider("ハイライト", -1.0, 1.0, 0.0, 0.05)
    with col2:
        shadows = st.slider("シャドウ", -1.0, 1.0, 0.0, 0.05)
        whites = st.slider("白レベル", -1.0, 1.0, 0.0, 0.05)
        blacks = st.slider("黒レベル", -1.0, 1.0, 0.0, 0.05)
        saturation = st.slider("彩度", 0.0, 2.0, 0.9, 0.05)
    with col3:
        warmth = st.slider("色温度", 0.5, 1.5, 0.85, 0.025)
        tint = st.slider("色合い", -1.0, 1.0, 0.0, 0.05)
        clarity = st.slider("クラリティ", -1.0, 1.0, 0.0, 0.05)
        vignette = st.slider("ビネット", 0.0, 1.0, 0.0, 0.05)
    
    # プリセットに基づいてパラメータを設定
    if preset == "サマー":
        exposure, brightness, contrast = 0.5, 1.3, 0.8
        highlights, shadows = -0.2, 0.3
        saturation, warmth = 0.9, 0.85
        clarity, vignette = 0.2, 0.1
    elif preset == "ビビッド":
        exposure, brightness, contrast = 0.3, 1.1, 1.3
        highlights, shadows = 0.2, 0.1
        saturation, warmth = 1.5, 1.1
        clarity, vignette = 0.4, 0.2
    elif preset == "モノクロ":
        saturation = 0.0
        contrast, clarity = 1.3, 0.3
    
    # スピナーを使用して処理中であることを表示
    with st.spinner('RAW画像を処理中です。しばらくお待ちください...'):
        # RAW画像の処理
        with rawpy.imread(uploaded_file) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)
        
        # 16ビットから0-1の範囲に正規化
        image = rgb.astype(np.float32) / 65535.0
        
        # 露出補正
        image = np.power(image, 1.0 / (2 ** exposure))
        
        # 明るさ調整
        image = image * brightness
        
        # コントラスト調整
        image = (image - 0.5) * contrast + 0.5
        
        # ハイライトとシャドウの調整
        def adjust_range(img, amount, range_min, range_max):
            mask = np.clip((img - range_min) / (range_max - range_min), 0, 1)
            return img + (amount * mask * (1 - img))
        
        image = adjust_range(image, highlights, 0.5, 1.0)
        image = adjust_range(image, shadows, 0.0, 0.5)
        
        # 白レベルと黒レベルの調整
        image = np.clip(image * (1 + whites) - blacks, 0, 1)
        
        # 彩度調整
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        image = np.clip(image + (image - gray[..., None]) * (saturation - 1), 0, 1)
        
        # 色温度調整（warmth）
        image[..., 2] = np.clip(image[..., 2] * (2 - warmth), 0, 1)  # 青チャンネルを調整
        image[..., 0] = np.clip(image[..., 0] * warmth, 0, 1)  # 赤チャンネルを調整
        
        # 色合い調整（tint）
        image[..., 1] = np.clip(image[..., 1] * (1 + tint), 0, 1)  # 緑チャンネルを調整
        
        # クラリティ（局所コントラスト）
        if clarity != 0:
            low_contrast = ndimage.gaussian_filter(image, sigma=3)
            high_contrast = image + (image - low_contrast) * clarity
            image = np.clip(high_contrast, 0, 1)
        
        # ビネット効果
        if vignette > 0:
            rows, cols = image.shape[:2]
            kernel_x = cv2.getGaussianKernel(cols, cols/2)
            kernel_y = cv2.getGaussianKernel(rows, rows/2)
            kernel = kernel_y * kernel_x.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            vignette_mask = np.clip(mask, 0, 1)
            image = image * (1 - vignette + vignette * vignette_mask[..., None])
        
        # 最終的なクリッピング
        processed = np.clip(image * 255, 0, 255).astype(np.uint8)
    
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