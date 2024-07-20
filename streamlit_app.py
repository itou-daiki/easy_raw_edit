import streamlit as st
import rawpy
import imageio
import numpy as np
from io import BytesIO
from scipy import ndimage

# ページ設定
st.set_page_config(page_title="RAW画像 サマープリセット変換アプリ", layout="wide")

# アプリのタイトルと制作者
st.title("RAW画像 サマープリセット変換アプリ")
st.caption("Created by Dit-Lab.(Daiki Ito)")

# 概要
st.markdown("""
## **概要**
このウェブアプリケーションでは、RAW画像にサマーカラープリセットを適用し、JPG形式でダウンロードすることができます。
iPadなどのデバイスでも対応しています。
""")

# ファイルアップローダー
uploaded_file = st.file_uploader("RAW画像をアップロードしてください", type=["raw", "arw", "cr2", "nef"])

def apply_summer_preset(image):
    # 明るさ調整
    brightness = 1.2
    image = image * brightness
    
    # コントラスト調整
    contrast = 1.2
    image = (image - 0.5) * contrast + 0.5
    
    # 彩度調整
    saturation = 1.1
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    image[..., :3] = np.clip(image[..., :3] + (image[..., :3] - gray[..., None]) * (saturation - 1), 0, 1)
    
    # 青みがかった冷たい色調の追加
    image[..., 2] = np.clip(image[..., 2] * 1.1, 0, 1)  # 青チャンネルを強調
    image[..., 0] = np.clip(image[..., 0] * 0.9, 0, 1)  # 赤チャンネルを抑制
    
    # ハイライトの強調
    highlight_threshold = 0.7
    highlights = np.where(image > highlight_threshold, image, 0)
    image = np.clip(image + highlights * 0.1, 0, 1)
    
    # 軽いぼかし効果
    image = ndimage.gaussian_filter(image, sigma=0.5)
    
    return np.clip(image, 0, 1)

if uploaded_file is not None:
    st.write("処理中...")
    
    # RAW画像の処理
    with rawpy.imread(uploaded_file) as raw:
        rgb = raw.postprocess()
    
    # 0-1の範囲に正規化
    rgb = rgb.astype(np.float32) / 255.0
    
    # サマーカラープリセットの適用
    processed = apply_summer_preset(rgb)
    
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