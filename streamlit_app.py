import streamlit as st
import rawpy
import imageio
import numpy as np
from io import BytesIO

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

if uploaded_file is not None:
    st.write("処理中...")
    
    # RAW画像の処理
    with rawpy.imread(uploaded_file) as raw:
        rgb = raw.postprocess()
    
    # 0-1の範囲に正規化
    rgb = rgb.astype(np.float32) / 255.0
    
    # サマーカラープリセットの適用
    brightness = 1.2
    warmth = 1.1
    
    rgb = rgb * brightness
    rgb[:,:,0] = rgb[:,:,0] * warmth  # 赤チャンネルを強調
    rgb[:,:,2] = rgb[:,:,2] / warmth  # 青チャンネルを抑制
    
    # 値を0-1の範囲に収める
    processed = np.clip(rgb, 0, 1)
    
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