import pandas as pd

# ==========================================
# 1. 目标设定：定义你的“东亚流行乐”研究池
# ==========================================
# Kaggle数据集中的歌手名通常是英文拼写或罗马音，这里可以根据你的展示主题自由增删
TARGET_ARTISTS = [
    'JJ Lin', 'Jay Chou', 'Eason Chan',          # 华语代表
    'BTS', 'BLACKPINK', 'EXO',                   # K-pop 代表
    'Kenshi Yonezu', 'Hikaru Utada', 'YOASOBI'   # J-pop 代表
]

def process_kaggle_dataset(input_csv, output_csv):
    """
    严谨的数据清洗 Pipeline：
    从海量开源数据中精准提取东亚歌手，并清洗为无瑕疵的标准格式。
    """
    print(f"正在读取庞大的开源数据集: {input_csv}，请稍候...")
    try:
        # 1. 加载全量数据
        df_raw = pd.read_csv(input_csv, encoding="utf-8-sig")
        
        # 2. 核心清洗逻辑：过滤出目标歌手
        # 不同的 Kaggle 数据集，歌手列可能叫 'artists' 或 'artist_name'
        artist_col = 'artists' if 'artists' in df_raw.columns else 'track_artist'
        track_col = 'track_name' if 'track_name' in df_raw.columns else 'name'
        
        # 数据集中的 artists 通常是字符串格式的列表，如 "['JJ Lin']" 或单字符串 "JJ Lin"
        # 我们使用模糊匹配来捕获所有包含目标歌手的曲目
        mask = df_raw[artist_col].astype(str).str.contains('|'.join(TARGET_ARTISTS), case=False, na=False)
        df_filtered = df_raw[mask].copy()
        
        if df_filtered.empty:
            print("[错误] 未能在数据集中找到目标歌手，请检查 TARGET_ARTISTS 的拼写是否与数据集一致。")
            return

        print(f"[完成] 成功从数据集中提取出 {len(df_filtered)} 首东亚流行曲目。")

        # 3. 特征对齐与重命名 (Feature Alignment)
        # 强制将 Kaggle 的全小写表头，映射为我们 Phase 2 (PCA代码) 中严格要求的首字母大写格式
        feature_mapping = {
            track_col: 'Track_Name',
            artist_col: 'Artist',
            'danceability': 'Danceability',
            'energy': 'Energy',
            'key': 'Key',
            'loudness': 'Loudness',
            'mode': 'Mode',
            'speechiness': 'Speechiness',
            'acousticness': 'Acousticness',
            'instrumentalness': 'Instrumentalness',
            'liveness': 'Liveness',
            'valence': 'Valence',
            'tempo': 'Tempo'
        }
        
        # 只保留我们需要的列，并丢弃含有 NaN (缺失值) 的残次数据
        df_final = df_filtered.rename(columns=feature_mapping)
        required_columns = list(feature_mapping.values())
        
        # 确保所有需要的列都在数据集中
        missing_cols = [col for col in required_columns if col not in df_final.columns]
        if missing_cols:
            print(f"[错误] 数据集缺少必要特征列: {missing_cols}")
            return

        df_final = df_final[required_columns].dropna()
        
        # 4. 导出完美的数据集
        df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"[完成] 数据清洗完成，已保存为: {output_csv}")
        print("\n数据概览 (前 3 行):")
        print(df_final[['Track_Name', 'Artist', 'Danceability', 'Energy', 'Acousticness']].head(3))
        
    except FileNotFoundError:
        print(f"[错误] 找不到文件 {input_csv}。请确认数据集已放在当前文件夹。")
    except Exception as e:
        print(f"[错误] 数据处理失败: {e}")

if __name__ == '__main__':
    # 假设你从 Kaggle 下载的文件名为 dataset.csv，我们将输出清洗后的 east_asian_pop_features.csv
    process_kaggle_dataset('dataset.csv', 'east_asian_pop_features.csv')