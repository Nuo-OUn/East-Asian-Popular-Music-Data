import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# 核心参数配置 (已全部硬编码对齐上一阶段)
# ==========================================
INPUT_CSV = 'east_asian_pop_features.csv'  # 上一步清洗完毕的数据集
OUTPUT_CSV = 'east_asian_pca_3d.csv'       # 降维后输出给网页渲染的 3D 数据集
LOADINGS_CSV = 'pca_loadings.csv'          # 主成分载荷矩阵(用于热力图解读各轴物理意义)

def run_pca_pipeline():
    """
    严谨的 PCA 降维流水线。
    将 9 维连续性声学特征压缩至 3 维可视化空间。
    """
    try:
        print(f"1. 正在读取清洗后的东亚音乐数据: {INPUT_CSV} ...")
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
        
        # 提取歌手名 (如果上一阶段有保留的话) 和 歌曲名
        # 为了防错，我们用 get 兜底，确保非数值列不会混入计算
        meta_cols = ['Track_Name']
        if 'Artist' in df.columns:
            meta_cols.append('Artist')
        elif 'artists' in df.columns:
            meta_cols.append('artists')
        elif 'track_artist' in df.columns:
            meta_cols.append('track_artist')
            
        df_meta = df[meta_cols].copy()
        
        # 2. 严格筛选连续型数值特征 (排除 Key, Mode 等离散分类特征)
        # 这些特征正是我们在上一阶段重命名并保留下来的
        numeric_features = [
            'Danceability', 'Energy', 'Loudness', 'Speechiness', 
            'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo'
        ]
        
        # 检查是否所有特征都存在于数据集中
        missing_features = [f for f in numeric_features if f not in df.columns]
        if missing_features:
            print(f"[错误] 输入数据缺少以下特征列：{missing_features}")
            return
            
        x = df.loc[:, numeric_features].values
        
        # 3. 数学标准化 (均值为0，方差为1)
        print("2. 正在执行数据标准化 (StandardScaler) ...")
        x_scaled = StandardScaler().fit_transform(x)
        
        # 4. 执行 PCA 降维 (保留 3 个主成分)
        print("3. 正在执行主成分分析 (PCA) 将维度压缩至 3D ...")
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(x_scaled)
        
        # 5. 重组数据：将降维后的 XYZ 坐标与原始歌曲名、原声度合并
        # 保留 Acousticness 和 Energy 是为了下一步在网页交互时，悬浮能显示原始数据
        principal_df = pd.DataFrame(data=principal_components, columns=['X', 'Y', 'Z'])
        
        # 拼接元数据、XYZ坐标，以及两个核心参考指标
        final_df = pd.concat(
            [df_meta, principal_df, df[['Acousticness', 'Energy', 'Valence']]], 
            axis=1
        )
        
        # 打印学术指标：方差解释率
        variance_ratio = pca.explained_variance_ratio_
        print("\n--- 学术指标检验 ---")
        print(f"X轴 (PC1) 保留信息量: {variance_ratio[0]*100:.2f}%")
        print(f"Y轴 (PC2) 保留信息量: {variance_ratio[1]*100:.2f}%")
        print(f"Z轴 (PC3) 保留信息量: {variance_ratio[2]*100:.2f}%")
        print(f"总体累计保留信息量: {sum(variance_ratio)*100:.2f}%")
        print("--------------------\n")
        
        # 6. 导出最终成果
        final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"[完成] 降维数据已保存至: {OUTPUT_CSV}")

        # 7. 导出主成分载荷矩阵 (PCA Loadings)
        # pca.components_ 形状为 (3, n_features),即每一行是一个主成分在原始特征上的系数
        # 转置后行=原始特征、列=PC1/PC2/PC3,便于热力图直接渲染
        loadings_df = pd.DataFrame(
            pca.components_.T,
            index=numeric_features,
            columns=['PC1', 'PC2', 'PC3'],
        )
        loadings_df.to_csv(LOADINGS_CSV, encoding='utf-8-sig')
        print(f"[完成] 主成分载荷矩阵已保存至: {LOADINGS_CSV}")
        print("\n--- 主成分载荷矩阵 (Loadings) ---")
        print(loadings_df.round(3))
        print("--------------------------------")
        
    except FileNotFoundError:
        print(f"[错误] 找不到文件 {INPUT_CSV}。请确认已生成该文件并与脚本同目录。")
    except Exception as e:
        print(f"[错误] PCA 流水线失败: {e}")

if __name__ == '__main__':
    run_pca_pipeline()