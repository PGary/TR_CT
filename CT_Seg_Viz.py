import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import filedialog

# --- 1. 自定义色卡 (黑色为背景) ---
cmap_red = LinearSegmentedColormap.from_list("black_red", ["black", "red"])
cmap_orange = LinearSegmentedColormap.from_list("black_orange", ["black", "orange"])

# --- 2. 状态管理 ---
if 'step' not in st.session_state: st.session_state.step = 'setup'
if 'path' not in st.session_state: st.session_state.path = ""
if 'crops' not in st.session_state:
    st.session_state.crops = {k: [100, 1200, 50, 985] for k in ['qian1', 'shang1', 'you1']}
if 'raw_data' not in st.session_state: st.session_state.raw_data = None
if 'bin1_data' not in st.session_state: st.session_state.bin1_data = None
if 'bin2_data' not in st.session_state: st.session_state.bin2_data = None

st.set_page_config(layout="wide", page_title="Thermal Runaway CT Expert")


# --- 3. 辅助函数 ---
def browse_folder():
    root = tk.Tk();
    root.withdraw();
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(master=root)
    root.destroy()
    if path: st.session_state.path = path


def get_seg_indices(total, m):
    size = int(total / (0.8 * m + 0.2))
    overlap = int(0.2 * size)
    step = size - overlap
    return [(int(i * step), int(min(i * step + size, total))) for i in range(m)]


def read_and_crop(args):
    path, h1, h2, w1, w2 = args
    img = cv2.imread(path, 0)
    return img[h1:h2, w1:w2] if img is not None else None


def load_data_threaded(folder_path, indices, crop, p_bar, status_txt, label, mode='raw', threshold_range=None):
    valid_exts = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
    w1, w2, h1, h2 = crop
    processed_avgs = []

    for idx, (s, e) in enumerate(indices):
        status_txt.text(f"计算 {label} - {mode}: 分段 {idx + 1}/{len(indices)}")
        subset = [os.path.join(folder_path, files[i]) for i in range(s, min(e, len(files)))]
        with ThreadPoolExecutor(max_workers=8) as exe:
            stack = list(exe.map(read_and_crop, [(p, h1, h2, w1, w2) for p in subset]))

        stack = [i for i in stack if i is not None]
        if not stack:
            processed_avgs.append(np.zeros((h2 - h1, w2 - w1), dtype=np.float32));
            continue

        if mode == 'raw':
            processed_avgs.append(np.mean(stack, axis=0))
        else:
            low, high = threshold_range
            bin_stack = [((img >= low) & (img <= high)).astype(np.float32) for img in stack]
            processed_avgs.append(np.mean(bin_stack, axis=0))
        p_bar.progress((idx + 1) / len(indices))
    return np.array(processed_avgs)


# --- 4. 侧边栏 ---
with st.sidebar:
    st.header("📂 1. 项目与路径")
    if st.button("📁 浏览本地文件夹"): browse_folder()
    st.session_state.path = st.text_input("项目路径:", st.session_state.path)
    m_seg = st.slider("分段数量 (m)", 5, 20, 9)

    st.divider()
    st.header("✂️ 2. 独立裁切")
    dirs_map = {'qian1': 'Front', 'shang1': 'Top', 'you1': 'Right'}
    for k, label in dirs_map.items():
        with st.expander(f"裁切: {label}"):
            c = st.session_state.crops[k]
            c[0] = st.number_input(f"W1 ({k})", 0, 5000, c[0], key=f"w1_{k}")
            c[1] = st.number_input(f"W2 ({k})", 0, 5000, c[1], key=f"w2_{k}")
            c[2] = st.number_input(f"H1 ({k})", 0, 5000, c[2], key=f"h1_{k}")
            c[3] = st.number_input(f"H2 ({k})", 0, 5000, c[3], key=f"h2_{k}")

    if st.button("🚀 加载原始灰度数据"):
        if os.path.exists(st.session_state.path):
            res = {}
            pb = st.progress(0);
            txt = st.empty()
            for k in dirs_map.keys():
                sub_p = os.path.join(st.session_state.path, k)
                files = [f for f in os.listdir(sub_p) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))]
                res[k] = load_data_threaded(sub_p, get_seg_indices(len(files), m_seg), st.session_state.crops[k], pb,
                                            txt, dirs_map[k], mode='raw')
            st.session_state.raw_data = res
            st.session_state.step = 'thre1_tuning';
            st.rerun()

# --- 5. 主流程 ---

# 步骤 1: 裁切预览
if st.session_state.step == 'setup':
    st.header("1. 裁切预览 (中位切片)")
    if st.session_state.path and os.path.exists(st.session_state.path):
        cols = st.columns(3)
        for i, (k, label) in enumerate(dirs_map.items()):
            sub_p = os.path.join(st.session_state.path, k)
            if os.path.exists(sub_p):
                files = sorted(
                    [f for f in os.listdir(sub_p) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
                if files:
                    img = cv2.imread(os.path.join(sub_p, files[len(files) // 2]), 0)
                    if img is not None:
                        fig, ax = plt.subplots();
                        ax.imshow(img, cmap='gray');
                        c = st.session_state.crops[k]
                        ax.add_patch(plt.Rectangle((c[0], c[2]), c[1] - c[0], c[3] - c[2], lw=2, ec='red', fc='none'))
                        ax.set_title(f"{label} View");
                        ax.axis('on');
                        cols[i].pyplot(fig);
                        plt.close(fig)

# 步骤 2: Thre1 确定 (左右布局)
elif st.session_state.step == 'thre1_tuning':
    st.header("2. 确定第一阈值范围 (Thre 1)")
    col_l, col_r = st.columns([1, 4])
    with col_l:
        l1 = st.slider("二值化 1 下限", 0, 255, 0)
        h1 = st.slider("二值化 1 上限", 0, 255, 120)
        if st.button("✅ 确认并进入下一步"):
            pb = st.progress(0);
            txt = st.empty()
            res_bin1 = {}
            for k in dirs_map.keys():
                sub_p = os.path.join(st.session_state.path, k)
                files = [f for f in os.listdir(sub_p) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))]
                res_bin1[k] = load_data_threaded(sub_p, get_seg_indices(len(files), m_seg), st.session_state.crops[k],
                                                 pb, txt, dirs_map[k], mode='bin', threshold_range=(l1, h1))
            st.session_state.bin1_data = res_bin1;
            st.session_state.r1 = (l1, h1);
            st.session_state.step = 'thre2_tuning';
            st.rerun()
    with col_r:
        sub_p = os.path.join(st.session_state.path, 'qian1')
        files = sorted([f for f in os.listdir(sub_p) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
        idxs = get_seg_indices(len(files), m_seg)[m_seg // 2]
        img_cols = st.columns(3)
        for i, s_idx in enumerate([idxs[0] + (idxs[1] - idxs[0]) // 4, idxs[0] + (idxs[1] - idxs[0]) // 2,
                                   idxs[0] + 3 * (idxs[1] - idxs[0]) // 4]):
            img = cv2.imread(os.path.join(sub_p, files[s_idx]), 0)[
                  st.session_state.crops['qian1'][2]:st.session_state.crops['qian1'][3],
                  st.session_state.crops['qian1'][0]:st.session_state.crops['qian1'][1]]
            with img_cols[i]:
                fig, ax = plt.subplots(2, 1, figsize=(4, 7))
                ax[0].imshow(img, cmap='gray');
                ax[0].axis('off')
                mask = ((img >= l1) & (img <= h1)).astype(np.float32)
                ax[1].imshow(mask, cmap=cmap_red);
                ax[1].axis('off');
                st.pyplot(fig);
                plt.close(fig)

# 步骤 3: Thre2 确定 (左右布局)
elif st.session_state.step == 'thre2_tuning':
    st.header("3. 确定第二阈值范围 (Thre 2)")
    col_l, col_r = st.columns([1, 4])
    with col_l:
        l2 = st.slider("二值化 2 下限", 0, 255, 90)
        h2 = st.slider("二值化 2 上限", 0, 255, 120)
        if st.button("✅ 确认并进入调窗预览"):
            pb = st.progress(0);
            txt = st.empty()
            res_bin2 = {}
            for k in dirs_map.keys():
                sub_p = os.path.join(st.session_state.path, k)
                files = [f for f in os.listdir(sub_p) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))]
                res_bin2[k] = load_data_threaded(sub_p, get_seg_indices(len(files), m_seg), st.session_state.crops[k],
                                                 pb, txt, dirs_map[k], mode='bin', threshold_range=(l2, h2))
            st.session_state.bin2_data = res_bin2;
            st.session_state.r2 = (l2, h2);
            st.session_state.step = 'scaling_preview';
            st.rerun()
    with col_r:
        sub_p = os.path.join(st.session_state.path, 'qian1')
        files = sorted([f for f in os.listdir(sub_p) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
        idxs = get_seg_indices(len(files), m_seg)[m_seg // 2]
        img_cols = st.columns(3)
        for i, s_idx in enumerate([idxs[0] + (idxs[1] - idxs[0]) // 4, idxs[0] + (idxs[1] - idxs[0]) // 2,
                                   idxs[0] + 3 * (idxs[1] - idxs[0]) // 4]):
            img = cv2.imread(os.path.join(sub_p, files[s_idx]), 0)[
                  st.session_state.crops['qian1'][2]:st.session_state.crops['qian1'][3],
                  st.session_state.crops['qian1'][0]:st.session_state.crops['qian1'][1]]
            with img_cols[i]:
                fig, ax = plt.subplots(2, 1, figsize=(4, 7))
                ax[0].imshow(img, cmap='gray');
                ax[0].axis('off')
                mask = ((img >= l2) & (img <= h2)).astype(np.float32)
                ax[1].imshow(mask, cmap=cmap_orange);
                ax[1].axis('off');
                st.pyplot(fig);
                plt.close(fig)

# 步骤 4: 调窗预览 (vmin/vmax 独立选择)
elif st.session_state.step == 'scaling_preview':
    st.header("4. 设定各图层显示亮度与对比度 (vmin/vmax 调窗)")
    st.info("💡 请拖动下方滑块，实时观察中间 Segment (Front) 的三层显示效果。")

    # 获取中间段数据
    mid_idx = m_seg // 2
    raw_mid = st.session_state.raw_data['qian1'][mid_idx]
    bin1_mid = st.session_state.bin1_data['qian1'][mid_idx]
    bin2_mid = st.session_state.bin2_data['qian1'][mid_idx]

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown("### 1. 原始灰度层")
        vmin_r = st.slider("Raw vmin", 0.0, 255.0, 0.0)
        vmax_r = st.slider("Raw vmax", 0.0, 255.0, 255.0)
    with col_s2:
        st.markdown("### 2. 二值分布层 1")
        vmin_b1 = st.slider("Bin1 vmin", 0.0, 1.0, 0.0, 0.01)
        vmax_b1 = st.slider("Bin1 vmax", 0.0, 1.0, 0.4, 0.01)
    with col_s3:
        st.markdown("### 3. 二值分布层 2")
        vmin_b2 = st.slider("Bin2 vmin", 0.0, 1.0, 0.0, 0.01)
        vmax_b2 = st.slider("Bin2 vmax", 0.0, 1.0, 0.4, 0.01)

    st.divider()

    # 实时预览
    prev_col1, prev_col2, prev_col3 = st.columns(3)
    with prev_col1:
        fig1, ax1 = plt.subplots();
        ax1.imshow(raw_mid, cmap='gray', vmin=vmin_r, vmax=vmax_r);
        ax1.set_title("Raw Scaling Preview");
        ax1.axis('off');
        st.pyplot(fig1);
        plt.close(fig1)
    with prev_col2:
        fig2, ax2 = plt.subplots();
        ax2.imshow(bin1_mid, cmap=cmap_red, vmin=vmin_b1, vmax=vmax_b1);
        ax2.set_title("Dist 1 Scaling Preview");
        ax2.axis('off');
        st.pyplot(fig2);
        plt.close(fig2)
    with prev_col3:
        fig3, ax3 = plt.subplots();
        ax3.imshow(bin2_mid, cmap=cmap_orange, vmin=vmin_b2, vmax=vmax_b2);
        ax3.set_title("Dist 2 Scaling Preview");
        ax3.axis('off');
        st.pyplot(fig3);
        plt.close(fig3)

    if st.button("🚀 生成结果大图"):
        st.session_state.scales = {'raw': (vmin_r, vmax_r), 'bin1': (vmin_b1, vmax_b1), 'bin2': (vmin_b2, vmax_b2)}
        st.session_state.step = 'report';
        st.rerun()

# 步骤 5: 最终报告
elif st.session_state.step == 'report':
    st.header("5. 高清CT分析结果")
    raw, b1, b2, sc = st.session_state.raw_data, st.session_state.bin1_data, st.session_state.bin2_data, st.session_state.scales
    h_px, w_px = raw['qian1'][0].shape
    aspect = h_px / w_px
    fig = plt.figure(figsize=(m_seg * 2.8, 9 * 2.8 * aspect))
    gs_m = gridspec.GridSpec(3, 1, hspace=0.1)
    for i, (k, label) in enumerate(dirs_map.items()):
        igs = gridspec.GridSpecFromSubplotSpec(3, m_seg, subplot_spec=gs_m[i], wspace=0.01, hspace=0.01)
        for col in range(m_seg):
            ax1 = fig.add_subplot(igs[0, col]);
            ax1.imshow(raw[k][col], cmap='gray', vmin=sc['raw'][0], vmax=sc['raw'][1]);
            ax1.axis('off')
            if col == 0: ax1.set_ylabel(f"{label}\nRaw", rotation=0, labelpad=40, fontsize=18, va='center',
                                        fontweight='bold')
            if i == 0: ax1.set_title(f"Seg {col + 1}", fontsize=14)
            ax2 = fig.add_subplot(igs[1, col]);
            ax2.imshow(b1[k][col], cmap=cmap_red, vmin=sc['bin1'][0], vmax=sc['bin1'][1]);
            ax2.axis('off')
            if col == 0: ax2.set_ylabel(f"Dist 1\n({st.session_state.r1})", rotation=0, labelpad=40, fontsize=14,
                                        va='center')
            ax3 = fig.add_subplot(igs[2, col]);
            ax3.imshow(b2[k][col], cmap=cmap_orange, vmin=sc['bin2'][0], vmax=sc['bin2'][1]);
            ax3.axis('off')
            if col == 0: ax3.set_ylabel(f"Dist 2\n({st.session_state.r2})", rotation=0, labelpad=40, fontsize=14,
                                        va='center')
    st.pyplot(fig)
    buf = BytesIO();
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=180)
    st.download_button("💾 下载结果大图", buf.getvalue(), f"Report_{os.path.basename(st.session_state.path)}.png",
                       "image/png")
    if st.button("🔄 重新开始分析"): st.session_state.step = 'setup'; st.rerun()
