import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# --- 1. 专业色卡设定 ---
cmap_red = LinearSegmentedColormap.from_list("black_red", ["black", "red"])
cmap_orange = LinearSegmentedColormap.from_list("black_orange", ["black", "orange"])

# --- 2. 状态管理与坐标默认值设定 ---
if 'step' not in st.session_state: st.session_state.step = 'setup'
if 'uploaded_files' not in st.session_state: st.session_state.uploaded_files = {}

# 严格按照用户要求的默认值初始化坐标
if 'crops' not in st.session_state:
    st.session_state.crops = {
        'qian1': [350, 1000, 50, 985],  # W1, W2, H1, H2
        'shang1': [150, 1000, 300, 800],
        'you1': [500, 800, 50, 985]
    }

st.set_page_config(layout="wide", page_title="CT Expert Web V11")


# --- 3. 核心内存读取函数 ---
def get_seg_indices(total, m):
    size = int(total / (0.8 * m + 0.2))
    overlap = int(0.2 * size)
    step = size - overlap
    return [(int(i * step), int(min(i * step + size, total))) for i in range(m)]


def decode_and_crop(args):
    file_bytes, h1, h2, w1, w2 = args
    file_bytes.seek(0)
    file_np = np.frombuffer(file_bytes.read(), np.uint8)
    img = cv2.imdecode(file_np, cv2.IMREAD_GRAYSCALE)
    return img[h1:h2, w1:w2] if img is not None else None


def process_uploaded_threaded(files, indices, crop, p_bar, status_txt, label, mode='raw', threshold_range=None):
    w1, w2, h1, h2 = crop
    processed_avgs = []
    for idx, (s, e) in enumerate(indices):
        status_txt.text(f"处理 {label} - {mode}: 分段 {idx + 1}/{len(indices)}")
        subset_files = files[s:e]
        args_list = [(f, h1, h2, w1, w2) for f in subset_files]
        with ThreadPoolExecutor(max_workers=8) as exe:
            stack = list(exe.map(decode_and_crop, args_list))
        stack = [i for i in stack if i is not None]
        if not stack:
            processed_avgs.append(np.zeros((h2 - h1, w2 - w1), dtype=np.float32));
            continue
        if mode == 'raw':
            processed_avgs.append(np.mean(stack, axis=0))
        else:
            l, h = threshold_range
            bin_stack = [((img >= l) & (img <= h)).astype(np.float32) for img in stack]
            processed_avgs.append(np.mean(bin_stack, axis=0))
        p_bar.progress((idx + 1) / len(indices))
    return np.array(processed_avgs)


# --- 4. 侧边栏：文件上传与坐标微调 ---
with st.sidebar:
    st.header("📤 1. 上传本地图片集")
    st.caption("提示：在文件框内 Ctrl+A 全选文件夹内图片上传")
    dirs_map = {'qian1': 'Front (前)', 'shang1': 'Top (上)', 'you1': 'Right (右)'}

    for k, label in dirs_map.items():
        files = st.file_uploader(f"上传 {label} 图片", accept_multiple_files=True, key=f"up_{k}")
        if files:
            st.session_state.uploaded_files[k] = sorted(files, key=lambda x: x.name)
            st.success(f"已加载 {len(files)} 张")

    m_seg = st.slider("分段数量 (m)", 5, 20, 9)

    st.divider()
    st.header("✂️ 2. 裁切坐标微调")
    for k, label in dirs_map.items():
        with st.expander(f"{label} 坐标设置"):
            c = st.session_state.crops[k]
            # 默认值已在初始化时设定
            c[0] = st.number_input(f"W1 ({k})", 0, 5000, c[0], key=f"w1_{k}")
            c[1] = st.number_input(f"W2 ({k})", 0, 5000, c[1], key=f"w2_{k}")
            c[2] = st.number_input(f"H1 ({k})", 0, 5000, c[2], key=f"h1_{k}")
            c[3] = st.number_input(f"H2 ({k})", 0, 5000, c[3], key=f"h2_{k}")

    if st.button("🚀 开始计算原始平均图"):
        if len(st.session_state.uploaded_files) == 3:
            res = {}
            pb = st.progress(0);
            txt = st.empty()
            for k in dirs_map.keys():
                files = st.session_state.uploaded_files[k]
                res[k] = process_uploaded_threaded(files, get_seg_indices(len(files), m_seg), st.session_state.crops[k],
                                                   pb, txt, dirs_map[k])
            st.session_state.raw_data = res
            st.session_state.step = 'thre1_tuning';
            st.rerun()
        else:
            st.error("请先上传所有方向的图片集")

# --- 5. 主流程界面 ---

# 步骤 1: 裁切预览
if st.session_state.step == 'setup':
    st.header("1. 独立裁切预览 (中位切片)")
    cols = st.columns(3)
    for i, (k, label) in enumerate(dirs_map.items()):
        if k in st.session_state.uploaded_files:
            files = st.session_state.uploaded_files[k]
            f = files[len(files) // 2];
            f.seek(0)
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 0)
            if img is not None:
                fig, ax = plt.subplots();
                ax.imshow(img, cmap='gray')
                c = st.session_state.crops[k]
                ax.add_patch(plt.Rectangle((c[0], c[2]), c[1] - c[0], c[3] - c[2], lw=2, ec='red', fc='none'))
                ax.set_title(f"{label}");
                ax.axis('on');
                cols[i].pyplot(fig);
                plt.close(fig)

# 步骤 2 & 3: 阈值确定 (左右布局 + 三点验证)
elif st.session_state.step in ['thre1_tuning', 'thre2_tuning']:
    is_s1 = st.session_state.step == 'thre1_tuning'
    st.header(f"{'2. 第一阈值设定' if is_s1 else '3. 第二阈值设定'}")
    col_l, col_r = st.columns([1, 4])
    with col_l:
        l = st.slider("下限", 0, 255, 0 if is_s1 else 60)
        h = st.slider("上限", 0, 255, 120)
        if st.button("✅ 确认并生成分布图"):
            pb = st.progress(0);
            txt = st.empty()
            res_bin = {}
            for k in dirs_map.keys():
                files = st.session_state.uploaded_files[k]
                res_bin[k] = process_uploaded_threaded(files, get_seg_indices(len(files), m_seg),
                                                       st.session_state.crops[k], pb, txt, dirs_map[k], mode='bin',
                                                       threshold_range=(l, h))
            if is_s1:
                st.session_state.bin1_data = res_bin;
                st.session_state.r1 = (l, h);
                st.session_state.step = 'thre2_tuning'
            else:
                st.session_state.bin2_data = res_bin;
                st.session_state.r2 = (l, h);
                st.session_state.step = 'scaling'
            st.rerun()
    with col_r:
        files = st.session_state.uploaded_files['qian1']
        idxs = get_seg_indices(len(files), m_seg)[m_seg // 2]
        img_cols = st.columns(3)
        samples = [idxs[0] + (idxs[1] - idxs[0]) // 4, idxs[0] + (idxs[1] - idxs[0]) // 2,
                   idxs[0] + 3 * (idxs[1] - idxs[0]) // 4]
        for i, s_idx in enumerate(samples):
            f = files[s_idx];
            f.seek(0)
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 0)[
                  st.session_state.crops['qian1'][2]:st.session_state.crops['qian1'][3],
                  st.session_state.crops['qian1'][0]:st.session_state.crops['qian1'][1]]
            with img_cols[i]:
                fig, ax = plt.subplots(2, 1, figsize=(4, 7))
                ax[0].imshow(img, cmap='gray');
                ax[0].axis('off')
                mask = ((img >= l) & (img <= h)).astype(np.float32)
                ax[1].imshow(mask, cmap=cmap_red if is_s1 else cmap_orange);
                ax[1].axis('off');
                st.pyplot(fig);
                plt.close(fig)

# 步骤 4: 调窗预览 (Step-by-step scaling)
elif st.session_state.step == 'scaling':
    st.header("4. 最终报告调窗预览")
    mid = m_seg // 2
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Raw 层")
        vr = st.slider("Raw vmin/max", 0.0, 255.0, (0.0, 255.0))
    with c2:
        st.subheader("Dist 1 层")
        v1 = st.slider("Bin1 vmin/max", 0.0, 1.0, (0.0, 0.4), 0.01)
    with c3:
        st.subheader("Dist 2 层")
        v2 = st.slider("Bin2 vmin/max", 0.0, 1.0, (0.0, 0.4), 0.01)

    p1, p2, p3 = st.columns(3)
    p1.imshow(st.session_state.raw_data['qian1'][mid], cmap='gray', vmin=vr[0], vmax=vr[1])
    p2.imshow(st.session_state.bin1_data['qian1'][mid], cmap=cmap_red, vmin=v1[0], vmax=v1[1])
    p3.imshow(st.session_state.bin2_data['qian1'][mid], cmap=cmap_orange, vmin=v2[0], vmax=v2[1])
    # 注意：Streamlit 1.10+ 直接支持 st.pyplot 或简单封装，此处为演示逻辑
    st.info("💡 预览满意后请点击下方按钮生成完整大图。")
    if st.button("🚀 生成最终报告"):
        st.session_state.sc = {'raw': vr, 'bin1': v1, 'bin2': v2};
        st.session_state.step = 'report';
        st.rerun()

# 步骤 5: 最终报告
elif st.session_state.step == 'report':
    st.header("5. 高清对比分析大图")
    raw, b1, b2, sc = st.session_state.raw_data, st.session_state.bin1_data, st.session_state.bin2_data, st.session_state.sc
    h_px, w_px = raw['qian1'][0].shape
    fig = plt.figure(figsize=(m_seg * 2.8, 9 * 2.8 * (h_px / w_px)))
    gs = gridspec.GridSpec(3, 1, hspace=0.1)
    for i, k in enumerate(['qian1', 'shang1', 'you1']):
        igs = gridspec.GridSpecFromSubplotSpec(3, m_seg, subplot_spec=gs[i], wspace=0.01, hspace=0.01)
        for col in range(m_seg):
            ax1 = fig.add_subplot(igs[0, col]);
            ax1.imshow(raw[k][col], cmap='gray', vmin=sc['raw'][0], vmax=sc['raw'][1]);
            ax1.axis('off')
            ax2 = fig.add_subplot(igs[1, col]);
            ax2.imshow(b1[k][col], cmap=cmap_red, vmin=sc['bin1'][0], vmax=sc['bin1'][1]);
            ax2.axis('off')
            ax3 = fig.add_subplot(igs[2, col]);
            ax3.imshow(b2[k][col], cmap=cmap_orange, vmin=sc['bin2'][0], vmax=sc['bin2'][1]);
            ax3.axis('off')
    st.pyplot(fig)
    buf = BytesIO();
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=180)
    st.download_button("💾 下载分析报告", buf.getvalue(), "Final_Report.png", "image/png")
    if st.button("🔄 重置"): st.session_state.step = 'setup'; st.rerun()
