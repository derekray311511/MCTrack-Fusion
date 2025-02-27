import os
import json
import matplotlib.pyplot as plt
import argparse
import math

def get_metric_from_json(data, metric, category=None):
    """
    根據指定的 metric 與 category 從 JSON 中提取數值。
    若 category 為 None，則直接從頂層取值，
    否則從 data["label_metrics"][metric][category] 中取。
    """
    if category is None:
        return data.get(metric, None)
    else:
        # 針對單項類別結果
        if "label_metrics" in data and metric in data["label_metrics"]:
            return data["label_metrics"][metric].get(category, None)
        else:
            return None

def extract_results(root_dir, methods, distances, metric, category):
    """
    從各方法的資料夾中提取對應距離下的指定 metric 數值，
    返回格式為 {method: {distance: value, ...}, ...}
    """
    results = {}
    for method in methods:
        results[method] = {}
        for d in distances:
            json_path = os.path.join(root_dir, method, f"eval_result_{d}_3.0_small_data2_1000frames", "metrics_summary.json")
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                value = get_metric_from_json(data, metric, category)
                results[method][d] = value
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                results[method][d] = None
    return results

def extract_all_results(root_dir, methods, distances, metrics, category):
    """
    對每個 metric 提取結果，返回格式為：
    { metric: { method: {distance: value, ...}, ... }, ... }
    """
    all_results = {}
    for metric in metrics:
        all_results[metric] = extract_results(root_dir, methods, distances, metric, category)
    return all_results

def plot_all_results(all_results, metrics, category, distances, method_aliases,
                     save_flag=False, save_dir=None, filename=None):
    """
    使用子圖繪製所有 metric 的結果。
    每個子圖 X 軸為驗證距離，Y 軸為對應 metric 的數值。
    採用動態 grid 排版，例如 4 個子圖會以 2×2 排列，6 個子圖則以 2×3 排列。
    若 save_flag 為 True，則儲存圖片到指定路徑。
    method_aliases: {原始方法名稱: "M1", ...} 用於替換圖例標籤。
    """
    n = len(metrics)
    # 動態決定 grid 排列（例如：4 個子圖 -> 2x2, 6 個子圖 -> 2x3）
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    
    # 若只有一個子圖，包裝成 list 處理
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for method, data in all_results[metric].items():
            sorted_distances = sorted(data.keys())
            values = [data[d] for d in sorted_distances]
            
            # 以簡短代號作為圖例標籤
            label = method_aliases.get(method, method)  
            ax.plot(sorted_distances, values, marker='o', label=label)
        
        ax.set_xlabel("Validation Distance (m)")
        ylabel = metric if category is None else f"{metric} ({category})"
        ax.set_ylabel(ylabel)
        ax.set_title(f"Comparison of {ylabel}")
        ax.legend()
        ax.grid(True)
    
    # 隱藏多餘的子圖（若 grid 超出實際數量）
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # 儲存圖片
    if save_flag:
        if save_dir is None:
            save_dir = os.path.join("results", "plots")
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            metrics_str = "_".join(metrics)
            # 增加判斷 category 是否為 None
            cat_str = "" if category is None else f"_{category}"
            filename = f"comparison{cat_str}_{metrics_str}_{min(distances)}-{max(distances)}m.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Figure saved at: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="自動提取並繪製不同方法於各驗證距離下的多個評估結果")
    parser.add_argument("--root_dir", type=str, default="results/nuscenes", help="結果資料夾的根目錄")
    parser.add_argument("--methods", type=str, nargs="+", default=["method1", "method2", "method3"],
                        help="要比較的方法，對應資料夾名稱")
    parser.add_argument("--distances", type=int, nargs="+", default=[50, 60, 70, 80, 90, 100],
                        help="驗證距離列表")
    parser.add_argument("--metrics", type=str, nargs="+", default=["recall"],
                        help="要提取的 metrics，例如 recall, amota, tp, fp, fn 等，可同時指定多個")
    parser.add_argument("--category", type=str, default=None,
                        help="如果要提取單項類別的結果，設定此參數，例如 car")
    # 新增儲存圖片的 flag 與參數
    parser.add_argument("--save", action="store_true", help="是否儲存繪製的圖片")
    parser.add_argument("--save_dir", type=str, default=None, help="儲存圖片的資料夾，預設為 results/plots")
    parser.add_argument("--filename", type=str, default=None, help="儲存圖片的檔名")
    args = parser.parse_args()

    # Hard code for testing（請根據實際情況調整方法名稱）
    methods = [
        "20250123_111959_Original",
        "20250211_111607_DBSCAN_threeStage",
        "20250221_075427_Mix_threeStage_th-0.25+Mix_Fix_BEST",
        "20250226_084614_Mix_threeStage_th-0.25+MixTrack_Nice",
    ]

    # 依序自動建立方法別名，如 M1, M2, M3 ...
    # 如果你想手動定義別名，也可以使用固定字典，例如:
    # method_aliases = {m: f"M{i+1}" for i, m in enumerate(methods)}
    method_aliases = {
        methods[0]: "MCTrack",
        methods[1]: "DBSCAN",
        methods[2]: "SegmentByDetection",
        methods[3]: "SegmentByTrack",
    }

    # 提取所有指定 metric 的結果
    all_results = extract_all_results(args.root_dir, methods, args.distances, args.metrics, args.category)
    print("提取結果：")
    for metric in args.metrics:
        print(metric, all_results[metric])
    
    # 繪製所有 metric 的折線圖，並依據 flag 決定是否儲存圖片
    plot_all_results(
        all_results, 
        args.metrics, 
        args.category, 
        args.distances, 
        method_aliases,
        args.save, 
        args.save_dir, 
        args.filename
    )
