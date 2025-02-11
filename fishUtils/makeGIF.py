from PIL import Image
import os

def create_gif(image_folder, output_gif, duration=2000):
    """
    將指定資料夾中的圖片製作成 GIF。
    
    :param image_folder: 圖片所在資料夾路徑。
    :param output_gif: 輸出的 GIF 檔案名稱。
    :param duration: 每張圖片的顯示時間（毫秒），預設為 500ms。
    """
    # images folder
    print(f"image_folder: {image_folder}")

    # 取得資料夾中的所有圖片檔案（按名稱排序）
    images = sorted([
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.endswith(('.png', '.jpg', '.jpeg'))
    ])
    # 將圖片分為兩組
    images_group1 = [img for img in images if 'large1-' in os.path.basename(img)]
    images_group2 = [img for img in images if 'large2-' in os.path.basename(img)]

    # 確保兩組圖片數量相同
    if len(images_group1) != len(images_group2):
        print("兩組圖片數量不一致，無法生成對比 GIF。")
        return

    # 合併兩組圖片
    combined_images = []
    for img1, img2 in zip(images_group1, images_group2):
        image1 = Image.open(img1)
        image2 = Image.open(img2)
        
        # 確保兩張圖片大小相同
        if image1.size != image2.size:
            print(f"圖片大小不一致：{img1} 和 {img2}")
            return
        
        # 創建一個新的圖片，寬度為兩張圖片寬度之和，高度不變
        combined_image = Image.new('RGB', (image1.width + image2.width, image1.height))

        # print images size and combined image size
        print(f"image1 size: {image1.size}")
        print(f"image2 size: {image2.size}")
        print(f"combined image size: {combined_image.size}")
        
        # 將兩張圖片拼接在一起
        combined_image.paste(image1, (0, 0))
        combined_image.paste(image2, (image1.width, 0))
        
        combined_images.append(combined_image)
    
    # 確保至少有一張合併後的圖片
    if not combined_images:
        print("沒有合併後的圖片，無法生成 GIF。")
        return

    print(f"合併後的圖片數量: {len(combined_images)}")
    
    # 打開第一張合併後的圖片作為基底
    base_image = combined_images[0]
    
    # 將其他合併後的圖片讀入為帧
    frames = combined_images[1:]
    
    # 儲存成 GIF
    base_image.save(
        output_gif,
        format='GIF',
        save_all=True,
        append_images=frames,
        duration=duration,
        loop=0  # 設定為 0 表示無限循環
    )
    print(f"GIF 已儲存為 {output_gif}")

# 設定圖片資料夾和輸出檔案名稱
image_folder = "/data/opencvScreenShot/"  # 替換為圖片所在的資料夾路徑
output_gif = os.path.join(image_folder, "comparison.gif")

# 呼叫函式生成 GIF
create_gif(image_folder, output_gif)
