import pandas as pd
import os
import shutil

# Đường dẫn tới thư mục chứa ảnh và tệp metadata
metadata_path = 'data/HAM10000_metadata.csv'
images_dir = 'data/HAM10000_images_part_1/'
output_dir = 'mel_images/'

# Đọc tệp metadata
df = pd.read_csv(metadata_path)

# Lọc các ảnh có nhãn 'mel'
mel_images = df[df['dx'] == 'mel']['image_id'].tolist()

# Tạo thư mục lưu ảnh 'mel' nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Sao chép các ảnh 'mel' vào thư mục mới
for image_id in mel_images:
    image_filename = f"{image_id}.jpg"
    src_path = os.path.join(images_dir, image_filename)
    dst_path = os.path.join(output_dir, image_filename)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Hiển thị 5 ảnh đầu tiên trong thư mục 'mel_images'
mel_image_files = os.listdir(output_dir)[:5]
for image_file in mel_image_files:
    img_path = os.path.join(output_dir, image_file)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(image_file)
    plt.axis('off')
    plt.show()
