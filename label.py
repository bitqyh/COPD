import numpy as np
import pandas as pd
import os
from os import getcwd

classes = ['Health', 'Lung']
sets = ["E:\桌面\文件\慢阻肺\CT\Combin_test"]
df = pd.read_excel('慢阻肺标注样例.xlsx', engine='openpyxl')
for index, row in df.iterrows():
    # image_name = row['B'] # 文件夹名Ct0000000
    # label = row['C'] #类别
    selected_columns = df[['img-ID', 'Label']]
    # 转换为二维数组
    label = np.array(selected_columns)
if __name__ == '__main__':
    wd = getcwd()
    for se in sets:
        list_file = open('cls_' + 'CT' + '.txt', 'w')

        datasets_path = se
        types_name = os.listdir(datasets_path)  # type_name = ['Health', 'Lung']
        for type_name in types_name:
            if type_name not in classes:
                continue
            CT_path = os.path.join(datasets_path, type_name)
            CT_IDes = os.listdir(CT_path)  # CT000000
            for CT_ID in CT_IDes:
                indices = np.where(label[:, 0] == CT_ID)
                if len(indices[0]) > 0:  # 如果找到了匹配的元素
                    index = indices[0][0]  # 获取第一个匹配元素的索引

                    cls_id = label[index, 1]
                    # if label[index, 1] != 1:
                    #     cls_id = 1 if label[index, 1] > 0 else 0
                    # else:
                    #     continue
                    # photos_path = os.path.join(CT_path, CT_ID)
                    # photo_names = os.listdir(photos_path)
                    # for photo_name in photo_names:
                    list_file.write(str(cls_id) + ';' + '%s' % os.path.join(CT_path, CT_ID))
                    list_file.write('\n')

list_file.close()
