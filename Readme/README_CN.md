# COPD Detection Project

## ��Ŀ���
����Ŀּ��ͨ�����ѧϰģ�ͼ��ͷ������������Էμ�����COPD����ʹ����ResNet18ģ�ͽ���ͼ����࣬�������������ǿ��Ԥ��������

## ��װ˵��
1. ��¡��Ŀ�ֿ⣺
    ```bash
    git clone https://github.com/bitqyh/copd-detection.git
    cd copd-detection
    ```

2. �������������⻷����
    ```bash
    conda create -n copd-env python=3.8
    conda activate copd-env
    ```

3. ��װ�����
    ```bash
    pip install -r requirements.txt
    ```

## ʹ��˵��
1. Ԥ�������ݣ�
    ```bash
    python DataSet.py
    ```

2. ѵ��ģ�ͣ�
    ```bash
    python main.py
    ```

3. ����ģ�ͣ�
    ```bash
    python evaluate.py
    ```

## �ļ�˵��
- `net.py`��������ResNet18ģ�͵Ľṹ��
- `label.py`�������ǩ���ݡ�
- `DataSet.py`����������Ԥ��������ֱ��ͼ���⻯��٤��У����
- `cutoff.py`���ü�ͼ�񲢽�����̬ѧ����
- `DeletePhoto.py`��ɾ��������������ͼ��
- `Divide.py`�������ݼ�����Ϊѵ��������֤���Ͳ��Լ���
- `Combine.py`���ϲ�ͼ�������ɶ�ͨ�����롣
- `main.py`����ѵ���ű�������ģ��ѵ������֤�߼���

## ������
- Python 3.8
- PyTorch
- torchvision
- numpy
- pandas
- opencv-python
- pydicom
- Pillow
- matplotlib
- seaborn
- openpyxl

## ����ָ��
��ӭ���״���ͱ������⡣���ύPull Request����Issues�б������⡣