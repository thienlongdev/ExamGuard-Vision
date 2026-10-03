import requests

for ds in ['cctvdataset/cctv-exam-monitor-dataset', 'shreyasudaya/scb-05-dataset']:
    u = f'https://www.kaggle.com/api/v1/datasets/download/{ds}'
    with requests.get(u, stream=True, allow_redirects=False) as res:
        print(ds, '->', res.status_code, res.headers.get('Location'))
