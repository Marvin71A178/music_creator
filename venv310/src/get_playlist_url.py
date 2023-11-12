import json


def extract_values(obj, key):
    """遞迴提取指定key的所有值"""
    arr = []

    def extract(obj, arr, key):
        """子函式進行遞迴搜索"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results


# 從JSON檔案讀取
with open('./response/放鬆心情.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

values = extract_values(data, 'playlistId')


unique_values = set(values)
print(unique_values)
print(len(unique_values))
with open("./playlistId_list/放鬆心情.json", 'w', encoding="utf-8") as wf:
    json.dump(list(unique_values), wf, ensure_ascii=False, indent=4)
