import requests
from bs4 import BeautifulSoup


def chinese_crawler(url):
    # 设置请求头模拟浏览器访问
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9'  # 声明接受中文内容
    }

    try:
        # 发送HTTP请求
        response = requests.get(url, headers=headers)
        response.encoding = response.apparent_encoding  # 自动检测编码

        # 使用BeautifulSoup解析中文内容
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取主要中文内容（示例：提取页面所有段落）
        content = []
        # soup.find_all('content-right_1THTn'):
        elements = soup.find_all(class_="content-right_1THTn")
        for index, paragraph in enumerate(elements):
            text = paragraph.get_text(strip=True)
            if text:
                content.append(f"信息{index}: " + text)

        # 保存到文件（使用UTF-8编码）
        with open('chinese_content.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

        print(f"成功爬取{len(content)}条中文内容")

    except Exception as e:
        print(f"爬取失败: {str(e)}")


# 使用示例（以维基百科中文版为例）
if __name__ == "__main__":
    # 具有行气消积功效的药物是( )。\nA.使君子\nB.苦楝皮\nC.槟榔\nD.贯众\nE.雷丸
    # https://baike.baidu.com/item/%E5%9F%9F%E5%90%8D%E7%B3%BB%E7%BB%9F?fromModule=lemma_search-box
    target_url = "https://www.baidu.com/s?ie=UTF-8&wd='1)．若属肾阴虚证，最具诊断意义的症状是（  ）。A．头晕B．乏力C．遗精D．盗汗E．心悸'"
    # target_url = "https://zh.wikipedia.org/wiki/语迟"
    chinese_crawler(target_url)