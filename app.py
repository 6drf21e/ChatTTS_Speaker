import base64
import os
import os.path

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from gradio_leaderboard import Leaderboard
from pandas import DataFrame
import torch
import pybase16384 as b14
import numpy as np
import lzma

load_dotenv()

# 获取环境变量
storage_mode = os.getenv("STORAGE_MODE")
storage_path = os.getenv("STORAGE_PATH")
storage_url = os.getenv("STORAGE_URL")

# 临时文件目录
tmp_dir = os.path.join(os.getcwd(), "tmp")
os.makedirs(tmp_dir, exist_ok=True)


def _encode_spk_emb(spk_emb: torch.Tensor) -> str:
    with torch.no_grad():
        arr: np.ndarray = spk_emb.to(dtype=torch.float16, device="cpu").numpy()
        s = b14.encode_to_string(
            lzma.compress(
                arr.tobytes(),
                format=lzma.FORMAT_RAW,
                filters=[
                    {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}
                ],
            ),
        )
        del arr
    return s

def pt2str(pt_path):
    spk_emb = torch.load(pt_path, map_location="cpu")
    return _encode_spk_emb(spk_emb)

def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def base64_to_file(base64_str, output_path):
    with open(output_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def convert_to_markdown(percentage_str):
    """
    将百分比字符串转换为 markdown 格式
    :param percentage_str:
    :return:
    """
    if not percentage_str:
        return ""
    if not isinstance(percentage_str, str):
        return percentage_str
    items = percentage_str.split(";")
    markdown_str = "  ".join([f"**{item.split(':')[0]}** {item.split(':')[1]}%" for item in items])
    return markdown_str

def convert_to_str(percentage_str):
    """
    将百分比字符串转换为 str
    :param percentage_str:
    :return:
    """
    if not percentage_str or not isinstance(percentage_str, str):
        return "未知"
    items = percentage_str.split(";")
    # sort by value
    items.sort(key=lambda x: float(x.split(':')[1]), reverse=True)
    keys = [item.split(':')[0] for item in items]
    if keys and keys[0]:
        return keys[0]
    else:
        return "未知"

# Load
df = pd.read_csv("evaluation_results.csv", encoding="utf-8")

df["rank_long"] = df["rank_long"].apply(lambda x: round(x, 2))
df["rank_multi"] = df["rank_multi"].apply(lambda x: round(x, 2))
df["rank_single"] = df["rank_single"].apply(lambda x: round(x, 2))
df["gender_filter"] = df["gender"].apply(convert_to_str)
df["gender"] = df["gender"].apply(convert_to_markdown)
df["age_filter"] = df["age"].apply(convert_to_str)
df["age"] = df["age"].apply(convert_to_markdown)
df["feature"] = df["feature"].apply(convert_to_markdown)
df["score"] = df["score"].apply(lambda x: round(x, 2))


def download_wav_file(seed_id, download_url, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = os.path.join(local_dir, f"{seed_id}.wav")
    file_url = f"{download_url}/{seed_id}_test.wav"
    if not os.path.exists(local_file_path):
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(local_file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {file_url} to {local_file_path}")
        else:
            print(f"Failed to download {file_url}: Status code {response.status_code}")
    return local_file_path


def restore_wav_file(seed_id):
    """
    根据给定的 seed_id 恢复 WAV 文件。如果 storage_mode 为 'local'，
    则从本地存储路径中获取文件。如果 storage_mode 为 'url'，
    则从远程 URL 下载文件到临时目录。
    :param seed_id:
    :return:
    """
    if not seed_id:
        return None

    if storage_mode == "local":
        local_file_path = os.path.join(storage_path, f"{seed_id}_test.wav")
        if os.path.exists(local_file_path):
            return local_file_path
        else:
            print(f"Local file {local_file_path} does not exist.")
            return None

    elif storage_mode == "url":
        try:
            return download_wav_file(seed_id, storage_url, tmp_dir)
        except Exception as e:
            print(f"Failed to download WAV file: {e}")
            return None

    else:
        print(f"Invalid storage mode: {storage_mode}")
        return None


def restore_pt_file(seed_id):
    """
    根据给定的 seed_id 恢复 PT 文件。
    :param seed_id:
    :return:
    """
    row = df[df["seed_id"] == seed_id]
    if row.empty:
        return None
    row = row.iloc[0]
    if not row.empty:
        emb_data = row["emb_data"]
        output_path = os.path.join(tmp_dir, f"{row['seed_id']}_restored_emb.pt")
        base64_to_file(emb_data, output_path)
        return output_path
    else:
        return None


def seed_change(evt: gr.SelectData, value=None):
    """
    处理种子ID变化事件，根据选择的种子ID返回对应的.pt文件下载按钮和试听音频。
    :param evt:
    :param value:
    """
    print(f"You selected {evt.value} at {evt.index} from {evt.target}")

    if not isinstance(evt.index, list) or evt.index[1] != 0:
        return [
            None,
            gr.DownloadButton(value=None, label="Download .pt File", visible=False),
            gr.Audio(None, visible=False),
        ]

    assert isinstance(value, DataFrame), "Expected value to be a DataFrame"

    # seed_id
    seed_id = evt.value
    print(f"Selected seed_id: {seed_id}")

    # 获取 pt 文件
    down_file = restore_pt_file(seed_id)

    # spk_emb_str
    spk_emb_str = pt2str(down_file)

    # 获取试听文件
    wav_file = restore_wav_file(seed_id)
    if wav_file and not os.path.exists(wav_file):
        print(f"WAV file {wav_file} does not exist.")
        wav_file = None

    return [
        evt.index,
        gr.DownloadButton(value=down_file, label=f"Download .pt File [{seed_id}]", visible=True),
        gr.Audio(wav_file, visible=wav_file is not None),
        spk_emb_str,
    ]


with gr.Blocks() as demo:
    gr.Markdown("# 🥇 ChatTTS Speaker Leaderboard ")
    gr.Markdown("""
    ### 🎤 [ChatTTS](https://github.com/2noise/ChatTTS): 稳定音色查找与音色打标（实验性）欢迎下载试听音色！

    本项目已开源：[ChatTTS_Speaker](https://github.com/6drf21e/ChatTTS_Speaker) 欢迎 PR 和 Star！

    评估基于通义实验室：[eres2netv2_sv_zh-cn](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)
    """)

    with gr.Tab(label="🏆Leaderboard"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
### 参数解释

- **rank_long**: 长句文本的音色稳定性评分。
- **rank_multi**: 多句文本的音色稳定性评分。
- **rank_single**: 单句文本的音色稳定性评分。

这三个参数用于衡量不同音色在生成不同类型文本时的一致性，数值越高表示音色越稳定。

- **score**: 音色性别、年龄、特征的可能性，越高越准确。
- **gender age feature**: 音色的性别、年龄、特征。（特征准确度不高 仅供参考）

### 如何下载音色

- 点选一个音色，点击最下方的 **Download .pt File** 按钮，即可下载对应的 .pt 文件。

### FAQ

- **Q**: 怎么使用 .pt 文件？
- **A**: 可以直接在一些项目：例如：[ChatTTS_colab](https://github.com/6drf21e/ChatTTS_colab)  中载入使用。
也可以使用类似代码载入：
```python
spk = torch.load(<PT-FILE-PATH>)
params_infer_code = {
    'spk_emb': spk,
}
略
```
- **Q**: 为什么有的音色打分高但是很难听？
- **A**: 评分只是衡量音色的稳定性，不代表音色的好坏。可以根据自己的需求选择合适的音色。举个简单的例子：如果一个沙哑且结巴的音色一直很稳定，那么它的评分就会很高。
- **Q**: 我使用 seed_id 去生成音频，但是生成的音频不稳定？
- **A**: seed_id 只是一个参考ID 不同的环境下音色不一定一致。还是推荐使用 .pt 文件载入音色。
- **Q**: 音色标的男女准确吗？
- **A**: 当前第一批测试的音色有 2000 条，根据声纹相似性简单打标，准确度不高（特别是特征一项），仅供参考。如果大家有更好的标注方法，欢迎 PR。

                    """)
            with gr.Column(scale=3, min_width=800):
                leaderboard = Leaderboard(
                    value=df,
                    datatype=["markdown"] * 12,
                    select_columns=["seed_id", "rank_long", "rank_multi", "rank_single", "score", "gender", "age",
                                    "feature"],
                    search_columns=["gender", "age"],
                    filter_columns=["rank_long", "rank_multi", "rank_single", "gender_filter", "age_filter"],
                    hide_columns=["emb_data", "gender_filter", "age_filter"],
                )
                stats = gr.State(value=[1])
                download_button = gr.DownloadButton("Download .pt File", visible=True)
                spk_emb_str = gr.Textbox("", label="音色码/speaker embedding", lines=10)
                test_audio = gr.Audio(visible=True)
                gr.Markdown("选择 seed_id 才能下载 .pt 文件和试听音频。")
                # download_button.click(download, inputs=[stats], outputs=[])
                leaderboard.select(seed_change, inputs=[leaderboard], outputs=[stats, download_button, test_audio, spk_emb_str])

    with gr.Tab(label="📊Details"):
        gr.Markdown("""
    # 音色稳定性初步评估
    
    ## 原理
    
    利用 通义实验室开源的[eres2netv2_sv_zh-cn](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary) **SERes2NetV2 说话人识别模型** ，对同一个音色进行测评，评估其在不同语音样本中的一致性。具体步骤如下：
    
    1. **样本**：选择三个不同类型的测试样本：单句文本、多句文本和长句文本。
    2. **音色一致性评分**：
        - 对每对音频文件进行评分，计算它们是否来自同一说话人。
        - 使用 eres2netv2 模型，对每对音频文件进行打分，获得相似度分数。
    3. **稳定性评估**：
        - 计算每组音频文件的平均相似度分数和标准差。
        - 通过综合平均分和标准差，计算稳定性指数，用于衡量音色的一致性。
    
    
    ## 样本如下
    
    ### 单句文本
    - 这是一段测试文本[uv_break] 用来测试多批次生产音频的稳定性。 X 6次
    
    ### 多句文本
    - 今天早晨，市中心的主要道路因突发事故造成了严重堵塞[uv_break]。请驾驶员朋友们注意绕行，并听从现场交警的指挥。
    - 亲爱的朋友们，无论你现在处于什么样的境地，都不要放弃希望[uv_break]。每一个伟大的成功，都是从不懈的努力和坚定的信念中诞生的。
    - 很久很久以前，在一个宁静的小村庄里，住着一只名叫小花的可爱小猫咪[uv_break]。小花每天都喜欢在花园里玩耍，有一天，它遇到了一只迷路的小鸟。
    - 您好，欢迎致电本公司客服中心。为了更好地服务您，请在听到提示音后选择所需服务[uv_break]。如果您需要咨询产品信息，[uv_break]请按一。
    - 夜色如墨[uv_break]，山间小道蜿蜒曲折。李逍遥轻踏树梢，身形如同幽灵一般，迅捷无声[uv_break]。他手中的宝剑在月光下闪烁着寒芒，心中却是一片平静。
    - 亲爱的，你今天工作怎么样？[uv_break]有没有遇到什么开心的事。[uv_break]对了，晚上我们一起去那个新开的餐厅试试吧。
    
    ### 长句文本
    - 今天早晨，市中心的主要道路因突发事故造成了严重堵塞[uv_break]。请驾驶员朋友们注意绕行，并听从现场交警的指挥[uv_break]。天气预报显示，未来几天将有大范围降雨[uv_break]，请大家出门记得携带雨具，注意安全。另据报道，本次事故已造成数人受伤[uv_break]，目前相关部门正在积极处理事故现场[uv_break]，确保道路尽快恢复通畅。
    - 亲爱的朋友们，无论你现在处于什么样的境地，都不要放弃希望[uv_break]。每一个伟大的成功，都是从不懈的努力和坚定的信念中诞生的[uv_break]。人生的道路上充满了挑战和困难[uv_break]，但正是这些考验成就了我们的成长[uv_break]。记住，每一个今天的努力，都会成为明天成功的基石[uv_break]，坚持下去，你将看到光明的未来。
    - 很久很久以前，在一个宁静的小村庄里，住着一只名叫小花的可爱小猫咪[uv_break]。小花每天都喜欢在花园里玩耍，有一天，它遇到了一只迷路的小鸟[uv_break]。小花决定帮助小鸟找到回家的路[uv_break]，于是它们一起穿过森林，翻过小山丘，经历了许多冒险[uv_break]。最终，在小花的帮助下，小鸟终于回到了自己的家[uv_break]，它们成为了最好的朋友，从此过上了快乐的生活。
    - 您好，欢迎致电本公司客服中心。为了更好地服务您，请在听到提示音后选择所需服务[uv_break]。如果您需要咨询产品信息，[uv_break]请按一[uv_break]；如果您需要售后服务，请按二[uv_break]；如果您需要与人工客服交流，请按零[uv_break]。感谢您的来电，我们将竭诚为您服务，祝您生活愉快[uv_break]。如有任何疑问，请随时联系我们。
    - 夜色如墨[uv_break]，山间小道蜿蜒曲折。李逍遥轻踏树梢，身形如同幽灵一般，迅捷无声[uv_break]。他手中的宝剑在月光下闪烁着寒芒，心中却是一片平静[uv_break]。突然，一声清脆的剑鸣打破了夜的静谧[uv_break]，一个黑衣人出现在前方，冷笑道：‘李逍遥，你终于来了。’李逍遥目光如电，淡淡道：‘既然来了，就不打算走了[uv_break]。今天，我们就一决高下。",
    - 亲爱的，你今天工作怎么样？[uv_break]有没有遇到什么开心的事。[uv_break]对了，晚上我们一起去那个新开的餐厅试试吧[uv_break]。我听说那里的牛排特别好吃，而且还有你最喜欢的巧克力蛋糕[uv_break]。啊，今天真的好累，但想到等会儿可以见到你，心情就好多了[uv_break]。你还记得上次我们去的那个公园吗？[uv_break]那里的樱花真的好美，我还拍了好多照片呢。
    
            """)

if __name__ == "__main__":
    demo.launch()
