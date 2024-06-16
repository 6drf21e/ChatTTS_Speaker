[English README](README.en.md) | [中文简体](README.md)

# 🥇 ChatTTS Speaker Leaderboard

## 🎤 :ChatTTS稳定音色评分与音色打标（实验性）

本项目基于 [ChatTTS](https://github.com/2noise/ChatTTS)

评估基于通义实验室的 [eres2netv2_sv_zh-cn](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)。

欢迎下载试听音色！本项目已开源：[ChatTTS_Speaker](https://github.com/6drf21e/ChatTTS_Speaker) 欢迎 PR。

## 马上体验

| 说明                | 链接                                                    |
|-------------------|-------------------------------------------------------| 
| **ModelScop(国内)** | https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker |
| **HuggingFace**   | https://huggingface.co/spaces/taa/ChatTTS_Speaker     |

## 参数解释

- **rank_long**: 长句文本的音色稳定性评分。
- **rank_multi**: 多句文本的音色稳定性评分。
- **rank_single**: 单句文本的音色稳定性评分。

这三个参数用于衡量音色在不同样本一致性，数值越高表示音色越稳定。

- **score**: 音色性别、年龄、特征的可能性，数值越高表示越准确。
- **gender age feature**: 音色的性别、年龄、特征。（特征准确度不高，仅供参考）

## 如何下载试听音色

1. 点选一个音色 seed_id 单元格。
2. 点击最下方的 **Download .pt File** 按钮，即可下载对应的 .pt 文件。

## 评估代码

稳定性评估代码详见：https://github.com/2noise/ChatTTS/pull/317

## FAQ

- **Q**: 怎么使用 .pt 文件？
- **A**: 可以直接在一些项目中载入使用，例如：[ChatTTS_colab](https://github.com/6drf21e/ChatTTS_colab)。 也可以使用类似代码载入：

```python
spk = torch.load(PT_FILE_PATH)
params_infer_code = {
    'spk_emb': spk,
}
```

- **Q**: 为什么有的音色打分高但是很难听？
- **A**: 评分只是衡量音色的稳定性，不代表音色的好坏。可以根据自己的需求选择合适的音色。例如：如果一个沙哑且结巴的音色一直很稳定，那么它的评分就会很高但是很难听。


- **Q**: 我使用 seed_id 去生成音频，但是生成的音频不稳定？
- **A**: seed_id 只是一个参考 ID，不同的环境下音色不一定一致。还是推荐使用 .pt 文件载入音色。


- **Q**: 音色打标准确吗？
- **A**: 当前第一批测试的音色有 2000 条，根据声纹相似性简单打标，准确度一般（特别是特征一项），仅供参考。如果大家有更好的标注方法，欢迎
  PR。

## 相关项目
- [ChatTTS](https://github.com/2noise/ChatTTS)
- [eres2netv2_sv_zh-cn](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)
- [3D-Speaker](https://github.com/modelscope/3D-Speaker)

## 贡献

欢迎大家贡献代码和音色！如果有任何问题或建议，请提交 issue 或 pull request。

