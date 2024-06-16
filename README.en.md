[English README](README.en.md) | [中文简体](README.md)

# 🥇 ChatTTS Speaker Leaderboard

## 🎤 ChatTTS Stable Speaker Evaluation and Labeling (Experimental)

This project is based on [ChatTTS](https://github.com/2noise/ChatTTS).

The evaluation is based on Tongyi Laboratory's [eres2netv2_sv_zh-cn](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary).

Feel free to download and listen to the voices! This project is open source: [ChatTTS_Speaker](https://github.com/6drf21e/ChatTTS_Speaker). Contributions are welcome.

## Try It Now

| Description       | Link                                                    |
|-------------------|---------------------------------------------------------| 
| **ModelScope (China)** | https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker |
| **HuggingFace**   | https://huggingface.co/spaces/taa/ChatTTS_Speaker       |

## Parameter Descriptions

- **rank_long**: Stability score for long sentence text.
- **rank_multi**: Stability score for multi-sentence text.
- **rank_single**: Stability score for single sentence text.

These three parameters are used to measure the consistency of the voice across different samples. The higher the value, the more stable the voice.

- **score**: Likelihood of the voice's gender, age, and characteristics. The higher the value, the more accurate it is.
- **gender age feature**: The gender, age, and characteristics of the voice. (Feature accuracy is low, for reference only)

## How to Download and Listen to Voices

1. Click on a voice seed_id cell.
2. Click the **Download .pt File** button at the bottom to download the corresponding .pt file.

## Evaluation Code

The stability evaluation code can be found at: https://github.com/2noise/ChatTTS/pull/317

## FAQ

- **Q**: How to use the .pt file?
- **A**: You can directly load it in some projects, such as: [ChatTTS_colab](https://github.com/6drf21e/ChatTTS_colab). You can also load it with similar code:

```python
spk = torch.load(PT_FILE_PATH)
params_infer_code = {
    'spk_emb': spk,
}
```

- **Q**: Why do some voices have high scores but sound bad?
- **A**: The score only measures the stability of the voice, not its quality. You can choose the appropriate voice according to your needs. For example, if a hoarse and stuttering voice is very stable, its score will be high but it will sound bad.

- **Q**: I used the seed_id to generate audio, but the generated audio is not stable?
- **A**: The seed_id is just a reference ID, and the voice may not be consistent in different environments. It is still recommended to use the .pt file to load the voice.

- **Q**: Is the voice labeling accurate?
- **A**: The first batch of test voices includes 2000 samples, simply labeled based on voiceprint similarity, with average accuracy (especially for the feature item), for reference only. If you have better labeling methods, please submit a PR.

## Related Projects
- [ChatTTS](https://github.com/2noise/ChatTTS)
- [eres2netv2_sv_zh-cn](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)
- [3D-Speaker](https://github.com/modelscope/3D-Speaker)

## Contribution

Contributions of code and voices are welcome! If you have any questions or suggestions, please submit an issue or pull request.
