## BLEU 

### (Bilingual Evaluation Understudy，双语评估辅助工具)

<table style="border-style:none">
    <tbody>
        <tr>
            <td><img src=./images/bleu-1.jpg width="500"></td>
            <td><img src=./images/bleu-2.jpg width="500"></td>
        </tr>
    </tbody>
</table>
<!-- <div align=center><img src=./images/bleu-1.jpg width="200" style={display:inline-block}></div>
</br>
<div align=center><img src=./images/bleu-2.jpg width="400"></div>
</br>
 -->
 
```python
from nltk.translate.bleu_score import sentence_bleu
def sentence_bleu(
    references,
    hypothesis,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
)

references = [["there", "is", "a", "cat", "on", "the", "table"]]
hypothesis = [ "a", "cat", "is", "on", "the", "table"]
bleu_1 = sentence_bleu(references, hypothesis, weights=(1, 0, 0, 0)) # exp(-1/6)
bleu_2 = sentence_bleu(references, hypothesis, weights=(0, 1, 0, 0)) # exp(-1/6)*3/5
bleu_4_avg = sentence_bleu(references, hypothesis) # 6.433932628423997e-78 --> 0 !!!

hypothesis = ["a", "a", "a", "a", "a", "a", "a"]
bleu_1 =  sentence_bleu(references, hypothesis, weights=(1, 0, 0, 0)) # 1/7 
```

