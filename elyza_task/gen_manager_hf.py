import os.path
from typing import List, Dict, Tuple, Optional, final, Union, Any, Literal
from tqdm.auto import tqdm

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset

from util.file_tools import JsonlHandler

BASE_SYSTEM_PROMPT = (
    "あなたは誠実で優秀なアシスタントです。"
)


def raw_instruction(instruction: str) -> str:
    return instruction


def llmjp3_instruction(instruction: str) -> str:
    role = f"<s> {BASE_SYSTEM_PROMPT}\n\n"
    return f"{role}### 指示: \n{instruction}\n\n### 応答: "


def gamma2_instruction(instruction: str) -> str:
    role = f"<bos><start_of_turn>model\n{BASE_SYSTEM_PROMPT}<end_of_turn>\n"
    return f"{role}<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


class ElyzaTasksHfGenerateManager(object):
    def __init__(
            self,
            hf_model: PreTrainedModel,
            hf_tokenizer: PreTrainedTokenizer,
            template_type: Optional[Literal['gamma2', 'llmjp3']] = None,
            tasks_file_path: Optional[str] = None,
            **gen_options,
    ):
        """

        Parameters
        ----------
        hf_model
        hf_tokenizer
        template_type
        tasks_file_path: str
            <評価基準データ(*.jsonl)>
            以下のようにElayze-tasks-100が 'input' 、'output'、'eval_aspect'に格納されたデータ
            この生成クラスには最低限 'input' keyがあれば良い

            {"input": "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。", "output": "1. 自分の仕事に対す...いるのかを知ること。", "eval_aspect": "- 熱意を取り...ば1点減点\n\n"}
            {"input": "クマが海辺に行ってアザラシと友達に...", "output": "クマは、...永遠に忘れない。", "eval_aspect": "- クマが海辺に行く\n- クマとアザラシが...て淡白な場合: -1点"}
                ...

        gen_options: dict

            dict(
                max_new_tokens=100,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                repetition_penalty=1.05,
            )

        """
        self._model = hf_model
        self._tokenizer = hf_tokenizer
        self._tasks_file = tasks_file_path
        self._options = gen_options
        template_func = {
            'gamma2': gamma2_instruction,
            'llmjp3': llmjp3_instruction,
        }
        # self._template = template_func.get(template_type, raw_instruction) if template_type else raw_instruction
        self._template = template_func[template_type] if template_type else raw_instruction

        self._jsonl_handler = JsonlHandler()  # tool for *.jesonl files.

        if self._tasks_file:
            '''
            Elayze-tasks-100が"input"、"output"、"eval_aspect"に格納されたデータ
            '''
            self._datasets = load_dataset('json', data_files=self._tasks_file, split='train')
        else:
            ''' "elyza/ELYZA-Tasks-100"にはSplit:testしかないみたい
             {'input': dtype='string','output': dtype='string','eval_aspect': dtype='string'} 
             '''
            self._datasets = load_dataset("elyza/ELYZA-Tasks-100", split='test')

        # print(self._datasets.features.keys())
        assert 'input' in list(self._datasets.features.keys()), "An 'input' key is required in the task file."
        assert len(self._datasets) <= 100, f"Fewer than 100 answers are generated on elyza tasks ({len(self._datasets)})."

    def __call__(
            self,
            gen_jsonl_path: str,
    ) -> None:
        return self.file_handling(gen_jsonl_path)

    def file_handling(
            self,
            gen_jsonl_path: str,
    ) -> None:
        """
        Parameters
        ----------
        gen_jsonl_path:str 出力データ
            <生成データ(*.jsonl)>
        """

        # output directory check
        out_dir = os.path.dirname(gen_jsonl_path)
        os.makedirs(out_dir, exist_ok=True)

        res_list = []
        for i, d in tqdm(enumerate(self._datasets)):
            # 生成工程において入力データの 'output' は不要なので削除しておく
            d.pop('output', None)

            # task_id = d.get('task_id', i + 1)
            task_id = d.get('task_id', i)
            input = d['input']
            t_input = self._template(input)
            # print(input)
            # print(task_id, input)

            tokenized_input = self._tokenizer.encode(
                t_input,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(
                    tokenized_input,
                    **self._options
                )[0]

            # 入力プロンプトの長さを利用して、生成された部分のみを切り出します
            input_length = len(tokenized_input[0])
            # input_length = len(tokenizer.encode(input, add_special_tokens=False))
            assistant_response = self._tokenizer.decode(output[input_length:], skip_special_tokens=True)

            res = {'task_id': task_id, 'output': assistant_response, **d}
            # print(res)
            res_list.append(res)

            self._jsonl_handler.write(res_list, gen_jsonl_path)


if __name__ == "__main__":
    """
    python -m elyza_task.gen_manager_hf
    """
    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    basicConfig(level=WARN)

    hf_model_name = 'llm-jp/llm-jp-3-3.7b'
    # hf_model_name = 'google/gemma-2-2b-jpn-it'

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, device_map="auto", torch_dtype=torch.bfloat16)

    options = dict(
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.05,
    )

    # elyza_jsonl = './data/test/elyza100_jp.jsonl'
    gen_jsonl = './data/output/llm-jp-3-3.7b_l512-hf_01.jsonl'

    gm = ElyzaTasksHfGenerateManager(
        hf_model=model,
        hf_tokenizer=tokenizer,
        template_type='llmjp3',
        # template_type='gamma2',
        # tasks_file_path=elyza_jsonl,
        **options,
    )

    gm(gen_jsonl_path=gen_jsonl)
