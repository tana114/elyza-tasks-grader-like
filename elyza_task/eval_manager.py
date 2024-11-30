import os.path
from typing import List, Dict, Tuple, Optional, final, Union, Any

from tqdm.auto import tqdm
import numpy as np
from langchain_core.language_models import BaseChatModel

from datasets import load_dataset

from client.concrete.elyza_grader import ElyzaTasksGrader
from util.file_tools import JsonlHandler


class ElyzaTasksEvaluateManager(object):
    def __init__(
            self,
            main_llm: BaseChatModel,
            eval_batch_size: int = 10,
            tasks_file_path: Optional[str] = None,
    ):
        """

        """
        self._model = main_llm
        self._eval = ElyzaTasksGrader(
            chat_model=self._model,
            use_task_num_check=True,
        )
        self._batch_size = eval_batch_size
        self._jsonl_handler = JsonlHandler()  # tool for *.jsonl files.
        self._tasks_file = tasks_file_path

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
        assert 'output' in list(self._datasets.features.keys()), "An 'output' key is required in the task file."
        assert 'eval_aspect' in list(
            self._datasets.features.keys()), "An 'eval_aspect' key is required in the task file."

    def grade_elyza100(self, task_seeds: List[Dict]) -> List[Dict]:
        # Elyza100Graderの仕様に合わせて辞書を用意する
        inst_seeds = [
            dict(
                instruction=d['input'],
                correct_answer=d['output'],
                eval_aspect=d['eval_aspect'],
                valuation_answer=d['response'],
            ) for d in task_seeds]

        # few-shotの文字列を作成
        few_shot = self._eval.encode_few_shot_prompt(inst_seeds)

        instruction = {
            "seed_info": few_shot,
        }

        score_list = []
        results = self._eval(instruction)

        assert len(results) == len(task_seeds), f"{results}, {task_seeds}"

        for r, s in zip(results, task_seeds):
            score = {'score': r['rating_score'], **s}
            score_list.append(score)

        return score_list

    def __call__(
            self,
            eval_jsonl_path: str,
    ) -> float:
        return self.file_handling(eval_jsonl_path)

    def file_handling(
            self,
            target_jsonl_path: str,
    ) -> float:
        """
        Parameters
        ----------
        target_jsonl_path:str
        <評価対象データ(*.jsonl)>
        'task_id' と 'output' が必要

        Returns
        -------
        score: float
        """

        # output directory check
        out_dir = os.path.dirname(target_jsonl_path)
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.basename(target_jsonl_path)
        output_file = out_dir + f"/eval_{base_name}"

        # 'output' and 'task_id'
        target_list = self._jsonl_handler.read(target_jsonl_path)

        # 'input', 'output' and 'eval_aspect'
        tasks_list = list(self._datasets)

        msg = f"The number of criteria data ({len(tasks_list)}) does not match the evaluation data ({len(target_list)})."
        assert len(target_list) == len(tasks_list), msg

        criteria_list = [
            dict(
                input=task['input'],
                output=task['output'],
                eval_aspect=task['eval_aspect'],
                response=target['output'],  # target['output']が評価対象となる
            ) for task, target in zip(tasks_list, target_list)]

        def batch_processor(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield self.grade_elyza100(data[i:i + batch_size])

        object_list = []
        for processed in tqdm(batch_processor(criteria_list, self._batch_size)):
            object_list.extend(processed)
            self._jsonl_handler.write(object_list, output_file)

        all_score = np.asarray([int(d['score']) for d in object_list])

        return all_score.mean()


if __name__ == "__main__":
    """
    python -m elyza_task.eval_manager
    """
    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    basicConfig(level=WARN)

    from model.groq_llm import GroqChatBase

    llm_main = GroqChatBase(
        model_name="llama-3.1-70b-versatile",
        requests_per_second=0.32,
        # max_tokens=2048,
        temperature=0.8,
    )

    # tasks_files_path = './data/test/elyza100_jp.jsonl'

    em = ElyzaTasksEvaluateManager(
        main_llm=llm_main,
        eval_batch_size=5,
        # eval_batch_size = 10,
        # tasks_file_path=tasks_files_path,
    )

    target_json_path = './data/output/llm-jp-3-13b-inst-1_00-4bit_gen.jsonl'

    score = em(eval_jsonl_path=target_json_path)
    print(score)
