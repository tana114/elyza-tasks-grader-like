import json
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from client.chain_base import ChainDirector, ConcreteChainBase

BASE_SYSTEM_PROMPT_FORMAT = (
    "You are an honest and competent AI assistant. Generate a #response# to the given #instruction#.\n\n"
    "Here are the requirements:\n" 
    "- Do not hallucinate in generating the #response#. Do not make up factual information.\n" 
    "- Think as carefully as you can and ensure that the #response# you generate are not bland and short.\n"
    "- All outputs must be in Japanese, not in English.\n\n" 
    "The user will give you a #seed_list# that containing several structured sets of #seed_id#,"
    " and #instruction#.\n"
    "Repeat process for the number of elements in the #seed_list# to generate a multiple structured list"
    " containing sets of #output_id# and #response#.\n\n"
)

HUMAN_PROMPT_FORMAT = (
    "#seed_list#: \n{seed_info}\n"
)


class AnswerData(BaseModel):
    """The answer record includes id and response."""
    output_id: int = Field(description="List number of the generated answer")
    response: str = Field(description="Generated answer to the instruction.")


class AnswerList(BaseModel):
    """List for contains AnswerData."""
    tasks: List[AnswerData] = Field(description="A list containing multiple AnswerData.")


class ElyzaTasksResponseGenerator(ConcreteChainBase):
    def __init__(
            self,
            chat_model: BaseChatModel,
            use_task_num_check: bool = False,
    ):
        """
        Parameters
        ----------
        chat_model
        use_task_num_check: bool
            生成するタスクの数が、与えたseedの要素数と一致しているかを確認し、一致しない場合は自動的に再試行する
        """
        super().__init__()
        self._llm = chat_model
        self._use_check = use_task_num_check

    @staticmethod
    def encode_few_shot_prompt(
            seed_instructions: List[Dict[str, str]]
    ) -> str:
        # Add key 'seed_id' and give serial number to value.
        seeds = [{"seed_id": i + 1, **d} for i, d in enumerate(seed_instructions)]

        few_shot_prompt = ""
        # convert dict type to string.
        for d in seeds:
            few_shot_prompt += json.dumps(d, indent=2, ensure_ascii=False)
            few_shot_prompt += "\n"

        return few_shot_prompt

    def _create_chain_director(
            self,
            director_config: Optional[Dict],
    ) -> ChainDirector:
        system_prompt = BASE_SYSTEM_PROMPT_FORMAT
        human_prompt = HUMAN_PROMPT_FORMAT

        return ChainDirector(
            chat_model=self._llm,
            system_prompt=system_prompt,
            human_prompt=human_prompt,  # {seed_info}
            struct_type=AnswerList,
        )

    def _invoke_handling(
            self,
            input: Dict[Literal["seed_info"], str],
            **kwargs
    ) -> List[Dict]:

        if self._use_check:
            res = self._inst_num_check(input, **kwargs)
            # print('num check:', res)
        else:
            res = self._chain_director.invoke(
                input,  # {seed_info}
                **kwargs,
            )

        # TaskList型を辞書型にdumpしたものを返す#
        task_list = [d.model_dump() for d in res.tasks]

        return task_list

    def _inst_num_check(
            self,
            input: Dict[Literal["seed_info"], str],
            **kwargs
    ):
        res = self._chain_director.invoke(input, **kwargs)
        if not res:
            return self._inst_num_check(input, **kwargs)

        # 文字列に含まれている要素の数を確認
        info_s = input['seed_info']
        # 文字列を個別のJSONオブジェクトに分割して要素数を数える
        objects_num = len(info_s.strip().split('\n}\n'))
        # 要素の数が想定した個数になっているかチェックするためにListの中身を確認
        tasks_list = res.tasks
        tasks_list = [e for e in tasks_list if e.response]  # 空文字は削除
        return self._inst_num_check(input, **kwargs) if len(tasks_list) != objects_num else res


if __name__ == "__main__":
    """
    python -m client.concrete.elyza_gen
    """

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    basicConfig(level=WARN)

    from model.groq_llm import GroqChatBase

    llm = GroqChatBase(
        model_name="llama-3.1-70b-versatile",
        temperature=0.8,
    )

    per_list = [
        {
            "input": "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。",
            "output": "1. 自分の仕事に対する興味を再発見するために、新しい技能や知識を学ぶこと。\n2. カレッジやセミナーなどで講演を聴くことで、仕事に対する新しいアイデアや視点を得ること。\n3. 仕事に対してストレスを感じている場合は、ストレスマネジメントのテクニックを学ぶこと。\n4. 仕事以外の楽しいことをすることで、ストレスを発散すること。\n5. 仕事に対して自己評価をすることで、自分がどのように進化しているのかを知ること。",
            "eval_aspect": "- 熱意を取り戻すのではなく、仕事の効率化・スキルアップのような文脈になっていたら1点減点\n- 出したアイデアが5つより多い、少ない場合は1点減点\n- 5つのアイデアのうち、内容が重複しているものがあれば1点減点\n\n"
        },
        {
            "input": "次に読むべきSFを10冊挙げて",
            "output": "それではSFというジャンルに大きな影響を与えた古典とされるSF小説を読んでみるのはどうでしょう？\n\n1. メアリー・シェリー著『フランケンシュタイン』（1818年）\n     - 死体の一部から怪物を作り出す科学者を描いたこの古典小説は、史上初のSF小説のひとつとされています。\n2. H.G.ウェルズ著『タイムマシン』（1895年）\n    - 未来に旅立ったタイムトラベラーが、平和なエロイ族と残忍なモーロック族に分断された世界を発見するこの小説は、SFジャンルの先駆的作品です。\n3. H.G.ウェルズ著『宇宙戦争』（1898年）\n    - 火星人の地球侵略を描いたこの小説は、異星人の生命体の生き生きとした描写と、帝国主義と技術進歩に対する論評で知られるウェルズのもうひとつの代表作です。\n4. フランク・ハーバート著『デューン 砂の惑星』（1965年）\n    - 人類が銀河系の他の惑星を植民地化した遠い未来を舞台に、ポール・アトレイデスという青年が砂漠の惑星アラキス（別名デューン）の政治、宗教、生態系の複雑な網に巻き込まれていく姿を描いた小説です。\n5. アイザック・アシモフ著『ファウンデーション』（1951年）\n    - 人類が銀河系に広がり、崩壊の危機に直面している遠い未来が舞台で。ハリ・セルドンという数学者が、未来を予測するために心理歴史学という新しい分野を開発し、差し迫った暗黒時代に知識と文化を守るための財団を設立する姿を描いています。\n6. ウィリアム・ギブソン著『ニューロマンサー』（1984年）\n    - この小説の舞台は、仮想現実とサイバースペースが日常生活の不可欠な一部となった近未来の世界です。この小説は、ケースという名の落ちぶれたコンピューター・ハッカーが、企業が巨大な権力を振るう世界でスパイ行為と裏切りの危険なゲームに巻き込まれていく様を描いています。\n7. フィリップ・K・ディック著『アンドロイドは電気羊の夢を見るか』（1968年）\n    - ネクサス6と呼ばれる高度なアンドロイドが人類に奉仕するために作られた黙示録後の世界を舞台に、悪質なアンドロイドを狩る任務を負った賞金稼ぎの物語を通して、人間性、共感、そして生きていることの意味を探るSF小説です。\n8. アーシュラ・K・ル＝グウィン著『闇の左手』（1969年）\n    - この小説の舞台はゲーセンと呼ばれる遠い惑星で、住民は性別がなく両性愛者です。人間の外交官がこの惑星の複雑な政治状況をナビゲートしようとする中で、セクシュアリティ、ジェンダー、権力のテーマを探求する話です。\n9. スタニスワフ・レム著『ソラリス』（1961年）\n    - 惑星ソラリスを周回する宇宙ステーションを舞台にした小説で、クリス・ケルヴィンという心理学者が奇妙な現象を調査するためにやってきます。謎めいた異星人との出会いを通して、意識、人間性、科学的理解の限界といったテーマの小説です。\n10. リチャード・K・モーガン著『オルタード・カーボン』（2002年）\n    - 人間の意識が身体間で移動できるようになった未来を舞台に、元兵士のタケシ・コバックスが殺人事件を捜査し、社会秩序全体を脅かす陰謀を暴いていく物語です。\n",
            "eval_aspect": "- 「読むべき」とあるように小説であるべきで、アバターなどのSF映画だと -2点\n- 実在しない架空の小説の場合 -2点\n- ドラゴンボールなどの漫画の場合も -2点\n- 10冊ではない場合、-2点\n- 作品名のみの記載で、作品を薦める記述がない場合は-1点\n"
        },
        {
            "input": "次の文章を読んで、その人がどの程度怒っているか、1〜10の尺度で評価してください。(1＝怒っていない、10＝非常に怒っている)。\n\n1. またテストで赤点か。君は全く...\n2. テストで赤点？今回は難しかったね。",
            "output": "それぞれの文章について、怒りの程度を10段階で評価します。\n\n1. 8 (怒っている): 話者は「君は全く...」というように呆れながら怒りを示しています。\n2. 2 (あまり怒っていない): 話者は「今回は難しかったね。」というようにテストで赤点を取ってしまったことはテストが難しかったためであるとし、怒りを示していません。\n",
            "eval_aspect": "- 1問目は5~9の間であれば正解\n- 2問目は1~4の間であれば正解\n- どちらかの問いが不正解なら-2点\n- 2つ正解しているが、理由の説明がない場合は4点"
        },
    ]

    seeds = []
    for d in per_list:
        seed = dict(
            instruction=d['input'],
        )
        seeds.append(seed)


    gen = ElyzaTasksResponseGenerator(
        chat_model=llm,
        use_task_num_check=True,
    )

    # few-shotの文字列を作成
    few_shot = gen.encode_few_shot_prompt(seeds)
    print(few_shot)

    inst = {
        "seed_info": few_shot,
    }


    result = gen(inst)
    # print(result)

    for r in result:
        print(r['output_id'])
        print(r['response'])

    """
    1
    仕事の熱意を取り戻すための5つのアイデアは以下の通りです。
    1. 目標設定：自分に合った目標を設定し、それを達成するために必要なステップを計画する。
    2. 時間管理：効率的な時間管理を行い、仕事に集中できる時間を確保する。
    3. 健康管理：健康的な生活習慣を実践し、仕事に必要なエネルギーを持続させる。
    4. 学習と発展：新しいスキルや知識を身につけることで、仕事に対する興味を維持する。
    5. 認識と評価：自分の仕事に対する評価を認識し、それを基に改善を図る。


    2
    次に読むべきSFを10冊挙げると、以下の作品が挙げられます。
    1. 『2001年宇宙の旅』 - アーサー・C・クラーク
    2. 『宇宙戦争』 - ヒューイ・H・ローチ
    3. 『未来世界』 - H・G・ウェルズ
    4. 『機械と幽鬼』 - テオドール・スターゲン
    5. 『ロボットと帝国』 - アイザック・アシモフ
    6. 『ファウンデーション』シリーズ - アイザック・アシモフ
    7. 『太陽の簒奪者』 - アイザック・アシモフ
    8. 『宇宙の果ての島』 - アーサー・C・クラーク
    9. 『2001年宇宙の旅（続）』 - アーサー・C・クラーク
    10. 『2001年宇宙の旅（完結）』 - アーサー・C・クラーク


    3
    文章を読んで、その人がどの程度怒っているかを評価すると、
    1. 10点
    2. 1点

    となります。
    """