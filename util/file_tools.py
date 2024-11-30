from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, List, Any
import codecs
from functools import partial
import json
import csv

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class FileObjWrapper(object):
    """
    ファイル path のみで 目的の動作を行う file object を生成できるようにラップする
    e.g.
        codecs.open(file_name, mode='rb', encoding='utf-8') -> codecs.StreamReaderWriter
        ↓
        fe = partial(codecs.open, mode='rb', encoding='utf-8')
        ↓
        fe(file_name) -> codecs.StreamReaderWriter
    """

    def __init__(
            self,
            file_opener: Any = codecs.open,
            **kwargs,
    ):
        self._partial_encoder = partial(file_opener, **kwargs)

    def __call__(self, file_name: str) -> Any:
        return self._partial_encoder(file_name)


class FileHandler(metaclass=ABCMeta):
    def __init__(
            self,
            reader: FileObjWrapper,
            writer: FileObjWrapper,
    ):
        self._reader = reader
        self._writer = writer

    def read(self, file_name: str):
        try:
            with self._reader(file_name) as srw:
                return self.read_handling(srw)
        except FileNotFoundError:
            logger.error(f"{file_name} can not be found ...")
        except OSError as e:
            logger.error(f"OS error occurred trying to read {file_name}")
            logger.error(e)

    def write(self, data, file_name: str):
        try:
            with self._writer(file_name) as srw:
                self.write_handling(data, srw)
        except OSError as e:
            logger.error(f"OS error occurred trying to write {file_name}")
            logger.error(e)

    @abstractmethod
    def read_handling(self, srw):
        raise NotImplementedError

    @abstractmethod
    def write_handling(self, data, srw):
        raise NotImplementedError


''' ------ Tools for text file ------ '''


class JsonlHandler(FileHandler):
    def __init__(self):
        reader = FileObjWrapper(codecs.open, mode='r', encoding='utf-8')
        writer = FileObjWrapper(codecs.open, mode='w', encoding='utf-8')
        super().__init__(reader, writer)

    def read_handling(self, srw: codecs.StreamReaderWriter) -> List[Dict]:
        jsonl_data = [json.loads(l) for l in srw.readlines()]
        return jsonl_data

    def write_handling(self, data: List[Dict], srw: codecs.StreamReaderWriter):
        data_cl = [json.dumps(d, ensure_ascii=False) + "\n" for d in data]
        srw.writelines(data_cl)


class JsonHandler(FileHandler):
    def __init__(self):
        reader = FileObjWrapper(codecs.open, mode='r', encoding='utf-8')
        writer = FileObjWrapper(codecs.open, mode='w', encoding='utf-8')
        super().__init__(reader, writer)

    def read_handling(self, srw: codecs.StreamReaderWriter) -> Dict:
        return json.load(srw)

    def write_handling(self, data: Dict, srw: codecs.StreamReaderWriter):
        json.dump(data, srw, ensure_ascii=False, indent=2)


class CsvHandler(FileHandler):
    def __init__(self):
        reader = FileObjWrapper(codecs.open, mode='r', encoding='utf-8')
        writer = FileObjWrapper(codecs.open, mode='w', encoding='utf-8')
        super().__init__(reader, writer)

    def read_handling(self, srw: codecs.StreamReaderWriter) -> List[List[Optional[Any]]]:
        reader = csv.reader(srw)
        line_list = []
        for row in reader:
            line_list.append(row)
        return line_list

    def write_handling(self, data: List[List[Optional[Any]]], srw: codecs.StreamReaderWriter):
        data_cl = [[str(s) for s in raw] for raw in data]
        data_cl = [','.join(row) for row in data_cl]
        # 要素内の改行文字に関しては、そのまま保持させるためにエスケープしておく
        data_cl = [e.replace('\n', r'\n') for e in data_cl]
        data_cl = [row + '\n' for row in data_cl]
        srw.writelines(data_cl)


if __name__ == "__main__":
    """
    python -m util.file_tools
    """

    from logging import DEBUG, INFO, basicConfig

    basicConfig(level=INFO)

    ''' csv read and write '''
    csv_h = CsvHandler()
    csv_file = "./data/sample_write.csv"

    data_list = [
        ["hoge", 3],
        ["fuga", 4],
    ]

    csv_h.write(data_list, csv_file)

    csv_data = csv_h.read(csv_file)
    print(csv_data)

    """ test for jsonl """
    jl_h = JsonlHandler()
    jl_file = "./data/sample_write.jsonl"

    dict_list = [
        {
            "id": "seed_task_10",
            "instruction": "朝食として、卵を使わず、タンパク質を含み、だいたい700～1000キロカロリーのものはありますか？",
        },
        {
            "id": "seed_task_11",
            "instruction": "与えられたペアの関係は？",
        },
        {
            "id": "seed_task_12",
            "instruction": "次の各人物について、それぞれ1文で説明しなさい。",
        }
    ]

    jl_h.write(dict_list, file_name=jl_file)

    jl_data = jl_h.read(jl_file)
    print(jl_data)

    """ test for json """
    j_h = JsonHandler()
    j_file = "./data/sample_write.json"

    dict_data = {"data": dict_list}

    j_h.write(dict_data, file_name=j_file)

    j_data = j_h.read(j_file)
    print(j_data)
